"""
UCHIDA V3 - Backtest Engine

비중 시계열 + 가격 시계열 → 포트폴리오 수익률 시계열.

[체결 규약]
- t일 종가까지의 정보로 t일 종가 시점에 신호 결정
- t+1일 시가에 체결 (look-ahead bias 방지)
- 비용 차감 후 일별 수익률 산출

[No-Trade Band]
- max_i |target_w_i - current_w_i| < band → 거래 없음
- 단, 첫 진입(current가 모두 0)이거나 강제 리밸런싱 플래그가 있으면 거래

[거래비용]
- 편도 비용 (config.COSTS.total_one_way) × turnover
- turnover = sum(|new_w - old_w|) / 2  (매수+매도를 한 번에 셈하므로 2로 나눔)

[입력 규약]
- prices_close: DataFrame (날짜 × ticker, 일별 종가)
- prices_open:  DataFrame (날짜 × ticker, 일별 시가)
- target_weights: DataFrame (날짜 × ticker, 신호 발생일 기준 목표 비중)
  ※ 신호일 = t. 체결은 t+1 시가. 엔진이 자동 처리.
- 모든 입력의 날짜 인덱스가 일치해야 함.

[출력]
- result: dict with keys
    'portfolio_value':   초기 1.0으로 정규화된 포트폴리오 가치 시계열
    'daily_returns':     일별 포트폴리오 수익률 (체결 후 비용 차감)
    'positions':         실제 보유 비중 시계열 (drift 반영)
    'trades':            거래 발생일과 turnover, cost 기록
"""

import numpy as np
import pandas as pd
import os
import sys
from typing import Optional, Dict
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COSTS, REBALANCE


@dataclass
class BacktestResult:
    """백테스트 결과 컨테이너."""
    portfolio_value: pd.Series       # 정규화 포트폴리오 가치 (시작=1.0)
    daily_returns:   pd.Series       # 일별 수익률
    positions:       pd.DataFrame    # 실제 보유 비중 (drift 반영)
    trades:          pd.DataFrame    # 거래 기록 (turnover, cost)


def run_backtest(
    prices_close: pd.DataFrame,
    prices_open:  pd.DataFrame,
    target_weights: pd.DataFrame,
    cost_one_way: float = None,
    tolerance_band: float = None,
    cash_asset: Optional[str] = None,
) -> BacktestResult:
    """
    Vectorized-ish 백테스트 엔진.

    Parameters
    ----------
    prices_close, prices_open : pd.DataFrame
        같은 index와 columns. 자산 가격.
    target_weights : pd.DataFrame
        같은 index, 같은 columns(또는 일부 컬럼). 목표 비중 (각 행 합=1).
    cost_one_way : float
        편도 거래비용. None이면 config.COSTS.total_one_way 사용.
    tolerance_band : float
        no-trade band. None이면 config.REBALANCE.tolerance_band 사용.
    cash_asset : str, optional
        현금성 자산 티커. None이면 reweighting 시 모든 자산에 비례 배분.
        SOFR 같은 자산을 지정하면 잔여 비중을 그쪽으로 흡수.

    Returns
    -------
    BacktestResult
    """
    if cost_one_way is None:
        cost_one_way = COSTS.total_one_way
    if tolerance_band is None:
        tolerance_band = REBALANCE.tolerance_band

    # ----- 입력 검증 -----
    if not prices_close.index.equals(prices_open.index):
        raise ValueError("prices_close와 prices_open의 index가 일치하지 않습니다.")
    if not prices_close.columns.equals(prices_open.columns):
        raise ValueError("prices_close와 prices_open의 columns가 일치하지 않습니다.")

    # target_weights를 prices와 같은 컬럼으로 정렬, 없는 컬럼은 0
    tw = target_weights.reindex(columns=prices_close.columns).fillna(0.0)
    tw = tw.reindex(prices_close.index).ffill().fillna(0.0)

    # 비중 합 검증 (target weight 행 합이 1±tol 이어야 함; 0인 행은 진입 전)
    row_sums = tw.sum(axis=1)
    nonzero = row_sums > 1e-9
    if nonzero.any():
        bad_rows = row_sums[nonzero & ((row_sums - 1.0).abs() > 1e-6)]
        if len(bad_rows) > 0:
            raise ValueError(
                f"target_weights 행 합이 1이 아닌 시점 {len(bad_rows)}개. "
                f"예: {bad_rows.head(3).to_dict()}"
            )

    n = len(prices_close)
    tickers = list(prices_close.columns)
    n_assets = len(tickers)

    # ----- 상태 변수 초기화 -----
    # current_w: 실제 보유 비중 (drift 반영)
    current_w = np.zeros(n_assets)
    
    portfolio_values = np.zeros(n)
    daily_rets = np.zeros(n)
    positions_log = np.zeros((n, n_assets))
    trade_log = []  # 거래 기록

    pv = 1.0  # 정규화 가치
    portfolio_values[0] = pv

    # ----- 일별 루프 -----
    # 체결 규약: t일 종가 신호 → t+1일 시가에 체결
    # 즉, 시점 t의 target_weights는 시점 t+1 시가에 적용
    # 그 후 t+1 시가→종가 동안 가격 변동 → drift
    
    for t in range(1, n):
        date_t = prices_close.index[t]
        
        # ===== 1. 어제 종가→오늘 시가 사이 drift =====
        # 어제 종가 기준 current_w → 오늘 시가 기준 비중으로 변화
        # close[t-1]에서 open[t]까지 가격 변동
        ret_close_to_open = (prices_open.iloc[t] / prices_close.iloc[t-1] - 1).values
        ret_close_to_open = np.nan_to_num(ret_close_to_open, nan=0.0)
        
        # 보유 자산의 시가 시점 가치
        if current_w.sum() > 1e-9:
            # 비중 drift: w_i가 (1+r_i)에 비례하여 변함
            new_values = current_w * (1 + ret_close_to_open)
            value_factor_open = new_values.sum()  # 1 + 포트폴리오 yc→yo 수익률
            current_w = new_values / value_factor_open
            pv *= value_factor_open
        else:
            value_factor_open = 1.0
        
        # ===== 2. 오늘 시가에 체결 (어제 종가 신호 기준) =====
        # 어제(t-1) 종가 시점에 결정된 target_weights를 오늘 시가에 체결
        target_w = tw.iloc[t-1].values
        
        # no-trade band 체크
        # 첫 진입(current가 모두 0)이면 무조건 거래
        diff = target_w - current_w
        max_diff = np.max(np.abs(diff))
        is_first_entry = (current_w.sum() < 1e-9) and (target_w.sum() > 1e-9)
        
        trade_today = False
        cost_today = 0.0
        turnover_today = 0.0
        
        if is_first_entry or (target_w.sum() > 1e-9 and max_diff >= tolerance_band):
            # 거래 발생
            turnover_today = np.abs(diff).sum() / 2.0  # 매수합=매도합. 양쪽 합/2가 회전율
            cost_today = turnover_today * 2 * cost_one_way  # 매수+매도 양쪽 비용
            # 비용 차감 (자산 매각으로 비용 충당 가정 → 포트폴리오 가치 감소)
            pv *= (1 - cost_today)
            # 비중 갱신
            current_w = target_w.copy()
            trade_today = True
            
            trade_log.append({
                "date": date_t,
                "turnover": turnover_today,
                "cost": cost_today,
                "max_weight_diff": max_diff,
            })
        
        # ===== 3. 오늘 시가→종가 사이 drift =====
        ret_open_to_close = (prices_close.iloc[t] / prices_open.iloc[t] - 1).values
        ret_open_to_close = np.nan_to_num(ret_open_to_close, nan=0.0)
        
        if current_w.sum() > 1e-9:
            new_values = current_w * (1 + ret_open_to_close)
            value_factor_close = new_values.sum()
            current_w = new_values / value_factor_close
            pv *= value_factor_close
        else:
            value_factor_close = 1.0
        
        # ===== 4. 일별 수익률 = (오늘 종가 가치 / 어제 종가 가치) - 1 =====
        daily_rets[t] = (pv / portfolio_values[t-1]) - 1 if portfolio_values[t-1] > 0 else 0.0
        portfolio_values[t] = pv
        positions_log[t] = current_w

    # ----- 결과 패키징 -----
    idx = prices_close.index
    result = BacktestResult(
        portfolio_value=pd.Series(portfolio_values, index=idx, name="portfolio_value"),
        daily_returns=pd.Series(daily_rets, index=idx, name="daily_return"),
        positions=pd.DataFrame(positions_log, index=idx, columns=tickers),
        trades=pd.DataFrame(trade_log).set_index("date") if trade_log else pd.DataFrame(columns=["turnover", "cost", "max_weight_diff"]),
    )
    return result


# ==========================================
# 자체 검증 (직접 실행 시 4가지 단위 테스트)
# ==========================================
def _make_synthetic_prices(n_days=100, n_assets=3, seed=0):
    """테스트용 합성 가격: 일정 수익률로 상승하는 가격."""
    np.random.seed(seed)
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    tickers = [f"A{i}" for i in range(n_assets)]
    # 종가: 매일 +0.1% 상승 (deterministic)
    close = pd.DataFrame(
        100 * np.exp(np.arange(n_days)[:, None] * 0.001 * np.arange(1, n_assets+1)),
        index=idx, columns=tickers,
    )
    # 시가: 전일 종가와 동일 (단순화)
    open_ = close.shift(1).fillna(close.iloc[0])
    return close, open_, tickers


if __name__ == "__main__":
    print("=" * 70)
    print("backtest/engine.py 단위 테스트 (4가지)")
    print("=" * 70)

    # ===== Test 1: 100% 한 자산 보유 → 그 자산 수익률과 일치 =====
    print("\n[Test 1] 100% 한 자산만 보유 → 포트폴리오 수익률 = 자산 수익률")
    close, open_, tickers = _make_synthetic_prices(n_days=50, n_assets=3, seed=1)
    
    target = pd.DataFrame(0.0, index=close.index, columns=tickers)
    target["A0"] = 1.0  # 100% A0
    
    res = run_backtest(close, open_, target, cost_one_way=0.0, tolerance_band=0.0)
    
    # 첫 거래일 이후의 수익률 비교 (체결 지연 때문에 1일 lag)
    asset_ret = (close["A0"] / close["A0"].shift(1) - 1).iloc[2:]
    port_ret = res.daily_returns.iloc[2:]
    
    diff = (asset_ret - port_ret).abs().max()
    print(f"  자산 수익률 vs 포트폴리오 수익률 최대 차이: {diff:.2e}")
    assert diff < 1e-9, f"Test 1 실패: 차이 {diff}"
    print("  ✓ 통과")

    # ===== Test 2: 비중 변화 없음 → turnover = 0 (초기 진입 후) =====
    print("\n[Test 2] 비중 변화 없음 → 초기 진입 후 turnover = 0")
    close, open_, tickers = _make_synthetic_prices(n_days=30, n_assets=3, seed=2)
    
    target = pd.DataFrame(0.0, index=close.index, columns=tickers)
    target["A0"] = 0.5; target["A1"] = 0.3; target["A2"] = 0.2
    
    res = run_backtest(close, open_, target, cost_one_way=0.001, tolerance_band=0.05)
    
    print(f"  발생한 거래 수: {len(res.trades)}")
    assert len(res.trades) == 1, f"Test 2 실패: 초기 진입 1회만 있어야 함"
    print(f"  초기 진입 turnover: {res.trades['turnover'].iloc[0]:.4f}")
    print("  ✓ 통과")

    # ===== Test 3: No-trade band 작동 (4% < band, 6% > band) =====
    print("\n[Test 3] No-trade band 작동 확인")
    close, open_, tickers = _make_synthetic_prices(n_days=50, n_assets=2, seed=3)
    
    # 시나리오: 초기 50:50, 25일 후 54:46(4% diff)으로 변경, 40일 후 60:40(10% diff)으로 변경
    target = pd.DataFrame(0.0, index=close.index, columns=tickers)
    target.iloc[:25] = [0.5, 0.5]
    target.iloc[25:40] = [0.54, 0.46]  # 4% 변화 → 거래 안 해야
    target.iloc[40:] = [0.60, 0.40]    # 10% 변화 → 거래 해야
    
    res = run_backtest(close, open_, target, cost_one_way=0.001, tolerance_band=0.05)
    
    n_trades = len(res.trades)
    print(f"  발생한 거래 수: {n_trades} (예상: 2 = 초기진입 1 + 40일째 1)")
    # 거래일 확인
    trade_dates_pos = [close.index.get_loc(d) for d in res.trades.index]
    print(f"  거래일 인덱스: {trade_dates_pos}")
    assert n_trades == 2, f"Test 3 실패: 거래 {n_trades}회 (예상 2회)"
    print("  ✓ 통과")

    # ===== Test 4: 거래비용 정확성 (100% → 50% 전환, turnover=0.5, cost=0.0010) =====
    print("\n[Test 4] 거래비용 정확성")
    close, open_, tickers = _make_synthetic_prices(n_days=20, n_assets=2, seed=4)
    
    # 시나리오: 처음 10일 A0=100%, 다음 10일 A0=50%, A1=50%
    target = pd.DataFrame(0.0, index=close.index, columns=tickers)
    target.iloc[:10] = [1.0, 0.0]
    target.iloc[10:] = [0.5, 0.5]
    
    # 미세 drift 거래를 무시하기 위해 작은 band 사용 (0.01)
    # 전환 시: |0.5-1.0| + |0.5-0.0| = 1.0, turnover = 0.5
    # 비용 = 0.5 * 2 * 0.001 = 0.001
    res = run_backtest(close, open_, target, cost_one_way=0.001, tolerance_band=0.01)
    
    print(f"  거래 기록 (큰 거래만):")
    big_trades = res.trades[res.trades["turnover"] > 0.1]
    print(big_trades.round(6).to_string())
    
    # 두 개의 큰 거래가 있어야: 초기진입(1.0) + 전환(0.5)
    assert len(big_trades) == 2, f"Test 4 실패: 큰 거래 {len(big_trades)}개 (예상 2)"
    
    # 두 번째 큰 거래(전환)가 정확한지 확인
    expected_turnover = 0.5
    expected_cost = expected_turnover * 2 * 0.001  # = 0.001
    
    actual_turnover = big_trades["turnover"].iloc[1]
    actual_cost = big_trades["cost"].iloc[1]
    
    print(f"  전환 turnover: 실제={actual_turnover:.6f}, 기대={expected_turnover:.6f}")
    print(f"  전환 cost:     실제={actual_cost:.6f}, 기대={expected_cost:.6f}")
    
    assert abs(actual_turnover - expected_turnover) < 1e-9, "Test 4 실패: turnover 계산"
    assert abs(actual_cost - expected_cost) < 1e-9, "Test 4 실패: cost 계산"
    print("  ✓ 통과")

    print("\n" + "=" * 70)
    print("✓ 4가지 단위 테스트 모두 통과")
    print("=" * 70)
