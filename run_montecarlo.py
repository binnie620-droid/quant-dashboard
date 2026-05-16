"""
UCHIDA V3 - 적립식 Monte Carlo 시뮬레이션

Stationary Bootstrap (Politis & Romano 1994, JASA) 기반.
기존 백테스트 일별 수익률에서 평균 21일 길이의 블록 샘플링.

[학계 근거]
- Efron (1979): 부트스트랩 원리
- Politis & Romano (1994), "The Stationary Bootstrap", JASA:
  블록 길이를 기하분포에서 추출하여 stationarity 보존
- Politis & White (2004), Econometric Reviews:
  자동 블록 길이 선택 (Newey-West 기반)

[시뮬레이션 가정]
- 초기 자본: 1,000만원 (사용자 현재 보유)
- 월 적립금: 80만원 (연 960만, ISA 한도 내 보수적)
- 시뮬레이션 기간: 15년 = 180개월 = 3,780 거래일 (월 21일 가정)
- 적립 시점: 매월 첫 거래일 (21일마다)
- 시뮬레이션 횟수: 10,000회

[비교 대상]
1. QQQ Buy & Hold
2. Baseline (룰베이스)
3. QQQ70/SCHD30 연 1회 재조정

각 전략의 일별 수익률 분포에서 부트스트랩 → 적립식 누적 수익 시뮬레이션.

[실행]
    python run_montecarlo.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import date
import time

from config import DATES, INITIAL_CAPITAL
from data.loader import load_all, get_close_prices, get_open_prices
from data.features import build_features
from models.baseline import predict_weights as baseline_weights, WEIGHT_MAP
from backtest.engine import run_backtest


# ==========================================
# 시뮬레이션 파라미터
# ==========================================
BACKTEST_START      = date(2008, 1, 1)
INITIAL_DEPOSIT     = 10_000_000     # 초기 자본 1,000만원
MONTHLY_DEPOSIT     = 800_000        # 월 적립 80만원
SIMULATION_YEARS    = 15
TRADING_DAYS_PER_YR = 252
TRADING_DAYS_PER_MO = 21             # 평균 (월 21 거래일 가정)
N_SIMULATIONS       = 10_000
RANDOM_SEED         = 42

# Stationary Bootstrap 평균 블록 길이 (AI 판단)
# 근거: Politis & White (2004) 자동 선택 공식이 복잡하므로 21일 채택.
# 21일 = Triple-Barrier horizon, 학계 거시 변동성 군집 표준 기준.
MEAN_BLOCK_LENGTH = 21


def stationary_bootstrap(
    returns: np.ndarray,
    n_periods: int,
    mean_block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Stationary Bootstrap (Politis & Romano 1994).
    
    블록 길이를 평균 mean_block_length인 기하분포에서 추출.
    각 블록 시작 위치는 균등분포.
    원본 길이를 넘으면 wrap-around (circular).
    
    Parameters
    ----------
    returns : np.ndarray, shape (T,)
        원본 일별 수익률.
    n_periods : int
        샘플링할 기간 (거래일 수).
    mean_block_length : int
        평균 블록 길이.
    rng : np.random.Generator
        난수 생성기 (재현성).
    
    Returns
    -------
    np.ndarray, shape (n_periods,)
        부트스트랩 샘플.
    """
    T = len(returns)
    p = 1.0 / mean_block_length  # 기하분포 모수: 매 시점 블록 종료 확률
    
    sampled = np.empty(n_periods)
    
    # 첫 시점: 임의 위치에서 시작
    idx = rng.integers(0, T)
    sampled[0] = returns[idx]
    
    for t in range(1, n_periods):
        # 확률 p로 새 블록 시작 (= 임의 위치로 점프)
        # 확률 1-p로 다음 인덱스 진행
        if rng.random() < p:
            idx = rng.integers(0, T)
        else:
            idx = (idx + 1) % T  # circular wrap
        sampled[t] = returns[idx]
    
    return sampled


def simulate_accumulation(
    daily_returns: np.ndarray,
    initial: float,
    monthly_deposit: float,
    n_months: int,
    days_per_month: int,
) -> np.ndarray:
    """
    적립식 자산 누적 시뮬레이션.
    
    매월 첫 거래일에 monthly_deposit 입금 후 daily_returns로 복리 누적.
    
    Returns
    -------
    np.ndarray, shape (n_months * days_per_month,)
        일별 자산가치 시계열.
    """
    n_days = n_months * days_per_month
    portfolio = np.empty(n_days)
    value = initial
    
    for t in range(n_days):
        # 매월 첫 거래일 입금
        if t % days_per_month == 0 and t > 0:  # t=0은 초기 자본
            value += monthly_deposit
        # 일별 수익률 적용
        value *= (1 + daily_returns[t])
        portfolio[t] = value
    
    return portfolio


def run_mc_for_strategy(
    strategy_returns: pd.Series,
    label: str,
    n_simulations: int = N_SIMULATIONS,
) -> pd.DataFrame:
    """
    단일 전략에 대한 적립식 MC 실행.
    
    Returns
    -------
    pd.DataFrame
        index: simulation_id (0 ~ n_simulations-1)
        columns: ['final_value', 'cagr', 'max_drawdown', 'total_deposit']
    """
    n_months = SIMULATION_YEARS * 12
    n_days = n_months * TRADING_DAYS_PER_MO
    
    returns_arr = strategy_returns.dropna().values
    if len(returns_arr) < 252:
        raise ValueError(f"[{label}] 수익률 데이터 부족: {len(returns_arr)}일")
    
    rng = np.random.default_rng(RANDOM_SEED)
    
    results = np.empty((n_simulations, 4))  # final, cagr, mdd, total_deposit
    
    total_deposit = INITIAL_DEPOSIT + MONTHLY_DEPOSIT * (n_months - 1)
    
    print(f"  [{label}] {n_simulations:,}회 시뮬레이션 중...")
    t0 = time.time()
    
    for sim in range(n_simulations):
        # 부트스트랩 샘플
        boot_returns = stationary_bootstrap(
            returns_arr, n_days, MEAN_BLOCK_LENGTH, rng
        )
        # 적립식 누적
        portfolio = simulate_accumulation(
            boot_returns, INITIAL_DEPOSIT, MONTHLY_DEPOSIT,
            n_months, TRADING_DAYS_PER_MO
        )
        
        final_value = portfolio[-1]
        # CAGR (원금 대비 단순)
        # 적립식은 dollar-weighted return이 정확하나 단순 비교 위해 final/total_deposit 사용
        if total_deposit > 0:
            ratio = final_value / total_deposit
            cagr = ratio ** (1 / SIMULATION_YEARS) - 1
        else:
            cagr = np.nan
        # MDD
        peak = np.maximum.accumulate(portfolio)
        dd = (portfolio - peak) / peak
        mdd = dd.min()
        
        results[sim] = [final_value, cagr, mdd, total_deposit]
    
    elapsed = time.time() - t0
    print(f"    완료 ({elapsed:.1f}초)")
    
    return pd.DataFrame(
        results,
        columns=["final_value", "cagr", "max_drawdown", "total_deposit"],
    )


def print_mc_summary(mc_results: dict):
    """
    MC 결과 분위수 요약 출력.
    """
    print("\n" + "="*100)
    print(f"[적립식 Monte Carlo 결과] {N_SIMULATIONS:,}회 시뮬레이션, {SIMULATION_YEARS}년")
    print(f"초기 자본 {INITIAL_DEPOSIT:,}원 + 월 {MONTHLY_DEPOSIT:,}원 적립")
    
    total = INITIAL_DEPOSIT + MONTHLY_DEPOSIT * (SIMULATION_YEARS * 12 - 1)
    print(f"총 납입 원금: {total:,}원")
    print("="*100)
    
    # 최종 자산 분위수
    print(f"\n[최종 자산 분위수]")
    print(f"{'전략':18s} | {'5%':>14s} | {'25%':>14s} | {'50% (중앙값)':>17s} | {'75%':>14s} | {'95%':>14s}")
    print("-"*105)
    
    for label, df in mc_results.items():
        q = df["final_value"].quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        print(f"{label:18s} | {q.iloc[0]:>13,.0f}원 | {q.iloc[1]:>13,.0f}원 | "
              f"{q.iloc[2]:>16,.0f}원 | {q.iloc[3]:>13,.0f}원 | {q.iloc[4]:>13,.0f}원")
    
    # CAGR 분위수
    print(f"\n[CAGR 분위수 (원금 대비 단순 환산)]")
    print(f"{'전략':18s} | {'5%':>8s} | {'25%':>8s} | {'50%':>8s} | {'75%':>8s} | {'95%':>8s}")
    print("-"*70)
    
    for label, df in mc_results.items():
        q = df["cagr"].quantile([0.05, 0.25, 0.50, 0.75, 0.95]) * 100
        print(f"{label:18s} | {q.iloc[0]:>7.2f}% | {q.iloc[1]:>7.2f}% | "
              f"{q.iloc[2]:>7.2f}% | {q.iloc[3]:>7.2f}% | {q.iloc[4]:>7.2f}%")
    
    # MDD 분위수
    print(f"\n[Max Drawdown 분위수]")
    print(f"{'전략':18s} | {'5%':>8s} | {'25%':>8s} | {'50%':>8s} | {'75%':>8s} | {'95%':>8s}")
    print("-"*70)
    
    for label, df in mc_results.items():
        # MDD는 음수, 5% = 가장 나쁜 케이스
        q = df["max_drawdown"].quantile([0.05, 0.25, 0.50, 0.75, 0.95]) * 100
        print(f"{label:18s} | {q.iloc[0]:>7.2f}% | {q.iloc[1]:>7.2f}% | "
              f"{q.iloc[2]:>7.2f}% | {q.iloc[3]:>7.2f}% | {q.iloc[4]:>7.2f}%")
    
    # 손실 확률 (원금 대비)
    print(f"\n[원금 보전 확률 (final_value >= total_deposit)]")
    print(f"{'전략':18s} | {'원금 보전':>10s} | {'2배 이상':>10s} | {'3배 이상':>10s}")
    print("-"*60)
    
    for label, df in mc_results.items():
        p_preserved = (df["final_value"] >= df["total_deposit"]).mean() * 100
        p_2x        = (df["final_value"] >= 2 * df["total_deposit"]).mean() * 100
        p_3x        = (df["final_value"] >= 3 * df["total_deposit"]).mean() * 100
        print(f"{label:18s} | {p_preserved:>9.1f}% | {p_2x:>9.1f}% | {p_3x:>9.1f}%")


def main():
    print("="*70)
    print("UCHIDA V3 - 적립식 Monte Carlo")
    print(f"기간: {SIMULATION_YEARS}년, 초기 {INITIAL_DEPOSIT:,}원 + 월 {MONTHLY_DEPOSIT:,}원")
    print(f"방식: Stationary Bootstrap (mean block {MEAN_BLOCK_LENGTH}일)")
    print(f"횟수: {N_SIMULATIONS:,}회")
    print("="*70)
    
    # 1. 데이터
    print("\n[STEP 1] 데이터 로드 + 백테스트 수익률 계산")
    df = load_all(start=DATES.train_start, use_cache=True)
    features_raw = build_features(df)
    BASELINE_SIGNALS = ["cpi_z", "credit_spread", "t10y2y", "vix", "dist_ma200_QQQ"]
    available_signals = [c for c in BASELINE_SIGNALS if c in features_raw.columns]
    features = features_raw.dropna(subset=available_signals)
    feat_bt = features.loc[BACKTEST_START:]
    
    close_prices = get_close_prices(df)
    open_prices  = get_open_prices(df)
    cb = close_prices.loc[BACKTEST_START:]
    ob = open_prices.loc[BACKTEST_START:]
    common_idx = cb.index.intersection(ob.index)
    cb = cb.loc[common_idx]
    ob = ob.loc[common_idx]
    
    from config import ASSETS
    ticker_to_key = {a.ticker_us: k for k, a in ASSETS.items()}
    cb = cb.rename(columns=ticker_to_key)
    ob = ob.rename(columns=ticker_to_key)
    common_assets = [a for a in cb.columns if a in ob.columns]
    
    # 1-1. QQQ B&H
    qqq_w = pd.DataFrame(0.0, index=common_idx, columns=common_assets)
    qqq_w["QQQ"] = 1.0
    r_qqq = run_backtest(cb, ob, qqq_w, cost_one_way=0.0, tolerance_band=0.0)
    
    # 1-2. Baseline
    bw = baseline_weights(feat_bt)
    common_bw = bw.index.intersection(cb.index)
    tw_bl = bw.loc[common_bw]
    cb_bl = cb.loc[common_bw]
    ob_bl = ob.loc[common_bw]
    common_a_bl = [a for a in tw_bl.columns if a in cb_bl.columns]
    tw_bl = tw_bl[common_a_bl]
    cb_bl = cb_bl[common_a_bl]
    ob_bl = ob_bl[common_a_bl]
    tw_bl = tw_bl.div(tw_bl.sum(axis=1).replace(0, 1), axis=0)
    r_bl = run_backtest(cb_bl, ob_bl, tw_bl)
    
    # 1-3. 70/30 Annual
    w_year = pd.DataFrame(index=common_idx, columns=common_assets, dtype=float)
    yearly_first = w_year.groupby(w_year.index.year).head(1).index
    w_year.loc[yearly_first, "QQQ"]  = 0.70
    w_year.loc[yearly_first, "SCHD"] = 0.30
    for c in common_assets:
        if c not in ("QQQ", "SCHD"):
            w_year.loc[yearly_first, c] = 0.0
    w_year = w_year.ffill().fillna(0.0)
    r_70 = run_backtest(cb, ob, w_year, tolerance_band=0.999)
    
    # 2. MC 실행
    print("\n[STEP 2] Stationary Bootstrap 적립식 시뮬레이션")
    
    mc_results = {
        "QQQ B&H":      run_mc_for_strategy(r_qqq.daily_returns, "QQQ B&H"),
        "70/30 Annual": run_mc_for_strategy(r_70.daily_returns,  "70/30 Annual"),
        "Baseline":     run_mc_for_strategy(r_bl.daily_returns,  "Baseline"),
    }
    
    # 3. 요약 출력
    print_mc_summary(mc_results)
    
    # 4. 핵심 비교
    print("\n" + "="*70)
    print("[핵심 인사이트]")
    print("="*70)
    
    # 5% 분위수 (worst case) 비교
    print("\n[Worst Case (5% 분위수) 최종 자산]")
    worst_qqq = mc_results["QQQ B&H"]["final_value"].quantile(0.05)
    worst_70  = mc_results["70/30 Annual"]["final_value"].quantile(0.05)
    worst_bl  = mc_results["Baseline"]["final_value"].quantile(0.05)
    print(f"  QQQ B&H:      {worst_qqq:>15,.0f}원")
    print(f"  70/30 Annual: {worst_70:>15,.0f}원")
    print(f"  Baseline:     {worst_bl:>15,.0f}원")
    
    # 50% 중앙값 비교
    print("\n[중앙값 (50% 분위수) 최종 자산]")
    med_qqq = mc_results["QQQ B&H"]["final_value"].quantile(0.50)
    med_70  = mc_results["70/30 Annual"]["final_value"].quantile(0.50)
    med_bl  = mc_results["Baseline"]["final_value"].quantile(0.50)
    print(f"  QQQ B&H:      {med_qqq:>15,.0f}원")
    print(f"  70/30 Annual: {med_70:>15,.0f}원")
    print(f"  Baseline:     {med_bl:>15,.0f}원")
    
    # 위험조정: 중앙값 / 5%분위수
    print("\n[위험조정 비율 (중앙값 / Worst Case)]")
    print(f"  값이 작을수록 일관성 높음 (안정적)")
    print(f"  QQQ B&H:      {med_qqq/worst_qqq:.3f}")
    print(f"  70/30 Annual: {med_70/worst_70:.3f}")
    print(f"  Baseline:     {med_bl/worst_bl:.3f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
