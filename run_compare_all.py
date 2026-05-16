"""
UCHIDA V3 - 통합 비교 백테스트

전략 비교:
  1. QQQ Buy & Hold (벤치마크)
  2. Baseline (룰베이스, 정적 임계값)
  3. LSTM (신경망, 임계값 자동 학습)
  4. HMM (Hidden Markov Model, 분포 기반 임계값)

[학술 + 운용 관점]
세 가지 다른 접근법을 같은 데이터/자산으로 비교.
결과 → 최종 모델 결정.

[실행 방법]
    cd uchida_v3
    pip install hmmlearn   # HMM 사용 시
    python run_compare_all.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import date

from config import DATES, INITIAL_CAPITAL
from data.loader import load_all, get_close_prices, get_open_prices
from data.features import build_features
from labels.labeler import label_market_regimes
from models.baseline import predict_weights as baseline_weights, WEIGHT_MAP
from backtest.engine import run_backtest
from backtest.metrics import compare_strategies
from backtest.isa_simulator import compare_isa_scenarios

# 옵션 import (없으면 해당 모델 건너뜀)
try:
    from models.lstm import LSTMConfig, run_walk_forward as lstm_walk_forward
    LSTM_AVAILABLE = True
except ImportError as e:
    print(f"[!] LSTM 사용 불가: {e}")
    LSTM_AVAILABLE = False

try:
    from models.hmm import HMMConfig, run_walk_forward as hmm_walk_forward
    HMM_AVAILABLE = True
except ImportError as e:
    print(f"[!] HMM 사용 불가: {e}")
    HMM_AVAILABLE = False


BACKTEST_START = date(2008, 1, 1)
INITIAL = float(INITIAL_CAPITAL)


def run_strategy(df, target_weights, label):
    """단일 전략 백테스트."""
    print(f"\n  [{label}] 실행 중...")

    close_prices = get_close_prices(df)
    open_prices = get_open_prices(df)

    close_bt = close_prices.loc[BACKTEST_START:]
    open_bt = open_prices.loc[BACKTEST_START:]

    common_idx = target_weights.index.intersection(close_bt.index)
    if len(common_idx) == 0:
        print(f"  [!] 공통 날짜 없음")
        return None, None

    tw = target_weights.loc[common_idx]
    cb = close_bt.loc[common_idx]
    ob = open_bt.loc[common_idx]

    # ticker_us(VYM, SHV 등) → WEIGHT_MAP 키(SCHD, SOFR 등)로 rename
    # config.ASSETS: key=WEIGHT_MAP명, ticker_us=실제 ticker
    from config import ASSETS
    ticker_to_key = {a.ticker_us: k for k, a in ASSETS.items()}
    cb = cb.rename(columns=ticker_to_key)
    ob = ob.rename(columns=ticker_to_key)

    common_assets = [a for a in tw.columns if a in cb.columns]

    tw = tw[common_assets]
    cb = cb[common_assets]
    ob = ob[common_assets]

    row_sums = tw.sum(axis=1).replace(0, 1)
    tw = tw.div(row_sums, axis=0)

    result = run_backtest(cb, ob, tw)
    pv = result.portfolio_value * INITIAL
    print(f"  거래 {len(result.trades)}회, 최종 {pv.iloc[-1]:,.0f}원")
    return result, pv


def main():
    print("="*70)
    print("UCHIDA V3 - 통합 비교 백테스트")
    print(f"기간: {BACKTEST_START} ~ today")
    print(f"초기 자본: {INITIAL:,.0f}원")
    print("="*70)

    # 1. 데이터
    print("\n[STEP 1] 데이터 로드")
    df = load_all(start=DATES.train_start, use_cache=True)
    print(f"  {df.index[0].date()} ~ {df.index[-1].date()} ({len(df):,}일)")

    print(f"\n[데이터 로드 진단]")
    from data.loader import get_close_prices as _gcp
    _close = _gcp(df)
    print(f"  close_prices 컬럼: {list(_close.columns)}")

    features_raw = build_features(df)
    # Baseline용: 5개 핵심 신호 기준 dropna (2008 데이터 보존)
    # LSTM/HMM은 전체 feature가 필요하므로 별도로 features_full 사용
    BASELINE_SIGNALS = ["cpi_z", "credit_spread", "t10y2y", "vix", "dist_ma200_QQQ"]
    available_signals = [c for c in BASELINE_SIGNALS if c in features_raw.columns]
    features = features_raw.dropna(subset=available_signals)
    # LSTM/HMM용: 전체 feature dropna (더 짧은 기간, 모델 내부에서 사용)
    features_full = features_raw.dropna()
    print(f"  {len(features.columns)}개 feature, {len(features):,}일")

    print("\n[STEP 3] Triple-Barrier 라벨링 (LSTM용)")
    labels_df = label_market_regimes(get_close_prices(df), baseline="QQQ", k=1.0, horizon=21)
    labels = labels_df["label"]

    # Baseline용 날짜 정렬 (features 기준 - 2008부터)
    common_base = features.index.intersection(labels.index)
    features = features.loc[common_base]

    # LSTM/HMM용 날짜 정렬 (features_full 기준 - 더 짧은 기간)
    common_full = features_full.index.intersection(labels.index)
    features_full = features_full.loc[common_full]
    labels_full = labels.loc[common_full]

    dist = labels_full.value_counts().sort_index()
    print(f"  분포: CRISIS {dist.get(0,0)} / DEFENSE {dist.get(1,0)} / ATTACK {dist.get(2,0)}")

    results = {}
    pv_dict = {}

    # ----- 4-1. QQQ Buy & Hold -----
    print("\n[STEP 4-1] QQQ Buy & Hold")
    close_prices = get_close_prices(df)
    open_prices = get_open_prices(df)
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
    qqq_w = pd.DataFrame(0.0, index=common_idx, columns=common_assets)
    if "QQQ" in common_assets:
        qqq_w["QQQ"] = 1.0
    r_qqq = run_backtest(cb, ob, qqq_w, cost_one_way=0.0, tolerance_band=0.0)
    results["QQQ B&H"] = r_qqq
    pv_dict["QQQ B&H"] = r_qqq.portfolio_value * INITIAL
    print(f"  최종 {pv_dict['QQQ B&H'].iloc[-1]:,.0f}원")

    # ----- 4-1b. QQQ70/SCHD30 (연 1회 재조정) -----
    # 근거: Jaconetti, Kinniry & Zilbering (2010, Vanguard) 연간 리밸런싱 권장
    # SCHD는 VYM proxy (config.ASSETS["SCHD"].ticker_us = "VYM")
    # VYM/QQQ 모두 auto_adjust=True → 배당 재투자 가격에 내재
    print("\n[STEP 4-1b] QQQ70/SCHD30 (연 1회 재조정)")
    if "QQQ" in common_assets and "SCHD" in common_assets:
        # 매년 1월 첫 영업일에만 30/70으로 reset, 나머지는 drift
        w_annual = pd.DataFrame(0.0, index=common_idx, columns=common_assets)
        w_annual.loc[:, "QQQ"]  = 0.70
        w_annual.loc[:, "SCHD"] = 0.30
        # 백테스트 엔진 tolerance_band=0으로 매일 30/70 유지 가능하지만
        # "연 1회 재조정" 시맨틱: 매년 첫 영업일에 target=30/70, 그 외는 drift 허용
        # 가장 정확한 방법은 weights를 연초에만 변경 신호로 두고 5%밴드 0으로
        # 하지만 엔진이 매 시점 weights 따라가므로 trick: 연초만 0.30/0.70, 나머지는 NaN→ffill
        w_year = pd.DataFrame(index=common_idx, columns=common_assets, dtype=float)
        # 매년 1월 첫 거래일 추출
        yearly_first = w_year.groupby(w_year.index.year).head(1).index
        w_year.loc[yearly_first, "QQQ"]  = 0.70
        w_year.loc[yearly_first, "SCHD"] = 0.30
        for c in common_assets:
            if c not in ("QQQ", "SCHD"):
                w_year.loc[yearly_first, c] = 0.0
        w_year = w_year.ffill().fillna(0.0)
        # tolerance_band=1.0 → 사실상 신호 변할 때만 거래 = 연초만 거래
        r_70_annual = run_backtest(cb, ob, w_year, tolerance_band=0.999)
        results["70/30 Annual"] = r_70_annual
        pv_dict["70/30 Annual"] = r_70_annual.portfolio_value * INITIAL
        print(f"  거래 {len(r_70_annual.trades)}회, 최종 {pv_dict['70/30 Annual'].iloc[-1]:,.0f}원")
    else:
        print("  [!] QQQ 또는 SCHD 데이터 없음, 스킵")

    # ----- 4-1c. QQQ70/SCHD30 (5% no-trade band) -----
    # 근거: Donohue & Yip (2003) 거래비용 기반 동적 밴드. 5%는 Baseline과 동일 조건
    print("\n[STEP 4-1c] QQQ70/SCHD30 (5% no-trade band)")
    if "QQQ" in common_assets and "SCHD" in common_assets:
        w_band = pd.DataFrame(0.0, index=common_idx, columns=common_assets)
        w_band.loc[:, "QQQ"]  = 0.70
        w_band.loc[:, "SCHD"] = 0.30
        r_70_band = run_backtest(cb, ob, w_band, tolerance_band=0.05)
        results["70/30 5%Band"] = r_70_band
        pv_dict["70/30 5%Band"] = r_70_band.portfolio_value * INITIAL
        print(f"  거래 {len(r_70_band.trades)}회, 최종 {pv_dict['70/30 5%Band'].iloc[-1]:,.0f}원")
    else:
        print("  [!] QQQ 또는 SCHD 데이터 없음, 스킵")

    # ----- 4-1d. All Weather (Ray Dalio / Bridgewater 공개 비율) -----
    # 근거: Dalio (2017) "Principles". 모든 시장 환경에 대응하는 SAA 포트폴리오
    # 원본: 주식 30% / 장기채 40% / 중기채 15% / 금 7.5% / 원자재 7.5%
    # 우리 자산 적용: QQQ 30% / IEF 55% (장기+중기 통합) / GLD 15%
    # 원자재 없음 (자산 universe 5개 한정)
    print("\n[STEP 4-1d] All Weather (Bridgewater 변형)")
    w_aw = pd.DataFrame(0.0, index=common_idx, columns=common_assets)
    w_aw.loc[:, "QQQ"] = 0.30
    w_aw.loc[:, "IEF"] = 0.55
    w_aw.loc[:, "GLD"] = 0.15
    r_aw = run_backtest(cb, ob, w_aw, tolerance_band=0.05)
    results["All Weather"] = r_aw
    pv_dict["All Weather"] = r_aw.portfolio_value * INITIAL
    print(f"  거래 {len(r_aw.trades)}회, 최종 {pv_dict['All Weather'].iloc[-1]:,.0f}원")

    # ----- 4-1e. Classic 60/40 (Bogleheads) -----
    # 근거: Bogle (1999) "Common Sense on Mutual Funds". 학계 권장 보수 SAA
    # QQQ 60% / IEF 40%
    print("\n[STEP 4-1e] 60/40 (QQQ/IEF)")
    w_6040 = pd.DataFrame(0.0, index=common_idx, columns=common_assets)
    w_6040.loc[:, "QQQ"] = 0.60
    w_6040.loc[:, "IEF"] = 0.40
    r_6040 = run_backtest(cb, ob, w_6040, tolerance_band=0.05)
    results["60/40"] = r_6040
    pv_dict["60/40"] = r_6040.portfolio_value * INITIAL
    print(f"  거래 {len(r_6040.trades)}회, 최종 {pv_dict['60/40'].iloc[-1]:,.0f}원")

    # ----- 4-2. Baseline -----
    print("\n[STEP 4-2] Baseline (룰베이스)")
    feat_bt = features.loc[BACKTEST_START:]
    bw = baseline_weights(feat_bt)
    r, pv = run_strategy(df, bw, "Baseline")
    if r is not None:
        results["Baseline"] = r
        pv_dict["Baseline"] = pv

    # 국면 분포 분석
    from models.baseline import compute_flags, compute_score, classify_regime, BaselineConfig
    cfg_bl = BaselineConfig()
    regime = classify_regime(compute_score(compute_flags(feat_bt, cfg_bl), cfg_bl), cfg_bl)

    dist = regime.value_counts()
    n_total = len(regime)
    print(f"\n  [국면 분포 - 전체 {BACKTEST_START}~today]")
    for r_name in ["ATTACK", "DEFENSE", "CRISIS"]:
        cnt = dist.get(r_name, 0)
        print(f"    {r_name:8s}: {cnt:4d}일 ({cnt/n_total*100:.1f}%)")

    # 실효 평균 비중 (국면 분포 × WEIGHT_MAP)
    print(f"\n  [실효 평균 비중 (국면 분포 가중 평균)]")
    atk = dist.get("ATTACK",  0) / n_total
    dfs = dist.get("DEFENSE", 0) / n_total
    crs = dist.get("CRISIS",  0) / n_total
    for asset in WEIGHT_MAP["ATTACK"].keys():
        eff = (WEIGHT_MAP["ATTACK"][asset]  * atk +
               WEIGHT_MAP["DEFENSE"][asset] * dfs +
               WEIGHT_MAP["CRISIS"][asset]  * crs)
        print(f"    {asset:6s}: {eff*100:.1f}%")

    # 연도별 국면 분포
    print(f"\n  [연도별 국면 분포]")
    print(f"    {'연도':>6s} | {'ATTACK':>7s} {'DEFENSE':>8s} {'CRISIS':>7s} | 주요국면")
    print(f"    {'-'*58}")
    regime_df = regime.to_frame("regime")
    regime_df["year"] = regime_df.index.year
    for yr, grp in regime_df.groupby("year"):
        vc = grp["regime"].value_counts()
        n  = len(grp)
        a  = vc.get("ATTACK",   0) / n * 100
        d  = vc.get("DEFENSE",  0) / n * 100
        c  = vc.get("CRISIS",   0) / n * 100
        dominant = max(["ATTACK","DEFENSE","CRISIS"], key=lambda x: vc.get(x,0))
        print(f"    {yr:>6d} | {a:>6.0f}% {d:>7.0f}% {c:>6.0f}% | {dominant}")

    # 월별 국면 (CRISIS 발생 월만)
    print(f"\n  [월별 국면 - CRISIS 발생 월]")
    print(f"    {'연월':>8s} | {'주요':>8s} | ATK  DEF  CRI")
    print(f"    {'-'*45}")
    regime_df["ym"] = regime_df.index.to_period("M")
    for ym, grp in regime_df.groupby("ym"):
        vc  = grp["regime"].value_counts()
        n_m = len(grp)
        if vc.get("CRISIS", 0) < 3:
            continue
        dominant = max(["ATTACK","DEFENSE","CRISIS"], key=lambda x: vc.get(x,0))
        a_p = vc.get("ATTACK",  0) / n_m * 100
        d_p = vc.get("DEFENSE", 0) / n_m * 100
        c_p = vc.get("CRISIS",  0) / n_m * 100
        print(f"    {str(ym):>8s} | {dominant:>8s} | {a_p:3.0f}% {d_p:3.0f}% {c_p:3.0f}%")

    # ----- 4-3. Baseline + Soft-scaling (QQQ 전용, Clipped Linear) -----
    # 근거: Moskowitz, Ooi, Pedersen (2012), JFE "Time Series Momentum"
    # Hard Cap(이진법) 대신 시장 12개월 수익률에 비례해서 QQQ만 부드럽게 조정
    # GLD/IEF/SCHD는 건드리지 않음 → 매크로 신호와 충돌 없음
    # floor=-20%: 수익률 -20% 이하에서 QQQ=0% (AI 판단)
    print("\n[STEP 4-3] Baseline + Soft-scaling (QQQ 전용, floor=-20%)")
    close_bt = get_close_prices(df).loc[BACKTEST_START:]
    from config import ASSETS as _ASSETS
    _ticker_to_key = {a.ticker_us: k for k, a in _ASSETS.items()}
    close_bt = close_bt.rename(columns=_ticker_to_key)
    bw_soft = baseline_weights(feat_bt, use_mom_soft=True, close_prices=close_bt,
                               mom_soft_floor=-0.20)
    r_soft, pv_soft = run_strategy(df, bw_soft, "Baseline+Soft")
    if r_soft is not None:
        results["Baseline+Soft"] = r_soft
        pv_dict["Baseline+Soft"] = pv_soft

    # soft-scaling 발동 현황
    if r_soft is not None:
        log_px = np.log(close_bt["QQQ"])
        mom_12 = (log_px - log_px.shift(252)).reindex(bw_soft.index)
        neg_pct  = (mom_12 < 0).mean() * 100
        deep_pct = (mom_12 < -0.20).mean() * 100
        avg_scale = (mom_12.clip(lower=-0.20, upper=0) / 0.20 + 1.0).clip(0, 1).mean() * 100
        print(f"\n  [QQQ Soft-scaling 발동 현황]")
        print(f"    12개월 수익률 음수 구간: {neg_pct:.1f}% (scale < 1.0)")
        print(f"    12개월 수익률 -20% 이하: {deep_pct:.1f}% (scale = 0.0)")
        print(f"    평균 QQQ scale:          {avg_scale:.1f}%")

    # ===== 리포트 =====
    print("\n" + "="*90)
    print("[STEP 5] 종합 비교 리포트")
    print("="*90)

    strategies = {n: r.daily_returns for n, r in results.items()}
    metrics_df = compare_strategies(strategies, risk_free_rate=0.03)

    names = list(results.keys())
    col_w = 16
    header = f"{'':20s}" + "".join(f"{n[:col_w-1]:>{col_w}s}" for n in names)
    print(f"\n{header}")
    print("-" * (20 + col_w * len(names)))

    rows = [
        ("최종 자산", "pretax_final", False),
        ("CAGR (%)", "cagr", True),
        ("MDD (%)", "mdd", True),
        ("Sharpe", "sharpe", False),
        ("Sortino", "sortino", False),
        ("Calmar", "calmar", False),
        ("CE α=1 (%)", "ce_a1.0", True),
        ("CE α=5 (%)", "ce_a5.0", True),
    ]

    def fmt(val, is_pct):
        if pd.isna(val): return "N/A"
        return f"{val*100:.2f}%" if is_pct else f"{val:.3f}"

    for label, col, is_pct in rows:
        if col == "pretax_final":
            line = f"  {label:18s}" + "".join(f"{pv_dict[n].iloc[-1]:>{col_w},.0f}" for n in names)
        elif col in metrics_df.columns:
            line = f"  {label:18s}" + "".join(f"{fmt(metrics_df.loc[n, col], is_pct):>{col_w}s}" for n in names)
        else:
            continue
        print(line)

    # 지표별 1위
    print(f"\n[지표별 우승 모델]")
    for ind in ["cagr", "mdd", "sharpe", "calmar"]:
        if ind in metrics_df.columns:
            if ind == "mdd":
                best = metrics_df[ind].abs().idxmin()
            else:
                best = metrics_df[ind].idxmax()
            print(f"  {ind.upper():10s}: {best}")

    # vs QQQ 알파 분석
    if "QQQ B&H" in metrics_df.index:
        qqq_cagr = metrics_df.loc["QQQ B&H", "cagr"]
        qqq_mdd = metrics_df.loc["QQQ B&H", "mdd"]
        print(f"\n[vs QQQ Buy&Hold (CAGR {qqq_cagr*100:.2f}%, MDD {qqq_mdd*100:.2f}%)]")
        for name in names:
            if name == "QQQ B&H":
                continue
            alpha = metrics_df.loc[name, "cagr"] - qqq_cagr
            mdd_diff = abs(metrics_df.loc[name, "mdd"]) - abs(qqq_mdd)
            print(f"  {name:20s}: CAGR 알파 {alpha*100:+.2f}%p  |  MDD 차이 {mdd_diff*100:+.2f}%p")

    # ISA 세후
    print(f"\n[ISA 세후 (서민형 비과세 400만원)]")
    isa_df = compare_isa_scenarios(pv_dict, initial_deposit=INITIAL)
    for name in isa_df.index:
        row = isa_df.loc[name]
        print(f"  {name:18s}: 세후 {row['ISA 세후 최종 자산 (원)']:>15,.0f}원  "
              f"(CAGR {row['ISA 세후 CAGR (%)']:.2f}%)")

    print(f"\n{'='*90}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n[ERROR] 오류: {e}")
        traceback.print_exc()
