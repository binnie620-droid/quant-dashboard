"""
UCHIDA V3 - Baseline Sensitivity Analysis

Baseline의 임계값/가중치 조합 비교.
SCHD는 모든 시나리오에서 CRISIS 20% 유지 (배당 복리 전략).

시나리오:
  A: 현재 (기준선)
  B: 임계값 완화 (CRISIS 더 자주 발동)
  C: VIX/추세 가중치 강화 (시장 신호 빠른 반응)
  D: B+C 결합
  + QQQ B&H 벤치마크

목표: MDD -22% 달성 + CAGR 최대 유지
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import date
from dataclasses import dataclass, replace

from config import DATES, INITIAL_CAPITAL
from data.loader import load_all, get_close_prices, get_open_prices
from data.features import build_features
from models.baseline import (
    BaselineConfig, predict_weights, WEIGHT_MAP
)
from backtest.engine import run_backtest
from backtest.metrics import compare_strategies
from backtest.isa_simulator import compare_isa_scenarios


BACKTEST_START = date(2008, 1, 1)
INITIAL = float(INITIAL_CAPITAL)


# ==========================================
# 시나리오 정의
# ==========================================
SCENARIOS = {
    "A. 현재": BaselineConfig(
        attack_threshold=1.5,
        crisis_threshold=3.0,
        w_cpi=1.5, w_spread=1.5, w_yield=1.0,
        w_vix=0.5, w_trend=0.5,
    ),
    "B. 임계값 완화": BaselineConfig(
        attack_threshold=1.0,
        crisis_threshold=2.5,
        w_cpi=1.5, w_spread=1.5, w_yield=1.0,
        w_vix=0.5, w_trend=0.5,
    ),
    "C. VIX 강화": BaselineConfig(
        attack_threshold=1.5,
        crisis_threshold=3.0,
        w_cpi=1.0, w_spread=1.0, w_yield=1.0,
        w_vix=1.5, w_trend=1.0,
    ),
    "D. B+C 결합": BaselineConfig(
        attack_threshold=1.0,
        crisis_threshold=2.5,
        w_cpi=1.0, w_spread=1.0, w_yield=1.0,
        w_vix=1.5, w_trend=1.0,
    ),
}


def run_scenario(df, features, name, cfg):
    close_prices = get_close_prices(df)
    open_prices  = get_open_prices(df)

    feat_bt  = features.loc[BACKTEST_START:]
    close_bt = close_prices.loc[BACKTEST_START:]
    open_bt  = open_prices.loc[BACKTEST_START:]

    common_idx = feat_bt.index.intersection(close_bt.index)
    feat_bt  = feat_bt.loc[common_idx]
    close_bt = close_bt.loc[common_idx]
    open_bt  = open_bt.loc[common_idx]

    # 비중 계산 (시나리오별 cfg)
    target_w = predict_weights(feat_bt, cfg=cfg, weight_map=WEIGHT_MAP)

    common_assets = [a for a in target_w.columns if a in close_bt.columns]
    target_w  = target_w[common_assets]
    close_bt  = close_bt[common_assets]
    open_bt   = open_bt[common_assets]

    row_sums = target_w.sum(axis=1).replace(0, 1)
    target_w = target_w.div(row_sums, axis=0)

    result = run_backtest(close_bt, open_bt, target_w)
    pv = result.portfolio_value * INITIAL

    # 국면 분포 확인
    from models.baseline import compute_flags, compute_score, classify_regime
    flags  = compute_flags(feat_bt, cfg)
    score  = compute_score(flags, cfg)
    regime = classify_regime(score, cfg)
    dist = regime.value_counts(normalize=True).to_dict()

    print(f"  [{name}] 거래 {len(result.trades)}회 | "
          f"ATTACK {dist.get('ATTACK',0)*100:.0f}% / "
          f"DEFENSE {dist.get('DEFENSE',0)*100:.0f}% / "
          f"CRISIS {dist.get('CRISIS',0)*100:.0f}%")

    return result, pv


# ==========================================
# 정적 분산 포트폴리오 정의
# ==========================================
# 동적 신호 없이 고정 비중 유지.
# baseline A와 성과가 비슷하면 신호 시스템의 가치 = 0.
#
# Static-60/40: 전통 60/40의 성장주 버전
#   근거: Brinson, Hood, Beebower (1986) — 자산배분이 수익의 90%
#         동적 조정이 정적 배분보다 낫다는 증거 약함
#
# Static-ATTACK: WEIGHT_MAP[ATTACK] 그대로 고정
#   근거: "신호가 ATTACK 88%를 판단한다면, 그냥 ATTACK 비중을 들고 있으면?"
#   baseline A와 비교 시 신호 시스템의 거래비용 손실을 측정
STATIC_PORTFOLIOS = {
    "E. Static-60/40": {
        "QQQ": 0.60, "SCHD": 0.15, "IEF": 0.15, "GLD": 0.10,
        "SOFR": 0.0, "EEM": 0.0, "TLT": 0.0, "OIL": 0.0,
    },
    "F. Static-ATTACK": {
        "QQQ": 0.70, "SCHD": 0.15, "GLD": 0.05, "IEF": 0.05,
        "SOFR": 0.0, "EEM": 0.05, "TLT": 0.0, "OIL": 0.0,
    },
}


def run_static(df, name, static_weights):
    """정적 분산 포트폴리오 백테스트. 신호 없이 고정 비중 유지."""
    close_prices = get_close_prices(df)
    open_prices  = get_open_prices(df)

    close_bt = close_prices.loc[BACKTEST_START:]
    open_bt  = open_prices.loc[BACKTEST_START:]
    common   = close_bt.index.intersection(open_bt.index)
    close_bt = close_bt.loc[common]
    open_bt  = open_bt.loc[common]

    common_assets = [a for a in close_bt.columns if a in open_bt.columns]
    w = pd.DataFrame(index=common, columns=common_assets, dtype=float)
    for asset in common_assets:
        w[asset] = static_weights.get(asset, 0.0)

    # 비중 합 = 1 정규화 (보유 가능한 자산만 있을 경우 대비)
    row_sum = w.sum(axis=1).replace(0, 1)
    w = w.div(row_sum, axis=0)

    # 정적 포트폴리오: 연 1회 리밸런싱 상정, 거래비용은 동일 적용
    result = run_backtest(close_bt, open_bt, w)
    pv = result.portfolio_value * INITIAL

    dist_str = " / ".join(f"{k}:{v*100:.0f}%" for k, v in static_weights.items() if v > 0)
    print(f"  [{name}] 거래 {len(result.trades)}회 | {dist_str}")
    return result, pv


def main():
    print("="*70)
    print("UCHIDA V3 - Baseline Sensitivity Analysis")
    print(f"기간: {BACKTEST_START} ~ today  |  초기: {INITIAL:,.0f}원")
    print("="*70)

    # 데이터
    print("\n[데이터 로드...]")
    df = load_all(start=DATES.train_start, use_cache=True)
    features = build_features(df)
    # NaN 제거: baseline 5개 핵심 신호 기준 (2008 데이터 보존)
    BASELINE_SIGNALS = ["cpi_z", "credit_spread", "t10y2y", "vix", "dist_ma200_QQQ"]
    available_signals = [c for c in BASELINE_SIGNALS if c in features.columns]
    features = features.dropna(subset=available_signals)
    print(f"  {df.index[0].date()} ~ {df.index[-1].date()}, feature {len(features.columns)}개")

    results = {}
    pv_dict = {}

    # QQQ 벤치마크
    print("\n[QQQ Buy & Hold]")
    close_prices = get_close_prices(df)
    open_prices  = get_open_prices(df)
    cb = close_prices.loc[BACKTEST_START:]
    ob = open_prices.loc[BACKTEST_START:]
    common = cb.index.intersection(ob.index)
    cb = cb.loc[common]
    ob = ob.loc[common]
    common_assets = [a for a in cb.columns if a in ob.columns]
    qqq_w = pd.DataFrame(0.0, index=common, columns=common_assets)
    if "QQQ" in common_assets:
        qqq_w["QQQ"] = 1.0
    r_qqq = run_backtest(cb, ob, qqq_w, cost_one_way=0.0, tolerance_band=0.0)
    results["QQQ B&H"] = r_qqq
    pv_dict["QQQ B&H"] = r_qqq.portfolio_value * INITIAL
    print(f"  최종 {pv_dict['QQQ B&H'].iloc[-1]:,.0f}원")

    # 시나리오별 실행
    print("\n[시나리오 비교]")
    for name, cfg in SCENARIOS.items():
        r, pv = run_scenario(df, features, name, cfg)
        results[name] = r
        pv_dict[name] = pv

    # 정적 분산 포트폴리오 (신호 없이 고정 비중)
    print("\n[정적 분산 포트폴리오 비교]")
    for name, weights in STATIC_PORTFOLIOS.items():
        r, pv = run_static(df, name, weights)
        results[name] = r
        pv_dict[name] = pv

    # 리포트
    print("\n" + "="*90)
    print("[결과 비교]")
    print("="*90)

    strategies = {n: r.daily_returns for n, r in results.items()}
    metrics_df = compare_strategies(strategies, risk_free_rate=0.03)

    names = list(results.keys())
    col_w = 15
    header = f"{'':22s}" + "".join(f"{n[:col_w-1]:>{col_w}s}" for n in names)
    print(f"\n{header}")
    print("-" * (22 + col_w * len(names)))

    rows = [
        ("최종 자산",   "pretax_final", False),
        ("CAGR (%)",    "cagr",         True),
        ("MDD (%)",     "mdd",          True),
        ("Sharpe",      "sharpe",       False),
        ("Calmar",      "calmar",       False),
        ("CE α=1 (%)",  "ce_a1.0",      True),
        ("CE α=5 (%)",  "ce_a5.0",      True),
    ]

    def fmt(val, is_pct):
        if pd.isna(val): return "N/A"
        return f"{val*100:.2f}%" if is_pct else f"{val:.3f}"

    for label, col, is_pct in rows:
        if col == "pretax_final":
            line = f"  {label:20s}" + "".join(
                f"{pv_dict[n].iloc[-1]:>{col_w},.0f}" for n in names)
        elif col in metrics_df.columns:
            line = f"  {label:20s}" + "".join(
                f"{fmt(metrics_df.loc[n, col], is_pct):>{col_w}s}" for n in names)
        else:
            continue
        print(line)

    # MDD 목표 달성 여부
    print(f"\n[MDD -22% 목표 달성 여부]")
    for name in names:
        if name == "QQQ B&H":
            continue
        mdd = metrics_df.loc[name, "mdd"] * 100
        cagr = metrics_df.loc[name, "cagr"] * 100
        passed = "✅" if mdd >= -22 else "❌"
        print(f"  {name:20s}: MDD {mdd:.2f}% {passed}  |  CAGR {cagr:.2f}%")

    # vs QQQ 알파
    qqq_cagr = metrics_df.loc["QQQ B&H", "cagr"]
    qqq_mdd  = metrics_df.loc["QQQ B&H", "mdd"]
    print(f"\n[vs QQQ B&H (CAGR {qqq_cagr*100:.2f}%, MDD {qqq_mdd*100:.2f}%)]")
    for name in names:
        if name == "QQQ B&H":
            continue
        alpha = (metrics_df.loc[name, "cagr"] - qqq_cagr) * 100
        mdd_diff = (abs(metrics_df.loc[name, "mdd"]) - abs(qqq_mdd)) * 100
        print(f"  {name:20s}: CAGR {alpha:+.2f}%p  |  MDD {mdd_diff:+.2f}%p")

    # ISA 세후
    print(f"\n[ISA 세후]")
    isa_df = compare_isa_scenarios(pv_dict, initial_deposit=INITIAL)
    for name in isa_df.index:
        row = isa_df.loc[name]
        print(f"  {name:20s}: 세후 {row['ISA 세후 최종 자산 (원)']:>15,.0f}원  "
              f"(CAGR {row['ISA 세후 CAGR (%)']:.2f}%)")

    print(f"\n{'='*90}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n❌ 오류: {e}")
        traceback.print_exc()
