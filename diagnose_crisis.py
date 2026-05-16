"""
UCHIDA V3 - 위기별 Baseline 작동 진단

사용자 의문 3가지에 대한 데이터 기반 답변:
1. DEFENSE에 QQQ 25%가 적정한가?
2. 백테스트 MDD 개선이 2022년 한 해의 결과인가?
3. 다른 위기(2008, 2011, 2020)에서도 DEFENSE/CRISIS가 잘 작동했나?
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import date

from config import DATES, INITIAL_CAPITAL
from data.loader import load_all, get_close_prices, get_open_prices
from data.features import build_features
from models.baseline import (
    predict_weights as baseline_weights,
    compute_flags, compute_score, classify_regime,
    BaselineConfig, WEIGHT_MAP,
)
from backtest.engine import run_backtest


BACKTEST_START = date(2008, 1, 1)
INITIAL = float(INITIAL_CAPITAL)

# 위기/박스권 구간 정의 (학계 + 시장 일반 통념)
# - 2008 GFC: NBER 침체 2007.12~2009.06, 시장 바닥 2009.03
# - 2011 유럽재정위기: 미국 신용등급 강등 8월
# - 2015~2016 박스권: SP500 횡보, 위안화 충격
# - 2018 Q4 조정: 연준 금리인상 + 무역분쟁
# - 2020 코로나: 2020.02~04
# - 2022 금리쇼크: 인플레 + 연준 금리인상
CRISIS_PERIODS = {
    "2008 GFC":          (date(2008, 1, 1),  date(2009, 6, 30)),
    "2011 유럽위기":     (date(2011, 7, 1),  date(2011, 12, 31)),
    "2015-16 박스권":    (date(2015, 1, 1),  date(2016, 6, 30)),
    "2018 Q4 조정":      (date(2018, 10, 1), date(2018, 12, 31)),
    "2020 코로나":       (date(2020, 2, 1),  date(2020, 5, 31)),
    "2022 금리쇼크":     (date(2022, 1, 1),  date(2022, 12, 31)),
}


def analyze_crisis_periods(daily_returns_dict, regime_series):
    """
    위기 구간별 Baseline vs QQQ B&H 성과 비교.
    """
    print("\n" + "="*100)
    print("[Q2/Q3 답변] 위기별 작동 진단")
    print("="*100)
    print(f"\n{'위기':22s} | {'기간':22s} | {'QQQ MDD':>9s} | {'BL MDD':>9s} | {'개선':>7s} | {'국면 분포':30s}")
    print("-"*120)

    rows = []
    for name, (start, end) in CRISIS_PERIODS.items():
        # 구간 추출
        qqq_ret = daily_returns_dict["QQQ B&H"].loc[start:end]
        bl_ret  = daily_returns_dict["Baseline"].loc[start:end]
        regime_p = regime_series.loc[start:end]

        if len(qqq_ret) == 0 or len(bl_ret) == 0:
            continue

        # 누적수익률
        qqq_cum = (1 + qqq_ret).cumprod()
        bl_cum  = (1 + bl_ret).cumprod()

        # MDD
        qqq_mdd = ((qqq_cum / qqq_cum.cummax()) - 1).min()
        bl_mdd  = ((bl_cum  / bl_cum.cummax())  - 1).min()
        improvement = abs(qqq_mdd) - abs(bl_mdd)

        # 국면 분포
        vc = regime_p.value_counts(normalize=True) * 100
        atk = vc.get("ATTACK",  0)
        dfs = vc.get("DEFENSE", 0)
        crs = vc.get("CRISIS",  0)

        regime_str = f"A{atk:.0f}% D{dfs:.0f}% C{crs:.0f}%"

        # 총수익률
        qqq_total = qqq_cum.iloc[-1] - 1
        bl_total  = bl_cum.iloc[-1]  - 1

        period_str = f"{start} ~ {end}"
        print(f"{name:22s} | {period_str:22s} | "
              f"{qqq_mdd*100:>8.1f}% | {bl_mdd*100:>8.1f}% | "
              f"{improvement*100:>+6.1f}p | {regime_str:30s}")

        rows.append({
            "name": name, "qqq_mdd": qqq_mdd, "bl_mdd": bl_mdd,
            "qqq_total": qqq_total, "bl_total": bl_total,
            "ATTACK%": atk, "DEFENSE%": dfs, "CRISIS%": crs,
        })

    # 2022 제거 시 전체 효과
    print("\n[2022 제거 효과 검증 - Q2 답변]")
    print("-"*70)
    bl_returns = daily_returns_dict["Baseline"]
    qqq_returns = daily_returns_dict["QQQ B&H"]

    # 전체 기간
    bl_total_mdd  = ((1+bl_returns).cumprod()  / (1+bl_returns).cumprod().cummax()  - 1).min()
    qqq_total_mdd = ((1+qqq_returns).cumprod() / (1+qqq_returns).cumprod().cummax() - 1).min()
    print(f"전체 기간     | QQQ MDD {qqq_total_mdd*100:.2f}% | BL MDD {bl_total_mdd*100:.2f}% | "
          f"개선 {abs(qqq_total_mdd)-abs(bl_total_mdd):+.2f}%p")

    # 2022 제외 (각 시리즈 인덱스 기준으로 별도 mask)
    bl_mask  = ~((bl_returns.index  >= pd.Timestamp("2022-01-01")) &
                 (bl_returns.index  <= pd.Timestamp("2022-12-31")))
    qqq_mask = ~((qqq_returns.index >= pd.Timestamp("2022-01-01")) &
                 (qqq_returns.index <= pd.Timestamp("2022-12-31")))
    bl_no22  = bl_returns[bl_mask]
    qqq_no22 = qqq_returns[qqq_mask]

    bl_mdd_no22  = ((1+bl_no22).cumprod()  / (1+bl_no22).cumprod().cummax()  - 1).min()
    qqq_mdd_no22 = ((1+qqq_no22).cumprod() / (1+qqq_no22).cumprod().cummax() - 1).min()
    print(f"2022 제외     | QQQ MDD {qqq_mdd_no22*100:.2f}% | BL MDD {bl_mdd_no22*100:.2f}% | "
          f"개선 {abs(qqq_mdd_no22)-abs(bl_mdd_no22):+.2f}%p")

    return rows


def sensitivity_defense_qqq(df, features, common_idx, ticker_to_key):
    """
    DEFENSE의 QQQ 비중 sensitivity analysis (Q1 답변).
    """
    print("\n" + "="*100)
    print("[Q1 답변] DEFENSE QQQ 비중 sensitivity")
    print("="*100)
    print("\n현재 DEFENSE: QQQ 25% / SCHD 20% / GLD 20% / IEF 25% / SOFR 10%")
    print("QQQ 비중을 바꾸면서 나머지는 비례 조정 (SCHD/GLD/IEF/SOFR 합 75% 유지)\n")

    close_prices = get_close_prices(df)
    open_prices = get_open_prices(df)
    cb = close_prices.loc[BACKTEST_START:]
    ob = open_prices.loc[BACKTEST_START:]
    cb = cb.rename(columns=ticker_to_key)
    ob = ob.rename(columns=ticker_to_key)
    common_assets = [a for a in cb.columns if a in ob.columns]

    feat_bt = features.loc[BACKTEST_START:]

    # QQQ 비중 변화: 15% ~ 50%
    qqq_weights_to_test = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    print(f"{'QQQ%':>5s} | {'SCHD%':>5s} | {'GLD%':>5s} | {'IEF%':>5s} | {'SOFR%':>5s} | "
          f"{'CAGR':>7s} | {'MDD':>7s} | {'Sharpe':>7s} | {'Final':>13s}")
    print("-"*100)

    for qqq_pct in qqq_weights_to_test:
        # 나머지 75% 분포: 현재 비율 유지 (SCHD 20/75 : GLD 20/75 : IEF 25/75 : SOFR 10/75)
        # → SCHD 26.67% / GLD 26.67% / IEF 33.33% / SOFR 13.33% (75% 기준)
        # 새 합: 1 - qqq_pct를 같은 비율로 분배
        rest = 1 - qqq_pct
        schd = rest * (20/75)
        gld  = rest * (20/75)
        ief  = rest * (25/75)
        sofr = rest * (10/75)

        # 커스텀 WEIGHT_MAP
        custom_wm = {
            "ATTACK": WEIGHT_MAP["ATTACK"],  # 그대로
            "DEFENSE": {
                "QQQ":  qqq_pct, "SCHD": schd, "GLD": gld,
                "IEF":  ief,     "SOFR": sofr,
            },
            "CRISIS": WEIGHT_MAP["CRISIS"],  # 그대로
        }
        assert abs(sum(custom_wm["DEFENSE"].values()) - 1.0) < 1e-9

        # 비중 생성
        cfg = BaselineConfig()
        flags = compute_flags(feat_bt, cfg)
        score = compute_score(flags, cfg)
        regime = classify_regime(score, cfg)

        assets = list(custom_wm["ATTACK"].keys())
        weights = pd.DataFrame(index=feat_bt.index, columns=assets, dtype=float)
        for d, r in regime.items():
            weights.loc[d] = custom_wm[r]

        # 백테스트
        common_bw = weights.index.intersection(cb.index)
        tw = weights.loc[common_bw]
        cb_b = cb.loc[common_bw]
        ob_b = ob.loc[common_bw]
        ca = [a for a in tw.columns if a in cb_b.columns]
        tw = tw[ca]
        cb_b = cb_b[ca]
        ob_b = ob_b[ca]
        tw = tw.div(tw.sum(axis=1).replace(0, 1), axis=0)
        r = run_backtest(cb_b, ob_b, tw)

        # 지표
        rets = r.daily_returns
        cum = (1 + rets).cumprod()
        cagr = cum.iloc[-1] ** (252/len(rets)) - 1
        mdd  = ((cum / cum.cummax()) - 1).min()
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        final = cum.iloc[-1] * INITIAL

        print(f"{qqq_pct*100:>4.0f}% | {schd*100:>4.1f}% | {gld*100:>4.1f}% | "
              f"{ief*100:>4.1f}% | {sofr*100:>4.1f}% | "
              f"{cagr*100:>6.2f}% | {mdd*100:>6.2f}% | {sharpe:>6.3f} | "
              f"{final:>12,.0f}")


def main():
    print("="*70)
    print("UCHIDA V3 - 위기별 Baseline 작동 진단")
    print("="*70)

    # 데이터
    print("\n[데이터 로드...]")
    df = load_all(start=DATES.train_start, use_cache=True)
    features_raw = build_features(df)
    SIGNALS = ["cpi_z", "credit_spread", "t10y2y", "vix", "dist_ma200_QQQ"]
    available = [c for c in SIGNALS if c in features_raw.columns]
    features = features_raw.dropna(subset=available)
    feat_bt = features.loc[BACKTEST_START:]
    print(f"  {df.index[0].date()} ~ {df.index[-1].date()}")

    # 백테스트
    close_prices = get_close_prices(df)
    open_prices = get_open_prices(df)
    cb_full = close_prices.loc[BACKTEST_START:]
    ob_full = open_prices.loc[BACKTEST_START:]
    common_idx = cb_full.index.intersection(ob_full.index)

    from config import ASSETS
    ticker_to_key = {a.ticker_us: k for k, a in ASSETS.items()}
    cb = cb_full.loc[common_idx].rename(columns=ticker_to_key)
    ob = ob_full.loc[common_idx].rename(columns=ticker_to_key)
    common_assets = [a for a in cb.columns if a in ob.columns]

    # QQQ B&H
    qqq_w = pd.DataFrame(0.0, index=common_idx, columns=common_assets)
    qqq_w["QQQ"] = 1.0
    r_qqq = run_backtest(cb, ob, qqq_w, cost_one_way=0.0, tolerance_band=0.0)

    # Baseline
    bw = baseline_weights(feat_bt)
    common_bw = bw.index.intersection(cb.index)
    tw_bl = bw.loc[common_bw]
    cb_bl = cb.loc[common_bw]
    ob_bl = ob.loc[common_bw]
    ca_bl = [a for a in tw_bl.columns if a in cb_bl.columns]
    tw_bl = tw_bl[ca_bl]
    cb_bl = cb_bl[ca_bl]
    ob_bl = ob_bl[ca_bl]
    tw_bl = tw_bl.div(tw_bl.sum(axis=1).replace(0, 1), axis=0)
    r_bl = run_backtest(cb_bl, ob_bl, tw_bl)

    daily_returns_dict = {
        "QQQ B&H":  r_qqq.daily_returns,
        "Baseline": r_bl.daily_returns,
    }

    # 국면 series
    cfg = BaselineConfig()
    regime = classify_regime(compute_score(compute_flags(feat_bt, cfg), cfg), cfg)

    # Q2, Q3 답변
    analyze_crisis_periods(daily_returns_dict, regime)

    # Q1 답변
    sensitivity_defense_qqq(df, features, common_idx, ticker_to_key)

    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
