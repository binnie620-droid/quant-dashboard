"""
UCHIDA V3 - 통합 백테스트 실행 스크립트

지금까지 만든 모든 모듈을 연결하여 실제 데이터로 백테스트 실행.

[실행 순서]
1. 데이터 다운로드 (yfinance + FRED)
2. Feature 생성
3. Triple-Barrier 라벨링
4. Baseline 룰베이스 백테스트
5. QQQ Buy&Hold 백테스트 (벤치마크)
6. 성과 비교 출력

[실행 방법]
    cd uchida_v3
    python run_backtest.py

[필요 패키지]
    pip install yfinance pandas-datareader pandas numpy scikit-learn pyarrow
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import date

# 프로젝트 모듈
from config import ASSETS, DATES, INITIAL_CAPITAL
from data.loader import load_all, get_close_prices, get_open_prices
from data.features import build_features
from labels.labeler import label_market_regimes
from models.baseline import predict_weights, BaselineConfig
from backtest.engine import run_backtest
from backtest.metrics import compute_all_metrics, compare_strategies
from backtest.isa_simulator import simulate_isa, compare_isa_scenarios


# ==========================================
# 0. 설정
# ==========================================
BACKTEST_START = date(2015, 1, 1)   # 백테스트 시작 (학습 기간 2010~는 feature 계산용)
BACKTEST_END   = date.today()
INITIAL        = float(INITIAL_CAPITAL)  # 1,000만원


# ==========================================
# 1. 데이터 다운로드
# ==========================================
def step1_load_data():
    print("\n" + "="*60)
    print("STEP 1. 데이터 다운로드")
    print("="*60)

    df = load_all(
        start=DATES.train_start,   # 2010-01-01 (feature 계산 위해 충분히 앞에서)
        end=BACKTEST_END,
        use_cache=True,
    )

    print(f"  기간: {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"  행 수: {len(df):,}")
    print(f"  컬럼 수: {len(df.columns)}")

    # NaN 체크
    nan_cols = df.isna().sum()
    nan_cols = nan_cols[nan_cols > 0]
    if len(nan_cols) > 0:
        print(f"  ⚠ NaN 있는 컬럼:\n{nan_cols}")
    else:
        print(f"  ✓ NaN 없음")

    return df


# ==========================================
# 2. Feature 생성
# ==========================================
def step2_build_features(df):
    print("\n" + "="*60)
    print("STEP 2. Feature 생성")
    print("="*60)

    features = build_features(df)

    # NaN 제거: baseline이 실제로 사용하는 5개 신호 컬럼 기준으로만 dropna.
    # 전체 dropna()를 쓰면 mom_12_1(231일 lookback) 등의 NaN이
    # 2008~2009년 데이터를 통째로 삭제하는 버그 수정 (2026-05).
    BASELINE_SIGNALS = ["cpi_z", "credit_spread", "t10y2y", "vix", "dist_ma200_QQQ"]
    available_signals = [c for c in BASELINE_SIGNALS if c in features.columns]
    features = features.dropna(subset=available_signals)

    print(f"  Feature 수: {len(features.columns)}개")
    print(f"  유효 기간: {features.index[0].date()} ~ {features.index[-1].date()}")
    print(f"  NaN 비율: {features.isna().mean().mean()*100:.2f}%")

    # Feature 군별 요약
    groups = {
        "가격 returns":   [c for c in features.columns if c.startswith("ret_")],
        "실현 변동성":    [c for c in features.columns if c.startswith("rvol_")],
        "모멘텀":         [c for c in features.columns if c.startswith("mom_")],
        "기술적 지표":    [c for c in features.columns if any(c.startswith(p) for p in ["rsi_","macd_","boll_","dist_","ma_"])],
        "매크로":         [c for c in features.columns if c in ["cpi_yoy","cpi_z","cpi_mom","t10y2y",
                           "dgs10","dgs10_chg","credit_spread","credit_spread_chg",
                           "unrate","unrate_chg","vix","vix_z","vix_chg","dxy","dxy_ret_5d"]],
        "자산간 관계":    [c for c in features.columns if c.startswith("corr_") or c.endswith("_ratio")],
    }
    for g, cols in groups.items():
        print(f"  {g:15s}: {len(cols):3d}개")

    # 핵심 신호 실제값 확인 (Level 1 진단용)
    sig_cols = ["credit_spread", "vix", "cpi_z", "t10y2y", "dist_ma200_QQQ"]
    available = [c for c in sig_cols if c in features.columns]
    print(f"\n  [핵심 신호 통계 - 백테스트 기간]")
    bt_feat = features.loc[str(BACKTEST_START):]
    for c in available:
        s = bt_feat[c].dropna()
        print(f"    {c:20s}: min={s.min():.2f}  p25={s.quantile(.25):.2f}  "
              f"median={s.median():.2f}  p75={s.quantile(.75):.2f}  max={s.max():.2f}")

    return features


# ==========================================
# 3. 라벨링
# ==========================================
def step3_label(df):
    print("\n" + "="*60)
    print("STEP 3. Triple-Barrier 라벨링")
    print("="*60)

    close_prices = get_close_prices(df)
    labels_df = label_market_regimes(
        close_prices,
        baseline="QQQ",   # SPY는 ASSETS에 미포함. QQQ로 대체 (상관 0.92).
        k=1.0,
        horizon=21,
        vol_window=60,
    )

    dist = labels_df["label"].value_counts().sort_index()
    names = {0: "CRISIS", 1: "DEFENSE/횡보", 2: "ATTACK"}
    total = len(labels_df)

    print(f"  라벨 기간: {labels_df.index[0].date()} ~ {labels_df.index[-1].date()}")
    print(f"  총 라벨 수: {total:,}")
    print(f"  분포:")
    for cls, cnt in dist.items():
        print(f"    {names[cls]:15s}: {cnt:5d}일 ({cnt/total*100:.1f}%)")

    print(f"  평균 touch_day: {labels_df['touch_day'].mean():.1f}일")

    return labels_df


# ==========================================
# 4. Baseline 룰베이스 백테스트
# ==========================================
def step4_baseline_backtest(df, features, use_momentum=False, use_vol_target=False, label_suffix="", _cfg=None):
    print("\n" + "="*60)
    label = "Baseline" if (not use_momentum and not use_vol_target) else "New System"
    if label_suffix:
        label = f"{label} ({label_suffix})"
    print(f"STEP 4. {label} 백테스트")
    print("="*60)

    close_prices = get_close_prices(df)
    open_prices  = get_open_prices(df)

    # 백테스트 기간 슬라이싱
    feat_bt  = features.loc[BACKTEST_START:]
    close_bt = close_prices.loc[BACKTEST_START:]
    open_bt  = open_prices.loc[BACKTEST_START:]

    # 공통 날짜 정렬
    common_idx = feat_bt.index.intersection(close_bt.index)
    feat_bt  = feat_bt.loc[common_idx]
    close_bt = close_bt.loc[common_idx]
    open_bt  = open_bt.loc[common_idx]

    # 비중 결정 (옵션에 따라 momentum/vol_target 적용)
    print(f"  비중 계산 중... (momentum={use_momentum}, vol_target={use_vol_target})")
    pw_kwargs = dict(
        use_momentum=use_momentum,
        use_vol_target=use_vol_target,
        close_prices=close_prices,
    )
    if _cfg is not None:
        pw_kwargs["cfg"] = _cfg
    target_w = predict_weights(feat_bt, **pw_kwargs)

    # 국면 분포 진단 (use_level1=True인 경우만)
    cfg_used = pw_kwargs.get("cfg", None)
    is_l1 = (cfg_used is None) or getattr(cfg_used, "use_level1", True)
    if is_l1:
        from models.baseline import classify_regime_level1, _score_credit, _score_panic, _score_inflation, _score_recovery, BaselineConfig
        _cfg_diag = cfg_used if cfg_used is not None else BaselineConfig()
        _regime = classify_regime_level1(feat_bt, _cfg_diag)
        _dist = _regime.value_counts()
        _n = len(_regime)
        print(f"  [국면 분포] ATTACK {_dist.get('ATTACK',0)}일({_dist.get('ATTACK',0)/_n*100:.0f}%) "
              f"DEFENSE {_dist.get('DEFENSE',0)}일({_dist.get('DEFENSE',0)/_n*100:.0f}%) "
              f"CRISIS {_dist.get('CRISIS',0)}일({_dist.get('CRISIS',0)/_n*100:.0f}%)")
        # 신호별 발동률
        _sc = _score_credit(feat_bt).mean()
        _sp = _score_panic(feat_bt).mean()
        _si = _score_inflation(feat_bt).mean()
        _sr = _score_recovery(feat_bt).mean()
        print(f"  [신호 발동률] 신용:{_sc*100:.1f}% 패닉:{_sp*100:.1f}% 인플레:{_si*100:.1f}% 회복:{_sr*100:.1f}%")
        # 첫 CRISIS 진입일
        crisis_days = _regime[_regime == "CRISIS"]
        if len(crisis_days) > 0:
            print(f"  [첫 CRISIS 진입] {crisis_days.index[0].date()}  "
                  f"마지막 CRISIS: {crisis_days.index[-1].date()}")

    # 자산 컬럼 일치 확인
    common_assets = [a for a in target_w.columns if a in close_bt.columns]
    target_w  = target_w[common_assets]
    close_bt  = close_bt[common_assets]
    open_bt   = open_bt[common_assets]

    # 없는 자산 제외 후 비중 합이 1이 되도록 정규화
    row_sums = target_w.sum(axis=1)
    row_sums = row_sums.replace(0, 1)  # 0 방지
    target_w = target_w.div(row_sums, axis=0)

    # 백테스트 실행
    print(f"  백테스트 실행 중... ({common_idx[0].date()} ~ {common_idx[-1].date()})")
    result = run_backtest(
        prices_close=close_bt,
        prices_open=open_bt,
        target_weights=target_w,
    )

    # 실제 금액으로 변환
    pv_series = result.portfolio_value * INITIAL

    print(f"  총 거래 횟수: {len(result.trades)}회")
    if len(result.trades) > 0:
        print(f"  평균 turnover: {result.trades['turnover'].mean():.3f}")
        print(f"  총 거래비용: {result.trades['cost'].sum() * INITIAL:,.0f}원")

    return result, pv_series


# ==========================================
# 5. QQQ Buy&Hold 백테스트 (벤치마크)
# ==========================================
def step5_qqq_benchmark(df):
    print("\n" + "="*60)
    print("STEP 5. QQQ Buy & Hold 벤치마크")
    print("="*60)

    close_prices = get_close_prices(df)
    open_prices  = get_open_prices(df)

    # 백테스트 기간
    close_bt = close_prices.loc[BACKTEST_START:]
    open_bt  = open_prices.loc[BACKTEST_START:]

    common_assets = [a for a in close_bt.columns if a in open_bt.columns]
    close_bt = close_bt[common_assets]
    open_bt  = open_bt[common_assets]

    # QQQ 100% 고정 (정규화 포함)
    qqq_w = pd.DataFrame(0.0, index=close_bt.index, columns=common_assets)
    if "QQQ" in common_assets:
        qqq_w["QQQ"] = 1.0
    else:
        print("  ⚠ QQQ 데이터 없음. 첫 번째 자산으로 대체.")
        qqq_w.iloc[:, 0] = 1.0

    result_qqq = run_backtest(
        prices_close=close_bt,
        prices_open=open_bt,
        target_weights=qqq_w,
        cost_one_way=0.0,      # Buy&Hold는 거래비용 없음 (초기 매수만)
        tolerance_band=0.0,
    )

    pv_qqq = result_qqq.portfolio_value * INITIAL
    print(f"  QQQ Buy&Hold 기간: {close_bt.index[0].date()} ~ {close_bt.index[-1].date()}")

    return result_qqq, pv_qqq


# ==========================================
# 6. 성과 비교 출력
# ==========================================
def step6_report(results: dict, pv_dict: dict):
    """
    Parameters
    ----------
    results : dict
        전략명 → BacktestResult
    pv_dict : dict
        전략명 → 포트폴리오 가치 시계열
    """
    print("\n" + "="*80)
    print("STEP 6. 성과 비교 리포트")
    print("="*80)

    strategies = {name: r.daily_returns for name, r in results.items()}
    metrics_df = compare_strategies(strategies, risk_free_rate=0.03)

    # ----- 핵심 지표 출력 -----
    names = list(results.keys())
    col_w = 18
    header = f"{'':25s}" + "".join(f"{n:>{col_w}s}" for n in names)
    print(f"\n{header}")
    print("-" * (25 + col_w * len(names)))

    rows = [
        ("최종 자산 (원)",       "pretax_final",  False),
        ("CAGR (%)",            "cagr",          True),
        ("MDD (%)",             "mdd",           True),
        ("Sharpe",              "sharpe",        False),
        ("Sortino",             "sortino",       False),
        ("Calmar",              "calmar",        False),
        ("Entropic (α=1.0)",    "entropic_a1.0", True),
        ("CE (α=1.0, %)",       "ce_a1.0",       True),
    ]

    def fmt(val, is_pct):
        if pd.isna(val):
            return "N/A"
        if is_pct:
            return f"{val*100:.2f}%"
        return f"{val:.3f}"

    for label, col, is_pct in rows:
        if col == "pretax_final":
            line = f"  {label:23s}" + "".join(f"{pv_dict[n].iloc[-1]:>{col_w},.0f}" for n in names)
            print(line)
        elif col in metrics_df.columns:
            line = f"  {label:23s}" + "".join(f"{fmt(metrics_df.loc[n, col], is_pct):>{col_w}s}" for n in names)
            print(line)

    # ----- ISA 세후 비교 -----
    print(f"\n[ISA 세후 (서민형, 비과세 400만원)]")
    isa_df = compare_isa_scenarios(pv_dict, initial_deposit=INITIAL)

    for name in isa_df.index:
        row = isa_df.loc[name]
        print(f"  {name}:")
        print(f"    세후 최종: {row['ISA 세후 최종 자산 (원)']:>15,.0f}원  "
              f"(세후 CAGR {row['ISA 세후 CAGR (%)']:.2f}%)")
        print(f"    절세액:    {row['ISA 절세액 (원)']:>15,.0f}원")

    # ----- Entropic sensitivity -----
    print(f"\n[Entropic Risk Sensitivity (α 변화) - 작을수록 좋음]")
    header2 = f"  {'α':>6s} |" + "".join(f"{n:>{col_w}s}" for n in names)
    print(header2)
    print(f"  {'-'*(8 + col_w * len(names))}")
    for a in [0.5, 1.0, 2.0, 5.0]:
        col = f"entropic_a{a}"
        if col in metrics_df.columns:
            vals = [metrics_df.loc[n, col] for n in names]
            line = f"  {a:>6.1f} |" + "".join(f"{v*100:>{col_w-1}.2f}%" for v in vals)
            print(line)

    print(f"\n{'='*80}")
    print("백테스트 완료.")
    print(f"{'='*80}")

    return metrics_df


# ==========================================
# 메인
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("UCHIDA V3 - 통합 백테스트")
    print(f"기간: {BACKTEST_START} ~ {BACKTEST_END}")
    print(f"초기 자본: {INITIAL:,.0f}원")
    print("="*60)

    try:
        df       = step1_load_data()
        features = step2_build_features(df)
        labels   = step3_label(df)

        # 4가지 전략 실행
        results = {}
        pv_dict = {}

        # 1. Baseline Level 1 (룰베이스만, Level 1 재설계)
        r, pv = step4_baseline_backtest(df, features, use_momentum=False, use_vol_target=False)
        results["Baseline L1"] = r
        pv_dict["Baseline L1"] = pv

        # 1-b. Baseline Legacy (구버전 가중합, 비교용)
        from models.baseline import BaselineConfig
        legacy_cfg = BaselineConfig(use_level1=False)
        r, pv = step4_baseline_backtest(df, features, use_momentum=False, use_vol_target=False,
                                         label_suffix="Legacy", _cfg=legacy_cfg)
        results["Baseline Legacy"] = r
        pv_dict["Baseline Legacy"] = pv

        # 2. Baseline + Momentum
        r, pv = step4_baseline_backtest(df, features, use_momentum=True, use_vol_target=False, label_suffix="+ Momentum")
        results["+ Momentum"] = r
        pv_dict["+ Momentum"] = pv

        # 3. Baseline + Momentum + Vol Target (전체)
        r, pv = step4_baseline_backtest(df, features, use_momentum=True, use_vol_target=True, label_suffix="+ Mom + Vol")
        results["+ Mom + Vol"] = r
        pv_dict["+ Mom + Vol"] = pv

        # 4. QQQ Buy & Hold
        r_qqq, pv_qqq = step5_qqq_benchmark(df)
        results["QQQ B&H"] = r_qqq
        pv_dict["QQQ B&H"] = pv_qqq

        # 리포트
        metrics = step6_report(results, pv_dict)

    except Exception as e:
        import traceback
        print(f"\n❌ 오류 발생: {e}")
        traceback.print_exc()
