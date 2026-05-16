"""
UCHIDA V3 - Performance Metrics

백테스트 결과 평가용 지표 모음.

[지표 분류]
1. 수익: CAGR, Total Return
2. 위험: Annualized Vol, MDD, MDD Recovery Time
3. 위험조정 수익: Sharpe, Sortino, Calmar
4. Entropic Risk Measure (Föllmer & Schied 2002)
5. Certainty Equivalent (entropic 해석)
6. Probabilistic Sharpe Ratio (Bailey & López de Prado 2012)

[입력 규약]
- daily_returns: pd.Series of daily simple returns (예: 0.01 = +1%)
  index는 DatetimeIndex 또는 영업일 순서
- 모든 함수는 NaN 자동 처리

[참고문헌]
- Sharpe (1966), "Mutual Fund Performance"
- Sortino & Price (1994), "Performance measurement in a downside risk framework"
- Young (1991), "Calmar Ratio: A Smoother Tool"
- Föllmer & Schied (2002), Stochastic Finance, Ch.4
- Bailey & López de Prado (2012), "The Sharpe Ratio Efficient Frontier"
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, List
from scipy import stats


TRADING_DAYS_PER_YEAR = 252


# ==========================================
# 1. 기본 수익률 지표
# ==========================================
def total_return(daily_returns: pd.Series) -> float:
    """전체 기간 누적 수익률. (1+r1)(1+r2)... - 1"""
    return (1 + daily_returns.dropna()).prod() - 1


def cagr(daily_returns: pd.Series) -> float:
    """
    Compound Annual Growth Rate (연복리수익률).
    (1 + total_return) ^ (1/years) - 1
    """
    r = daily_returns.dropna()
    if len(r) == 0:
        return np.nan
    years = len(r) / TRADING_DAYS_PER_YEAR
    return (1 + total_return(r)) ** (1 / years) - 1


# ==========================================
# 2. 위험 지표
# ==========================================
def annualized_vol(daily_returns: pd.Series) -> float:
    """연율화 변동성. std × √252"""
    return daily_returns.dropna().std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(daily_returns: pd.Series) -> float:
    """
    Maximum Drawdown (최대 낙폭, 음수).
    누적수익률 곡선의 peak-to-trough 최대 손실.
    """
    r = daily_returns.dropna()
    if len(r) == 0:
        return np.nan
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    return drawdown.min()


def mdd_recovery_days(daily_returns: pd.Series) -> Union[int, float]:
    """
    MDD 발생 후 전고점 회복까지 걸린 영업일 수.
    회복 못했으면 NaN 반환.
    """
    r = daily_returns.dropna()
    if len(r) == 0:
        return np.nan
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak

    trough_idx = drawdown.idxmin()
    if pd.isna(trough_idx):
        return np.nan
    peak_value = peak.loc[trough_idx]

    # trough 이후 peak 수준 회복 시점 찾기
    after_trough = cum.loc[trough_idx:]
    recovered = after_trough[after_trough >= peak_value]
    if len(recovered) == 0:
        return np.nan
    recovery_idx = recovered.index[0]

    # 영업일 수: 인덱스가 datetime이면 위치 차이로 계산
    pos_trough = cum.index.get_loc(trough_idx)
    pos_recovery = cum.index.get_loc(recovery_idx)
    return int(pos_recovery - pos_trough)


# ==========================================
# 3. 위험조정 수익 지표
# ==========================================
def sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Annualized Sharpe Ratio.
    (CAGR - rf) / annualized_vol

    Parameters
    ----------
    risk_free_rate : float
        연율 무위험수익률 (예: 0.03 = 3%). 기본 0.
    """
    vol = annualized_vol(daily_returns)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return (cagr(daily_returns) - risk_free_rate) / vol


def sortino_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Annualized Sortino Ratio.
    (CAGR - rf) / annualized_downside_vol

    downside_vol = std of returns where return < 0, 연율화.
    상방 변동성에 페널티 안 줌 → Sharpe의 비대칭 개선.
    """
    r = daily_returns.dropna()
    downside = r[r < 0]
    if len(downside) == 0:
        return np.nan
    downside_vol = downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    if downside_vol == 0:
        return np.nan
    return (cagr(r) - risk_free_rate) / downside_vol


def calmar_ratio(daily_returns: pd.Series) -> float:
    """
    Calmar Ratio = CAGR / |MDD|.
    Young (1991). MDD를 직접 분모로 → 장기투자자 선호.
    """
    mdd = max_drawdown(daily_returns)
    if mdd is None or np.isnan(mdd) or mdd == 0:
        return np.nan
    return cagr(daily_returns) / abs(mdd)


# ==========================================
# 4. Entropic Risk Measure (학교 수업 핵심)
# ==========================================
def entropic_risk_measure(
    daily_returns: pd.Series,
    alpha: float = 1.0,
    annualize: bool = True,
) -> float:
    """
    Entropic Risk Measure.

        ρ_α(X) = (1/α) · log E[exp(-α X)]

    Parameters
    ----------
    daily_returns : pd.Series
        일별 수익률.
    alpha : float
        위험회피 계수 (α > 0). 클수록 좌측꼬리(손실)에 큰 가중.
    annualize : bool
        True면 일별 → 연율 환산 (× 252).
        엄밀 정의는 단순 곱이 아니지만, 일별 entropic risk × 252는 실무 관행.

    Returns
    -------
    float
        risk measure. 작을수록 좋음.

    Notes
    -----
    - log-sum-exp trick으로 수치 안정화 (alpha 크거나 outlier 있을 때 overflow 방지).
    - 평가 지표로 사용: 모델 비교 시 entropic risk가 작은 모델이 우수.

    References
    ----------
    Föllmer & Schied (2002), Stochastic Finance, Ch.4.
    Bühler et al. (2019), "Deep Hedging", Quantitative Finance.
    """
    if alpha <= 0:
        raise ValueError("alpha는 양수여야 합니다.")

    r = daily_returns.dropna().values
    n = len(r)
    if n == 0:
        return np.nan

    # log-sum-exp 안정화: log(mean(exp(z))) = logsumexp(z) - log(n)
    z = -alpha * r
    z_max = z.max()
    log_mean_exp = z_max + np.log(np.exp(z - z_max).mean())

    rho_daily = log_mean_exp / alpha

    if annualize:
        return rho_daily * TRADING_DAYS_PER_YEAR
    return rho_daily


def certainty_equivalent(
    daily_returns: pd.Series,
    alpha: float = 1.0,
    annualize: bool = True,
) -> float:
    """
    Certainty Equivalent (확실성 등가).
        CE_α(X) = -ρ_α(X) = -(1/α) · log E[exp(-α X)]

    "위험 자산 X와 효용 동등한 확실한 수익률".
    클수록 좋음 (entropic risk의 부호 반전).

    실무 해석: "이 포트폴리오를 운용하는 사람에게 entropic 관점에서 동등한 무위험 수익률"
    """
    return -entropic_risk_measure(daily_returns, alpha=alpha, annualize=annualize)


# ==========================================
# 5. Probabilistic Sharpe Ratio (보너스)
# ==========================================
def probabilistic_sharpe_ratio(
    daily_returns: pd.Series,
    benchmark_sharpe: float = 0.0,
) -> float:
    """
    Probabilistic Sharpe Ratio.
    "관측된 Sharpe가 benchmark_sharpe보다 클 확률"을 통계적으로 계산.

    Bailey & López de Prado (2012).

    Parameters
    ----------
    benchmark_sharpe : float
        비교 기준 (연율). 기본 0 (수익률이 양수일 확률).

    Returns
    -------
    float
        0~1 사이 확률. 0.95 이상이면 통계적 유의 (95% 신뢰).

    Notes
    -----
    표본 Sharpe는 noise에 약함. 짧은 백테스트의 높은 Sharpe는 우연일 수 있음.
    PSR은 표본 크기, skewness, kurtosis까지 반영하여 통계적 유의성 판단.
    """
    r = daily_returns.dropna()
    n = len(r)
    if n < 30:
        return np.nan

    sr = sharpe_ratio(r)
    if np.isnan(sr):
        return np.nan

    # 일별 Sharpe로 환산 (PSR 공식은 연율 단위 일관성 유지)
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=True)  # excess kurtosis

    # 연율 단위로 작업
    sr_daily = sr / np.sqrt(TRADING_DAYS_PER_YEAR)
    benchmark_daily = benchmark_sharpe / np.sqrt(TRADING_DAYS_PER_YEAR)

    # PSR 분모 (Sharpe의 표준오차에 skew/kurt 보정)
    denom = np.sqrt(1 - skew * sr_daily + (kurt / 4) * sr_daily ** 2)
    if denom == 0 or np.isnan(denom):
        return np.nan
    z = (sr_daily - benchmark_daily) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


# ==========================================
# 6. 통합 리포트 (모델 비교용)
# ==========================================
def compute_all_metrics(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
    alphas: List[float] = [0.5, 1.0, 2.0, 5.0],
) -> Dict[str, float]:
    """
    한 번에 모든 지표 계산.

    Parameters
    ----------
    daily_returns : pd.Series
        일별 수익률.
    risk_free_rate : float
        연율 무위험수익률 (Sharpe/Sortino용).
    alphas : list of float
        Entropic risk sensitivity analysis용 α 값들.

    Returns
    -------
    dict
        지표명 → 값.
    """
    result = {
        "total_return":        total_return(daily_returns),
        "cagr":                cagr(daily_returns),
        "annualized_vol":      annualized_vol(daily_returns),
        "mdd":                 max_drawdown(daily_returns),
        "mdd_recovery_days":   mdd_recovery_days(daily_returns),
        "sharpe":              sharpe_ratio(daily_returns, risk_free_rate),
        "sortino":             sortino_ratio(daily_returns, risk_free_rate),
        "calmar":              calmar_ratio(daily_returns),
        "psr_vs_zero":         probabilistic_sharpe_ratio(daily_returns, 0.0),
    }
    # Entropic risk sensitivity
    for a in alphas:
        result[f"entropic_a{a}"] = entropic_risk_measure(daily_returns, alpha=a)
        result[f"ce_a{a}"]       = certainty_equivalent(daily_returns, alpha=a)

    return result


def compare_strategies(
    strategies: Dict[str, pd.Series],
    risk_free_rate: float = 0.0,
    alphas: List[float] = [0.5, 1.0, 2.0, 5.0],
) -> pd.DataFrame:
    """
    여러 전략의 지표를 한 표로 비교.

    Parameters
    ----------
    strategies : dict[str, pd.Series]
        전략 이름 → 일별 수익률.

    Returns
    -------
    pd.DataFrame
        index: 전략명, columns: 지표명.
    """
    rows = {name: compute_all_metrics(ret, risk_free_rate, alphas)
            for name, ret in strategies.items()}
    return pd.DataFrame(rows).T


# ==========================================
# 7. 자체 검증 (수동 실행 시)
# ==========================================
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 70)
    print("backtest/metrics.py 단독 실행 테스트 (시뮬레이션 전략 3개 비교)")
    print("=" * 70)

    np.random.seed(42)
    n = 2520  # 10년 영업일
    idx = pd.date_range("2015-01-01", periods=n, freq="B")

    # 시나리오 1: QQQ 모방 (고수익, 고변동성, 깊은 MDD)
    qqq_like = pd.Series(np.random.normal(0.0006, 0.014, n), index=idx)
    # 폭락 구간 삽입
    qqq_like.iloc[800:850] = np.random.normal(-0.01, 0.025, 50)

    # 시나리오 2: 60/40 모방 (중수익, 중변동성)
    bal_like = pd.Series(np.random.normal(0.0004, 0.008, n), index=idx)
    bal_like.iloc[800:850] = np.random.normal(-0.005, 0.015, 50)

    # 시나리오 3: UCHIDA 가정 (방어형, 낮은 MDD)
    uchida_like = pd.Series(np.random.normal(0.0005, 0.009, n), index=idx)
    uchida_like.iloc[800:850] = np.random.normal(-0.002, 0.012, 50)  # 위기 회피

    strategies = {
        "QQQ_like":    qqq_like,
        "60_40_like":  bal_like,
        "UCHIDA_like": uchida_like,
    }

    df = compare_strategies(strategies, risk_free_rate=0.03)

    # ----- 보기 좋게 포맷 -----
    print(f"\n[전략 비교 표]")
    formatted = df.copy()
    pct_cols = ["total_return", "cagr", "annualized_vol", "mdd"] + \
               [c for c in df.columns if c.startswith("entropic_") or c.startswith("ce_")]
    for col in pct_cols:
        formatted[col] = (df[col] * 100).round(2).astype(str) + "%"
    other_cols = ["sharpe", "sortino", "calmar", "psr_vs_zero", "mdd_recovery_days"]
    for col in other_cols:
        if col == "mdd_recovery_days":
            formatted[col] = df[col].round(0)
        else:
            formatted[col] = df[col].round(3)

    print(formatted.T.to_string())

    print(f"\n[Entropic Risk Sensitivity 해석]")
    print("  α 변화에 따라 entropic risk가 어떻게 바뀌는지 (작을수록 좋음)")
    print("  → α 클수록 폭락 회피 전략(UCHIDA_like)이 더 유리해 보여야 정상")
    for a in [0.5, 1.0, 2.0, 5.0]:
        print(f"  α={a}: ", end="")
        ranked = df[f"entropic_a{a}"].sort_values()
        for i, (name, val) in enumerate(ranked.items()):
            print(f"{i+1}위 {name}({val*100:.2f}%)  ", end="")
        print()

    print("\n✓ metrics.py 검증 완료")
