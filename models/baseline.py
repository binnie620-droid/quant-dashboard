"""
UCHIDA V3 - Rule-Based Baseline Model

기존 UCHIDA 룰을 새 인프라(features.py) 위에서 재구현.
LSTM과 비교할 베이스라인 역할.

[핵심 원칙]
- features.py 출력을 입력으로 사용 (look-ahead 없음)
- 임계값은 하드코딩이지만 파라미터로 노출 (sensitivity analysis 가능)
- 비중 맵은 한 곳에만 정의 (기존 코드의 불일치 해결)
- 출력 형식이 LSTM과 동일 → 백테스트 엔진 공통 사용 가능

[국면 분류 로직]
1. 각 신호가 임계값을 넘으면 flag=1
2. 비대칭 유예: 최근 lookback일 내에 flag=1이 있으면 현재도 1로 유지
   (위기 신호는 빠르게 반응, 해제는 천천히)
3. 가중합산 → 총 위험 점수
4. 점수 구간에 따라 국면 결정

[비중 맵]
ATTACK:  QQQ 60% / SPY 25% / GLD 10% / IEF 5%
DEFENSE: QQQ 20% / SPY 10% / GLD 25% / IEF 30% / SOFR 15%
CRISIS:  GLD 30% / SOFR 50% / TLT 20%

[참고문헌]
기존 UCHIDA v1/v2 코드 (uchida.py, uchida_backtest.py).
임계값 근거: 각 함수 docstring 참조.
"""

import numpy as np
import pandas as pd
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REBALANCE, CRISIS_THR


# ==========================================
# 1. 임계값 설정
# ==========================================
@dataclass
class BaselineConfig:
    """
    룰베이스 모델 임계값 설정.
    모든 값을 파라미터로 노출하여 sensitivity analysis 가능.

    [임계값 근거]
    - cpi_z_threshold (2.3): 기존 UCHIDA 코드 유지. Z-score 2.3은 약 99퍼센타일.
      인플레이션이 역사적으로 극단적인 수준일 때만 위기로 분류.
    - credit_spread_threshold (3.5%): BAA-10Y 스프레드 3.5% 이상.
      Kindleberger (1978), "Manias, Panics, and Crashes"에서 4% 이상을 금융위기
      전조로 제시. 3.5%는 사전 경보 기준으로 하향.
    - t10y2y_threshold (0%): 장단기 역전. Campbell & Shiller (1991),
      "Yield Spreads and Interest Rate Movements" — 역전은 경기침체 선행지표.
      0% 기준은 학계 표준.
    - vix_threshold (22): VIX 22 이상. Whaley (2009), "Understanding the VIX"
      에서 VIX 20을 "investor fear gauge" 임계점으로 제시. 22는 약간 보수적 기준.
    - ma_dist_threshold (-0.03): QQQ가 MA200 대비 -3% 이하.
      기존 UCHIDA 코드 유지. 0.97 = 3% 이탈.

    [가중치 근거]
    CPI/SPREAD는 구조적 위기 (가중치 높음),
    VIX/TREND는 단기 노이즈 (가중치 낮음). 기존 코드 참조.
    """
    # 임계값
    cpi_z_threshold:      float = 1.5  # [2026-05] 2.3→1.5. 실측: 2022 인플레 위기 cpi_z max=2.0
    credit_spread_threshold: float = 3.5
    t10y2y_threshold:     float = 0.0
    vix_threshold:        float = 22.0
    ma_dist_threshold:    float = -0.03   # (price/MA200 - 1) < -0.03

    # 가중치 (2026-05 재조정 — 실측 데이터 기반)
    # 기존 cpi=1.5는 22년간 단 한 번도 발동 안 함 (임계값 2.3 비현실적)
    # 변경 후: VIX/Trend는 모든 위기에서 작동, yield는 2022 같은 금리위기에 작동
    w_cpi:    float = 0.5   # 0.5로 낮춤 (임계값 완화로 발동률↑ → 가중치는 낮게)
    w_spread: float = 1.5   # 유지 (2008 신용위기 핵심)
    w_yield:  float = 1.0   # 유지 (2022 금리위기 핵심)
    w_vix:    float = 1.0   # 0.5→1.0 (모든 위기 공통)
    w_trend:  float = 1.0   # 0.5→1.0 (모든 위기 공통)
    # 합계 = 5.0 (변경 없음)

    # 비대칭 유예 기간 (영업일): 위기 신호는 이 기간 동안 유지
    lookback_cpi:    int = 3
    lookback_spread: int = 3
    lookback_yield:  int = 1
    lookback_vix:    int = 5
    lookback_trend:  int = 5

    # 국면 전환 임계 점수
    attack_threshold:  float = 1.5   # score < 1.5 → ATTACK
    crisis_threshold:  float = 2.5   # score >= 2.5 → CRISIS
    # [2026-05 수정] 3.0 → 2.5. 새 가중치 기준 예상 점수:
    # - 2008/2020 (spread+vix+trend): 1.5+1.0+1.0 = 3.5 ✓
    # - 2022 (cpi+yield+vix+trend): 0.5+1.0+1.0+1.0 = 3.5 ✓

    # [D-019] SIDEWAYS 박스권 신호
    # ADX < adx_threshold AND Bollinger Width < p30 rolling
    use_sideways:          bool  = True
    adx_threshold:         float = 20.0   # Wilder(1978): <20 = 추세 없음
    boll_width_quantile:   float = 0.30   # 하위 30% = 변동성 압축 (AI 판단)
    boll_width_window:     int   = 756    # 3년 rolling (동적 임계값)
    sideways_target:       str   = "QQQ"  # 박스권 신호 기준 자산
    # use_dynamic_threshold=True: 고정값 대신 rolling quantile 사용
    # look-ahead bias 방지: rolling().quantile().shift(1) 적용
    # window 3년(756일): 학계 일반 권장 + AI 판단
    use_dynamic_threshold: bool  = False
    dynamic_window_days:   int   = 756    # 3년 영업일 (AI 판단)

    # 분위수 위치 (학계 권장 범위 내, AI 판단)
    # [2026-05 실험 1] 분위수만 보수적으로 (window 3년 유지)
    # 이유: Baseline-D 결과 2015/2016 false positive 과다
    vix_quantile:          float = 0.90   # p80→p90 (상위 10%만 위험)
    spread_quantile:       float = 0.97   # p95→p97 (상위 3%만)
    yield_quantile:        float = 0.05   # p10→p05 (하위 5%만)
    cpi_quantile:          float = 0.92   # p85→p92 (상위 8%만)
    ma_dist_quantile:      float = 0.10   # p20→p10 (하위 10%만)
    # True (기본): 신용/패닉/인플레 각각 임계값 + recovery 점수
    # False: 구버전 가중합 방식 (비교/디버깅 용)
    use_level1: bool = False  # Level 1 재설계 포기. Legacy 가중합 방식 확정 (2026-05)

    # Level 1 상태 전환 lookback (None이면 config.CRISIS_THR 값 사용)
    entry_lookback: int = 5    # AI 판단: 5일 연속 (CRISIS_THR.entry_lookback과 동기화)
    exit_lookback:  int = 10   # AI 판단: 10일 (CRISIS_THR.exit_lookback과 동기화)

    @property
    def max_score(self) -> float:
        return self.w_cpi + self.w_spread + self.w_yield + self.w_vix + self.w_trend


# 기본 설정 싱글턴
DEFAULT_CONFIG = BaselineConfig()


# ==========================================
# 2. 비중 맵
# ==========================================
# [국면 정의 명시]
# ATTACK  : 성장 추구. SPY 상단 barrier 도달. QQQ/SPY 비중 최대.
# DEFENSE : 횡보/중립. Triple-Barrier T=21일 만료 (큰 방향성 없음).
#           "방어"라는 이름이지만 실제 의미는 "중립 균형 유지".
#           SOFR(현금성) + IEF(쿠폰 수익)로 횡보장 수익 확보.
# CRISIS  : 위기/방어. SPY 하단 barrier 도달. GLD/SOFR/TLT로 대피.
# ATTACK / DEFENSE / CRISIS 각각의 자산 비중
# 키는 config.ASSETS의 키와 일치해야 함
#
# [재설계 근거 - 2026.05]
# 사용자 성향: "MDD 희생해도 CAGR 올리고 싶음" + "SCHD 위기에도 일정 유지"
# - ATTACK: QQQ 70% + SCHD 15% (분산 + 성장)
# - DEFENSE: QQQ 25% + SCHD 20% (배당 + 안정)
# - CRISIS: SCHD 20% (위기에도 배당 수익 확보)
# - EEM은 신흥국 분산: ATTACK 5% / DEFENSE 5% / CRISIS 0%
# - SPY 제거 (QQQ와 상관 0.92로 중복), SCHD가 진짜 분산 (상관 0.65)
# [2026-05] 자산 정리: EEM/TLT/OIL 제거, 5자산으로 재배분
# 재배분 원칙:
# - EEM 자리 → ATTACK은 QQQ, DEFENSE는 IEF로 (안전자산 강화)
# - TLT 자리 → IEF/GLD로 (중기채/금이 장기채 대체)
# [D-019 폐기] 4-class -> 3-class 복원 (SIDEWAYS/JEPQ 제거)
# 3-class: ATTACK / DEFENSE / CRISIS. DECISION_LOG D-019 참조.
WEIGHT_MAP: Dict[str, Dict[str, float]] = {
    "ATTACK": {
        "QQQ": 0.75, "SCHD": 0.15, "GLD": 0.05, "IEF": 0.05, "SOFR": 0.0,
    },
    "DEFENSE": {
        "QQQ": 0.25, "SCHD": 0.20, "GLD": 0.20, "IEF": 0.25, "SOFR": 0.10,
    },
    "CRISIS": {
        "QQQ": 0.0,  "SCHD": 0.20, "GLD": 0.30, "IEF": 0.25, "SOFR": 0.25,
    },
}

# 4-class WEIGHT_MAP (CAUTION 추가, 실험용)
# CAUTION: 점수 0.8~1.5 구간 (과매수 경계)
# AI 판단: ATTACK(75%) ~ DEFENSE(25%) 중간값 50%로 시작
# 다른 자산은 ATTACK/DEFENSE 평균
WEIGHT_MAP_CAUTION: Dict[str, Dict[str, float]] = {
    "ATTACK": {
        "QQQ": 0.75, "SCHD": 0.15, "GLD": 0.05, "IEF": 0.05, "SOFR": 0.0,
    },
    "CAUTION": {
        "QQQ": 0.50, "SCHD": 0.18, "GLD": 0.12, "IEF": 0.15, "SOFR": 0.05,
    },
    "DEFENSE": {
        "QQQ": 0.25, "SCHD": 0.20, "GLD": 0.20, "IEF": 0.25, "SOFR": 0.10,
    },
    "CRISIS": {
        "QQQ": 0.0,  "SCHD": 0.20, "GLD": 0.30, "IEF": 0.25, "SOFR": 0.25,
    },
}

# 비중 합 = 1 검증
for _regime, _w in WEIGHT_MAP.items():
    assert abs(sum(_w.values()) - 1.0) < 1e-9, f"WEIGHT_MAP[{_regime}] 합 != 1"
for _regime, _w in WEIGHT_MAP_CAUTION.items():
    assert abs(sum(_w.values()) - 1.0) < 1e-9, f"WEIGHT_MAP_CAUTION[{_regime}] 합 != 1"


# ==========================================
# 3. 신호 계산
# ==========================================
def compute_flags(
    features: pd.DataFrame,
    cfg: BaselineConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    features에서 각 신호의 flag 계산 (0 또는 1).
    비대칭 유예 기간 적용: 최근 lookback일 내에 1이 있으면 1 유지.

    Parameters
    ----------
    features : pd.DataFrame
        data/features.py의 build_features() 출력.

    Returns
    -------
    pd.DataFrame
        columns: ['flag_cpi', 'flag_spread', 'flag_yield', 'flag_vix', 'flag_trend']
        values: 0 or 1
    """
    def _with_lookback(raw_flag: pd.Series, lookback: int) -> pd.Series:
        """rolling max: lookback 기간 내 1이 하나라도 있으면 1."""
        return raw_flag.rolling(lookback, min_periods=1).max()

    def _dynamic_thr(series: pd.Series, quantile: float, window: int,
                     upper: bool = True) -> pd.Series:
        """
        Rolling quantile 임계값 계산.
        look-ahead bias 방지: shift(1) 적용 (어제까지 분포로 오늘 판단).
        upper=True:  series > quantile (VIX, spread, cpi 등 상위 임계)
        upper=False: series < quantile (yield, ma_dist 등 하위 임계)
        """
        thr = series.rolling(window, min_periods=window // 3).quantile(quantile).shift(1)
        if upper:
            return (series > thr).astype(int)
        else:
            return (series < thr).astype(int)

    w = cfg.dynamic_window_days
    flags = pd.DataFrame(index=features.index)

    # CPI Z-score
    if "cpi_z" in features.columns:
        if cfg.use_dynamic_threshold:
            raw = _dynamic_thr(features["cpi_z"], cfg.cpi_quantile, w, upper=True)
        else:
            raw = (features["cpi_z"] > cfg.cpi_z_threshold).astype(int)
        flags["flag_cpi"] = _with_lookback(raw, cfg.lookback_cpi)
    else:
        flags["flag_cpi"] = 0

    # Credit spread
    if "credit_spread" in features.columns:
        if cfg.use_dynamic_threshold:
            raw = _dynamic_thr(features["credit_spread"], cfg.spread_quantile, w, upper=True)
        else:
            raw = (features["credit_spread"] > cfg.credit_spread_threshold).astype(int)
        flags["flag_spread"] = _with_lookback(raw, cfg.lookback_spread)
    else:
        flags["flag_spread"] = 0

    # Yield curve inversion
    if "t10y2y" in features.columns:
        if cfg.use_dynamic_threshold:
            raw = _dynamic_thr(features["t10y2y"], cfg.yield_quantile, w, upper=False)
        else:
            raw = (features["t10y2y"] < cfg.t10y2y_threshold).astype(int)
        flags["flag_yield"] = _with_lookback(raw, cfg.lookback_yield)
    else:
        flags["flag_yield"] = 0

    # VIX
    if "vix" in features.columns:
        if cfg.use_dynamic_threshold:
            raw = _dynamic_thr(features["vix"], cfg.vix_quantile, w, upper=True)
        else:
            raw = (features["vix"] > cfg.vix_threshold).astype(int)
        flags["flag_vix"] = _with_lookback(raw, cfg.lookback_vix)
    else:
        flags["flag_vix"] = 0

    # MA200 추세
    if "dist_ma200_QQQ" in features.columns:
        if cfg.use_dynamic_threshold:
            raw = _dynamic_thr(features["dist_ma200_QQQ"], cfg.ma_dist_quantile, w, upper=False)
        else:
            raw = (features["dist_ma200_QQQ"] < cfg.ma_dist_threshold).astype(int)
        flags["flag_trend"] = _with_lookback(raw, cfg.lookback_trend)
    else:
        flags["flag_trend"] = 0

    return flags


def compute_score(
    flags: pd.DataFrame,
    cfg: BaselineConfig = DEFAULT_CONFIG,
) -> pd.Series:
    """
    flag 가중합산 → 위험 점수.

    Returns
    -------
    pd.Series
        위험 점수 시계열 (0 ~ max_score)
    """
    score = (
        flags["flag_cpi"]    * cfg.w_cpi    +
        flags["flag_spread"] * cfg.w_spread +
        flags["flag_yield"]  * cfg.w_yield  +
        flags["flag_vix"]    * cfg.w_vix    +
        flags["flag_trend"]  * cfg.w_trend
    )
    return score.rename("risk_score")


# ==========================================
# 4. 국면 분류
# ==========================================
def classify_regime(
    score: pd.Series,
    cfg: BaselineConfig = DEFAULT_CONFIG,
) -> pd.Series:
    """
    위험 점수 → 국면 문자열 시계열.

    Returns
    -------
    pd.Series
        values: 'ATTACK' | 'DEFENSE' | 'CRISIS'
    """
    regime = pd.Series("DEFENSE", index=score.index, name="regime")
    regime[score < cfg.attack_threshold]  = "ATTACK"
    regime[score >= cfg.crisis_threshold] = "CRISIS"
    return regime


def classify_regime_caution(
    score: pd.Series,
    caution_threshold: float = 0.8,
    cfg: BaselineConfig = DEFAULT_CONFIG,
) -> pd.Series:
    """
    4-class: ATTACK / CAUTION / DEFENSE / CRISIS.

    CAUTION: 점수가 ATTACK 임계 가까이 올라옴 (과매수 경계).
    부드러운 비중 감소를 위한 버퍼 구간.

    [근거]
    - Black & Litterman (1992): 시그널 강도에 비례한 비중 조정
    - 본인 사용자 요구: ATTACK -> DEFENSE 절벽 완화

    Parameters
    ----------
    caution_threshold : float
        CAUTION 진입 임계값. AI 판단 (기본 0.8).
        점수 0.8~1.5 구간을 CAUTION으로.

    Returns
    -------
    pd.Series
        values: 'ATTACK' | 'CAUTION' | 'DEFENSE' | 'CRISIS'
    """
    regime = pd.Series("DEFENSE", index=score.index, name="regime")
    regime[score < caution_threshold]     = "ATTACK"
    regime[(score >= caution_threshold) & (score < cfg.attack_threshold)] = "CAUTION"
    regime[score >= cfg.crisis_threshold] = "CRISIS"
    return regime


def classify_regime_rsi_filter(
    score: pd.Series,
    features: pd.DataFrame,
    rsi_threshold: float = 75.0,
    ma_dist_threshold: float = 0.15,
    cfg: BaselineConfig = DEFAULT_CONFIG,
) -> pd.Series:
    """
    RSI 기반 과열 필터 (최종 판단 필터).

    [설계 원칙]
    - 1단계: 기존 Baseline 점수로 국면 판정 (변경 없음)
    - 2단계: ATTACK 국면 중 RSI + MA200 거리 동시 극단 → CAUTION 격하
    - DEFENSE/CRISIS는 건드리지 않음

    [근거]
    - Wilder (1978): RSI 70 이상 과매수 신호
    - Connors & Alvarez (2008): RSI 75+ 단기 고점 신호
    - MA200 거리 +15%: 추세 과도 이탈 (Faber 2007 연장)
    - 두 조건 AND: false positive 최소화 (AI 판단)

    [CAUTION 비중]
    - QQQ 50% (ATTACK 75%와 DEFENSE 25% 중간)
    - 나머지는 WEIGHT_MAP_CAUTION 참조

    Parameters
    ----------
    rsi_threshold : float
        RSI 과열 임계값. 기본 75.0 (AI 판단, Wilder 70보다 보수적).
    ma_dist_threshold : float
        MA200 거리 임계값. 기본 +15% (AI 판단).

    Returns
    -------
    pd.Series
        values: 'ATTACK' | 'CAUTION' | 'DEFENSE' | 'CRISIS'
    """
    # 1단계: 기존 Baseline 국면
    regime = classify_regime(score, cfg)

    # 2단계: 과열 조건 계산
    # rsi_QQQ, dist_ma200_QQQ 둘 다 features에 있음
    if "rsi_QQQ" not in features.columns or "dist_ma200_QQQ" not in features.columns:
        return regime  # 피처 없으면 그대로

    common_idx = regime.index.intersection(features.index)
    rsi     = features.loc[common_idx, "rsi_QQQ"]
    ma_dist = features.loc[common_idx, "dist_ma200_QQQ"]

    # 과열 조건: RSI > 75 AND MA200 거리 > +15% (동시 충족)
    overheated = (rsi > rsi_threshold) & (ma_dist > ma_dist_threshold)

    # ATTACK 국면에서만 CAUTION으로 격하
    result = regime.copy()
    result.loc[common_idx] = regime.loc[common_idx].copy()
    caution_mask = (regime.loc[common_idx] == "ATTACK") & overheated
    result.loc[common_idx[caution_mask]] = "CAUTION"

    return result


def classify_regime_naaim_filter(
    score: pd.Series,
    naaim_path: str = "data/naaim.xlsx",
    naaim_threshold: float = 90.0,
    naaim_window: int = 20,
    cfg: BaselineConfig = DEFAULT_CONFIG,
) -> pd.Series:
    """
    NAAIM Exposure Index 기반 과열 필터 (최종 판단 필터).

    [설계 원칙]
    - 1단계: 기존 Baseline 점수로 국면 판정 (변경 없음)
    - 2단계: ATTACK 국면 중 NAAIM 4주 평균 > 90 → CAUTION 격하
    - DEFENSE/CRISIS는 건드리지 않음

    [근거]
    - NAAIM: 액티브 펀드매니저 실제 주식 익스포저 (0~200%)
    - 4주 평균 > 90: 펀드매니저 대부분이 풀 베팅 = 과열
    - 발동 빈도: 12.4% (RSI 필터 2.8%보다 많음)
    - 데이터: 2006~현재, 주간 발표 (NAAIM.org 무료)

    Parameters
    ----------
    naaim_threshold : float
        과열 임계값. 기본 90.0 (AI 판단, 상위 ~12%).
    naaim_window : int
        이동평균 윈도우 (영업일). 기본 20 (4주).
    """
    # 1단계: 기존 Baseline 국면
    regime = classify_regime(score, cfg)

    # NAAIM 데이터 로드
    try:
        df_n = pd.read_excel(naaim_path)
        df_n = df_n[['Date', 'NAAIM Number']].dropna()
        df_n['Date'] = pd.to_datetime(df_n['Date'])
        df_n = df_n.drop_duplicates('Date').sort_values('Date')
        df_n = df_n.set_index('Date')

        # 주간 → 일별 forward fill
        idx = pd.date_range(df_n.index.min(), df_n.index.max(), freq='B')
        df_daily = df_n.reindex(idx).ffill()

        # 4주 이동평균 + look-ahead bias 방지 (shift 1)
        naaim_4w = df_daily['NAAIM Number'].rolling(naaim_window).mean().shift(1)
        naaim_4w.index = pd.DatetimeIndex(naaim_4w.index)

    except Exception as e:
        print(f"[NAAIM 로드 실패] {e}")
        return regime

    # 공통 인덱스
    common_idx = regime.index.intersection(naaim_4w.index)
    overheated = naaim_4w.loc[common_idx] > naaim_threshold

    # ATTACK 국면에서만 CAUTION으로 격하
    result = regime.copy()
    caution_mask = (regime.loc[common_idx] == "ATTACK") & overheated
    result.loc[common_idx[caution_mask]] = "CAUTION"

    return result


def classify_regime_4class(
    features: pd.DataFrame,
    cfg: BaselineConfig = DEFAULT_CONFIG,
) -> pd.Series:
    """
    4-class 국면 분류: ATTACK / SIDEWAYS / DEFENSE / CRISIS

    분류 로직:
    1. 기존 3-class (ATTACK/DEFENSE/CRISIS) 먼저 판단
    2. ATTACK 판정 구간 중 박스권 조건 충족 시 → SIDEWAYS로 교체
       (DEFENSE/CRISIS는 박스권 여부와 무관하게 유지 — 위험 신호 우선)

    박스권 조건 (이중, 보수적):
    - ADX(QQQ) < 20 (Wilder 1978)
    - Bollinger Width(QQQ) < rolling p30 (3년 window, look-ahead 방지)

    CRISIS 동적 임계값: VIX/spread 신호만 p85/p96 분위수 기반
    나머지 신호(cpi, yield, ma_dist)는 고정 임계값 유지 (ceteris paribus)
    """
    # 3-class 기본 분류
    flags  = compute_flags(features, cfg)
    score  = compute_score(flags, cfg)
    regime = classify_regime(score, cfg)

    if not cfg.use_sideways:
        return regime

    # --- 박스권 신호 계산 ---
    tgt = cfg.sideways_target  # 기준 자산 (QQQ)

    # ADX
    adx_col = f"adx_{tgt}"
    if adx_col not in features.columns:
        return regime  # ADX 없으면 3-class 유지

    adx = features[adx_col]
    is_low_adx = adx < cfg.adx_threshold

    # Bollinger Width
    bw_col = f"boll_w_{tgt}"
    if bw_col not in features.columns:
        return regime  # BB Width 없으면 3-class 유지

    bw = features[bw_col]
    bw_thr = (bw.rolling(cfg.boll_width_window, min_periods=cfg.boll_width_window // 3)
                .quantile(cfg.boll_width_quantile)
                .shift(1))  # look-ahead 방지
    is_compressed = bw < bw_thr

    # 박스권 = ADX 낮음 AND 변동성 압축
    is_sideways = is_low_adx & is_compressed

    # ATTACK 구간 중 박스권이면 SIDEWAYS로 교체
    # DEFENSE/CRISIS는 유지 (위험 신호 우선)
    sideways_mask = (regime == "ATTACK") & is_sideways
    regime = regime.copy()
    regime[sideways_mask] = "SIDEWAYS"

    return regime


# ==========================================
# 4-b. Level 1 재설계: 위기 유형별 분리 분류 (D-017)
# ==========================================
def _score_credit(features: pd.DataFrame) -> pd.Series:
    """
    신용 위기 점수.
    조건: BAA10Y 스프레드 > 2.5%p AND VIX > 25
    근거: Gilchrist & Zakrajšek (2012, AER) — 신용 스프레드 위기 식별
    반환: bool Series (True = 신용 위기 조건 충족)
    """
    spread = features.get("credit_spread", pd.Series(0.0, index=features.index))
    vix    = features.get("vix",           pd.Series(0.0, index=features.index))
    return (spread > CRISIS_THR.credit_spread) & (vix > CRISIS_THR.credit_vix)


def _score_panic(features: pd.DataFrame) -> pd.Series:
    """
    패닉/추세 위기 점수.
    조건: VIX > 30  OR  (VIX > 25 AND dist_ma200_QQQ < -10%)
    근거: Bloom (2009, Econometrica); Whaley (2000, JPM)
    반환: bool Series
    """
    vix  = features.get("vix",           pd.Series(0.0, index=features.index))
    ma   = features.get("dist_ma200_QQQ", pd.Series(0.0, index=features.index))
    high = vix > CRISIS_THR.panic_vix_high
    mid  = (vix > CRISIS_THR.panic_vix_mid) & (ma < CRISIS_THR.panic_ma200)
    return high | mid


def _score_inflation(features: pd.DataFrame) -> pd.Series:
    """
    인플레이션 위기 점수.
    조건: cpi_z > 2.0 AND t10y2y < 0.5
    근거: Estrella & Mishkin (1998, RES) — yield curve; cpi_z 기준은 AI 판단(2-sigma)
    반환: bool Series
    """
    cpi    = features.get("cpi_z",  pd.Series(0.0, index=features.index))
    t10y2y = features.get("t10y2y", pd.Series(9.9, index=features.index))
    return (cpi > CRISIS_THR.inflation_cpi_z) & (t10y2y < CRISIS_THR.inflation_yield)


def _score_recovery(features: pd.DataFrame) -> pd.Series:
    """
    Recovery 점수 (CRISIS → DEFENSE 전환 허용 조건).
    조건: spread < 1.5%p AND VIX < 20 AND dist_ma200 > -5%
    근거: Chauvet & Piger (2008, JBES) — 침체 종료 판정 지연
    반환: bool Series (True = 회복 조건 충족)
    """
    spread = features.get("credit_spread", pd.Series(0.0, index=features.index))
    vix    = features.get("vix",           pd.Series(99.0, index=features.index))
    ma     = features.get("dist_ma200_QQQ", pd.Series(-1.0, index=features.index))
    return (spread < CRISIS_THR.recovery_spread) & \
           (vix    < CRISIS_THR.recovery_vix)    & \
           (ma     > CRISIS_THR.recovery_ma200)


def classify_regime_level1(
    features: pd.DataFrame,
    cfg: BaselineConfig = DEFAULT_CONFIG,
) -> pd.Series:
    """
    Level 1 재설계: 유형별 분리 점수 + recovery 비대칭 상태 전환.

    분류 로직:
    1. 위기 유형 3개 (신용/패닉/인플레) 중 하나라도 entry_lookback일 연속 충족 → CRISIS
    2. CRISIS 상태에서 recovery 조건이 exit_lookback일 연속 충족 → DEFENSE
    3. 그 외: 기존 compute_score + classify_regime으로 ATTACK/DEFENSE 결정

    Parameters
    ----------
    features : pd.DataFrame
    cfg : BaselineConfig

    Returns
    -------
    pd.Series
        values: 'ATTACK' | 'DEFENSE' | 'CRISIS'
    """
    n = len(features)
    idx = features.index

    # 각 유형 raw signal (bool → int)
    s_credit  = _score_credit(features).astype(int)
    s_panic   = _score_panic(features).astype(int)
    s_inflat  = _score_inflation(features).astype(int)
    s_recover = _score_recovery(features).astype(int)

    # entry: rolling min (연속 entry_lookback일 모두 1이어야 진입)
    # exit: rolling min (연속 exit_lookback일 모두 1이어야 해제)
    # entry/exit lookback: CRISIS_THR 우선, cfg는 override용
    el = getattr(cfg, 'entry_lookback', None) or CRISIS_THR.entry_lookback
    xl = getattr(cfg, 'exit_lookback',  None) or CRISIS_THR.exit_lookback

    crisis_entry = (
        s_credit.rolling(el,  min_periods=el).min().fillna(0).astype(bool) |
        s_panic.rolling(el,   min_periods=el).min().fillna(0).astype(bool) |
        s_inflat.rolling(el,  min_periods=el).min().fillna(0).astype(bool)
    )

    recovery_exit = s_recover.rolling(xl, min_periods=xl).min().fillna(0).astype(bool)

    # 상태 머신: 이전 상태에 의존하는 순차 루프
    # CRISIS 유지/해제는 상태 의존적이므로 vectorize 불가 → 루프 (n ≤ 5,500일, 허용)
    regime_arr = ["DEFENSE"] * n

    # ATTACK/DEFENSE 기준선: 기존 compute_score + classify_regime 재사용
    flags    = compute_flags(features, cfg)
    score    = compute_score(flags, cfg)
    base_arr = classify_regime(score, cfg).values  # ATTACK/DEFENSE only here

    in_crisis = False

    for i in range(n):
        if not in_crisis:
            if crisis_entry.iloc[i]:
                in_crisis = True
                regime_arr[i] = "CRISIS"
            else:
                regime_arr[i] = base_arr[i]   # ATTACK or DEFENSE
        else:
            # CRISIS 상태: recovery_exit는 이미 rolling(xl).min() 적용됨
            # → True이면 xl일 연속 회복 조건 충족 → 즉시 탈출
            if recovery_exit.iloc[i]:
                in_crisis = False
                regime_arr[i] = base_arr[i]
            else:
                regime_arr[i] = "CRISIS"

    return pd.Series(regime_arr, index=idx, name="regime")


# ==========================================
# 5. Momentum & Volatility Targeting (alpha layers)
# ==========================================
def apply_momentum_filter(
    weights: pd.DataFrame,
    close_prices: pd.DataFrame,
    momentum_window: int = 252,
    risk_free_asset: str = "SOFR",
) -> pd.DataFrame:
    """
    Absolute Momentum 필터 (Antonacci 2014).

    각 자산의 momentum_window일 수익률이 0보다 작으면 비중을 risk_free_asset으로 이동.

    Parameters
    ----------
    weights : pd.DataFrame
        baseline의 비중 (날짜 × 자산)
    close_prices : pd.DataFrame
        종가 (날짜 × 자산), weights와 동일 컬럼 포함해야 함
    momentum_window : int
        모멘텀 계산 윈도우 (영업일). 기본 252 = 1년.
    risk_free_asset : str
        모멘텀 음수 자산의 비중을 옮길 자산.

    Returns
    -------
    pd.DataFrame
        모멘텀 필터 적용된 비중. 각 행 합 = 1 보장.

    References
    ----------
    Antonacci (2014), "Dual Momentum Investing"
    Jegadeesh & Titman (1993), Journal of Finance.
    """
    # 모멘텀 계산: log(P_t / P_{t-window})
    log_prices = np.log(close_prices)
    momentum = log_prices - log_prices.shift(momentum_window)

    # 날짜 정렬
    common_idx = weights.index.intersection(momentum.index)
    weights = weights.loc[common_idx].copy()
    momentum = momentum.loc[common_idx]

    # 각 시점에서 모멘텀 음수 자산의 비중을 risk_free_asset으로 이동
    result = weights.copy()
    for asset in weights.columns:
        if asset == risk_free_asset:
            continue
        if asset not in momentum.columns:
            continue
        # momentum이 음수이거나 NaN인 시점
        bad_momentum = (momentum[asset] < 0) | momentum[asset].isna()
        # 그 자산의 비중을 risk_free_asset으로 이동
        result.loc[bad_momentum, risk_free_asset] += result.loc[bad_momentum, asset]
        result.loc[bad_momentum, asset] = 0.0

    return result


def apply_momentum_soft_scaling(
    weights: pd.DataFrame,
    close_prices: pd.DataFrame,
    target_asset: str = "QQQ",
    risk_free_asset: str = "SOFR",
    momentum_window: int = 252,
    floor: float = -0.20,
) -> pd.DataFrame:
    """
    QQQ 전용 Soft-scaling 모멘텀 필터.

    Hard Cap(이진법)과 달리 시장 수익률에 비례해서 비중을 부드럽게 조정.
    GLD/IEF/SCHD 등 안전자산은 건드리지 않아 매크로 신호와 충돌 없음.

    [작동 방식 - Clipped Linear]
    - 기준: target_asset(QQQ)의 시장 12개월 수익률 (개인 평단 무관)
    - 수익률 >= 0%:           scale = 1.0 (강세장, 비중 그대로)
    - 0% > 수익률 > floor:    scale = 1 + momentum / |floor|  (선형 감소)
    - 수익률 <= floor(-20%):  scale = 0.0 (하한 cap)

    예시 (floor=-20%):
      수익률  0% → scale 1.00 → QQQ 비중 그대로
      수익률 -5% → scale 0.75 → QQQ 25% 감소, SOFR으로 이동
      수익률-10% → scale 0.50 → QQQ 50% 감소
      수익률-20% → scale 0.00 → QQQ 전부 SOFR

    [설계 근거]
    - target_asset만 적용: GLD/IEF는 위기 시 상승하므로 모멘텀 적용 부적절
    - Clipped Linear: 단순하고 해석 가능 (AI 판단)
    - floor=-20%: 닷컴 버블(-78%), GFC(-49%) 대비 보수적 완충 (AI 판단)
    - 학계: Moskowitz, Ooi, Pedersen (2012), JFE "Time Series Momentum"
      → 12개월 수익률 기반 비중 조정의 이론적 근거

    Parameters
    ----------
    weights : pd.DataFrame
        baseline의 목표 비중 (날짜 x 자산)
    close_prices : pd.DataFrame
        시장 종가 (yfinance 기준, 개인 평단 무관)
    target_asset : str
        soft-scaling 적용 자산 (기본 QQQ만)
    risk_free_asset : str
        줄어든 비중을 받을 자산 (기본 SOFR)
    momentum_window : int
        모멘텀 계산 윈도우 (기본 252일 = 1년)
    floor : float
        scale=0이 되는 수익률 하한 (기본 -20%, AI 판단)

    Returns
    -------
    pd.DataFrame
        soft-scaling 적용된 비중. 각 행 합 = 1 보장.
    """
    # 시장 수익률 계산 (log return, 개인 평단 무관)
    if target_asset not in close_prices.columns:
        return weights

    log_px = np.log(close_prices[target_asset])
    momentum = (log_px - log_px.shift(momentum_window)).reindex(weights.index)

    result = weights.copy()

    # scale 계산 (Clipped Linear)
    # 수익률 >= 0: scale=1, floor <= 수익률 < 0: 선형, 수익률 < floor: scale=0
    scale = momentum.clip(lower=floor, upper=0.0) / abs(floor) + 1.0
    scale = scale.clip(lower=0.0, upper=1.0)
    scale = scale.fillna(1.0)  # 초기 252일: 모멘텀 계산 불가 → 건드리지 않음

    # QQQ 비중 조정 (줄어든 만큼 SOFR으로)
    if target_asset in result.columns and risk_free_asset in result.columns:
        original_qqq = result[target_asset].copy()
        scaled_qqq   = original_qqq * scale
        delta        = original_qqq - scaled_qqq  # 줄어든 양

        result[target_asset]  = scaled_qqq
        result[risk_free_asset] = result[risk_free_asset] + delta

    return result


def apply_volatility_target(
    weights: pd.DataFrame,
    close_prices: pd.DataFrame,
    vol_target: float = 0.12,
    vol_window: int = 60,
    risk_free_asset: str = "SOFR",
    max_leverage: float = 1.0,
) -> pd.DataFrame:
    """
    Volatility Targeting (Moreira & Muir 2017).

    포트폴리오의 실현 변동성이 vol_target을 초과하면 비중을 축소,
    부족분은 risk_free_asset으로 이동. 레버리지 금지 (max_leverage=1.0).

    Parameters
    ----------
    weights : pd.DataFrame
        모멘텀 필터 후 비중
    close_prices : pd.DataFrame
        종가
    vol_target : float
        목표 연율 변동성 (기본 0.12 = 12%)
    vol_window : int
        변동성 계산 윈도우 (영업일). 기본 60일.
    risk_free_asset : str
        남는 비중을 옮길 자산.
    max_leverage : float
        최대 비중 배수. 1.0 = 레버리지 금지.

    Returns
    -------
    pd.DataFrame
        변동성 타겟팅 적용된 비중. 각 행 합 = 1.

    References
    ----------
    Moreira & Muir (2017), "Volatility-Managed Portfolios",
    Journal of Finance.
    """
    # 1. 각 자산의 일별 log return
    log_ret = np.log(close_prices / close_prices.shift(1))

    # 2. 포트폴리오의 실현 변동성 (단순화: 자산별 vol × 비중)
    # 정확한 portfolio vol은 covariance 필요하지만, 학계 표준 단순화:
    # σ_port ≈ Σ w_i × σ_i (상관관계 고려 안 함)
    asset_vol = log_ret.rolling(vol_window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    common_idx = weights.index.intersection(asset_vol.index)
    weights = weights.loc[common_idx].copy()
    asset_vol = asset_vol.loc[common_idx]

    # 가중 vol (포트폴리오 vol 근사)
    common_assets = [a for a in weights.columns if a in asset_vol.columns]
    port_vol = (weights[common_assets] * asset_vol[common_assets]).sum(axis=1)

    # 3. Scaling factor
    scaling = vol_target / port_vol.replace(0, np.nan)
    scaling = scaling.clip(upper=max_leverage).fillna(1.0)

    # 4. 비중 적용
    result = weights.copy()
    for asset in result.columns:
        if asset == risk_free_asset:
            continue
        result[asset] = result[asset] * scaling

    # 남는 비중 → risk_free_asset
    used = result.drop(columns=[risk_free_asset]).sum(axis=1)
    result[risk_free_asset] = 1.0 - used

    return result


TRADING_DAYS_PER_YEAR = 252  # 모듈 전역


# ==========================================
# 6. 비중 결정 (메인 인터페이스 - 확장)
# ==========================================
def predict_weights(
    features: pd.DataFrame,
    cfg: BaselineConfig = DEFAULT_CONFIG,
    tolerance_band: float = None,
    use_momentum: bool = False,
    use_mom_soft: bool = False,
    use_vol_target: bool = False,
    close_prices: pd.DataFrame = None,
    momentum_window: int = 252,
    mom_soft_floor: float = -0.20,
    vol_target: float = 0.12,
    vol_window: int = 60,
    weight_map: Dict[str, Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    features → 목표 비중 DataFrame.

    백테스트 엔진(engine.py)의 target_weights 입력으로 사용.

    Parameters
    ----------
    features : pd.DataFrame
        build_features() 출력
    cfg : BaselineConfig
        임계값 설정
    tolerance_band : float
        비중 변화 임계 (None이면 config 사용)
    use_momentum : bool
        True면 Hard Cap 절대 모멘텀 필터 (Antonacci 2014). 이진법.
    use_mom_soft : bool
        True면 Soft-scaling 모멘텀 (QQQ만, Clipped Linear).
        use_momentum과 동시 사용 불가. 둘 다 True면 soft 우선.
    use_vol_target : bool
        True면 volatility targeting 적용 (Moreira & Muir 2017)
    close_prices : pd.DataFrame
        모멘텀/변동성 계산에 필요. use_momentum/use_mom_soft/use_vol_target=True면 필수.
    momentum_window : int
        모멘텀 윈도우 (영업일). 기본 252.
    mom_soft_floor : float
        soft-scaling 하한 수익률. 기본 -0.20 (AI 판단).
    vol_target : float
        목표 변동성. 기본 0.12 (12%).
    vol_window : int
        변동성 계산 윈도우. 기본 60.
    weight_map : dict
        국면별 비중 맵. None이면 모듈 전역 WEIGHT_MAP 사용.

    Returns
    -------
    pd.DataFrame
        index: features.index, columns: 자산명
        값: 목표 비중 (각 행 합=1)
    """
    if tolerance_band is None:
        tolerance_band = REBALANCE.tolerance_band
    if weight_map is None:
        weight_map = WEIGHT_MAP

    flags  = compute_flags(features, cfg)
    score  = compute_score(flags, cfg)

    # Level 1 재설계 (D-017): 유형별 분리 점수 사용 여부
    # [D-019 폐기] use_sideways 분기 제거, 3-class로 고정
    if cfg.use_level1:
        regime = classify_regime_level1(features, cfg)
    else:
        regime = classify_regime(score, cfg)

    # 국면 → 비중 매핑
    assets = list(weight_map["ATTACK"].keys())
    weights = pd.DataFrame(index=features.index, columns=assets, dtype=float)

    for date, r in regime.items():
        weights.loc[date] = weight_map[r]

    # Alpha layer 1a: Soft-scaling (QQQ 전용, Clipped Linear)
    if use_mom_soft:
        if close_prices is None:
            raise ValueError("use_mom_soft=True이면 close_prices 인자 필수")
        weights = apply_momentum_soft_scaling(
            weights, close_prices,
            momentum_window=momentum_window,
            floor=mom_soft_floor,
        )
    # Alpha layer 1b: Hard Cap (use_mom_soft가 없을 때만)
    elif use_momentum:
        if close_prices is None:
            raise ValueError("use_momentum=True이면 close_prices 인자 필수")
        weights = apply_momentum_filter(
            weights, close_prices,
            momentum_window=momentum_window,
        )

    # Alpha layer 2: Volatility targeting
    if use_vol_target:
        if close_prices is None:
            raise ValueError("use_vol_target=True이면 close_prices 인자 필수")
        weights = apply_volatility_target(
            weights, close_prices,
            vol_target=vol_target,
            vol_window=vol_window,
        )

    return weights


# ==========================================
# 6. 전체 신호 요약 (디버깅/분석용)
# ==========================================
def get_signal_summary(
    features: pd.DataFrame,
    cfg: BaselineConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    features → 전체 신호 요약 DataFrame.
    flags + score + regime + weights 통합.
    """
    flags  = compute_flags(features, cfg)
    score  = compute_score(flags, cfg)
    regime = classify_regime(score, cfg)
    weights = predict_weights(features, cfg)

    # Level 1 신호 (디버깅용)
    l1_signals = pd.DataFrame({
        "l1_credit":   _score_credit(features).astype(int),
        "l1_panic":    _score_panic(features).astype(int),
        "l1_inflation":_score_inflation(features).astype(int),
        "l1_recovery": _score_recovery(features).astype(int),
        "regime_l1":   classify_regime_level1(features, cfg),
    }, index=features.index)

    summary = pd.concat([flags, score, regime, l1_signals, weights], axis=1)
    return summary


# ==========================================
# 7. 자체 검증
# ==========================================
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 70)
    print("models/baseline.py 단독 실행 테스트 (더미 feature)")
    print("=" * 70)

    np.random.seed(42)
    n = 500
    idx = pd.date_range("2015-01-01", periods=n, freq="B")

    # 더미 feature: 평시 + 위기 구간 삽입
    features = pd.DataFrame({
        "cpi_z":          np.random.normal(0, 1, n),
        "credit_spread":  np.random.uniform(1.5, 3.0, n),
        "t10y2y":         np.random.uniform(-0.5, 2.5, n),
        "vix":            np.random.uniform(12, 25, n),
        "dist_ma200_QQQ": np.random.uniform(-0.02, 0.10, n),
    }, index=idx)

    # 위기 구간: 300~350일 (모든 신호 동시 발동)
    features.loc[idx[300:350], "cpi_z"]          = 3.0   # CPI 과열
    features.loc[idx[300:350], "credit_spread"]   = 4.5   # 신용위기
    features.loc[idx[300:350], "t10y2y"]          = -0.5  # 역전
    features.loc[idx[300:350], "vix"]             = 35.0  # 공포
    features.loc[idx[300:350], "dist_ma200_QQQ"]  = -0.10 # 추세 이탈

    # 방어 구간: 150~200일 (일부 신호)
    features.loc[idx[150:200], "vix"]             = 25.0
    features.loc[idx[150:200], "t10y2y"]          = -0.3

    # 예측
    weights = predict_weights(features)
    flags   = compute_flags(features)
    score   = compute_score(flags)
    regime  = classify_regime(score)

    # ----- 분포 확인 -----
    dist = regime.value_counts()
    print(f"\n[국면 분포]")
    for r in ["ATTACK", "DEFENSE", "CRISIS"]:
        cnt = dist.get(r, 0)
        print(f"  {r:8s}: {cnt:4d}일 ({cnt/n*100:.1f}%)")

    # ----- 위기 구간 검증 -----
    print(f"\n[위기 구간(300~350일) 국면 확인]")
    crisis_regime = regime.iloc[300:355]
    crisis_count = (crisis_regime == "CRISIS").sum()
    print(f"  CRISIS 판정: {crisis_count}일 / {len(crisis_regime)}일")
    assert crisis_count > 40, "위기 구간에 CRISIS가 충분히 감지되어야 함"
    print("  ✓ 통과")

    # ----- 비중 합 검증 -----
    print(f"\n[비중 합 검증]")
    weight_sums = weights.sum(axis=1)
    max_dev = (weight_sums - 1.0).abs().max()
    print(f"  최대 비중 합 편차: {max_dev:.2e}")
    assert max_dev < 1e-9, "비중 합이 1이 아닌 시점 존재"
    print("  ✓ 통과")

    # ----- 위기 구간 비중 확인 -----
    print(f"\n[위기 구간 비중 샘플 (인덱스 310~312일)]")
    print(weights.iloc[310:313].round(3).to_string())

    # ----- 점수 통계 -----
    print(f"\n[위험 점수 통계]")
    print(f"  평시(0~150) 평균: {score.iloc[:150].mean():.2f}")
    print(f"  방어(150~200) 평균: {score.iloc[150:200].mean():.2f}")
    print(f"  위기(300~350) 평균: {score.iloc[300:350].mean():.2f}")
    print(f"  최대 가능 점수: {DEFAULT_CONFIG.max_score:.1f}")

    # ----- Level 1 검증 -----
    print(f"\n[Level 1 재설계 국면 분포 (use_level1=True)]")
    regime_l1 = classify_regime_level1(features)
    dist_l1 = regime_l1.value_counts()
    for r in ["ATTACK", "DEFENSE", "CRISIS"]:
        cnt = dist_l1.get(r, 0)
        print(f"  {r:8s}: {cnt:4d}일 ({cnt/n*100:.1f}%)")

    print(f"\n[Level 1 위기 구간(300~350일) 검증]")
    crisis_l1 = (regime_l1.iloc[300:355] == "CRISIS").sum()
    print(f"  CRISIS 판정: {crisis_l1}일 / 55일")
    assert crisis_l1 > 40, "Level 1: 위기 구간에 CRISIS 미감지"
    print("  ✓ 통과")

    # 신호 타입별 발동 확인
    s_c = _score_credit(features)
    s_p = _score_panic(features)
    s_i = _score_inflation(features)
    print(f"\n[Level 1 신호 발동 (위기 구간 300~350)]")
    print(f"  신용 위기 신호: {s_c.iloc[300:350].sum()}일")
    print(f"  패닉 신호:     {s_p.iloc[300:350].sum()}일")
    print(f"  인플레 신호:   {s_i.iloc[300:350].sum()}일")

    print("\n✓ baseline.py 검증 완료 (구버전 + Level 1 통과)")
