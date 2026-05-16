"""
UCHIDA V3 - Central Configuration

모든 프로젝트 모듈이 참조하는 중앙 설정 파일.
하드코딩된 값을 다른 파일에 직접 쓰지 말고 반드시 여기서 import 할 것.

[설계 원칙]
1. 학습/백테스트는 미국 티커 (yfinance/FRED, 데이터 품질 우수)
2. 실전 매매는 한국 ETF 코드 (ISA 계좌에서 거래 가능)
3. 동일 역할 자산은 ASSETS dict의 같은 key로 매핑
"""

from dataclasses import dataclass, field
from typing import Dict
from datetime import date


# ==========================================
# 1. 자산 정의 (Asset Universe)
# ==========================================
# 5 코어 + 2 위성 = 7개 자산
# 학습용 ticker (미국 티커, yfinance), 실전용 code (한국 ETF, KIS API)

@dataclass(frozen=True)
class Asset:
    """단일 자산 정의."""
    role: str           # 자산의 역할 (코어/위성)
    name_kr: str        # 한국 ETF 명
    code_kr: str        # 한국 ETF 코드 (6자리)
    ticker_us: str      # 미국 티커 (학습/백테스트용)
    fred_id: str = ""   # FRED 매크로 데이터 ID (해당 시)


ASSETS: Dict[str, Asset] = {
    # ----- 코어 (항상 보유 후보) -----
    "QQQ":   Asset(role="core_growth",   name_kr="TIGER 미국나스닥100",       code_kr="133690", ticker_us="QQQ"),
    "SCHD":  Asset(role="core_dividend", name_kr="TIGER 미국배당다우존스",      code_kr="458730", ticker_us="VYM"),
    "GLD":   Asset(role="core_hedge",    name_kr="ACE KRX금현물",             code_kr="411060", ticker_us="GLD"),
    "IEF":   Asset(role="core_bond",     name_kr="TIGER 미국채10년선물",      code_kr="305080", ticker_us="IEF"),
    "SOFR":  Asset(role="core_cash",     name_kr="TIGER 미국달러SOFR금리액티브(합성)", code_kr="456610", ticker_us="SHV"),
    # [D-019 폐기, D-020] SIDEWAYS 4-class 폐기로 JEPQ 제거
    # 3-class (ATTACK/DEFENSE/CRISIS)로 복원
    # [2026-05] 자산 정리: EEM/TLT/OIL 제거
    # - EEM: 백테스트 기간 underperform (QQQ 대비 -12%p/년)
    # - TLT: ATTACK 0%, CRISIS 10%로 거의 미사용. IEF로 통합
    # - OIL: 이미 미사용
}
# [메모] SCHD 백테스트 ticker는 VYM 사용 (2006년부터 데이터, SCHD와 상관 0.96)
# 실전 매수는 TIGER 미국배당다우존스 (코드 458730)


# ==========================================
# 2. 매크로 데이터 (FRED)
# ==========================================
# 시장 국면 판단용 거시지표 ID
MACRO_FRED_IDS: Dict[str, str] = {
    "CPIAUCSL": "Consumer Price Index, urban consumers",
    "BAA10Y":   "Moody's Baa Corporate Bond Yield - 10Y Treasury (credit spread)",
    "T10Y2Y":   "10Y - 2Y Treasury yield spread (yield curve)",
    "DGS10":    "10-Year Treasury Constant Maturity Rate",
    "UNRATE":   "Unemployment Rate",
    "VIXCLS":   "CBOE Volatility Index",
    "DTWEXBGS": "Trade Weighted USD Index (Broad)",
}

# CPI는 발표 지연 반영 (실제 발표 약 한 달 + 보수적으로 45일)
CPI_PUBLICATION_LAG_DAYS = 45


# ==========================================
# 3. 백테스트 / 학습 기간
# ==========================================
@dataclass(frozen=True)
class DateConfig:
    train_start: date = date(2003, 1, 1)    # 2003: 모든 자산(EEM 포함) 데이터 존재
    backtest_start: date = date(2008, 1, 1) # 2008 글로벌 금융위기 포함 백테스트
    # end는 코드 실행 시점에 today()로 설정


DATES = DateConfig()


# ==========================================
# 4. 거래 비용 모델
# ==========================================
@dataclass(frozen=True)
class CostConfig:
    """편도(one-way) 거래 비용. 매수/매도 각각 적용."""
    commission: float = 0.00015      # 한국 증권사 평균 매매수수료 (0.015%)
    slippage: float = 0.001          # ETF 호가창 보수적 추정 (0.1%)
    
    @property
    def total_one_way(self) -> float:
        return self.commission + self.slippage


COSTS = CostConfig()


# ==========================================
# 5. ISA 세제 (서민형 기준)
# ==========================================
@dataclass(frozen=True)
class ISAConfig:
    """투자중개형 ISA, 서민형 기준."""
    tax_free_limit: int = 4_000_000          # 비과세 한도 (서민형)
    separate_tax_rate: float = 0.099         # 초과분 분리과세율 (9.9%)
    annual_deposit_limit: int = 20_000_000   # 연간 납입한도
    cumulative_deposit_limit: int = 100_000_000  # 누적 납입한도 (5년 1억)
    mandatory_period_years: int = 3          # 의무 가입기간


ISA = ISAConfig()


# ==========================================
# 6. 리밸런싱 정책
# ==========================================
@dataclass(frozen=True)
class RebalanceConfig:
    """
    리밸런싱 빈도와 no-trade band.
    
    - frequency_days: 정기 점검 주기 (사용자 답변 "1~2개월" 중 보수 측인 30일 채택)
    - tolerance_band: 5% 이상 비중 차이 시에만 거래 (사용자 지정)
    - 학계 근거: Sun et al. (2006), Donohue & Yip (2003) — tolerance band가 calendar-only보다 우월
    """
    frequency_days: int = 30
    tolerance_band: float = 0.05
    
    # 이벤트 트리거 (Phase 후반 구현 예정)
    vix_emergency_level: float = 30.0
    daily_drawdown_emergency: float = -0.03  # 일일 -3% 이상


REBALANCE = RebalanceConfig()


# ==========================================
# 7. 초기 자본 (백테스트/시뮬레이션용)
# ==========================================
INITIAL_CAPITAL: int = 10_000_000  # 1,000만원


# ==========================================
# 8. 데이터 / 모델 저장 경로
# ==========================================
import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data_cache")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model_cache")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


# ==========================================
# 9. 알림 설정 (실전 운용)
# ==========================================
# 보안: 실제 토큰은 환경변수로 관리. config에는 키 이름만.
TELEGRAM_TOKEN_ENV = "UCHIDA_TG_TOKEN"
TELEGRAM_CHAT_ID_ENV = "UCHIDA_TG_CHAT_ID"


# ==========================================
# Self-check (import 시 무결성 검증)
# ==========================================
def _validate_config():
    """설정 모순 검출. import 시 자동 실행."""
    # 자산 코드 중복 검증
    codes = [a.code_kr for a in ASSETS.values()]
    assert len(codes) == len(set(codes)), "ASSETS에 중복된 한국 ETF 코드가 있습니다"
    
    tickers = [a.ticker_us for a in ASSETS.values()]
    assert len(tickers) == len(set(tickers)), "ASSETS에 중복된 미국 티커가 있습니다"
    
    # 날짜 순서 검증
    assert DATES.train_start < DATES.backtest_start, "train_start가 backtest_start 이후입니다"
    
    # 비용 음수 검증
    assert COSTS.commission >= 0 and COSTS.slippage >= 0, "거래비용이 음수입니다"
    
    # 밴드 범위 검증
    assert 0 < REBALANCE.tolerance_band < 1, "tolerance_band는 0~1 범위여야 합니다"


_validate_config()


# ==========================================
# 10. Level 1 재설계 임계값 (DECISION_LOG D-017)
# ==========================================
# [설계 근거] 단일 가중합 → 위기 유형별 분리 점수
# 각 유형이 독립적으로 CRISIS 트리거 (OR 구조)
# 구체 수치는 AI 판단 (학계 근거: 신호 유효성 + 분포 기반)
# 상세 근거: DECISION_LOG.md D-017 참조

@dataclass(frozen=True)
class CrisisThresholds:
    # --- 신용 위기 (Gilchrist & Zakrajšek 2012, AER) ---
    # BAA10Y(이미 스프레드) > 2.5%p AND VIX > 25
    credit_spread: float = 2.5   # AI 판단: GZ 위기 평균 3%p 대비 보수적 사전 경보
    credit_vix:    float = 25.0  # AI 판단: Bloom(2009) 기준 + 단독 발동 방지

    # --- 패닉/추세 위기 (Bloom 2009, Econometrica; Whaley 2000, JPM) ---
    # VIX > 30  OR  (VIX > 25 AND dist_ma200 < -10%)
    panic_vix_high: float = 30.0   # 학계: Whaley(2000) VIX 30 = historical ~95퍼센타일
    panic_vix_mid:  float = 25.0   # AI 판단: Bloom(2009) uncertainty shock 다수 관측 구간
    panic_ma200:    float = -0.10  # AI 판단: -10% = 추세 이탈 확인 수준

    # --- 인플레이션 위기 (Estrella & Mishkin 1998, RES) ---
    # cpi_z > 2.0 AND t10y2y < 0.5
    inflation_cpi_z: float = 2.0  # AI 판단: 2-sigma 통계적 기준
    inflation_yield: float = 0.5  # AI 판단: 역전 직전 평탄화 포함 (Estrella 0%보다 선행)

    # --- Recovery (CRISIS → DEFENSE 전환) ---
    # Chauvet & Piger(2008, JBES): NBER 침체 종료 판정 평균 7~8개월 지연
    # [2026-05 2차 수정] 실제 데이터 분포 기반 재설정 (AI 판단)
    # credit_spread 실측: min=1.36, p25=1.80, median=2.07, p75=2.37
    # spread<1.5 = 전체 기간 ~10% 충족 → rolling 10일 min이면 ~0% → CRISIS 탈출 불가
    # 수정: spread<2.5 (p75 수준, "정상 시장" 기준)
    recovery_spread: float = 2.5   # AI 판단: 실측 p75 기준 (1.5→2.5)
    recovery_vix:    float = 25.0  # AI 판단: 공포 진정 기준
    recovery_ma200:  float = -0.08 # AI 판단: 추세 부분 회복 기준

    # --- 상태 전환 lookback ---
    entry_lookback: int = 5    # AI 판단: 5일 연속 (완화된 recovery와 균형 맞춤)
    exit_lookback:  int = 10   # AI 판단: 10일 (완화된 recovery 조건이므로 단축)


CRISIS_THR = CrisisThresholds()


if __name__ == "__main__":
    # 설정 미리보기
    print("=" * 60)
    print("UCHIDA V3 Configuration")
    print("=" * 60)
    print(f"\n[자산 {len(ASSETS)}개]")
    for key, asset in ASSETS.items():
        print(f"  {key:6s} | {asset.role:25s} | {asset.code_kr} {asset.name_kr}")
    print(f"\n[기간] 학습 {DATES.train_start} ~ / 백테스트 {DATES.backtest_start} ~")
    print(f"[비용] 편도 {COSTS.total_one_way*100:.3f}% (수수료 {COSTS.commission*100:.3f}% + 슬리피지 {COSTS.slippage*100:.3f}%)")
    print(f"[ISA] 비과세 {ISA.tax_free_limit:,}원 / 초과분 {ISA.separate_tax_rate*100:.1f}% 분리과세")
    print(f"[리밸런싱] {REBALANCE.frequency_days}일 주기 / {REBALANCE.tolerance_band*100:.0f}% 밴드")
