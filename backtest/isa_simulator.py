"""
UCHIDA V3 - ISA Tax Simulator

투자중개형 ISA (서민형) 세제 시뮬레이션.

[ISA 세제 구조 - 서민형 기준]
- 비과세 한도: 400만원
- 초과분 분리과세: 9.9%
- 의무 가입기간: 3년
- 연간 납입한도: 2,000만원
- 세금 정산: 만기 해지 시 일괄 (연간 과세 아님)
- 손실 발생 시: 세금 없음

[시뮬레이터의 역할]
1. 백테스트 수익률 시계열 → 세전/세후 수익률 비교
2. 만기 도래 시 세금 정산 (3년 주기)
3. 납입 한도 초과 여부 추적
4. 세제 효과 정량화 (일반 과세 대비 얼마나 이익인지)

[일반 과세 vs ISA 비교]
- 일반 계좌: 배당소득세 15.4% (매년)
- ISA: 비과세 400만원 + 초과 9.9% (만기 일괄)
→ 장기투자일수록 ISA가 유리

[한계]
- ETF 분배금은 별도 추적 필요 (이 시뮬레이터는 총수익률 기준)
- 실제 세액은 세무사 확인 필요
- 2024년 기준 법령. 향후 변경 가능.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ISA, INITIAL_CAPITAL


# ==========================================
# 1. 세금 계산 (핵심 함수)
# ==========================================
def calc_isa_tax(
    profit: float,
    tax_free_limit: float = None,
    tax_rate: float = None,
) -> Tuple[float, float, float]:
    """
    ISA 만기 해지 시 세금 계산.

    Parameters
    ----------
    profit : float
        총 실현 손익 (양수 = 이익, 음수 = 손실)
    tax_free_limit : float
        비과세 한도. None이면 config.ISA.tax_free_limit
    tax_rate : float
        분리과세율. None이면 config.ISA.separate_tax_rate

    Returns
    -------
    (tax_free_amount, taxable_amount, tax_paid) tuple
        tax_free_amount: 비과세로 처리된 금액
        taxable_amount: 과세 대상 금액
        tax_paid: 납부 세금
    """
    if tax_free_limit is None:
        tax_free_limit = ISA.tax_free_limit
    if tax_rate is None:
        tax_rate = ISA.separate_tax_rate

    if profit <= 0:
        # 손실이거나 0이면 세금 없음
        return 0.0, 0.0, 0.0

    tax_free_amount = min(profit, tax_free_limit)
    taxable_amount  = max(0.0, profit - tax_free_limit)
    tax_paid        = taxable_amount * tax_rate

    return tax_free_amount, taxable_amount, tax_paid


def calc_general_tax(
    profit: float,
    dividend_tax_rate: float = 0.154,
) -> float:
    """
    일반 계좌 배당소득세 계산 (비교용).
    배당소득세 15.4% (소득세 14% + 지방소득세 1.4%).

    Parameters
    ----------
    dividend_tax_rate : float
        배당소득세율. 기본 15.4%.
    """
    if profit <= 0:
        return 0.0
    return profit * dividend_tax_rate


# ==========================================
# 2. ISA 계좌 상태 추적
# ==========================================
@dataclass
class ISAAccount:
    """
    ISA 계좌 상태 추적.

    Parameters
    ----------
    initial_deposit : float
        초기 납입금
    open_date : pd.Timestamp
        계좌 개설일
    tax_free_limit : float
        비과세 한도 (서민형 기본 400만원)
    tax_rate : float
        분리과세율 (기본 9.9%)
    mandatory_years : int
        의무 가입기간 (기본 3년)
    annual_deposit_limit : float
        연간 납입한도 (기본 2000만원)
    """
    initial_deposit: float
    open_date: pd.Timestamp
    tax_free_limit: float = field(default_factory=lambda: float(ISA.tax_free_limit))
    tax_rate: float = field(default_factory=lambda: ISA.separate_tax_rate)
    mandatory_years: int = field(default_factory=lambda: ISA.mandatory_period_years)
    annual_deposit_limit: float = field(default_factory=lambda: float(ISA.annual_deposit_limit))

    def maturity_date(self) -> pd.Timestamp:
        """만기일 = 개설일 + 의무 가입기간."""
        return self.open_date + pd.DateOffset(years=self.mandatory_years)

    def is_mature(self, date: pd.Timestamp) -> bool:
        """해당 날짜에 만기 여부."""
        return date >= self.maturity_date()


# ==========================================
# 3. 백테스트 결과에 ISA 세제 적용
# ==========================================
@dataclass
class ISAResult:
    """ISA 시뮬레이션 결과."""
    # 원본 (세전)
    pretax_final_value: float
    pretax_profit: float
    pretax_cagr: float

    # ISA 세후
    isa_tax_paid: float
    isa_tax_free_amount: float
    isa_taxable_amount: float
    isa_after_tax_value: float
    isa_after_tax_cagr: float

    # 일반 계좌 세후 (비교)
    general_tax_paid: float
    general_after_tax_value: float
    general_after_tax_cagr: float

    # ISA 절세 효과
    tax_saving: float       # 일반 과세 대비 절세액
    effective_tax_rate: float   # 실효세율 (ISA)


def simulate_isa(
    portfolio_value: pd.Series,
    initial_deposit: float = None,
    open_date: Optional[pd.Timestamp] = None,
    tax_free_limit: float = None,
    tax_rate: float = None,
    mandatory_years: int = None,
) -> ISAResult:
    """
    포트폴리오 가치 시계열에 ISA 세제 적용.

    Parameters
    ----------
    portfolio_value : pd.Series
        포트폴리오 가치 시계열 (index: DatetimeIndex).
        engine.BacktestResult.portfolio_value를 그대로 사용.
        단, 이 시리즈는 정규화(시작=1.0)가 아닌 실제 금액이어야 함.
    initial_deposit : float
        초기 납입금. None이면 config.INITIAL_CAPITAL.
    open_date : pd.Timestamp
        계좌 개설일. None이면 portfolio_value 첫 날짜.
    """
    if initial_deposit is None:
        initial_deposit = float(INITIAL_CAPITAL)
    if open_date is None:
        open_date = portfolio_value.index[0]
    if tax_free_limit is None:
        tax_free_limit = float(ISA.tax_free_limit)
    if tax_rate is None:
        tax_rate = ISA.separate_tax_rate
    if mandatory_years is None:
        mandatory_years = ISA.mandatory_period_years

    # 실제 금액으로 변환 (정규화된 시리즈라면 초기 납입금 곱함)
    pv = portfolio_value
    if abs(pv.iloc[0] - 1.0) < 1e-6:
        # 정규화된 시리즈 (시작=1.0)
        pv = pv * initial_deposit

    final_value  = float(pv.iloc[-1])
    total_profit = final_value - initial_deposit
    years = (pv.index[-1] - pv.index[0]).days / 365.0

    # ----- 세전 CAGR -----
    pretax_cagr = (final_value / initial_deposit) ** (1 / years) - 1 if years > 0 else 0.0

    # ----- ISA 세금 -----
    tax_free_amount, taxable_amount, isa_tax = calc_isa_tax(
        total_profit, tax_free_limit, tax_rate
    )
    isa_after_tax_value = final_value - isa_tax
    isa_after_tax_cagr = (isa_after_tax_value / initial_deposit) ** (1 / years) - 1 if years > 0 else 0.0

    # ----- 일반 계좌 세금 (비교) -----
    # 매년 수익에 15.4% 복리 과세 가정 (보수적 추정)
    # 실제로는 분배 시점에 과세이지만, 비교용으로 단순화
    general_tax = calc_general_tax(total_profit)
    general_after_tax_value = final_value - general_tax
    general_after_tax_cagr = (general_after_tax_value / initial_deposit) ** (1 / years) - 1 if years > 0 else 0.0

    # ----- 절세 효과 -----
    tax_saving = general_tax - isa_tax
    effective_tax_rate = isa_tax / total_profit if total_profit > 0 else 0.0

    return ISAResult(
        pretax_final_value=final_value,
        pretax_profit=total_profit,
        pretax_cagr=pretax_cagr,
        isa_tax_paid=isa_tax,
        isa_tax_free_amount=tax_free_amount,
        isa_taxable_amount=taxable_amount,
        isa_after_tax_value=isa_after_tax_value,
        isa_after_tax_cagr=isa_after_tax_cagr,
        general_tax_paid=general_tax,
        general_after_tax_value=general_after_tax_value,
        general_after_tax_cagr=general_after_tax_cagr,
        tax_saving=tax_saving,
        effective_tax_rate=effective_tax_rate,
    )


# ==========================================
# 4. 시나리오 비교 (여러 전략 한 번에)
# ==========================================
def compare_isa_scenarios(
    strategies: dict,
    initial_deposit: float = None,
) -> pd.DataFrame:
    """
    여러 전략의 ISA 세후 결과를 한 표로 비교.

    Parameters
    ----------
    strategies : dict[str, pd.Series]
        전략명 → 포트폴리오 가치 시계열 (정규화 또는 실제 금액)

    Returns
    -------
    pd.DataFrame
        index: 전략명, columns: 주요 지표
    """
    if initial_deposit is None:
        initial_deposit = float(INITIAL_CAPITAL)

    rows = {}
    for name, pv in strategies.items():
        r = simulate_isa(pv, initial_deposit=initial_deposit)
        rows[name] = {
            "세전 최종 자산 (원)":     r.pretax_final_value,
            "세전 CAGR (%)":           r.pretax_cagr * 100,
            "총 수익 (원)":             r.pretax_profit,
            "ISA 비과세 적용 (원)":    r.isa_tax_free_amount,
            "ISA 과세 대상 (원)":      r.isa_taxable_amount,
            "ISA 납부 세금 (원)":      r.isa_tax_paid,
            "ISA 세후 최종 자산 (원)": r.isa_after_tax_value,
            "ISA 세후 CAGR (%)":       r.isa_after_tax_cagr * 100,
            "일반 계좌 세금 (원)":     r.general_tax_paid,
            "ISA 절세액 (원)":         r.tax_saving,
            "실효세율 (%)":            r.effective_tax_rate * 100,
        }

    return pd.DataFrame(rows).T


# ==========================================
# 5. 자체 검증
# ==========================================
if __name__ == "__main__":
    print("=" * 70)
    print("backtest/isa_simulator.py 단독 실행 테스트")
    print("=" * 70)

    # ----- Test 1: 세금 계산 수치 검증 -----
    print("\n[Test 1] 세금 계산 수치 검증")
    
    # 케이스 A: 이익 300만원 (비과세 한도 내)
    tf, tx, paid = calc_isa_tax(3_000_000)
    print(f"  이익 300만원 → 비과세 {tf/1e4:.0f}만원, 세금 {paid/1e4:.1f}만원")
    assert paid == 0.0, "300만원은 비과세 한도 내"
    
    # 케이스 B: 이익 600만원 (한도 초과 200만원 × 9.9%)
    tf, tx, paid = calc_isa_tax(6_000_000)
    expected_tax = 2_000_000 * 0.099
    print(f"  이익 600만원 → 비과세 {tf/1e4:.0f}만원, 과세 {tx/1e4:.0f}만원, 세금 {paid/1e4:.2f}만원 (기대: {expected_tax/1e4:.2f}만원)")
    assert abs(paid - expected_tax) < 1, "600만원 세금 계산"
    
    # 케이스 C: 손실 100만원 → 세금 없음
    tf, tx, paid = calc_isa_tax(-1_000_000)
    assert paid == 0.0, "손실이면 세금 없음"
    print(f"  손실 100만원 → 세금 {paid}원 (세금 없음)")
    print("  ✓ 통과")

    # ----- Test 2: 실제 시뮬레이션 -----
    print("\n[Test 2] 전략별 ISA 효과 시뮬레이션 (10년)")
    np.random.seed(42)
    n = 2520  # 10년 영업일
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    initial = 10_000_000

    # 전략 3개
    def make_pv(daily_mu, daily_sigma):
        ret = np.random.normal(daily_mu, daily_sigma, n)
        return pd.Series(initial * np.exp(np.cumsum(np.log(1+ret))), index=idx)

    np.random.seed(42)
    strategies = {
        "공격형 (QQQ류)":  make_pv(0.0006, 0.014),
        "균형형 (60/40)":  make_pv(0.0004, 0.008),
        "UCHIDA 목표":     make_pv(0.0005, 0.009),
    }

    df = compare_isa_scenarios(strategies, initial_deposit=initial)
    
    # 출력 포맷팅
    print(f"\n  초기 납입금: {initial:,}원")
    print()
    
    int_cols = ["세전 최종 자산 (원)", "총 수익 (원)", "ISA 비과세 적용 (원)",
                "ISA 과세 대상 (원)", "ISA 납부 세금 (원)", "ISA 세후 최종 자산 (원)",
                "일반 계좌 세금 (원)", "ISA 절세액 (원)"]
    pct_cols = ["세전 CAGR (%)", "ISA 세후 CAGR (%)", "실효세율 (%)"]

    for name, row in df.iterrows():
        print(f"  [{name}]")
        print(f"    세전 최종: {row['세전 최종 자산 (원)']:>15,.0f}원 (CAGR {row['세전 CAGR (%)']:.2f}%)")
        print(f"    ISA 세금:  {row['ISA 납부 세금 (원)']:>15,.0f}원 (실효세율 {row['실효세율 (%)']:.2f}%)")
        print(f"    ISA 세후:  {row['ISA 세후 최종 자산 (원)']:>15,.0f}원 (CAGR {row['ISA 세후 CAGR (%)']:.2f}%)")
        print(f"    절세액:    {row['ISA 절세액 (원)']:>15,.0f}원 (vs 일반 계좌)")
        print()

    # ----- Test 3: 비과세 한도 의미 -----
    print("[Test 3] 비과세 한도 400만원 효과")
    print("  이익 구간별 세금 비교 (ISA vs 일반)")
    print(f"  {'이익':>12} | {'ISA 세금':>10} | {'일반 세금':>10} | {'절세':>10}")
    print(f"  {'-'*50}")
    for profit in [2_000_000, 4_000_000, 6_000_000, 10_000_000, 20_000_000]:
        _, _, isa_t = calc_isa_tax(profit)
        gen_t = calc_general_tax(profit)
        saving = gen_t - isa_t
        print(f"  {profit:>12,.0f} | {isa_t:>10,.0f} | {gen_t:>10,.0f} | {saving:>10,.0f}")
    
    print("\n✓ isa_simulator.py 검증 완료")
