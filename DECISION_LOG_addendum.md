# UCHIDA V3 — Decision Log (Addendum 2026-05)

기존 DECISION_LOG.md 끝에 추가할 결정 이력.

---

## D-017: Level 1 재설계 (위기 유형별 분리) — **폐기**

**시도**: 위기 유형(credit/panic/inflation/recovery)별 분리 점수 사용.

**근거**: Reinhart & Rogoff (2009), *This Time Is Different*: 위기마다 유형 다름 (2008 신용, 2020 패닉, 2022 인플레).

**결과**: 실패. CAGR 14.5%, Sharpe 0.7, CRISIS 86% 고착.

**원인**: 
- Recovery 조건 너무 엄격
- 강세장 false positive
- 단일 점수 vs 분리 점수의 본질적 차이 없음 (가중합이 어차피 정보 통합)

**결정**: 단일 점수 가중합 (기존 방식) 유지. Level 1 폐기.

---

## D-018: 동적 임계값 (Baseline-D) — **폐기**

**시도**: 5개 신호 임계값을 rolling quantile로 동적화.

**근거**: 
- Gama et al. (2014), "A Survey on Concept Drift Adaptation", *ACM Computing Surveys*: 시장 구조 변화에 자동 적응 방법론
- Engle & Manganelli (2004), "CAViaR: Conditional Autoregressive Value at Risk by Regression Quantiles", *JBES*: rolling quantile 학계 표준

**파라미터**:
- Window: 756일 (3년, AI 판단)
- Quantile: vix p80, spread p95, yield p10, cpi p85, ma_dist p20 (학계 권장 범위 내, AI 판단)
- Look-ahead 방지: `.rolling(756).quantile(q).shift(1)` 적용

**실험**:
1. 1차 (p80/p95/p10/p85/p20): CAGR 14.51%, MDD -36.73%, Sharpe 0.804
2. 2차 (보수적 p90/p97/p05/p92/p10): CAGR 14.67%, MDD -36.89%, Sharpe 0.747
3. 결과: 모든 시도가 Baseline(고정) 대비 열위

**원인 (구조적 한계)**:
- 저변동성 구간(2012~2014) 이후 임계값 낮아짐 → 2015/2016 false positive
- 고변동성 환경(2022)에서 임계값 함께 올라감 → 진짜 위기 미감지
- rolling quantile은 **빈번한 패턴** 학습용. **드물고 강렬한 위기** 감지에 구조적으로 불리

**결정**: 동적 임계값 폐기. 고정 임계값 Baseline 유지. ceteris paribus 정신으로 변수 하나씩 실험한 결과.

**향후**: 데이터 30년+ 확보 또는 다른 구조(RL/Deep Hedging) 시 재시도.

---

## D-019: SIDEWAYS 4-class 확장 — **결정 보류 (조정 필요)**

**시도**: ATTACK/DEFENSE/CRISIS 3-class → ATTACK/SIDEWAYS/DEFENSE/CRISIS 4-class 확장.

**근거**:
- Pagan & Sossounov (2003), *Journal of Applied Econometrics*: bull/bear/sideways 분류 학계 표준
- Wilder (1978): ADX < 20 = 추세 부재 (박스권)
- 박스권에서 carry 자산(SCHD/JEPQ/IEF) 비중 강화 → 학계 일반 견해

**신호 정의**:
- ADX(QQQ) < 20 **AND** Bollinger Width(QQQ) < rolling p30 (3년)
- 이중 조건 (보수적). DEFENSE/CRISIS는 위험 신호 우선이라 SIDEWAYS로 교체 안 함

**신호 구현 (features.py)**:
- `_bollinger_width()`: (Upper - Lower) / MA. 변동성 압축 측정
- `_adx()`: 종가 기반 ADX 근사. Wilder 1978 공식의 종가 변형 (AI 판단, 고가/저가 미사용)

**WEIGHT_MAP SIDEWAYS (AI 판단)**:
```
QQQ 0%, SCHD 30%, GLD 10%, IEF 20%, SOFR 10%, JEPQ 30%
```
- QQQ 0%: 박스권에서 인덱스 상승 기대 안 함
- SCHD 30% + JEPQ 30%: carry 추구
- 합 100% 검증 완료

**JEPQ 합성** (loader.py):
- JEPQ 실제 데이터: 2022-05 출시 (4년만)
- 2008~2022 합성: `QQQ_return × 0.65 + 일 0.033% 프리미엄`
- 0.65: 콜 매도로 상승 35% 차단 (AI 판단)
- 일 0.033%: 연 10% 분배율 ÷ 252일 (JEPQ 실제 분배율 근사, AI 판단)
- **한계**: 백테스트 검증 가능 구간이 사실상 2022 이후 4년뿐. 합성 가정 검증 어려움

**결과**:
```
3-class Baseline: CAGR 16.25%, MDD -23.38%, Sharpe 0.875, 거래 79회
4-class Baseline: CAGR 14.48%, MDD -23.53%, Sharpe 0.848, 거래 211회
```

**문제**:
1. CAGR -1.77%p 하락
2. 거래 횟수 79 → 211회 (거래비용 폭증)
3. 2012(SIDEWAYS 18%), 2013(SIDEWAYS 24%) 강세장에서도 박스권 분류
4. SIDEWAYS 진입/이탈 빈번 (lookback 없음)

**원인 분석 중**:
- ADX/Bollinger Width 조건이 강세장 초입 일시적 변동성 압축도 박스권으로 분류
- SIDEWAYS 진입/이탈에 lookback 없음 → 잦은 전환

**결정 보류**: 다음 옵션 중 선택 필요
- (a) SIDEWAYS 진입/이탈에 lookback 추가 (거래 빈도 감소)
- (b) SIDEWAYS 조건에 `|dist_ma200| < 3%` 추가 (강세장 제외)
- (c) 4-class 폐기 → 3-class Baseline 확정

**현재 채택 기준 (§7.1) 통과 여부**:
- 3-class: 4/4 통과 (확정 가능)
- 4-class: 3/4 통과 (CAGR 미달)

---

## D-020: 자산 정리 — **확정**

**시도**: EEM, TLT, OIL 제거. 5자산(QQQ, SCHD, GLD, IEF, SOFR) 체제.

**근거 (실증)**:
- EEM: 미국 대비 underperform 지속 (2008~2025)
- TLT: 거의 사용 안 됨 (CRISIS 비중 10% 이하)
- OIL: ATTACK/DEFENSE/CRISIS 모든 국면에서 0%

**결과**:
```
이전 (7자산): CAGR 15.24%, MDD -26.04%, Sharpe 0.812
이후 (5자산): CAGR 16.25%, MDD -23.38%, Sharpe 0.875
```

**의의**: 모든 지표 동시 개선. EEM/TLT가 dead weight였음 실증.

**결정**: 자산 5개 확정. EEM/TLT/OIL 제거.

---

## D-021: LSTM 폐기 — **확정**

**시도**: 2-layer LSTM (hidden 64-32) walk-forward.

**결과**: 평균 정확도 36.9%, Fold 11 정확도 8.3%. 사실상 랜덤(33%) 수준.

**원인**:
- 데이터 22년 부족 (학계 권장 30년+)
- 클래스 불균형 (ATTACK 80%)
- 132 feature × 22년 / 63,000 파라미터 = 데이터:파라미터 비율 부족

**결정**: 폐기. 데이터 30년+ 확보 또는 학교 졸업 후 Deep Hedging 직접 구현 시 재시도.

---

## D-022: HMM 폐기 — **확정**

**시도**: Gaussian HMM (3 states, full covariance) walk-forward.

**결과**: CAGR 8.52%, MDD -22.15%. CAGR 목표(15%) 미달.

**원인**:
- 3-state로 ATTACK/DEFENSE/CRISIS 매핑 불안정
- 위기 후 회복 구간을 CRISIS로 오분류

**결정**: 폐기.

---

## D-023: ISA API 자동매매 — **결정 보류**

**시도**: 한국투자증권 KIS Open API로 ISA 자동매매 가능성 조사.

**검색 결과**:
- KIS API 공식 계좌상품코드: 01(종합), 03(국내선물옵션), 08(해외선물옵션), 22(개인연금), 29(퇴직연금)
- **ISA 코드 명시 안 됨**
- 일부 블로그: "계좌종류는 API와 관계없다" 주장. 출처 비공식.

**결정**: 
- 텔레그램 봇으로 **수동 매매 알림** 우선 구현
- ISA 자동매매는 한국투자증권 챗봇/콜센터 직접 확인 후 결정
- 월간 리밸런싱이라 자동매매는 over-engineering 가능성

---

## D-024: 3-System Framework 보류 — **단계적 진행**

**계획**: UCHIDA 60% + MITSUDA 20% + SEKINO 20% (총 자본 5,000만원)

**짚을 점**:
- MITSUDA(KOSPI LightGBM): 백테스트 검증 안 됨
- SEKINO(미국 섹터): 학계 회의적 (Brinson 1986: 자산배분 90%, 섹터 10%)
- 세 시스템 동시 위기 시 분산 효과 무너짐 (Longin & Solnik 2001, *JoF*)

**결정 (단계적)**:
1. **1단계 (현재)**: UCHIDA 단독 완성 → 실전 운용
2. **2단계 (UCHIDA 6개월 후)**: SEKINO 추가 검토. 실전 데이터로 충돌 룰 설계
3. **3단계 (그 후)**: MITSUDA 별도 백테스트 후 추가

**근거**: 동시 설계는 결정 과부하. UCHIDA 실전 데이터 없이 SEKINO 설계하면 추측 기반.

---

## 향후 결정 보류 사항 (업데이트)

- [ ] D-019 SIDEWAYS 조정 방향 결정 (lookback / dist_ma200 / 폐기)
- [ ] 텔레그램 봇 API 키 관리 방식
- [ ] ISA API 자동매매 가능 여부 (한국투자증권 직접 확인)
- [ ] 적립식 MC 파라미터 (월 입금액, 변동성 분포)
- [ ] HMM state 수 BIC 최적화 (HMM 자체 폐기로 우선순위 낮음)
- [ ] 거래시간 자동화 (수동 매매 유지 vs 자동)
- [ ] Long-only 제약 유지 (현재) vs 일부 short 허용
