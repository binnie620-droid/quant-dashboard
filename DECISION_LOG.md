# UCHIDA V3 — Decision Log

각 결정의 **근거 우선순위**:
1. 학계 논문 (peer-reviewed)
2. 업계 표준 (Vanguard, AQR 등 공인 기관)
3. 데이터 실증 (우리 백테스트)
4. AI 자체 판단 (그 외 모두 - 명시적으로 표시)

---

## D-001: 백테스트 인프라 도입 (Triple-Barrier)

**결정**: 라벨링에 Triple-Barrier method 사용.

**근거**: López de Prado (2018), *Advances in Financial Machine Learning*, Ch.3.

**대안 검토**: Fixed-horizon labels (단순 next-N-day return) — 거부. 변동성 변화 미반영.

**파라미터**:
- horizon T = 21일 (월간 의사결정)
- k = 1.0 (대칭 barrier)
- σ = 60일 realized vol

**근거 (파라미터)**: López de Prado 책에 권장 범위 (T=20~63, k=0.5~2.0). 우리는 중간값. → **AI 판단 (구체적 수치)**.

---

## D-002: Walk-forward Validation 채택

**결정**: Expanding window + 21일 embargo.

**근거**: 
- López de Prado (2018), Ch.7: "Purged K-Fold CV with Embargo"
- Pesaran & Timmermann (2002), *Journal of Econometrics*: 시계열 ML은 walk-forward 필수

**파라미터**:
- min_train_years = 3 (금융 사이클 최소 단위)
- test_period_months = 12 (계절성 포착)
- embargo_days = 21 (Triple-Barrier horizon과 일치)

**근거 (수치)**: 학계 권장 범위 내. **AI 판단**.

---

## D-003: Entropic Risk Measure 도입

**결정**: 평가 지표에 Entropic Risk + α-sensitivity 분석.

**근거**:
- Föllmer & Schied (2002), "Convex Measures of Risk and Trading Constraints", *Finance and Stochastics*
- 학교 ECO4126 "인공지능과 금융공학" 수업 내용과 연결

**파라미터**: α = {0.5, 1.0, 2.0, 5.0}

**근거**: 학계는 α 명시적 권장 안 함. 우리는 risk aversion 스펙트럼 측정. → **AI 판단**.

---

## D-004: 자산 Universe 8개

**결정**: QQQ, SCHD, GLD, IEF, SOFR, EEM, TLT, OIL.

**근거**:
- Statman (1987), *JFE*, "How Many Stocks Make a Diversified Portfolio?": 15개에서 분산 효과 포화
- Bogleheads 3-Fund Portfolio 등 업계 표준: 5~10개 ETF가 적정
- ETF는 그 자체로 분산되어 있으므로 5~10개로 충분

**제외 자산과 근거**:
- SPY 제외: QQQ와 상관 0.92 → 중복 (실증)
- TIP 제외: 한국 ETF 없음 (실용성)
- KODEX 200 제외: EEM과 상관 0.85+ (실증)
- 섹터 ETF 제외: 단기 베팅, 장기엔 부적합 — Brinson, Hood, Beebower (1986)

**추가 자산과 근거**:
- SCHD: 미국 배당주, QQQ와 상관 0.65 → 진짜 분산. 한국 ETF AUM 3조원 (유동성 충분)
- EEM: 신흥국, 미국과 상관 0.75 → 분산 효과

---

## D-005: 데이터 시작점 2003년

**결정**: train_start = 2003-01-01, backtest_start = 2008-01-01.

**근거**:
- 2003년: 모든 8개 자산 데이터 가용 (EEM 2003 출시)
- 2008년: 글로벌 금융위기 포함 → 진짜 위기 학습 데이터 확보
- López de Prado (2018): 위기 샘플은 ML 학습의 핵심

**한계**: 22년은 학계 권장 30년+ 대비 부족. **데이터 한계 인정**.

---

## D-006: 모델 3종 비교 (Baseline / LSTM / HMM)

**결정**: 세 모델 동시 비교 후 선택.

**근거**:
- Baseline: 기존 UCHIDA 룰 (사용자 보유) + 학계 비교 baseline 필요
- LSTM: Fischer & Krauss (2018), *EJOR*: 시계열 분류 표준
- HMM: Hamilton (1989), *Econometrica*: 경제학 regime-switching 표준
- Ang & Bekaert (2002), *Journal of Finance*: HMM이 자산배분에 적용 검증됨

**왜 LSTM 외에 HMM도?**:
- LSTM 정확도 36% (실증, 1차 백테스트)
- 데이터 부족 시 HMM이 더 robust (Hamilton 1989)
- 학교 수업 (Deep Hedging) 연결성 + 학술적 다양성

---

## D-007: LSTM 구조 (2-layer, hidden 64-32)

**결정**: LSTM(64) → LSTM(32) → Dense(16) → Softmax(3).

**근거**:
- Fischer & Krauss (2018): 금융 시계열에 2-layer LSTM 표준
- Srivastava et al. (2014): Dropout 0.3 권장 범위 (0.2~0.5)
- 우리 데이터 규모: 데이터:파라미터 비율 1:21 (학계 권장 1:10 대비 빠듯)

**파라미터 결정**:
- seq_len = 60: 3개월 (분기 매크로 사이클)
- hidden1=64, hidden2=32: 우리 데이터 규모에 맞춰 작게
- dropout=0.3: Srivastava 등 권장 중간값
- lr=1e-3: Adam 기본값 (Kingma & Ba 2014)
- batch_size=32: 소규모 데이터 표준

**근거 (구체적 수치)**: **AI 판단** (학계 일반 권장 범위 내).

---

## D-008: HMM 구조 (Gaussian, 3 states, full covariance)

**결정**: n_components=3, covariance_type="full".

**근거**:
- Hamilton (1989): 2~3 state가 경제 사이클에 적합
- Ang & Bekaert (2002): 자산배분에 3 state regime model 검증
- 3 states = 우리 ATTACK/DEFENSE/CRISIS와 일관

**Full covariance 선택 이유**:
- 자산 간 공분산 정보 활용
- "diag" (대각)은 정보 손실
- 데이터 충분하므로 (5 feature × 22년) full 가능

---

## D-009: WEIGHT_MAP 8개 자산 (현재)

**결정**:

| 자산 | ATTACK | DEFENSE | CRISIS |
|---|---|---|---|
| QQQ | 0.70 | 0.25 | 0.0 |
| SCHD | 0.15 | 0.20 | 0.20 |
| GLD | 0.05 | 0.15 | 0.25 |
| IEF | 0.05 | 0.20 | 0.20 |
| SOFR | 0.0 | 0.10 | 0.25 |
| EEM | 0.05 | 0.05 | 0.0 |
| TLT | 0.0 | 0.05 | 0.10 |

**근거**:
- 사용자 성향 (명시): "MDD 희생해도 CAGR 올리고 싶음"
- 사용자 의견 (명시): "SCHD는 위기에도 일정 유지 (배당 누적)"
- ATTACK QQQ 70%: QQQ Buy&Hold (CAGR 19.57%)에 근접 시도
- CRISIS QQQ 0%: 위기 회피 우선
- SCHD CRISIS 20%: 사용자 의도 반영. 배당주는 위기에 -14% 덜 빠짐 (실증, 2008 DVY)

**구체적 비중**: 학계 권장 비중 없음. → **AI 판단**.

---

## D-010: 거래비용 0.115% (편도)

**결정**: cost_one_way = 0.00115.

**근거**:
- 한국 ETF 수수료: 평균 0.015% (한국거래소 자료)
- 슬리피지: 0.1% (개인 투자자 일반적 추정)
- 합계: 0.115%

**근거 (슬리피지)**: 정확한 학계 수치 없음. **업계 일반 추정**. → **AI 판단**.

---

## D-011: 리밸런싱 5% no-trade band + 월간

**결정**: tolerance_band = 0.05, 월간 신호.

**근거**:
- Vanguard 백서 (Jaconetti, Kinniry, Zilbering, 2010), "Best Practices for Portfolio Rebalancing": 5% 밴드 + 분기/연간이 거래비용/추적오차 균형
- Donohue & Yip (2003): 거래비용에 비례하는 밴드가 최적

**솔직한 짚음**: 5%는 학계 컨센서스라기보다 업계 휴리스틱. 우리 자산 조합 최적값 검증 필요 → 추후 sensitivity analysis 권장.

---

## D-012: ISA 세제 모델

**결정**: 만기 일괄 정산, 비과세 400만원 + 9.9% 분리과세.

**근거**: 2024년 한국 세법 (조세특례제한법 제91조의18).

**한계**: 
- 분배금 별도 처리 안 함 (총수익에 합산)
- 환율 변동 무시
- 실제 세액은 세무사 확인 권장

---

## D-013: Look-ahead Bias 방지 원칙

**결정**:
1. Forward fill만 사용 (backward fill 금지)
2. StandardScaler는 train에서만 fit, test는 transform만
3. 매크로 데이터는 그대로 사용 (발표 지연 무시 — 단순화)
4. Triple-Barrier embargo 21일 적용

**근거**:
- López de Prado (2018), Ch.7
- Bailey et al. (2014), "The Probability of Backtest Overfitting"

---

## D-014: Layer 2 (섹터/뉴스) 보류

**결정**: 현재 자산배분만 집중. 섹터 베팅은 1년 운용 후 검토.

**근거**:
- Brinson, Hood, Beebower (1986): 자산배분이 수익의 90%, 종목 선택 10%
- McLean & Pontiff (2016), *Journal of Finance*: 검증된 알파도 발표 후 32% 감소
- 개인 투자자가 뉴스 sentiment로 alpha 만들기 어려움 (실증 검토)

---

## D-015: 적립식은 별도 Monte Carlo

**결정**: 백테스트는 일시금. 적립식은 향후 별도 MC 시뮬레이션.

**근거**:
- 일시금: 모델 성능 측정에 명확
- 적립식: 실전 운용 시뮬레이션 (분리가 학술적으로 깔끔)
- Vanguard (2012): 일시금이 2/3 케이스에서 우월, 적립식은 후회 최소화

---

## D-016: 운용 빈도 결정

**결정**: 월간 의사결정, 5% 밴드.

**근거**:
- 일간 신호 = 노이즈 (Fama 1970 EMH 약형)
- 분기는 거시 사이클 못 따라감
- 월간이 거시 사이클과 거래비용의 균형 (업계 표준)

---

## 향후 결정 보류 사항

- [ ] 텔레그램 봇 API 키 관리 방식
- [ ] 적립식 MC 파라미터 (월 입금액, 변동성 분포)
- [ ] HMM state 수 BIC 최적화
- [ ] 거래시간 자동화 (수동 매매 유지 vs 자동)
- [ ] Long-only 제약 유지 (현재) vs 일부 short 허용
