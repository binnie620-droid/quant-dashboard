# UCHIDA V3 — System Architecture

## 1. 프로젝트 정체성

**목적**: 개인 투자자(연세대 경제학과 학생, ISA 서민형 계좌)의 10~15년 장기 자산배분 시스템.

**목표 함수**: QQQ Buy & Hold 대비
- CAGR 알파 (음수여도 ≤ -3%p)
- MDD 축소 (목표 -22% 이내)
- 위험조정 수익 (Sharpe ≥ 0.85)

**의사결정 빈도**: 월간 리밸런싱 (5% no-trade band)

**운용 자본**: 1,000만~2,000만원 시작, 연 2,000만원 ISA 한도 내 적립.

---

## 2. 진화 이력

### v1: 초기 룰베이스 (uchida.py)
- 5개 신호 룰 기반 분류 (CPI/스프레드/역전/VIX/추세)
- 비대칭 유예 (lookback)
- ATTACK/DEFENSE/CRISIS 3-class
- **문제**: backtest.py와 실전 비중이 불일치

### v2: 백테스트 + 몬테카를로 (uchida_backtest.py, uchida_montecarlo.py)
- backtest.py: 별도 비중 맵 사용
- montecarlo.py: GBM 시뮬레이션 (가정값 입력형)
- **문제**: 백테스트 ≠ 실전 ≠ 몬테카를로 (3중 불일치)

### v3: 통합 재설계 (현재)
- 단일 비중 맵 (WEIGHT_MAP)
- 학계 인프라 도입 (Triple-Barrier, Walk-forward, Entropic Risk)
- baseline / LSTM / HMM 3가지 비교

---

## 3. 디렉토리 구조

```
uchida_v3/
├── config.py                    # 자산/기간/비용/ISA 정의
├── data/
│   ├── loader.py                # yfinance + FRED 통합 로드
│   └── features.py              # 132개 feature 생성
├── labels/
│   └── labeler.py               # Triple-Barrier 라벨링
├── backtest/
│   ├── metrics.py               # CAGR, MDD, Sharpe, Entropic Risk
│   ├── engine.py                # t+1 Open 체결, 5% 밴드, 비용
│   ├── walkforward.py           # Expanding window, 21일 embargo
│   └── isa_simulator.py         # ISA 세제 시뮬레이션
├── models/
│   ├── baseline.py              # 룰베이스 5개 신호 + WEIGHT_MAP
│   ├── lstm.py                  # 2-layer LSTM (PyTorch)
│   └── hmm.py                   # Gaussian HMM (hmmlearn)
├── run_backtest.py              # baseline 단독 백테스트
├── run_lstm.py                  # LSTM walk-forward + 비교
├── run_compare_all.py           # 4모델 통합 비교
├── ARCHITECTURE.md              # 이 문서
├── DECISION_LOG.md              # 결정 이력
└── INSTRUCTIONS.md              # AI 협업 지시사항
```

---

## 4. 자산 Universe (8개)

| 자산 | 역할 | 학습 ticker | 한국 ETF 코드 |
|---|---|---|---|
| QQQ | 성장 핵심 | QQQ | 133690 |
| SCHD | 배당 + 분산 | VYM (proxy) | 458730 |
| GLD | 인플레이션 헤지 | GLD | 411060 |
| IEF | 미국 중기 채권 | IEF | 305080 |
| SOFR | 현금성 | SHV (proxy) | 456610 |
| EEM | 신흥국 분산 | EEM | 195980 |
| TLT | 장기 채권 (위기 헤지) | TLT | 458250 |
| OIL | 원자재 (현재 미사용) | USO | 261220 |

**자산 선정 원칙**: 글로벌 분산 + ISA 매수 가능 + 데이터 2003년 가용.

---

## 5. 데이터 파이프라인

```
yfinance (가격) ─┐
                 ├─→ load_all() → Close_/Open_ + 매크로
FRED API (매크로) ─┘                    │
                                        ↓
                                build_features() → 132개 feature
                                        │
                                        ├─→ labeler → Triple-Barrier 라벨
                                        │
                                        └─→ 모델 입력
```

**Look-ahead bias 방지**:
- forward fill만 (backward fill 금지)
- StandardScaler는 train에서만 fit
- 매크로 데이터의 발표 지연 고려 (CPI 1개월)

---

## 6. 라벨링 시스템

**Triple-Barrier** (López de Prado 2018):
- 기준 자산: SPY
- horizon: 21일
- barrier: k=1.0 × σ × √(21/252)
- σ: 60일 realized volatility

**3-class 정의**:
- CRISIS (0): SPY 21일 -k×σ 도달
- DEFENSE (1): 21일 만료 (횡보)
- ATTACK (2): SPY 21일 +k×σ 도달

---

## 7. 모델 비교

| 측면 | Baseline | LSTM | HMM |
|---|---|---|---|
| 분류 방식 | 5개 임계값 가중합 | 132 feature 신경망 | 5 feature 분포 학습 |
| 임계값 | 사람이 정함 | 학습 | 자동 (분포 기반) |
| 파라미터 수 | ~10개 | ~63,000개 | ~30개 |
| 데이터 요구 | 적음 | 매우 많음 | 적음 |
| 해석 가능성 | 높음 | 낮음 (블랙박스) | 높음 |
| 학습 시간 | 즉시 | 10~15분/fold | 1~2초/fold |

**비중 결정** (LSTM/HMM 공통):
```
target_w = P_crisis × W_crisis + P_defense × W_defense + P_attack × W_attack
```

**WEIGHT_MAP** (모든 모델 공통):

| 자산 | ATTACK | DEFENSE | CRISIS |
|---|---|---|---|
| QQQ | 0.70 | 0.25 | 0.0 |
| SCHD | 0.15 | 0.20 | 0.20 |
| GLD | 0.05 | 0.15 | 0.25 |
| IEF | 0.05 | 0.20 | 0.20 |
| SOFR | 0.0 | 0.10 | 0.25 |
| EEM | 0.05 | 0.05 | 0.0 |
| TLT | 0.0 | 0.05 | 0.10 |

SCHD가 CRISIS에도 20% 유지 (배당 누적 + 회복기 알파).

---

## 8. 백테스트 엔진

**체결 규약**:
- t일 종가 신호 → t+1일 Open 체결
- 5% no-trade band (max 자산별 비중 차이)
- 거래비용 0.115% 편도 (수수료 0.015% + 슬리피지 0.1%)

**Walk-forward Validation**:
- 분할: Expanding window
- min_train_years: 3
- test_period_months: 12
- embargo: 21일 (Triple-Barrier horizon과 일치)

---

## 9. 평가 지표

**1순위 (학계 검증)**:
- CAGR
- MDD (Max Drawdown)
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Entropic Risk Measure (α=0.5, 1, 2, 5) — Föllmer-Schied (2002)
- Certainty Equivalent (CE)

**2순위 (실전 운용)**:
- ISA 세후 수익 (서민형 비과세 400만원 + 9.9% 분리과세)
- 총 거래 횟수
- 일반 계좌 대비 절세액

---

## 10. 실행 흐름

```
1. config.py 설정 확인
2. python run_backtest.py     # baseline만 빠르게
3. python run_lstm.py          # LSTM 단독 (~15분)
4. python run_compare_all.py   # 모든 모델 비교 (~20분)
5. 결과 분석 → 모델 선택
6. 실전 운용 (텔레그램 봇은 별도 구현 예정)
```

---

## 11. 알려진 한계

1. **자산 universe 한정**: 미국 ETF 중심, 8개
2. **데이터 길이**: 22년 (2003~2025) — 학계 권장 30년 대비 부족
3. **알파 원천**: 자산배분 alpha는 본질적으로 작음
4. **횡보장 분류**: baseline은 진짜 횡보 일부만 잡음 (HMM 우월 예상)
5. **거래비용**: 한국 ETF 실제 비용은 추정치 (0.115%)
6. **환율**: 미국 ticker 백테스트, 한국 ETF 실전 → 환율 영향 미반영

---

## 12. 향후 확장

**우선순위 높음**:
- 텔레그램 봇 (live/signal_generator.py)
- 적립식 Monte Carlo 시뮬레이션

**우선순위 중간**:
- HMM state 수 최적화 (BIC 기반)
- Feature selection (BorutaShap)

**우선순위 낮음 (1년 후)**:
- Layer 2: 섹터 베팅 (별도 시스템)
- 뉴스 sentiment 통합
- Deep Hedging 스타일 end-to-end 학습
