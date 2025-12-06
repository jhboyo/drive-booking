# 강화학습 프로젝트: 자동차 시승 예약 통합 시스템

> **Interactive Conversational Recommendation System using Reinforcement Learning**

## 📋 프로젝트 개요

자동차 브랜드 홈페이지의 시승 예약 과정에서 고객이 겪는 번거로움을 **강화학습 기반 대화형 추천 시스템**으로 해결함.

### 강화학습 관점의 문제 정의

본 프로젝트는 **Agent-Environment 상호작용** 구조로 설계됨.

- **Environment**: 고객(응답자), 차량 데이터베이스(재고), 시승 센터 상태(스케줄)로 구성
- **Agent**: 강화학습 기반 의사결정 주체
- **State**: 고객 응답 히스토리, 후보 차량 목록, 센터 가용 상태
- **Action**: 질문 선택, 차량 추천, 일정 배정
- **Reward**: 고객 만족도 + 예약 성사 - 질문 수 - 대기시간

### Two-Phase RL System

본 시스템은 두 개의 연속된 강화학습 Agent로 구성됨.

- **Phase 1 (Recommendation Agent)**: Q-Learning 기반, 최소 질문으로 최적 차량 추천
- **Phase 2 (Scheduling Agent)**: DQN 기반, 고객과 센터 모두 만족하는 일정 배정

두 Phase는 순차적으로 실행되며, Phase 1의 추천 결과가 Phase 2의 입력으로 전달됨.

### 학습 목표

Agent가 학습을 통해 스스로 터득해야 하는 것:

1. 어떤 질문이 고객 선호 파악에 효과적인가?
2. 몇 개의 질문 후 추천해야 정확도와 효율의 균형이 맞는가?
3. 어떤 차량을 추천해야 고객 만족도가 높은가?
4. 어떤 시간대에 배정해야 고객과 센터 모두 만족하는가?

### 차별화 포인트

1. **대화형 추천**: 단순 필터링이 아닌 Sequential Decision Making 기반 추천
2. **Two-Phase 통합**: 추천과 스케줄링을 연결한 실용적 시스템
3. **시뮬레이션 환경**: 도메인 지식 기반 현실적인 고객·센터 시뮬레이션

### 구현 범위

- **분석 대상**: Hyundai, Kia, Genesis 3개 브랜드 시승 예약 프로세스
- **구현 대상**: **Hyundai** 브랜드 기준으로 구현 (7단계 프로세스 → 2단계로 단순화)
- **확장성**: 차량 DB와 질문셋만 교체하면 다른 브랜드에도 적용 가능한 구조로 설계

---

## 🚨 현재 문제

### 브랜드별 시승 예약 프로세스 분석

| 브랜드 | 단계 수 | 주요 단계 |
|--------|---------|-----------|
| **Hyundai** | 7단계 | 모델 → 장소 → 방법 → 일정 → 운전경력 → 보유차종 → 요청사항 |
| **Genesis** | 5단계 | 차량 → 드라이빙라운지 → 일정 → 유의사항 → 확인 |
| **Kia** | 7단계 | 모델 → 거점 → 방법 → 일정 → 동의 → 시승자 → 설문 |

> 📁 실제 화면 캡처: `resource/image/brand/` 참조

### 고객의 번거로움 (Pain Points)

1. **차량 선택의 어려움**: 수십 개 모델 중 어떤 차가 나에게 맞는지 모름
2. **복잡한 단계**: 5~7단계를 모두 직접 입력해야 함
3. **일정 선택의 불편함**: 캘린더에서 가능한 시간대를 직접 확인 필요
4. **정보 입력 반복**: 운전경력, 보유차종 등 매번 입력

### 강화학습으로 해결

```
AS-IS: 고객이 모든 것을 직접 선택 (5~7단계)
         ↓
TO-BE: AI가 최소 질문으로 추천 + 최적 일정 자동 배정 (2단계)
```

- **Phase 1**: 2~3개 질문만으로 최적 시승 차량 추천
- **Phase 2**: 고객 선호 시간 + 센터 가용 상황 고려하여 최적 일정 자동 배정

---

## 🤖 왜 강화학습인가?

### 단순 추천 vs 대화형 추천

| 구분 | 단순 추천 | 대화형 추천 (본 프로젝트) |
|------|-----------|---------------------------|
| 방식 | 프로필 → 즉시 추천 | 질문 → 응답 → 질문 → ... → 추천 |
| 적합 알고리즘 | Collaborative Filtering, Rule-based | **강화학습 (RL)** |
| 의사결정 | 1회성 | **Sequential Decision Making** |

### 대화형 추천의 RL 적합성

본 프로젝트는 **"20 Questions" 게임**과 유사한 구조를 가짐:

```
Agent: "주로 어떤 용도로 사용하세요?"     ← Action 1: 질문 선택
고객: "출퇴근이요"                        ← 환경 응답 (State 변화)
Agent: "가족은 몇 명이세요?"              ← Action 2: 추가 질문 or 추천?
고객: "4명이요"                           ← 환경 응답 (State 변화)
Agent: "싼타페 하이브리드 추천드립니다"    ← Action 3: 추천 결정
고객: "마음에 들어요!" (만족도 80%)        ← Reward 발생
```

### RL이 학습하는 것

| 학습 목표 | 설명 |
|-----------|------|
| **질문 최소화** | 몇 개의 질문으로 정확히 추천할 수 있는지 |
| **질문 순서 최적화** | 어떤 질문이 정보량(Information Gain)이 높은지 |
| **추천 타이밍** | 언제 질문을 멈추고 추천해야 하는지 |
| **Exploration-Exploitation** | 새로운 질문 전략 탐색 vs 검증된 전략 활용 |

### 강화학습 적용의 핵심 요소

| 요소 | 자율주행 | 대화형 추천 (본 프로젝트) |
|------|----------|---------------------------|
| Sequential Decision | ✅ 연속 제어 | ✅ 질문 → 질문 → 추천 |
| Delayed Reward | ✅ 목적지 도착 | ✅ 추천 후 만족도 확인 |
| State Transition | ✅ 차량 위치 변화 | ✅ 고객 응답으로 정보 축적 |
| Exploration-Exploitation | ✅ 새 경로 탐색 | ✅ 더 질문할지 vs 바로 추천할지 |

---

## 🎯 MDP 설계

상세 State, Action, Reward 설계는 [MDP_DESIGN.md](./resource/docs/MDP_DESIGN.md) 참조.

---

## 🏗️ 시스템 아키텍처

```
고객 입력
    ↓
┌─────────────────────────────┐
│  Phase 1: 시승 차량 추천    │
│  - Q-Learning               │
│  - State: 고객 프로필       │
│  - Action: 질문 또는 추천   │
└─────────────────────────────┘
    ↓ (추천 차량 목록)
┌─────────────────────────────┐
│  Phase 2: 스케줄링          │
│  - DQN                      │
│  - State: 센터 상태         │
│  - Action: 시간 배정        │
└─────────────────────────────┘
    ↓
최종 스케줄 + 예약 확정
```

---

## 🛠️ 기술 스택

- **Python** 3.12+
- **Gymnasium** (OpenAI Gym)
- **PyTorch** (DQN)
- **NumPy, Pandas, Matplotlib, Seaborn**

---

## 📊 평가 지표

| Phase | 지표 | 설명 |
|-------|------|------|
| Phase 1 | 추천 정확도 | 고객 선호도와 추천 차량 매칭 점수 |
| Phase 1 | 질문 효율성 | 평균 질문 수 |
| Phase 2 | 평균 대기시간 | 고객 요청 ~ 예약 확정 |
| Phase 2 | 차량 활용률 | 실제 사용 시간 / 가용 시간 |
| 통합 | Total Reward | Episode당 누적 보상 |

---

## 🚀 실행 방법

```bash
# 환경 설정
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Phase 1 학습
python src/train_phase1.py --episodes 1000

# Phase 2 학습
python src/train_phase2.py --episodes 1000

# 통합 시스템 학습
python src/train_integrated.py --episodes 2000

# 평가
python src/evaluate.py --model checkpoints/integrated_model.pth
```

---

## 📅 로드맵

상세 구현 일정은 [ROADMAP.md](./ROADMAP.md) 참조.

---

## 🎯 목표 성과

- [ ] 시승 차량 추천 정확도 80% 이상
- [ ] 평균 질문 수 3개 이하
- [ ] 베이스라인(Random, Rule-based) 대비 20% 성능 향상

---

## 📂 프로젝트 구조

```
driving-test/
├── README.md
├── ROADMAP.md
├── requirements.txt
├── data/
│   ├── vehicles.json
│   └── customer_profiles.json
├── src/
│   ├── env/
│   ├── agents/
│   ├── baselines/
│   ├── utils/
│   └── visualization/
└── resource/
    ├── docs/
    │   └── MDP_DESIGN.md
    └── image/brand/
```

---

## 📖 참고 자료

- [Gymnasium Documentation](https://gymnasium.farama.org)
- [Spinning Up in Deep RL](https://spinningup.openai.com)
