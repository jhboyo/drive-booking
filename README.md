# 강화학습 프로젝트: 자동차 시승 예약 통합 시스템



## 📋 프로젝트 개요

자동차 브랜드 홈페이지에서 시승 예약 메뉴가 존재 함. 기존에는 고객이 직접 차량 선택 및 일정 등을 입력하였으나 
본 프로젝트에서는 고객에게 최적의 차량을 추천하고, 시승 일정을 효율적으로 배정하는 통합 시스템을 강화학습으로 구현합니다.

### 핵심 아이디어
- **Phase 1**: 고객 상담을 통한 맞춤 시승 차량 추천 (Vehicle Recommendation)
- **Phase 2**: 추천된 차량의 시승 스케줄링 최적화 (Test Drive Scheduling)
- **통합**: 두 단계를 연결하여 end-to-end 고객 경험 최적화

### 차별화 포인트
1. 단순 추천이 아닌 실제 예약까지 고려한 **실용적 시스템**
2. Multi-stage decision making으로 **현실적인 문제 모델링**
3. Hyundai, Kia, Genesis **3개 브랜드 통합 비교**
4. 추천-스케줄링 간 **시너지 효과** 분석

---

## 🚨 현재 문제

### 브랜드별 시승 예약 프로세스 분석

현재 각 브랜드 웹사이트의 시승 예약 과정을 분석한 결과:

| 브랜드 | 단계 수 | 주요 단계 |
|--------|---------|-----------|
| **Hyundai** | 7단계 | 모델 → 장소 → 방법 → 일정 → 운전경력 → 보유차종 → 요청사항 |
| **Genesis** | 5단계 | 차량 → 드라이빙라운지 → 일정 → 유의사항 → 확인 |
| **Kia** | 7단계 | 모델 → 거점 → 방법 → 일정 → 동의 → 시승자 → 설문 |

> 📁 실제 화면 캡처: `resource/image/brand/` 참조

### 고객의 번거로움 (Pain Points)

1. **차량 선택의 어려움**: 수십 개 모델 중 어떤 차가 나에게 맞는지 모름
2. **복잡한 단계**: 5~7단계를 모두 직접 입력해야 함
3. **일정 선택의 불편함**: 캘린더에서 가능한 시간대를 직접 확인 (이미 예약된 슬롯 많음)
4. **정보 입력 반복**: 운전경력, 보유차종 등 매번 입력

### 강화학습으로 해결

```
AS-IS: 고객이 모든 것을 직접 선택 (5~7단계)
         ↓
TO-BE: AI가 최소 질문으로 추천 + 최적 일정 자동 배정 (2단계)
```

- **Phase 1**: "어떤 용도로 사용하세요?" 등 2~3개 질문만으로 최적 시승 차량 추천
- **Phase 2**: 고객 선호 시간 + 센터 가용 상황 고려하여 최적 일정 자동 배정

---

## 🎯 문제 정의

### Phase 1: 시승 차량 추천 시스템

**목표**: 고객과의 대화를 통해 최소한의 질문으로 최적의 시승 차량 추천

**State**:
```python
{
    'customer_profile': {
        'age': int,              # 고객 연령
        'budget': float,         # 예산 (만원)
        'family_size': int,      # 가족 구성원 수
        'commute_distance': int, # 출퇴근 거리 (km)
        'priorities': [float],   # [안전성, 연비, 성능, 디자인]
    },
    'conversation_history': [
        {'question': str, 'answer': str}
    ],
    'candidate_cars': {
        'car_name': {'match_score': float, 'available': bool}
    },
    'questions_asked': int,
    'max_questions': 5
}
```

**Action**:
- 0-9: 고객 선호도 질문 (주 사용 목적, 연료 타입, 차량 크기 등)
- 10-13: 시승 차량 추천 (1개, 2개, 3개, 대체 차량)

**Reward**:
```
R1 = 10 × (고객 만족도)
   + 5 × (시승 예약 의향)
   - 1 × (질문 수)
   - 10 × (재고 없는 시승 차량 추천)
```

---

### Phase 2: 시승 스케줄링 최적화

**목표**: 추천된 차량들의 시승 일정을 최적으로 배정하여 대기시간 최소화 및 센터 효율 극대화

**State**:
```python
{
    'recommended_cars': [str],  # Phase 1의 추천 결과
    'customer_preference': {
        'preferred_date': str,
        'time_window': (int, int),  # 선호 시간대
        'flexibility': float        # 0~1, 유연성
    },
    'center_state': {
        'available_vehicles': {
            'car_name': {
                'count': int,
                'current_schedule': [(start, end, customer_id)],
                'next_available': str
            }
        },
        'staff_available': [{'id': int, 'free_at': str}],
        'pending_requests': int
    },
    'current_time': str,
    'date': str
}
```

**Action**:
- 시간대 배정: (차량 인덱스, 시간 슬롯) 조합
- 대체 시간 제안
- 대기 (더 나은 슬롯 기다림)

**Reward**:
```
R2 = 15 × (예약 성사)
   + 5 × (선호 시간 매칭)
   - 2 × (대기시간 / 30분)
   + 3 × (차량 활용률)
   - 10 × (스케줄 충돌)
```

---

### 통합 보상 함수

```
Total_Reward = R1 + R2 + Synergy_Bonus

Synergy_Bonus = 5 × (즉시 예약 가능)
              + 3 × (추천-스케줄 매칭도)
```

---

## 🏗️ 시스템 아키텍처

```
고객 입력
    ↓
┌─────────────────────────────┐
│  Phase 1: 시승 차량 추천         │
│  - Q-Learning / DQN         │
│  - State: 고객 프로필       │
│  - Action: 질문 또는 추천   │
└─────────────────────────────┘
    ↓ (추천 차량 목록)
┌─────────────────────────────┐
│  Phase 2: 스케줄링          │
│  - DQN / A2C                │
│  - State: 센터 상태         │
│  - Action: 시간 배정        │
└─────────────────────────────┘
    ↓
최종 스케줄 + 예약 확정
```

---

## 📅 구현 로드맵

상세 로드맵은 [ROADMAP.md](./ROADMAP.md)를 참조하세요.

---

## 🛠️ 기술 스택

### 환경
- Python 3.9+
- OpenAI Gym / Gymnasium

### 강화학습
- **Phase 1**: Q-Learning (Tabular or Function Approximation)
- **Phase 2**: DQN (Deep Q-Network)
- **선택**: PPO, A2C (시간 여유 시)

### 라이브러리
```txt
gymnasium>=0.29.0
numpy>=1.24.0
torch>=2.0.0
matplotlib>=3.7.0
pandas>=2.0.0
seaborn>=0.12.0
tqdm>=4.65.0
```

### 데이터
- 차량 정보: JSON 파일 (직접 구축)
- 고객 프로필: 시뮬레이션 생성 (정규분포, 포아송 분포 등)

---

## 📊 평가 지표

### Phase 1: 시승 차량 추천
- **추천 정확도**: 고객 선호도와 추천 차량 매칭 점수
- **질문 효율성**: 평균 질문 수
- **예약 전환율**: 추천 후 실제 예약 비율

### Phase 2: 스케줄링
- **평균 대기시간**: 고객 요청 ~ 예약 확정
- **차량 활용률**: 실제 사용 시간 / 가용 시간
- **예약 성사율**: 전체 요청 대비 예약 확정 비율

### 통합 시스템
- **Total Reward**: Episode당 누적 보상
- **고객 만족도**: 추천 + 스케줄 종합 점수
- **시스템 효율성**: 시간당 처리 고객 수

---

## 🎨 예상 시각화

1. **학습 곡선**: Episode vs Cumulative Reward
2. **알고리즘 비교**: Bar chart (RL vs Baseline)
3. **고객 여정**: Flowchart (상담 → 추천 → 예약)
4. **스케줄 간트 차트**: 시간대별 차량 배정
5. **브랜드별 성능**: Hyundai vs Kia vs Genesis
6. **Ablation Study**: Phase별 기여도

---

## 📝 보고서 구성 (PPT)

1. **표지**: 팀원 정보, GitHub 링크
2. **프로젝트 소개**: 문제 정의, 목표
3. **환경 설계**: State, Action, Reward 상세 설명
4. **알고리즘**: Q-Learning, DQN 설명
5. **실험 설계**: 시나리오, 하이퍼파라미터
6. **실험 결과**: 그래프, 테이블, 분석
7. **Ablation Study**: 각 컴포넌트 기여도
8. **토의 및 결론**: 인사이트, 개선사항
9. **팀원 기여사항**: 각자 역할

---

## 🚀 실행 방법

### 환경 설정
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 학습 실행
```bash
# Phase 1: 시승 차량 추천 학습
python src/train_phase1.py --episodes 1000 --algorithm qlearning

# Phase 2: 스케줄링 학습
python src/train_phase2.py --episodes 1000 --algorithm dqn

# 통합 시스템 학습
python src/train_integrated.py --episodes 2000
```

### 평가 및 시각화
```bash
# 학습된 모델 평가
python src/evaluate.py --model checkpoints/integrated_model.pth

# 시각화 생성
python src/visualization/generate_plots.py
```
 
---

## 📖 참고 자료

- [Gymnasium Documentation](https://gymnasium.farama.org)
- [Spinning Up in Deep RL](https://spinningup.openai.com)
- [DQN Paper](https://www.nature.com/articles/nature14236)
- 프로젝트 핸드아웃: `RL_project_handout_r1.pdf`

---

## 📌 주의사항

- 코드는 직접 구현하여 표절 방지
- AI 코딩 도구 사용 가능하나 반드시 검증
- Random seed를 바꿔가며 실험 (신뢰구간 계산)
- 작은 문제라도 명확히 정의하고 해결하는 것이 중요

---

## 🎯 목표 성과

- [ ] 시승 차량 추천 정확도 80% 이상
- [ ] 평균 대기시간 30분 이하
- [ ] 차량 활용률 70% 이상
- [ ] 베이스라인 대비 20% 성능 향상
- [ ] 3개 브랜드 통합 비교 완료

---