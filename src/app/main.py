"""
Brand 자동차 시승 예약 챗봇 - Streamlit 앱
Hyundai Test Drive Reservation Chatbot

학습된 강화학습 모델을 사용하여 대화형으로 차량을 추천하고
시승 예약을 진행하는 챗봇 인터페이스임.
"""

import streamlit as st
import json
import sys
import random
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# 프로젝트 루트 경로 설정 (로컬/Streamlit Cloud 호환)
def get_project_root() -> Path:
    """프로젝트 루트 경로 반환 (다양한 실행 환경 지원)"""
    # 방법 1: __file__ 기반 (로컬 실행)
    if '__file__' in globals():
        return Path(__file__).resolve().parent.parent.parent

    # 방법 2: 현재 작업 디렉토리에서 src 폴더 탐색
    cwd = Path.cwd()
    if (cwd / 'src').is_dir():
        return cwd

    # 방법 3: sys.path에서 src 폴더가 있는 경로 탐색
    for p in sys.path:
        path = Path(p)
        if (path / 'src').is_dir():
            return path

    # 기본값
    return cwd

project_root = get_project_root()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 에이전트 및 환경 임포트
from src.agents.q_learning_agent import QLearningAgent
from src.agents.scheduling_agent import DQNAgent
from src.env.recommendation_env import VehicleRecommendationEnv
from src.env.scheduling_env import SchedulingEnv

# ============================================================================
# 페이지 설정
# ============================================================================

st.set_page_config(
    page_title="Brand 차 시승 예약",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# CSS 스타일 로드
# ============================================================================

def load_css():
    """외부 CSS 파일 로드"""
    css_path = project_root / "resource" / "styles" / "main.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            return f"<style>{f.read()}</style>"
    return ""

st.markdown(load_css(), unsafe_allow_html=True)


# ============================================================================
# 데이터 로드
# ============================================================================

def load_questions():
    """질문 데이터 로드"""
    with open(project_root / "data" / "questions.json", "r", encoding="utf-8") as f:
        return json.load(f)["questions"]

@st.cache_data
def load_vehicles():
    """차량 데이터 로드"""
    with open(project_root / "data" / "vehicles.json", "r", encoding="utf-8") as f:
        return json.load(f)["vehicles"]

# ============================================================================
# 에이전트 로드
# ============================================================================

@st.cache_resource
def load_agents():
    """학습된 에이전트 로드"""
    try:
        # Phase 1 환경 생성 (에이전트 초기화용)
        phase1_env = VehicleRecommendationEnv()

        # Phase 1 에이전트 (Q-Learning)
        phase1_agent = QLearningAgent(
            n_actions=phase1_env.action_space.n,
            seed=42
        )

        # 학습된 모델 로드 시도
        checkpoint_path = project_root / "checkpoints" / "chatbot" / "chatbot_q_learning.json"
        if checkpoint_path.exists():
            phase1_agent.load(str(checkpoint_path))
            model_loaded = True
        else:
            # 체크포인트 없으면 새로 시작
            model_loaded = False

        return phase1_agent, phase1_env, model_loaded

    except Exception as e:
        st.error(f"에이전트 로드 실패: {e}")
        return None, None, False

# ============================================================================
# 세션 상태 초기화
# ============================================================================

def init_session_state():
    """세션 상태 초기화"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "phase" not in st.session_state:
        st.session_state.phase = "greeting"  # greeting, questioning, recommending, scheduling, complete

    if "answers" not in st.session_state:
        st.session_state.answers = {}  # 사용자 응답 저장

    if "current_question_idx" not in st.session_state:
        st.session_state.current_question_idx = None

    if "questions_asked" not in st.session_state:
        st.session_state.questions_asked = []  # 이미 한 질문들

    if "recommended_vehicle" not in st.session_state:
        st.session_state.recommended_vehicle = None

    if "recommended_history" not in st.session_state:
        st.session_state.recommended_history = []  # 이미 추천한 차량 ID 목록

    if "observation" not in st.session_state:
        st.session_state.observation = None

    # Reward 추적 (RL 시각화용)
    if "reward" not in st.session_state:
        st.session_state.reward = 0.0  # 누적 보상

    # RL Trajectory 추적 (모델 학습용)
    if "trajectory" not in st.session_state:
        st.session_state.trajectory = []  # [(observation, action, reward), ...]

    if "episode_step_reward" not in st.session_state:
        st.session_state.episode_step_reward = 0.0  # 현재 스텝의 보상

    # 현재 Action 표시용
    if "current_action" not in st.session_state:
        st.session_state.current_action = "대기 중"

    # MDP 시각화용 추가 변수
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0  # 현재 에피소드 스텝

    if "policy_mode" not in st.session_state:
        st.session_state.policy_mode = "대기"  # 탐험/활용/대기

init_session_state()

# ============================================================================
# RL 모델 연동 함수
# ============================================================================

def build_observation() -> np.ndarray:
    """
    챗봇 상태를 RL 에이전트용 observation 벡터로 변환

    Returns:
        69차원 observation 벡터
    """
    obs = np.zeros(69)

    # [0-4]: 고객 정보 (기본값 사용 - 챗봇에서는 수집 안 함)
    obs[0] = 0.0  # 나이 (정규화, 기본: 중년)
    obs[1] = 0.0  # 성별 (기본: 중립)
    obs[2] = 0.0  # 외국인 여부
    obs[3] = 0.0  # 직장인 여부
    obs[4] = 1.0  # 관심차량 있음 (시승 예약이므로)

    # [5-44]: 질문 응답 (8질문 x 5옵션, one-hot)
    # questions.json의 attribute와 매핑
    attribute_to_idx = {
        "usage": 0, "fuel_type": 1, "family_size": 2, "budget": 3,
        "priority": 4, "size": 5, "body_type": 6, "color": 7
    }

    for attr, q_idx in attribute_to_idx.items():
        if attr in st.session_state.answers:
            # 해당 질문의 옵션 인덱스 찾기
            for q in questions:
                if q.get("attribute") == attr:
                    answer = st.session_state.answers[attr]
                    if answer in q["options"]:
                        opt_idx = q["options"].index(answer)
                        # one-hot 인코딩
                        base_idx = 5 + q_idx * 5
                        if opt_idx < 5:  # 최대 5개 옵션
                            obs[base_idx + opt_idx] = 1.0
                    break

    # [45]: 질문 횟수 비율 (0~1)
    max_questions = 8
    obs[45] = len(st.session_state.questions_asked) / max_questions

    # [46-68]: 차량 점수 (간단히 균등 분포)
    obs[46:69] = 0.5

    return obs


def get_action_for_question(question_attr: str) -> int:
    """질문 attribute를 RL action 인덱스로 변환"""
    attr_to_action = {
        "usage": 0, "fuel_type": 1, "family_size": 2, "budget": 3,
        "priority": 4, "size": 5, "body_type": 6, "color": 7, "region": 7
    }
    return attr_to_action.get(question_attr, 0)


def get_action_name(action_type: str, detail: str = "") -> str:
    """액션 타입을 사람이 읽기 쉬운 이름으로 변환"""
    action_names = {
        "usage": "용도 질문",
        "fuel_type": "연료타입 질문",
        "family_size": "가족구성원 질문",
        "budget": "예산 질문",
        "priority": "우선순위 질문",
        "size": "크기 질문",
        "body_type": "차체타입 질문",
        "color": "컬러 질문",
        "region": "지역 질문",
        "recommend": "차량 추천",
        "schedule": "일정 배정",
        "complete": "예약 완료",
        "waiting": "대기 중"
    }
    name = action_names.get(action_type, action_type)
    if detail:
        return f"{name} ({detail})"
    return name


def update_rl_model(final_reward: float, terminated: bool = True):
    """
    에피소드 종료 시 RL 모델 업데이트

    Args:
        final_reward: 최종 보상
        terminated: 정상 종료 여부
    """
    if phase1_agent is None or len(st.session_state.trajectory) == 0:
        return

    trajectory = st.session_state.trajectory

    # Trajectory의 각 스텝에 대해 Q-Learning 업데이트
    for i, (obs, action, step_reward) in enumerate(trajectory):
        if i < len(trajectory) - 1:
            next_obs = trajectory[i + 1][0]
            phase1_agent.update(obs, action, step_reward, next_obs, False, False)
        else:
            # 마지막 스텝: 최종 보상 포함
            final_obs = build_observation()
            total_step_reward = step_reward + final_reward
            phase1_agent.update(obs, action, total_step_reward, final_obs, terminated, False)

    # 에피소드 종료 처리
    phase1_agent.end_episode()

    # 모델 저장 (매 에피소드마다)
    save_model()


def save_model():
    """학습된 모델 저장 (5 에피소드마다 standalone 모델과 동기화)"""
    if phase1_agent is None:
        return

    # 챗봇 모델 저장 (매 에피소드)
    chatbot_path = project_root / "checkpoints" / "chatbot" / "chatbot_q_learning.json"
    chatbot_path.parent.mkdir(parents=True, exist_ok=True)
    phase1_agent.save(str(chatbot_path))

    # 5 에피소드마다 standalone 모델과 동기화
    if phase1_agent.episode_count % 5 == 0:
        standalone_path = project_root / "checkpoints" / "standalone" / "q_learning_model.json"
        standalone_path.parent.mkdir(parents=True, exist_ok=True)
        phase1_agent.save(str(standalone_path))


def record_trajectory(action: int, reward: float):
    """현재 상태와 액션을 trajectory에 기록"""
    obs = build_observation()
    st.session_state.trajectory.append((obs.copy(), action, reward))


# ============================================================================
# 헤더
# ============================================================================

st.markdown("""
<div class="header-container">
    <div class="header-inner">
        <div class="header-left">
            <div class="header-title">Brand 차 시승 예약</div>
            <div class="header-subtitle">AI가 최적의 차량을 추천해 드립니다</div>
        </div>
        <div class="header-icon">🗓️</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 에이전트 및 데이터 로드
# ============================================================================

questions = load_questions()
vehicles = load_vehicles()
phase1_agent, phase1_env, model_loaded = load_agents()

# 레이어드 카드 - 상태 표시
status_text = "🧠 Q-Learning 기반" if model_loaded else "🧠 강화학습 모델"
phase_text = {
    "greeting": "🎯 시작",
    "questioning": "💬 선호도 분석",
    "recommending": "🚗 최적 차량 추천",
    "scheduling": "📅 일정 최적화",
    "complete": "✅ 예약 완료"
}.get(st.session_state.phase, "")

# Reward 색상 (양수: 초록, 음수: 빨강, 0: 회색)
reward = st.session_state.reward
if reward > 0:
    reward_color = "#16A34A"
    reward_bg = "#DCFCE7"
elif reward < 0:
    reward_color = "#DC2626"
    reward_bg = "#FEE2E2"
else:
    reward_color = "#6B7280"
    reward_bg = "#F3F4F6"

# 모델 통계
episode_count = phase1_agent.episode_count if phase1_agent else 0
q_table_size = len(phase1_agent.q_table) if phase1_agent else 0
epsilon = phase1_agent.epsilon if phase1_agent else 1.0

# 현재 MDP 상태
current_action = st.session_state.current_action
current_step = st.session_state.current_step
policy_mode = st.session_state.policy_mode
state_progress = f"{len(st.session_state.questions_asked)}/8"

# Policy 색상
if policy_mode == "탐험":
    policy_color = "#7C3AED"
    policy_bg = "#EDE9FE"
    policy_icon = "🔍"
elif policy_mode == "활용":
    policy_color = "#059669"
    policy_bg = "#D1FAE5"
    policy_icon = "🎯"
else:
    policy_color = "#6B7280"
    policy_bg = "#F3F4F6"
    policy_icon = "⏸️"

st.markdown(f"""
<div class="layered-card" style="padding: 1.2rem;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
        <h3 style="color: #002C5F; margin: 0; font-size: 1.1rem; font-weight: 600;">👋 안녕하세요!</h3>
        <div>
            <span class="status-badge">{status_text}</span>
            <span class="status-badge">{phase_text}</span>
        </div>
    </div>
    <p style="color: #555; font-size: 0.85rem; margin: 0 0 0.8rem 0; text-align: center;">Brand 차 시승 예약 도우미입니다.</p>
    <div style="display: flex; gap: 0.5rem; margin-bottom: 0.5rem;">
        <div style="flex: 1; background: #FEF3C7; border-radius: 12px; padding: 0.5rem 0.8rem; display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #92400E; font-size: 0.75rem; font-weight: 500;">🎯 Action</span>
            <span style="color: #B45309; font-size: 0.85rem; font-weight: 600;">{current_action}</span>
        </div>
        <div style="flex: 1; background: {reward_bg}; border-radius: 12px; padding: 0.5rem 0.8rem; display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #374151; font-size: 0.75rem; font-weight: 500;">🏆 Reward</span>
            <span style="color: {reward_color}; font-size: 1rem; font-weight: 700;">{reward:+.1f}</span>
        </div>
    </div>
    <div style="display: flex; gap: 0.5rem; margin-bottom: 0.5rem;">
        <div style="flex: 1; background: #F0FDF4; border-radius: 12px; padding: 0.4rem 0.6rem; display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #166534; font-size: 0.7rem; font-weight: 500;">📋 State</span>
            <span style="color: #15803D; font-size: 0.8rem; font-weight: 600;">{state_progress}</span>
        </div>
        <div style="flex: 1; background: #FDF4FF; border-radius: 12px; padding: 0.4rem 0.6rem; display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #86198F; font-size: 0.7rem; font-weight: 500;">🎲 ε</span>
            <span style="color: #A21CAF; font-size: 0.8rem; font-weight: 600;">{epsilon:.2f}</span>
        </div>
        <div style="flex: 1; background: #FFF7ED; border-radius: 12px; padding: 0.4rem 0.6rem; display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #9A3412; font-size: 0.7rem; font-weight: 500;">👣 Step</span>
            <span style="color: #C2410C; font-size: 0.8rem; font-weight: 600;">{current_step}</span>
        </div>
        <div style="flex: 1; background: {policy_bg}; border-radius: 12px; padding: 0.4rem 0.6rem; display: flex; justify-content: space-between; align-items: center;">
            <span style="color: {policy_color}; font-size: 0.7rem; font-weight: 500;">{policy_icon} Policy</span>
            <span style="color: {policy_color}; font-size: 0.8rem; font-weight: 600;">{policy_mode}</span>
        </div>
    </div>
    <div style="background: #EFF6FF; border-radius: 12px; padding: 0.5rem 1rem; margin-bottom: 0.5rem; display: flex; justify-content: space-around; align-items: center;">
        <span style="color: #3B82F6; font-size: 0.75rem; font-weight: 500;">📊 Episodes: {episode_count}</span>
        <span style="color: #3B82F6; font-size: 0.75rem; font-weight: 500;">🧠 Q-states: {q_table_size}</span>
    </div>
    <p style="margin: 0; color: #6B7280; font-size: 0.7rem; text-align: center;">추가질문 -1 | 다른차량 -5 | 예약확정 +15</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 차량 추천 함수
# ============================================================================

def get_vehicle_recommendation(answers: dict, exclude_ids: list = None) -> dict:
    """사용자 응답 기반 차량 추천

    Args:
        answers: 사용자 응답 딕셔너리
        exclude_ids: 제외할 차량 ID 목록 (이미 추천한 차량)
    """
    if exclude_ids is None:
        exclude_ids = []

    # 간단한 규칙 기반 추천 (실제로는 학습된 모델 사용)
    scores = {}

    for vehicle in vehicles:
        # 이미 추천한 차량은 제외
        if vehicle["id"] in exclude_ids:
            continue

        score = 0

        # 예산 매칭
        if "budget" in answers:
            budget_map = {
                "3000만원 이하": 3000,
                "3000-4500만원": 4500,
                "4500-6000만원": 6000,
                "6000만원 이상": 10000
            }
            max_budget = budget_map.get(answers["budget"], 5000)
            if vehicle["price_range"]["min"] <= max_budget:
                score += 2

        # 연료 타입 매칭
        if "fuel_type" in answers:
            fuel_map = {
                "가솔린": "gasoline",
                "하이브리드": "hybrid",
                "전기차": "electric",
                "상관없음": None
            }
            preferred_fuel = fuel_map.get(answers["fuel_type"])
            if preferred_fuel is None or vehicle["fuel_type"] == preferred_fuel:
                score += 2

        # 가족 구성원 매칭
        if "family_size" in answers:
            size_map = {"1명": 2, "2명": 4, "3-4명": 5, "5명 이상": 7}
            needed_seats = size_map.get(answers["family_size"], 5)
            if vehicle["seats"] >= needed_seats:
                score += 2

        # 차량 크기 매칭
        if "size" in answers:
            if answers["size"] == "상관없음" or vehicle["size"] in answers["size"].lower():
                score += 1

        # 차체 타입 매칭
        if "body_type" in answers:
            body_map = {"세단": "sedan", "SUV": "suv", "MPV": "mpv", "상관없음": None}
            preferred_body = body_map.get(answers["body_type"])
            if preferred_body is None or vehicle["category"] == preferred_body:
                score += 2

        # 우선순위 매칭
        if "priority" in answers:
            priority_map = {"안전성": "safety", "연비": "fuel_efficiency", "성능": "performance", "디자인": "design"}
            priority_key = priority_map.get(answers["priority"])
            if priority_key and priority_key in vehicle["features"]:
                score += vehicle["features"][priority_key] * 3

        scores[vehicle["id"]] = score

    # 추천할 차량이 없으면 None 반환
    if not scores:
        return None

    # 최고 점수 차량 반환
    best_id = max(scores, key=scores.get)
    for v in vehicles:
        if v["id"] == best_id:
            return v

    return None

# ============================================================================
# 채팅 인터페이스
# ============================================================================

chat_container = st.container()

with chat_container:
    # 채팅 히스토리 표시
    for chat in st.session_state.chat_history:
        if chat["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(chat["content"])
        else:
            with st.chat_message("user", avatar="🙂"):
                st.markdown(chat["content"])

# ============================================================================
# 대화 흐름 관리
# ============================================================================

# 인사 단계
if st.session_state.phase == "greeting":
    # 현재 Action: 대기 중
    st.session_state.current_action = get_action_name("waiting")
    st.session_state.policy_mode = "대기"

    if st.button("🚀 시작하기", type="secondary"):
        st.session_state.phase = "questioning"
        st.session_state.current_step = 0  # 에피소드 시작 시 Step 초기화
        st.session_state.chat_history.append({"role": "user", "content": "시작할게요!"})
        st.rerun()

# 질문 단계
elif st.session_state.phase == "questioning":
    # 아직 질문할 게 있는지 확인
    remaining_questions = [q for q in questions if q["id"] not in st.session_state.questions_asked]

    # 필수 질문 목록 (처음 3개 + 지역 질문)
    required_attributes = ["usage", "fuel_type", "family_size", "region"]
    required_questions = [q for q in questions if q.get("attribute") in required_attributes]
    required_asked = [q for q in required_questions if q["id"] in st.session_state.questions_asked]

    # 모든 질문 완료 또는 남은 질문 없음 → 추천 단계로
    if len(remaining_questions) == 0:
        st.session_state.phase = "recommending"
        st.rerun()
    else:
        # 다음 질문 선택
        if st.session_state.current_question_idx is None:
            # 필수 질문 중 아직 안 한 것 우선
            remaining_required = [q for q in remaining_questions if q.get("attribute") in required_attributes]

            if len(remaining_required) > 0:
                next_q = remaining_required[0]
            else:
                # 필수 질문 완료, 나머지 질문 진행
                next_q = remaining_questions[0]

            st.session_state.current_question_idx = next_q["id"]

            # 현재 Action 업데이트
            question_attr = next_q.get("attribute", "")
            st.session_state.current_action = get_action_name(question_attr)

            # 질문 메시지 추가
            q_msg = f"**{next_q['text']}**"
            st.session_state.chat_history.append({"role": "assistant", "content": q_msg})
            st.rerun()

        # 현재 질문에 대한 옵션 버튼 표시
        current_q = questions[st.session_state.current_question_idx]

        # 필수 질문 4개 완료 후 스킵 버튼 표시
        show_skip_btn = len(required_asked) >= 4

        st.markdown("##### 답변을 선택해주세요:")

        # 버튼들 가로 배치 (스킵 버튼 포함)
        num_cols = len(current_q["options"]) + (1 if show_skip_btn else 0)
        cols = st.columns(num_cols)

        for i, option in enumerate(current_q["options"]):
            with cols[i]:
                if st.button(option, key=f"opt_{current_q['id']}_{i}", type="secondary"):
                    # 응답 저장
                    st.session_state.answers[current_q["attribute"]] = option
                    st.session_state.questions_asked.append(current_q["id"])
                    st.session_state.current_question_idx = None

                    # Step 증가
                    st.session_state.current_step += 1

                    # Policy 모드 결정 (ε-greedy)
                    if phase1_agent and phase1_agent.epsilon > 0:
                        if random.random() < phase1_agent.epsilon:
                            st.session_state.policy_mode = "탐험"
                        else:
                            st.session_state.policy_mode = "활용"

                    # Reward 감소 (필수 4개 질문 이후 추가 질문만 -1)
                    is_required_question = current_q.get("attribute") in required_attributes
                    step_reward = 0.0
                    if not is_required_question:
                        st.session_state.reward -= 1.0
                        step_reward = -1.0

                    # RL Trajectory 기록
                    action = get_action_for_question(current_q.get("attribute", ""))
                    record_trajectory(action, step_reward)

                    # 사용자 응답 추가
                    st.session_state.chat_history.append({"role": "user", "content": option})

                    # 응답 확인 메시지
                    confirm_msg = f"'{option}'을(를) 선택하셨네요! 👍"
                    st.session_state.chat_history.append({"role": "assistant", "content": confirm_msg})

                    st.rerun()

        # "바로 추천" 스킵 버튼 (필수 4개 질문 완료 후 표시)
        if show_skip_btn:
            with cols[-1]:
                if st.button("✅ 바로 추천!", key="skip_btn", type="secondary"):
                    st.session_state.chat_history.append({"role": "user", "content": "바로 추천해주세요!"})
                    st.session_state.phase = "recommending"
                    st.rerun()

# 추천 단계
elif st.session_state.phase == "recommending":
    # 현재 Action 업데이트
    st.session_state.current_action = get_action_name("recommend")

    if st.session_state.recommended_vehicle is None:
        # 차량 추천 수행 (이미 추천한 차량 제외)
        recommended = get_vehicle_recommendation(
            st.session_state.answers,
            exclude_ids=st.session_state.recommended_history
        )

        if recommended is None:
            # 더 이상 추천할 차량이 없음
            no_more_msg = "죄송합니다. 조건에 맞는 다른 차량이 없습니다. 처음부터 다시 상담을 진행해 주세요."
            st.session_state.chat_history.append({"role": "assistant", "content": no_more_msg})
            st.session_state.phase = "complete"
            st.rerun()

        st.session_state.recommended_vehicle = recommended
        # 추천 히스토리에 추가
        st.session_state.recommended_history.append(recommended['id'])

        # 추천 메시지 생성
        rec_msg = f"""고객님의 응답을 분석한 결과, **{recommended['name']}**을(를) 추천드립니다! 🎉

📌 **{recommended['name']}**
- 차종: {recommended['category'].upper()}
- 연료: {recommended['fuel_type']}
- 좌석: {recommended['seats']}인승
- 가격: {recommended['price_range']['min']:,}만원 ~ {recommended['price_range']['max']:,}만원

이 차량으로 시승 예약을 진행하시겠습니까?"""

        st.session_state.chat_history.append({"role": "assistant", "content": rec_msg})
        st.rerun()

    # 예약 진행 버튼
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 시승 예약하기", use_container_width=True):
            st.session_state.current_step += 1
            st.session_state.policy_mode = "활용"
            st.session_state.chat_history.append({"role": "user", "content": "시승 예약할게요!"})
            st.session_state.phase = "scheduling"
            st.rerun()
    with col2:
        if st.button("🔄 다른 차량 보기", use_container_width=True):
            st.session_state.current_step += 1
            st.session_state.policy_mode = "탐험"
            st.session_state.chat_history.append({"role": "user", "content": "다른 차량도 보고 싶어요"})
            # Reward 감소 (다른 차량 요청 -5)
            st.session_state.reward -= 5.0
            # 현재 Action 업데이트 (다른 차량 추천 중)
            st.session_state.current_action = "다른 차량 탐색"
            # 다른 차량 추천 로직 (간단히 처리)
            st.session_state.recommended_vehicle = None
            st.rerun()

# 스케줄링 단계
elif st.session_state.phase == "scheduling":
    # 현재 Action 업데이트
    st.session_state.current_action = get_action_name("schedule")

    # 지역 기반 시승센터 매핑
    center_map = {
        "강남/서초": "강남 시승센터 (서울 강남구 테헤란로 152)",
        "송파/강동": "송파 시승센터 (서울 송파구 올림픽로 300)",
        "영등포/마포": "영등포 시승센터 (서울 영등포구 여의대로 108)",
        "성동/광진": "성수 시승센터 (서울 성동구 왕십리로 50)"
    }

    # 사용자가 선택한 지역에 해당하는 시승센터 추천
    selected_region = st.session_state.answers.get("region", "강남/서초")
    recommended_center = center_map.get(selected_region, center_map["강남/서초"])

    schedule_msg = f"""**{st.session_state.recommended_vehicle['name']}** 시승 예약을 진행합니다.

📍 **추천 시승센터**: {recommended_center}

원하시는 날짜와 시간을 선택해주세요."""

    if not any("시승 예약을 진행합니다" in c["content"] for c in st.session_state.chat_history):
        st.session_state.chat_history.append({"role": "assistant", "content": schedule_msg})
        st.rerun()

    # 날짜/시간 선택
    col1, col2 = st.columns(2)
    with col1:
        selected_date = st.date_input(
            "시승 날짜",
            min_value=datetime.now().date(),
            max_value=datetime.now().date() + timedelta(days=21)
        )
    with col2:
        selected_time = st.selectbox(
            "시승 시간",
            ["09:00", "10:00", "11:00", "13:00", "14:00", "15:00", "16:00", "17:00"]
        )

    if st.button("📅 예약 확정", use_container_width=True):
        # Step 증가 및 Policy 업데이트
        st.session_state.current_step += 1
        st.session_state.policy_mode = "활용"

        # Reward 증가 (예약 확정 +15)
        st.session_state.reward += 15.0

        # RL 모델 업데이트 (에피소드 종료 - 성공)
        update_rl_model(final_reward=15.0, terminated=True)

        # 예약 완료
        st.session_state.chat_history.append({
            "role": "user",
            "content": f"{selected_date.strftime('%Y년 %m월 %d일')} {selected_time}"
        })

        vehicle = st.session_state.recommended_vehicle
        fuel_type_kr = {"gasoline": "가솔린", "hybrid": "하이브리드", "electric": "전기"}.get(vehicle['fuel_type'], vehicle['fuel_type'])
        category_kr = {"sedan": "세단", "suv": "SUV", "mpv": "MPV"}.get(vehicle['category'], vehicle['category'])

        complete_msg = f"""🎉 **예약이 완료되었습니다!**

📌 **예약 정보**
- 차량: {vehicle['name']}
- 차종: {category_kr}
- 연료: {fuel_type_kr}
- 좌석: {vehicle['seats']}인승
- 가격대: {vehicle['price_range']['min']:,}~{vehicle['price_range']['max']:,}만원
- 장소: {recommended_center}
- 날짜: {selected_date.strftime('%Y년 %m월 %d일')}
- 시간: {selected_time}

예약 확인 문자가 발송될 예정입니다.
시승 당일 운전면허증을 지참해 주세요. 감사합니다! 🙏"""

        st.session_state.chat_history.append({"role": "assistant", "content": complete_msg})
        st.session_state.phase = "complete"
        st.rerun()

# 완료 단계
elif st.session_state.phase == "complete":
    # 현재 Action 업데이트
    st.session_state.current_action = get_action_name("complete")

    st.toast("🎉 시승 예약이 완료되었습니다!", icon="✅")

    if st.button("🔄 새로운 상담 시작", use_container_width=True):
        # 세션 초기화
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ============================================================================
# 사이드바 - 디버그 정보 (개발용)
# ============================================================================

with st.sidebar:
    st.markdown("### 🔧 디버그 정보")
    st.json({
        "phase": st.session_state.phase,
        "answers": st.session_state.answers,
        "questions_asked": st.session_state.questions_asked,
        "model_loaded": model_loaded,
        "trajectory_length": len(st.session_state.trajectory),
        "reward": st.session_state.reward,
        "episode_count": phase1_agent.episode_count if phase1_agent else 0,
        "q_table_size": len(phase1_agent.q_table) if phase1_agent else 0
    })
