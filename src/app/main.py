"""
Brand 자동차 시승 예약 챗봇 - Streamlit 앱
Hyundai Test Drive Reservation Chatbot

학습된 강화학습 모델을 사용하여 대화형으로 차량을 추천하고
시승 예약을 진행하는 챗봇 인터페이스임.
"""

# =============================================================================
# 중요: sys.path 설정을 모든 로컬 import보다 먼저 수행
# Streamlit Cloud 배포 시 모듈 임포트 문제 방지
# =============================================================================
import sys
from pathlib import Path

def _setup_path():
    """프로젝트 루트를 sys.path에 추가 (모든 import 전에 실행)"""
    # 현재 파일 기준 프로젝트 루트 찾기: src/app/main.py -> 프로젝트 루트
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent

    # 프로젝트 루트에 src 폴더가 있는지 확인
    if not (project_root / 'src').is_dir():
        # 폴백: 현재 작업 디렉토리 또는 sys.path에서 찾기
        for candidate in [Path.cwd()] + [Path(p) for p in sys.path]:
            if (candidate / 'src').is_dir():
                project_root = candidate
                break

    # sys.path에 프로젝트 루트 추가
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    return project_root

# 다른 import 전에 path 설정 실행
project_root = _setup_path()

# =============================================================================
# 이제 표준 라이브러리 및 로컬 모듈 import 가능
# =============================================================================
import streamlit as st
import json
import random
from datetime import datetime, timedelta
import numpy as np

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

@st.cache_data
def load_questions():
    """질문 데이터 로드"""
    try:
        with open(project_root / "data" / "questions.json", "r", encoding="utf-8") as f:
            return json.load(f)["questions"]
    except Exception as e:
        st.error(f"질문 데이터 로드 실패: {e}")
        st.error(f"project_root: {project_root}")
        return []

@st.cache_data
def load_vehicles():
    """차량 데이터 로드"""
    try:
        with open(project_root / "data" / "vehicles.json", "r", encoding="utf-8") as f:
            return json.load(f)["vehicles"]
    except Exception as e:
        st.error(f"차량 데이터 로드 실패: {e}")
        return []

# ============================================================================
# 에이전트 로드
# ============================================================================

@st.cache_resource
def load_agents():
    """학습된 에이전트 로드 (Phase 1 + Phase 2)"""
    try:
        # === Phase 1: 차량 추천 에이전트 ===
        phase1_env = VehicleRecommendationEnv()
        phase1_agent = QLearningAgent(
            n_actions=phase1_env.action_space.n,
            seed=42
        )

        # Phase 1 모델 로드
        phase1_path = project_root / "checkpoints" / "chatbot" / "chatbot_q_learning.json"
        phase1_loaded = False
        if phase1_path.exists():
            phase1_agent.load(str(phase1_path))
            phase1_loaded = True

        # === Phase 2: 스케줄링 에이전트 ===
        phase2_env = SchedulingEnv()
        phase2_agent = DQNAgent(
            state_dim=phase2_env.observation_space.shape[0],
            action_dim=phase2_env.action_space.n,
            seed=42
        )

        # Phase 2 모델 로드
        phase2_path = project_root / "checkpoints" / "dqn_scheduling.pth"
        phase2_loaded = False
        if phase2_path.exists():
            phase2_agent.load(str(phase2_path))
            phase2_loaded = True

        return phase1_agent, phase1_env, phase2_agent, phase2_env, phase1_loaded, phase2_loaded

    except Exception as e:
        st.error(f"에이전트 로드 실패: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None, None, False, False

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

    # === Phase 2: 스케줄링 관련 상태 ===
    if "scheduling_step" not in st.session_state:
        st.session_state.scheduling_step = "select_day"  # select_day, select_time, dqn_recommend, confirm

    if "preferred_day_type" not in st.session_state:
        st.session_state.preferred_day_type = None  # 0: 이번주 평일, 1: 이번주 주말, 2: 다음주 평일, 3: 다음주 주말

    if "preferred_time_type" not in st.session_state:
        st.session_state.preferred_time_type = None  # 0: 오전, 1: 오후, 2: 저녁

    if "dqn_recommendation" not in st.session_state:
        st.session_state.dqn_recommendation = None  # DQN이 추천한 슬롯 정보

    if "scheduling_attempts" not in st.session_state:
        st.session_state.scheduling_attempts = 0  # 대안 제시 횟수

    if "selected_center" not in st.session_state:
        st.session_state.selected_center = None  # 선택된 시승센터

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
phase1_agent, phase1_env, phase2_agent, phase2_env, phase1_loaded, phase2_loaded = load_agents()

# 데이터 로드 확인
if len(questions) == 0:
    st.error(f"질문 데이터가 비어있음! project_root: {project_root}")
if len(vehicles) == 0:
    st.error(f"차량 데이터가 비어있음! project_root: {project_root}")

# 하위 호환성을 위한 변수
model_loaded = phase1_loaded

# 레이어드 카드 - 상태 표시
phase1_status = "✅" if phase1_loaded else "🔄"
phase2_status = "✅" if phase2_loaded else "🔄"
status_text = f"P1{phase1_status} P2{phase2_status}"
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

        # 현재 질문에 대한 옵션 버튼 표시 (id로 질문 찾기)
        current_q = next((q for q in questions if q["id"] == st.session_state.current_question_idx), None)
        if current_q is None:
            st.error(f"질문을 찾을 수 없음: id={st.session_state.current_question_idx}")
            st.stop()

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

# 스케줄링 단계 (Phase 2: DQN 기반)
elif st.session_state.phase == "scheduling":
    st.session_state.current_action = get_action_name("schedule")

    # 지역 기반 시승센터 매핑
    region_to_center = {
        "강남/서초": {"id": "gangnam", "name": "강남 시승센터", "address": "서울 강남구 테헤란로 152"},
        "송파/강동": {"id": "songpa", "name": "송파 시승센터", "address": "서울 송파구 올림픽로 300"},
        "영등포/마포": {"id": "yeongdeungpo", "name": "영등포 시승센터", "address": "서울 영등포구 국제금융로 10"},
        "성동/광진": {"id": "mapo", "name": "마포 시승센터", "address": "서울 마포구 월드컵북로 396"}
    }

    selected_region = st.session_state.answers.get("region", "강남/서초")
    center_info = region_to_center.get(selected_region, region_to_center["강남/서초"])

    # 시간 슬롯 매핑
    time_slots = ["09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00"]
    day_type_labels = ["이번주 평일", "이번주 주말", "다음주 평일", "다음주 주말"]
    time_type_labels = ["오전 (09:00~12:00)", "오후 (13:00~16:00)", "저녁 (16:00~18:00)"]

    # === Step 1: 요일 선택 ===
    if st.session_state.scheduling_step == "select_day":
        if not any("시승 예약을 진행" in c["content"] for c in st.session_state.chat_history):
            schedule_msg = f"""**{st.session_state.recommended_vehicle['name']}** 시승 예약을 진행함.

📍 **시승센터**: {center_info['name']} ({center_info['address']})

원하시는 **요일**을 선택해주세요."""
            st.session_state.chat_history.append({"role": "assistant", "content": schedule_msg})
            st.rerun()

        st.markdown("##### 원하시는 요일을 선택해주세요:")
        cols = st.columns(4)
        for i, label in enumerate(day_type_labels):
            with cols[i]:
                if st.button(label, key=f"day_{i}", use_container_width=True):
                    st.session_state.preferred_day_type = i
                    st.session_state.chat_history.append({"role": "user", "content": label})
                    st.session_state.scheduling_step = "select_time"
                    st.rerun()

    # === Step 2: 시간대 선택 ===
    elif st.session_state.scheduling_step == "select_time":
        if not any("시간대를 선택" in c["content"] for c in st.session_state.chat_history):
            time_msg = "원하시는 **시간대**를 선택해주세요."
            st.session_state.chat_history.append({"role": "assistant", "content": time_msg})
            st.rerun()

        st.markdown("##### 원하시는 시간대를 선택해주세요:")
        cols = st.columns(3)
        for i, label in enumerate(time_type_labels):
            with cols[i]:
                if st.button(label, key=f"time_{i}", use_container_width=True):
                    st.session_state.preferred_time_type = i
                    st.session_state.chat_history.append({"role": "user", "content": label})
                    st.session_state.scheduling_step = "dqn_recommend"
                    st.rerun()

    # === Step 3: DQN 분석 및 추천 ===
    elif st.session_state.scheduling_step == "dqn_recommend":
        # DQN 분석 수행
        if st.session_state.dqn_recommendation is None:
            day_type = st.session_state.preferred_day_type
            time_type = st.session_state.preferred_time_type

            # 날짜 계산 (day_type 기반)
            from datetime import datetime, timedelta
            today = datetime.now()
            if day_type == 0:  # 이번주 평일
                days_ahead = (7 - today.weekday()) % 7
                if days_ahead == 0 or days_ahead > 4:
                    days_ahead = 1
                target_date = today + timedelta(days=days_ahead)
            elif day_type == 1:  # 이번주 주말
                days_ahead = (5 - today.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                target_date = today + timedelta(days=days_ahead)
            elif day_type == 2:  # 다음주 평일
                days_ahead = (7 - today.weekday()) + 1
                target_date = today + timedelta(days=days_ahead)
            else:  # 다음주 주말
                days_ahead = (7 - today.weekday()) + 5
                target_date = today + timedelta(days=days_ahead)

            # 시간 계산 (time_type 기반)
            if time_type == 0:  # 오전
                slot_idx = 1  # 10:00
            elif time_type == 1:  # 오후
                slot_idx = 5  # 14:00
            else:  # 저녁
                slot_idx = 7  # 16:00

            recommended_time = time_slots[slot_idx]

            # DQN 에이전트로 최적 슬롯 분석 (실제 환경에서는 phase2_env 사용)
            dqn_action = 0  # 기본: 예약 확정
            if phase2_agent is not None:
                try:
                    # 환경 초기화 및 observation 생성
                    obs, _ = phase2_env.reset(options={
                        'vehicle_id': st.session_state.recommended_vehicle.get('id', 'avante'),
                        'prefill_ratio': 0.5
                    })
                    dqn_action = phase2_agent.select_action(obs, training=False)
                except Exception:
                    dqn_action = 0  # 에러 시 기본값

            # 추천 결과 저장
            st.session_state.dqn_recommendation = {
                "date": target_date,
                "time": recommended_time,
                "slot_idx": slot_idx,
                "day_type": day_type,
                "dqn_action": dqn_action,
                "center": center_info
            }

            # DQN 분석 결과 메시지
            day_name = ["월", "화", "수", "목", "금", "토", "일"][target_date.weekday()]
            if dqn_action == 0:
                analysis_msg = f"""🤖 **DQN 분석 완료**

선호하신 시간대를 분석한 결과, 다음 일정을 추천드림:

📅 **{target_date.strftime('%Y년 %m월 %d일')} ({day_name})** {recommended_time}
📍 {center_info['name']}

이 시간에 예약하시겠습니까?"""
            else:
                # 대안 제시
                alt_time = time_slots[min(slot_idx + 1, 8)]
                st.session_state.dqn_recommendation["alt_time"] = alt_time
                analysis_msg = f"""🤖 **DQN 분석 완료**

선호하신 시간대({recommended_time})는 예약이 많습니다.

**추천 대안**: {target_date.strftime('%Y년 %m월 %d일')} ({day_name}) **{alt_time}**
📍 {center_info['name']}

대안 시간으로 예약하시겠습니까?"""

            st.session_state.chat_history.append({"role": "assistant", "content": analysis_msg})
            st.rerun()

        # 예약 확정/거절 버튼
        rec = st.session_state.dqn_recommendation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 예약 확정", use_container_width=True):
                st.session_state.scheduling_step = "confirm"
                st.session_state.chat_history.append({"role": "user", "content": "예약 확정할게요!"})
                st.rerun()
        with col2:
            if st.button("🔄 다른 시간", use_container_width=True):
                st.session_state.scheduling_attempts += 1
                st.session_state.reward -= 2.0  # 대안 요청 패널티
                st.session_state.dqn_recommendation = None
                st.session_state.scheduling_step = "select_day"
                st.session_state.chat_history.append({"role": "user", "content": "다른 시간으로 할게요"})
                st.rerun()

    # === Step 4: 예약 확정 ===
    elif st.session_state.scheduling_step == "confirm":
        rec = st.session_state.dqn_recommendation
        vehicle = st.session_state.recommended_vehicle

        # Reward 증가
        st.session_state.reward += 15.0
        st.session_state.current_step += 1
        st.session_state.policy_mode = "활용"

        # RL 모델 업데이트
        update_rl_model(final_reward=15.0, terminated=True)

        fuel_type_kr = {"gasoline": "가솔린", "hybrid": "하이브리드", "electric": "전기"}.get(vehicle.get('fuel_type', ''), vehicle.get('fuel_type', ''))
        category_kr = {"sedan": "세단", "suv": "SUV", "mpv": "MPV"}.get(vehicle.get('category', ''), vehicle.get('category', ''))
        day_name = ["월", "화", "수", "목", "금", "토", "일"][rec["date"].weekday()]

        complete_msg = f"""🎉 **예약이 완료되었습니다!**

📌 **예약 정보**
- 차량: {vehicle['name']}
- 차종: {category_kr}
- 연료: {fuel_type_kr}
- 좌석: {vehicle.get('seats', 5)}인승
- 가격대: {vehicle['price_range']['min']:,}~{vehicle['price_range']['max']:,}만원
- 장소: {rec['center']['name']} ({rec['center']['address']})
- 날짜: {rec['date'].strftime('%Y년 %m월 %d일')} ({day_name})
- 시간: {rec.get('alt_time', rec['time'])}

예약 확인 문자가 발송될 예정임.
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
