"""
현대자동차 시승 예약 챗봇 - Streamlit 앱
Hyundai Test Drive Reservation Chatbot

학습된 강화학습 모델을 사용하여 대화형으로 차량을 추천하고
시승 예약을 진행하는 챗봇 인터페이스임.
"""

import streamlit as st
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent.parent.parent
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
# 현대자동차 브랜드 컬러 CSS
# ============================================================================

st.markdown("""
<style>
    /* 현대자동차 브랜드 컬러 */
    :root {
        --hyundai-blue: #002C5F;
        --active-blue: #00AAD2;
        --light-blue: #E8F4F8;
        --dark-gray: #333333;
        --light-gray: #F4F4F4;
    }

    /* Streamlit 헤더 - 현대 블루 */
    [data-testid="stHeader"] {
        background: #002C5F;
    }

    header[data-testid="stHeader"] {
        background: #002C5F;
    }

    [data-testid="stHeader"]::after {
        display: none;
    }

    [data-testid="stToolbar"] {
        background: #002C5F;
    }

    /* Deploy 버튼 숨김 */
    [data-testid="stToolbar"] button[kind="header"],
    [data-testid="stToolbar"] > div > button,
    button[data-testid="baseButton-header"] {
        display: none !important;
    }

    /* 메인 배경 */
    .main {
        background: #F4F4F4;
    }

    .main > div {
        padding-top: 0rem;
    }

    .block-container {
        padding-top: 1rem;
    }

    /* 헤더 컨테이너 - 단색 블루 (전체 너비) */
    .header-container {
        background: #002C5F;
        padding: 2.5rem 1.5rem 4rem 1.5rem;
        margin: -1rem calc(-50vw + 50%) 0 calc(-50vw + 50%);
        width: 100vw;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .header-inner {
        max-width: 1200px;
        width: 100%;
        padding: 0 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .header-icon {
        font-size: 2.5rem;
    }

    .header-left {
        flex: 1;
        color: white;
    }

    .header-title {
        color: white;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
        margin-bottom: 0.3rem;
    }

    .header-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.9rem;
        margin: 0;
        font-weight: 400;
    }

    .header-icon {
        font-size: 3rem;
        flex-shrink: 0;
        margin-left: 0.5rem;
    }

    /* 레이어드 카드 */
    .layered-card {
        background: white;
        padding: 1rem 1.2rem;
        border-radius: 18px;
        box-shadow: 0 4px 20px rgba(0, 44, 95, 0.15);
        margin: -3rem 1rem 1rem 1rem;
        text-align: center;
        position: relative;
        z-index: 10;
    }

    /* 상태 배지 */
    .status-badge {
        display: inline-block;
        background: #E8F4F8;
        color: #002C5F;
        padding: 0.4rem 0.8rem;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.2rem;
    }

    /* 채팅 메시지 스타일 */
    .stChatMessage {
        border-radius: 18px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* 버튼 스타일 - 현대 블루 */
    .stButton > button {
        background: #002C5F;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background: #00AAD2;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 170, 210, 0.3);
    }

    /* 옵션 버튼 그리드 */
    .option-button {
        background: white;
        border: 2px solid #E8F4F8;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s;
        margin: 0.3rem;
    }

    .option-button:hover {
        border-color: #00AAD2;
        background: #E8F4F8;
    }

    /* 차량 카드 */
    .vehicle-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 44, 95, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #002C5F;
    }

    .vehicle-name {
        color: #002C5F;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .vehicle-info {
        color: #666;
        font-size: 0.9rem;
        margin: 0.3rem 0;
    }

    .vehicle-price {
        color: #00AAD2;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    /* 진행 상태 바 */
    .progress-container {
        background: #E8F4F8;
        border-radius: 10px;
        padding: 0.3rem;
        margin: 1rem 0;
    }

    .progress-bar {
        background: linear-gradient(90deg, #002C5F 0%, #00AAD2 100%);
        height: 8px;
        border-radius: 8px;
        transition: width 0.3s ease;
    }

    /* 빠른 선택 칩 */
    .quick-chip {
        display: inline-block;
        background: #E8F4F8;
        color: #002C5F;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.2rem;
        cursor: pointer;
        transition: all 0.2s;
    }

    .quick-chip:hover {
        background: #00AAD2;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 데이터 로드
# ============================================================================

@st.cache_data
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
        checkpoint_path = project_root / "checkpoints" / "integrated" / "phase1_q_learning.json"
        if checkpoint_path.exists():
            phase1_agent.load(str(checkpoint_path))
            model_loaded = True
        else:
            # 체크포인트 없으면 간단히 학습
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

init_session_state()

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

st.markdown(f"""
<div class="layered-card" style="padding: 1.2rem;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
        <h3 style="color: #002C5F; margin: 0; font-size: 1.1rem; font-weight: 600;">👋 안녕하세요!</h3>
        <div>
            <span class="status-badge">{status_text}</span>
            <span class="status-badge">{phase_text}</span>
        </div>
    </div>
    <div style="color: #555; font-size: 0.9rem; line-height: 1.8;">
        <p style="margin: 0 0 0.6rem 0;">Brand 차 시승 예약 도우미입니다.</p>
        <p style="margin: 0 0 0.6rem 0;">몇 가지 질문을 통해 고객님께 딱 맞는 차량을 추천해 드리겠습니다.</p>
        <p style="margin: 0; color: #002C5F; font-weight: 500;">준비되셨으면 아래 <strong>'시작'</strong> 버튼을 눌러주세요!</p>
    </div>
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
            with st.chat_message("assistant", avatar="🚗"):
                st.markdown(chat["content"])
        else:
            with st.chat_message("user", avatar="👤"):
                st.markdown(chat["content"])

# ============================================================================
# 대화 흐름 관리
# ============================================================================

# 인사 단계
if st.session_state.phase == "greeting":
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 시작하기", use_container_width=True):
            st.session_state.phase = "questioning"
            st.session_state.chat_history.append({"role": "user", "content": "시작할게요!"})
            st.rerun()

# 질문 단계
elif st.session_state.phase == "questioning":
    # 아직 질문할 게 있는지 확인
    remaining_questions = [q for q in questions if q["id"] not in st.session_state.questions_asked]

    if len(st.session_state.questions_asked) >= 3 or len(remaining_questions) == 0:
        # 충분한 정보 수집 → 추천 단계로
        st.session_state.phase = "recommending"
        st.rerun()
    else:
        # 다음 질문 선택 (에이전트 사용 또는 순차)
        if st.session_state.current_question_idx is None:
            # 아직 안 한 질문 중 첫 번째 선택
            next_q = remaining_questions[0]
            st.session_state.current_question_idx = next_q["id"]

            # 질문 메시지 추가
            q_msg = f"**{next_q['text']}**"
            st.session_state.chat_history.append({"role": "assistant", "content": q_msg})
            st.rerun()

        # 현재 질문에 대한 옵션 버튼 표시
        current_q = questions[st.session_state.current_question_idx]

        st.markdown("##### 답변을 선택해주세요:")
        cols = st.columns(len(current_q["options"]))

        for i, option in enumerate(current_q["options"]):
            with cols[i]:
                if st.button(option, key=f"opt_{current_q['id']}_{i}", use_container_width=True):
                    # 응답 저장
                    st.session_state.answers[current_q["attribute"]] = option
                    st.session_state.questions_asked.append(current_q["id"])
                    st.session_state.current_question_idx = None

                    # 사용자 응답 추가
                    st.session_state.chat_history.append({"role": "user", "content": option})

                    # 응답 확인 메시지
                    confirm_msg = f"'{option}'을(를) 선택하셨네요! 👍"
                    st.session_state.chat_history.append({"role": "assistant", "content": confirm_msg})

                    st.rerun()

# 추천 단계
elif st.session_state.phase == "recommending":
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
            st.session_state.chat_history.append({"role": "user", "content": "시승 예약할게요!"})
            st.session_state.phase = "scheduling"
            st.rerun()
    with col2:
        if st.button("🔄 다른 차량 보기", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": "다른 차량도 보고 싶어요"})
            # 다른 차량 추천 로직 (간단히 처리)
            st.session_state.recommended_vehicle = None
            st.rerun()

# 스케줄링 단계
elif st.session_state.phase == "scheduling":
    schedule_msg = f"""**{st.session_state.recommended_vehicle['name']}** 시승 예약을 진행합니다.

원하시는 날짜와 시간을 선택해주세요."""

    if not any("시승 예약을 진행합니다" in c["content"] for c in st.session_state.chat_history):
        st.session_state.chat_history.append({"role": "assistant", "content": schedule_msg})
        st.rerun()

    # 날짜 선택
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
        # 예약 완료
        st.session_state.chat_history.append({
            "role": "user",
            "content": f"{selected_date.strftime('%Y년 %m월 %d일')} {selected_time}"
        })

        complete_msg = f"""🎉 **예약이 완료되었습니다!**

📌 **예약 정보**
- 차량: {st.session_state.recommended_vehicle['name']}
- 날짜: {selected_date.strftime('%Y년 %m월 %d일')}
- 시간: {selected_time}

예약 확인 문자가 발송될 예정입니다.
시승 당일 운전면허증을 지참해 주세요. 감사합니다! 🙏"""

        st.session_state.chat_history.append({"role": "assistant", "content": complete_msg})
        st.session_state.phase = "complete"
        st.rerun()

# 완료 단계
elif st.session_state.phase == "complete":
    st.balloons()

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
        "model_loaded": model_loaded
    })
