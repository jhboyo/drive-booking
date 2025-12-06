"""
eScooterAI - 안전 가이드 챗봇 페이지
Safety Chatbot Page - Mobile First Design

RAG 기반 헬멧 안전 질의응답 시스템
"""

# Streamlit 사이드바 메뉴 이름
title = "💬 안전 챗봇"

import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# .env 파일 로드
load_dotenv(project_root / ".env")

from src.rag.vector_store import FAISSVectorStore
from src.rag.query_engine import RAGQueryEngine

# ============================================================================
# 페이지 설정
# ============================================================================

st.set_page_config(
    page_title="안전 챗봇 - eScooterAI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# 커스텀 CSS - 모바일 디자인
# ============================================================================

st.markdown("""
<style>
    /* Streamlit 최상단 헤더 영역 - 파란색 */
    [data-testid="stHeader"] {
        background: #3B82F6;
    }

    header[data-testid="stHeader"] {
        background: #3B82F6;
    }

    /* 헤더 하단 구분선 제거 */
    [data-testid="stHeader"]::after {
        display: none;
    }

    /* Toolbar 배경도 파란색 */
    [data-testid="stToolbar"] {
        background: #3B82F6;
    }

    /* Deploy 버튼 숨김 */
    [data-testid="stToolbar"] button[kind="header"],
    [data-testid="stToolbar"] > div > button,
    button[data-testid="baseButton-header"] {
        display: none !important;
    }

    /* 메인 컨테이너 배경 */
    .main {
        background: #F8FAFC;
    }

    .main > div {
        padding-top: 0rem;
    }

    /* 상단 여백 조정 */
    .block-container {
        padding-top: 1rem;
    }

    /* 헤더 - 파란색, 좌우 레이아웃 */
    .header-container {
        background: #3B82F6;
        padding: 2.5rem 1.5rem 4rem 1.5rem;
        margin: -1rem -1rem 0 -1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .header-left {
        flex: 1;
        color: white;
    }

    .header-title {
        color: white;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0;
        margin-bottom: 0.3rem;
    }

    .header-subtitle {
        color: rgba(255, 255, 255, 0.95);
        font-size: 0.85rem;
        margin: 0;
        font-weight: 400;
    }

    .header-icon {
        font-size: 3rem;
        flex-shrink: 0;
        margin-left: 0.5rem;
    }

    /* 레이어드 카드 - 헤더와 겹치기 */
    .layered-card {
        background: white;
        padding: 0.9rem 1rem;
        border-radius: 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
        margin: -3rem 1rem 1rem 1rem;
        text-align: center;
        position: relative;
        z-index: 10;
    }

    /* 채팅 메시지 */
    .stChatMessage {
        border-radius: 18px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* 버튼 스타일 - 배지 크기 */
    .stButton > button {
        width: 100%;
        height: 24px;
        font-size: 0.65rem;
        font-weight: 500;
        border-radius: 12px;
        border: none;
        background: #EFF6FF;
        color: #3B82F6;
        box-shadow: none;
        transition: all 0.15s;
        padding: 0 0.6rem;
        line-height: 24px;
    }

    .stButton > button:hover {
        background: #DBEAFE;
        transform: scale(1.05);
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2);
    }

    /* 성능 배지 - 작게 */
    .status-badge {
        display: inline-block;
        background: #EFF6FF;
        color: #3B82F6;
        padding: 0.4rem 0.8rem;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.2rem;
    }

    /* 질문 예시 카드 */
    .question-card {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
        text-align: center;
        transition: all 0.2s;
    }

    .question-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# RAG 엔진 초기화 (캐싱)
# ============================================================================

@st.cache_resource
def load_rag_engine():
    """RAG 엔진 로드 (한 번만 실행)"""
    try:
        # .env 파일 재로드 (캐시 함수 내에서도 확실히 로드)
        load_dotenv(project_root / ".env")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None, "OpenAI API 키가 설정되지 않았습니다."

        vector_db_dir = project_root / "vector_db"
        vector_store = FAISSVectorStore(api_key=api_key)
        vector_store.load(vector_db_dir)

        rag_engine = RAGQueryEngine(
            vector_store=vector_store,
            model="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=500,
            top_k=3,
            api_key=api_key
        )

        return rag_engine, None

    except Exception as e:
        return None, f"RAG 엔진 초기화 실패: {str(e)}"


# ============================================================================
# 세션 상태 초기화
# ============================================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "show_sources" not in st.session_state:
    st.session_state.show_sources = True

# ============================================================================
# 헤더
# ============================================================================

st.markdown("""
<div class="header-container">
    <div class="header-left">
        <div class="header-title">안전 가이드 챗봇</div>
        <div class="header-subtitle">헬멧 법규·착용법·사고 사례 질의응답</div>
    </div>
    <div class="header-icon">💬</div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# RAG 엔진 로드 확인
# ============================================================================

rag_engine, error_msg = load_rag_engine()

if error_msg:
    st.error(f"❌ {error_msg}")
    st.info("📝 .env 파일에 OPENAI_API_KEY를 설정하세요")
    st.stop()

# 레이어드 카드 - 헤더와 겹치는 스타일
st.markdown("""
<div class="layered-card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
        <h3 style="color: #1E293B; margin: 0; font-size: 0.9rem; font-weight: 600;">안전 가이드 챗봇</h3>
        <div>
            <span style="background: #DCFCE7; color: #16A34A; padding: 0.2rem 0.5rem; border-radius: 8px; font-size: 0.65rem; font-weight: 600; margin-right: 0.3rem;">● RAG</span>
            <span style="background: #DCFCE7; color: #16A34A; padding: 0.2rem 0.5rem; border-radius: 8px; font-size: 0.65rem; font-weight: 600;">🤖 AI</span>
        </div>
    </div>
    <p style="color: #64748B; font-size: 0.7rem; line-height: 1.2; margin: 0 0 0.4rem 0;">
        RAG 기반으로 헬멧 관련 법규와 안전 가이드를 제공합니다
    </p>
    <div>
        <span class="status-badge">📚 35개 문서</span>
        <span class="status-badge">🔍 FAISS</span>
        <span class="status-badge">⚡ GPT</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 빠른 질문 배지 - 레이어드 카드 내부 스타일
st.markdown("""
<div style="text-align: center; margin: -0.3rem 1rem 1rem 1rem; padding: 0.3rem 0;">
    <span style="color: #94A3B8; font-size: 0.65rem; font-weight: 500; margin-bottom: 0.3rem; display: block;">빠른 질문</span>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1], gap="small")
with col1:
    if st.button("💰 과태료", key="chip_fine"):
        st.session_state.current_question = "헬멧을 안 쓰면 과태료가 얼마인가요?"
        st.rerun()
with col2:
    if st.button("🎓 착용법", key="chip_how"):
        st.session_state.current_question = "헬멧을 올바르게 착용하는 방법을 알려주세요"
        st.rerun()
with col3:
    if st.button("📊 사고", key="chip_accident"):
        st.session_state.current_question = "전동킥보드 사고 통계를 알려주세요"
        st.rerun()

# ============================================================================
# 채팅 인터페이스
# ============================================================================

# 채팅 히스토리 표시
chat_container = st.container()

with chat_container:
    for chat in st.session_state.chat_history:
        # 사용자 질문
        with st.chat_message("user", avatar="👤"):
            st.markdown(chat["question"])

        # AI 답변
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(chat["answer"])

            # 출처 문서 표시
            if st.session_state.show_sources and "sources" in chat and chat["sources"]:
                with st.expander("📄 참고 문서", expanded=False):
                    for i, source in enumerate(chat["sources"], 1):
                        st.markdown(f"""
                        **{i}. {source.get('category', 'Unknown')}**
                        (유사도: {source.get('score', 0):.2f})

                        > {source.get('content', '')[:150]}...

                        *출처: {source.get('source', 'Unknown')}*
                        """)

# 질문 입력 폼
question = st.chat_input("💬 헬멧 관련 질문을 입력하세요...", key="question_input")

# 예시 버튼 클릭 처리
if "current_question" in st.session_state:
    question = st.session_state.current_question
    del st.session_state.current_question

# 질문 처리
if question:
    # 사용자 질문 표시
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    # AI 답변 생성
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤔 답변 생성 중..."):
            try:
                # RAG 쿼리 실행
                result = rag_engine.query(question)

                answer = result.get("answer", "답변을 생성할 수 없습니다.")
                sources = result.get("sources", [])

                # 답변 표시
                st.markdown(answer)

                # 출처 문서 표시
                if st.session_state.show_sources and sources:
                    with st.expander("📄 참고 문서", expanded=False):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"""
                            **{i}. {source.get('category', 'Unknown')}**
                            (유사도: {source.get('score', 0):.2f})

                            > {source.get('content', '')[:150]}...

                            *출처: {source.get('source', 'Unknown')}*
                            """)

                # 히스토리에 추가
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "sources": sources,
                    "timestamp": datetime.now().isoformat()
                })

                st.rerun()

            except Exception as e:
                st.error(f"❌ 오류 발생: {str(e)}")
