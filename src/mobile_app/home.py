"""
eScooterAI - 전동킥보드 헬멧 안전 통합 플랫폼
Home Page - Mobile First Design
"""

import streamlit as st

# ============================================================================
# 페이지 설정
# ============================================================================

st.set_page_config(
    page_title="eScooterAI",
    page_icon="🛴",
    layout="wide",
    initial_sidebar_state="collapsed",  # 모바일에서 사이드바 숨김
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

    /* 헤더 - 파란색, 크게 */
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

    .header-greeting {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white;
    }

    .header-subtitle {
        color: rgba(255, 255, 255, 0.95);
        font-size: 0.9rem;
        font-weight: 400;
        color: white;
    }

    .header-icon {
        font-size: 5rem;
    }

    /* 레이어드 환영 카드 - 헤더와 겹치기 */
    .layered-card {
        background: white;
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
        margin: -3rem 1rem 1.5rem 1rem;
        text-align: center;
        position: relative;
        z-index: 10;
    }

    .welcome-text {
        color: #1E293B;
        font-size: 0.95rem;
        font-weight: 500;
        margin-bottom: 0.8rem;
        line-height: 1.5;
    }

    /* Streamlit columns 모바일에서도 2열 유지 - 우선순위 강화 */
    div.row-widget.stHorizontalBlock,
    .row-widget.stHorizontalBlock,
    [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        gap: 1rem !important;
        flex-wrap: nowrap !important;
    }

    div[data-testid="column"],
    [data-testid="column"],
    .stHorizontalBlock [data-testid="column"] {
        width: calc(50% - 0.5rem) !important;
        flex: 1 1 calc(50% - 0.5rem) !important;
        min-width: calc(50% - 0.5rem) !important;
        max-width: calc(50% - 0.5rem) !important;
    }

    .stHorizontalBlock > div,
    div.stHorizontalBlock > div {
        flex: 1 !important;
        min-width: 0 !important;
    }

    /* 모바일에서도 2열 강제 유지 */
    @media (max-width: 768px) {
        div[data-testid="column"],
        [data-testid="column"] {
            width: calc(50% - 0.5rem) !important;
            flex: 1 1 calc(50% - 0.5rem) !important;
            max-width: calc(50% - 0.5rem) !important;
        }

        div.row-widget.stHorizontalBlock,
        .row-widget.stHorizontalBlock {
            flex-direction: row !important;
            flex-wrap: nowrap !important;
        }
    }

    @media (max-width: 640px) {
        div[data-testid="column"],
        [data-testid="column"] {
            width: calc(50% - 0.5rem) !important;
            flex: 1 1 calc(50% - 0.5rem) !important;
            max-width: calc(50% - 0.5rem) !important;
        }
    }

    /* 기능 카드 - 2열 그리드 */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 18px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
        transition: all 0.2s;
        text-align: center;
        height: 100%;
    }

    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }

    .feature-icon {
        font-size: 2.8rem;
        margin-bottom: 0.8rem;
    }

    .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        color: #1E293B;
    }

    /* 버튼 스타일 - 작게 */
    .stButton > button {
        width: 100%;
        height: 45px;
        font-size: 0.9rem;
        font-weight: 600;
        border-radius: 12px;
        border: none;
        background: #3B82F6;
        color: white;
        box-shadow: 0 2px 6px rgba(59, 130, 246, 0.3);
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background: #2563EB;
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(59, 130, 246, 0.4);
    }

    /* 성능 배지 - 작게 */
    .performance-badge {
        display: inline-block;
        background: #EFF6FF;
        color: #3B82F6;
        padding: 0.4rem 0.8rem;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.2rem;
    }

    /* 푸터 */
    .footer {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        margin-top: 1rem;
        text-align: center;
    }

    .footer-title {
        color: #1E293B;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .footer-desc {
        color: #64748B;
        font-size: 0.75rem;
        line-height: 1.5;
        margin-bottom: 0.8rem;
    }

    .footer-tech {
        color: #94A3B8;
        font-size: 0.7rem;
        margin-bottom: 0.8rem;
    }

    .footer-copyright {
        color: #94A3B8;
        font-size: 0.7rem;
        padding-top: 0.8rem;
        border-top: 1px solid #E2E8F0;
    }

    /* 가이드 카드 - 컴팩트 */
    .guide-card {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        margin-bottom: 0.8rem;
    }

    .guide-title {
        color: #1E293B;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-align: center;
    }

    .guide-step {
        color: #1E293B;
        font-size: 0.8rem;
        margin: 0;
        text-align: center;
        line-height: 1.4;
    }

    /* 안전 통계 카드 - 컴팩트 */
    .stat-card {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        margin-bottom: 0.8rem;
    }

    .stat-title {
        color: #1E293B;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-align: center;
    }

    .stat-item {
        color: #3B82F6;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.3rem 0;
        text-align: center;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 헤더 - 좌우 레이아웃
# ============================================================================

st.markdown("""
<div class="header-container">
    <div class="header-left">
        <div class="header-greeting">안전한 라이딩, 함께해요! 👋</div>
        <div class="header-subtitle">스마트 헬멧 안전 케어 서비스</div>
    </div>
    <div class="header-icon">🛴</div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 레이어드 카드 (흰색 배경)
# ============================================================================

st.markdown("""
<div class="layered-card">
    <div class="welcome-text">전동킥보드 헬멧 착용을 AI로 자동 감지하고 안전 가이드를 제공합니다</div>
    <div>
        <span class="performance-badge">🎯 93.7%</span>
        <span class="performance-badge">⚡ 실시간</span>
        <span class="performance-badge">🤖 RAG</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 주요 기능 - 2열 그리드
# ============================================================================

col1, col2 = st.columns(2)

# 헬멧 탐지
with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📸</div>
        <div class="feature-title">헬멧 탐지</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("시작하기", key="btn_detection", use_container_width=True):
        st.switch_page("pages/helmet_detection.py")

# 안전 챗봇
with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">💬</div>
        <div class="feature-title">안전 챗봇</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("질문하기", key="btn_chatbot", use_container_width=True):
        st.switch_page("pages/safety_chatbot.py")

# ============================================================================
# 사용 가이드
# ============================================================================

st.markdown("""
<div class="guide-card">
    <div class="guide-title">💡 3단계로 시작하기</div>
    <div class="guide-step">1️⃣ 촬영 → 2️⃣ AI 분석 → 3️⃣ 안전 확인</div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 안전 통계 & 법적 주의사항 (2열)
# ============================================================================

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-title">🛡️ 알고 계셨나요?</div>
        <div class="stat-item">헬멧 착용 시 머리 부상 85% ↓</div>
        <div class="stat-item">킥보드 사고의 60%가 머리 부상</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        margin-bottom: 0.8rem;
    ">
        <div style="
            color: #991B1B;
            font-size: 0.95rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            text-align: center;
        ">⚠️ 법적 의무</div>
        <div style="
            color: #DC2626;
            font-size: 0.8rem;
            font-weight: 600;
            margin: 0.3rem 0;
            text-align: center;
            line-height: 1.4;
        ">안전모 착용 필수</div>
        <div style="
            color: #DC2626;
            font-size: 0.8rem;
            font-weight: 600;
            margin: 0.3rem 0;
            text-align: center;
            line-height: 1.4;
        ">미착용 시 과태료 2만원</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# Footer
# ============================================================================

st.markdown("""
<div class="footer">
    <div class="footer-title">🛴 eScooterAI</div>
    <div class="footer-desc">
        딥러닝 객체 탐지와 RAG 기반 NLP를 융합한<br/>
        스마트 헬멧 안전 케어 서비스
    </div>
    <div class="footer-tech">
        YOLOv8n • FAISS • GPT-3.5 • Streamlit
    </div>
    <div class="footer-copyright">
        © 2025 eScooterAI Team. All rights reserved.
    </div>
</div>
""", unsafe_allow_html=True)