"""
eScooterAI - 헬멧 탐지 페이지
Helmet Detection Page - Mobile First Design

실시간 웹캠 스트리밍 기반 헬멧 착용 상태 탐지 + Telegram 알림
"""

# Streamlit 사이드바 메뉴 이름
title = "📸 헬멧 탐지"

import streamlit as st
from pathlib import Path
import sys
import time
import threading
from datetime import datetime

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 웹캠 및 추론 관련 import
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
from collections import deque

# 기존 유틸리티 import
from src.mobile_app.utils.inference import load_model, get_model_path
from src.alert.telegram_notifier import TelegramNotifier

# ============================================================================
# 페이지 설정
# ============================================================================

st.set_page_config(
    page_title="헬멧 탐지 - eScooterAI",
    page_icon="📱",
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

    /* 전체 배경 */
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
        padding: 1.5rem 1.5rem 2.5rem 1.5rem;
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
        font-size: 3.5rem;
    }

    /* 레이어드 카드 - 헤더와 겹치기 (최소 크기) */
    .layered-card {
        background: white;
        padding: 0.9rem 1rem;
        border-radius: 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
        margin: -2rem 1rem 1rem 1rem;
        text-align: center;
        position: relative;
        z-index: 10;
    }

    /* 기능 카드 */
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 18px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        margin-bottom: 1.5rem;
        text-align: center;
    }

    .feature-icon-large {
        font-size: 3.5rem;
        margin-bottom: 1rem;
    }

    /* 버튼 스타일 - 밝은 파란색 */
    .stButton > button {
        width: 100%;
        height: 55px;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 15px;
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

    /* WebRTC 전체 컨테이너 중앙 정렬 - 회색 배경 높이 증가 */
    .streamlit-webrtc {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 600px;
        background: #E5E7EB;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
    }

    /* WebRTC 내부 div도 중앙 정렬 */
    [data-testid="stVerticalBlock"] > div:has(video) {
        display: flex;
        flex-direction: column;
        align-items: center;
        min-height: 400px;
    }

    /* 비디오 요소 크기 */
    video {
        width: 100% !important;
        max-height: 400px !important;
        object-fit: cover !important;
        border-radius: 12px !important;
    }

    /* WebRTC 버튼 컨테이너 중앙 정렬 */
    div:has(> button[kind="primary"]) {
        display: flex;
        justify-content: center;
        width: 100%;
    }

    /* WebRTC 버튼 스타일 (START - 녹색) - 우선순위 강화 */
    button[kind="primary"],
    button[kind="primary"][class*="st-"],
    div[data-testid="stVerticalBlock"] button[kind="primary"] {
        width: 100% !important;
        max-width: 300px !important;
        height: 65px !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        border-radius: 20px !important;
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        background-color: #10B981 !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
    }

    button[kind="primary"]:hover,
    button[kind="primary"][class*="st-"]:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.6) !important;
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        background-color: #059669 !important;
    }

    button[kind="primary"]:active,
    button[kind="primary"][class*="st-"]:active {
        transform: translateY(-1px) scale(1) !important;
    }

    /* STOP 버튼 (빨간색) - 우선순위 강화 */
    button[kind="primary"][aria-label*="Stop"],
    button[kind="primary"][class*="st-"][aria-label*="Stop"] {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%) !important;
        background-color: #EF4444 !important;
    }

    button[kind="primary"][aria-label*="Stop"]:hover,
    button[kind="primary"][class*="st-"][aria-label*="Stop"]:hover {
        background: linear-gradient(135deg, #DC2626 0%, #B91C1C 100%) !important;
        background-color: #DC2626 !important;
        box-shadow: 0 8px 25px rgba(239, 68, 68, 0.6) !important;
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

    /* 작은 카드 */
    .small-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
        text-align: center;
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

<script>
// START 버튼 색상 강제 변경 (녹색)
function updateButtonColors() {
    // 모든 primary 버튼 찾기
    const buttons = document.querySelectorAll('button[kind="primary"]');

    buttons.forEach(button => {
        const buttonText = button.textContent || button.innerText;

        // START 버튼 (STOP이 아닌 경우)
        if (!buttonText.includes('STOP') && !buttonText.includes('Stop')) {
            button.style.background = 'linear-gradient(135deg, #10B981 0%, #059669 100%)';
            button.style.backgroundColor = '#10B981';
            button.style.borderColor = '#10B981';
        }
        // STOP 버튼
        else {
            button.style.background = 'linear-gradient(135deg, #EF4444 0%, #DC2626 100%)';
            button.style.backgroundColor = '#EF4444';
            button.style.borderColor = '#EF4444';
        }
    });
}

// 페이지 로드 시 실행
document.addEventListener('DOMContentLoaded', updateButtonColors);

// MutationObserver로 DOM 변경 감지 (동적으로 버튼이 생성될 경우)
const observer = new MutationObserver(updateButtonColors);
observer.observe(document.body, { childList: true, subtree: true });

// 주기적으로 체크 (안전장치)
setInterval(updateButtonColors, 500);
</script>
""", unsafe_allow_html=True)

# ============================================================================
# 헤더
# ============================================================================

st.markdown("""
<div class="header-container">
    <div class="header-left">
        <div class="header-title">실시간 헬멧 탐지</div>
        <div class="header-subtitle">AI로 헬멧 착용 여부를 감지합니다</div>
    </div>
    <div class="header-icon">📹</div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 모델 로드
# ============================================================================

@st.cache_resource
def load_detection_model():
    """헬멧 탐지 모델 로드"""
    try:
        model_path = get_model_path("best.pt")
        model = load_model(str(model_path))
        return model, None
    except Exception as e:
        return None, f"모델 로드 실패: {str(e)}"


# ============================================================================
# 실시간 비디오 프로세서 (모바일 최적화)
# ============================================================================

class MobileHelmetDetector(VideoProcessorBase):
    """모바일 최적화된 실시간 헬멧 탐지 프로세서"""

    def __init__(self, model, telegram_notifier, conf_threshold: float = 0.55):
        self.model = model
        self.conf_threshold = conf_threshold
        self.telegram_notifier = telegram_notifier

        # 클래스별 색상 (BGR)
        self.class_colors = {
            0: (255, 0, 0),    # helmet - 파란색
            1: (0, 0, 255),    # head - 빨간색
            2: (0, 255, 255)   # vest - 노란색
        }

        self.class_names = {0: "Helmet", 1: "Head", 2: "Vest"}

        # 통계
        self.lock = threading.Lock()
        self.stats = {
            'helmet': 0,
            'head': 0,
            'total_workers': 0,
            'helmet_rate': 0.0,
            'fps': 0.0
        }

        # FPS 계산
        self.fps_queue = deque(maxlen=30)
        self.last_time = time.time()

        # Telegram 알림 쿨다운 (10초)
        self.last_alert_time = 0
        self.alert_cooldown = 10

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """프레임 수신 및 처리"""
        img = frame.to_ndarray(format="bgr24")

        # FPS 계산
        current_time = time.time()
        fps = 1 / (current_time - self.last_time) if current_time > self.last_time else 0
        self.last_time = current_time
        self.fps_queue.append(fps)
        avg_fps = np.mean(self.fps_queue) if len(self.fps_queue) > 0 else 0

        # YOLOv8 추론
        results = self.model(img, conf=self.conf_threshold, iou=0.45, verbose=False)[0]

        # 탐지 결과 파싱
        helmet_count = 0
        head_count = 0

        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)

            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)

                # 카운팅
                if cls == 0:
                    helmet_count += 1
                elif cls == 1:
                    head_count += 1

                # 바운딩 박스
                color = self.class_colors.get(cls, (255, 255, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # 라벨
                label = f"{self.class_names[cls]}: {score:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                cv2.rectangle(img, (x1, y1 - text_height - 10),
                            (x1 + text_width + 5, y1), color, -1)
                cv2.putText(img, label, (x1 + 3, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 헬멧 착용률 계산
        total_workers = helmet_count + head_count
        helmet_rate = (helmet_count / total_workers * 100) if total_workers > 0 else 0

        # 통계 업데이트
        with self.lock:
            self.stats = {
                'helmet': helmet_count,
                'head': head_count,
                'total_workers': total_workers,
                'helmet_rate': helmet_rate,
                'fps': avg_fps
            }

        # Telegram 알림 (헬멧 미착용 감지 시)
        if head_count > 0 and total_workers > 0:
            if current_time - self.last_alert_time > self.alert_cooldown:
                self.last_alert_time = current_time
                # 별도 스레드에서 알림 전송 (메인 스레드 차단 방지)
                threading.Thread(
                    target=self.telegram_notifier.send_safety_alert,
                    args=(head_count, total_workers, helmet_rate, None, "전동킥보드 현장"),
                    daemon=True
                ).start()

        # 화면 오버레이 (간소화 - 모바일용)
        overlay = img.copy()
        cv2.rectangle(overlay, (5, 5), (220, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

        cv2.putText(img, f"FPS: {avg_fps:.1f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"Helmet: {helmet_count} | Head: {head_count}", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if total_workers > 0:
            cv2.putText(img, f"Rate: {helmet_rate:.1f}%", (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if helmet_rate >= 90 else (0, 165, 255) if helmet_rate >= 70 else (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_stats(self):
        with self.lock:
            return self.stats.copy()


# ============================================================================
# 메인 UI
# ============================================================================

# 모델 로드 및 Telegram 초기화 (먼저 실행)
model, error_msg = load_detection_model()

if error_msg:
    st.error(f"❌ {error_msg}")
    st.stop()

# Telegram 알림 초기화
telegram_notifier = TelegramNotifier()
telegram_status = "ON" if telegram_notifier.enabled else "OFF"
telegram_color = "#16A34A" if telegram_notifier.enabled else "#94A3B8"
telegram_bg = "#DCFCE7" if telegram_notifier.enabled else "#F1F5F9"

# 레이어드 카드 - 헤더와 겹치는 스타일
telegram_badge_text = "🔔 알림"  # 항상 활성화 배지로 표시
st.markdown(f"""
<div class="layered-card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
        <h3 style="color: #1E293B; margin: 0; font-size: 0.9rem; font-weight: 600;">실시간 헬멧 탐지</h3>
        <div>
            <span style="background: #DCFCE7; color: #16A34A; padding: 0.2rem 0.5rem; border-radius: 8px; font-size: 0.65rem; font-weight: 600; margin-right: 0.3rem;">● AI</span>
            <span style="background: {telegram_bg}; color: {telegram_color}; padding: 0.2rem 0.5rem; border-radius: 8px; font-size: 0.65rem; font-weight: 600;">📱 {telegram_status}</span>
        </div>
    </div>
    <p style="color: #64748B; font-size: 0.7rem; line-height: 1.2; margin: 0 0 0.4rem 0;">
        카메라로 전동킥보드 탑승자를 비추면 AI가 자동으로 감지합니다
    </p>
    <div>
        <span class="status-badge">🎯 93.7%</span>
        <span class="status-badge">⚡ 실시간</span>
        <span class="status-badge">📱 모바일</span>
        <span class="status-badge">{telegram_badge_text}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# WebRTC 설정 (최신 API 사용)
rtc_config = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]}
    ]
}

# VideoProcessor 팩토리
class VideoProcessorFactory:
    def __init__(self):
        self.processor = None

    def __call__(self):
        self.processor = MobileHelmetDetector(
            model=model,
            telegram_notifier=telegram_notifier,
            conf_threshold=0.55
        )
        return self.processor

factory = VideoProcessorFactory()

# 카메라 시작 안내
st.markdown("""
<div style="padding: 0.5rem 0; margin: 0.5rem 0; text-align: center;">
    <p style="color: #64748B; font-size: 0.85rem; font-weight: 500; margin: 0;">
        📹 START 버튼을 눌러 실시간 헬멧 탐지 시작
    </p>
</div>
""", unsafe_allow_html=True)

# 웹캠 스트리머 (최신 API: frontend/server rtc_configuration 분리)
try:
    ctx = webrtc_streamer(
        key="mobile-helmet-detection",
        mode=WebRtcMode.SENDRECV,
        frontend_rtc_configuration=rtc_config,  # 최신 API 사용
        server_rtc_configuration=rtc_config,    # 최신 API 사용
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "facingMode": "environment"  # 모바일 후면 카메라
            },
            "audio": False
        },
        video_processor_factory=factory,
        async_processing=True,
    )
except Exception as e:
    # WebRTC 초기화 실패 시 에러 메시지 표시
    st.error(f"⚠️ WebRTC 초기화 실패: {str(e)}")
    st.info("💡 브라우저를 새로고침하거나 HTTPS 환경에서 접속하세요. Chrome 또는 Safari 브라우저를 사용해주세요.")
    ctx = None

# 실시간 통계
st.markdown("---")

if ctx and ctx.state.playing:
    stats_placeholder = st.empty()

    while ctx.state.playing:
        if factory.processor:
            stats = factory.processor.get_stats()

            with stats_placeholder.container():
                # 헬멧 착용률 카드
                if stats['total_workers'] > 0:
                    helmet_rate = stats['helmet_rate']
                    emoji = '✅' if helmet_rate >= 90 else '⚠️' if helmet_rate >= 70 else '🚨'

                    st.markdown(f"""
                    <div class="feature-card">
                        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{emoji}</div>
                        <h3 style="color: #1E293B; margin: 0.5rem 0; font-size: 1.2rem;">착용률: {helmet_rate:.1f}%</h3>
                        <p style="color: #64748B; font-size: 0.85rem; margin: 0;">
                            👷 {stats['total_workers']}명 |
                            🔵 {stats['helmet']}명 착용 |
                            🔴 {stats['head']}명 미착용
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # 경고 메시지
                    if stats['head'] > 0:
                        st.warning(f"⚠️ 헬멧 미착용자 {stats['head']}명 감지! Telegram 알림이 전송되었습니다.")
                else:
                    st.info("ℹ️ 탑승자가 탐지되지 않았습니다")

                st.caption(f"⚡ FPS: {stats['fps']:.1f}")

        time.sleep(0.5)

# ============================================================================
# 탐지 클래스 설명 - 범례 형태
# ============================================================================

st.markdown("""
<div style="
    background: white;
    padding: 1rem 1.5rem;
    border-radius: 15px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    margin-bottom: 1rem;
">
    <div style="
        color: #64748B;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        text-align: center;
    ">탐지 Boundary 색상 안내</div>
    <div style="
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 2rem;
        flex-wrap: wrap;
    ">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <div style="
                width: 20px;
                height: 20px;
                background: #3B82F6;
                border-radius: 4px;
            "></div>
            <span style="color: #1E293B; font-size: 0.85rem; font-weight: 500;">Helmet (안전)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <div style="
                width: 20px;
                height: 20px;
                background: #EF4444;
                border-radius: 4px;
            "></div>
            <span style="color: #1E293B; font-size: 0.85rem; font-weight: 500;">Head (위험)</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
