"""
Streamlit Cloud 진입점
"""
import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 전역 예외 처리기 설정
def exception_handler(exc_type, exc_value, exc_tb):
    import traceback
    st.error(f"예외 발생: {exc_type.__name__}: {exc_value}")
    st.code("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))

sys.excepthook = exception_handler

# 메인 앱 실행
try:
    from src.app import main  # noqa: F401
except Exception as e:
    import traceback
    st.error(f"앱 로드 실패: {e}")
    st.code(traceback.format_exc())
