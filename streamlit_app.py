"""
Streamlit Cloud 진입점
"""
import streamlit as st
import sys
from pathlib import Path

st.set_page_config(page_title="Debug", layout="wide")

st.write(f"**Python version:** {sys.version}")

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

st.write(f"**Project root:** {project_root}")

# PyTorch import 테스트
try:
    import torch
    st.success(f"PyTorch {torch.__version__} 로드 성공")
except Exception as e:
    st.error(f"PyTorch 로드 실패: {e}")

# Gymnasium import 테스트
try:
    import gymnasium
    st.success(f"Gymnasium {gymnasium.__version__} 로드 성공")
except Exception as e:
    st.error(f"Gymnasium 로드 실패: {e}")

# 메인 앱 import
try:
    from src.app import main  # noqa: F401
    st.success("메인 앱 로드 성공")
except Exception as e:
    st.error(f"메인 앱 로드 실패: {e}")
    import traceback
    st.code(traceback.format_exc())
