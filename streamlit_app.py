"""
Streamlit Cloud 진입점
프로젝트 루트에서 실행되어 모듈 임포트 문제 해결
"""
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (현재 파일이 루트에 있음)
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.app import main  # noqa: F401
