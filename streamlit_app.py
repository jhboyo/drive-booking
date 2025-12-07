"""
Streamlit Cloud 진입점
"""
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 메인 앱 실행
from src.app import main  # noqa: F401
