"""
Streamlit Cloud 진입점
src 폴더에서 실행되어 모듈 임포트 문제 해결
"""
import sys
import runpy
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (src의 부모 폴더)
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# main.py를 모듈로 실행 (__file__ 변수 자동 설정)
main_path = project_root / "src" / "app" / "main.py"
runpy.run_path(str(main_path), run_name="__main__")
