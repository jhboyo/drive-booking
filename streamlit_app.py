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

# 이제 src 모듈을 직접 import 가능
# main.py의 코드를 실행 (exec 대신 importlib 사용)
import importlib.util

main_path = project_root / "src" / "app" / "main.py"
spec = importlib.util.spec_from_file_location("main", str(main_path))
main_module = importlib.util.module_from_spec(spec)

# __file__ 설정
main_module.__file__ = str(main_path)

# 모듈 실행
spec.loader.exec_module(main_module)
