"""
Streamlit Cloud 진입점
src 폴더에서 실행되어 모듈 임포트 문제 해결
"""
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (src의 부모 폴더)
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# main.py 실행 (exec 사용 - __file__ 변수 설정)
main_path = project_root / "src" / "app" / "main.py"
with open(main_path, "r", encoding="utf-8") as f:
    code = compile(f.read(), str(main_path), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(main_path)})
