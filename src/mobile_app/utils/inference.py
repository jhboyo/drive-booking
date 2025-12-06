"""
eScooterAI - 추론 유틸리티
YOLOv8 모델 로드 및 추론 관련 함수
"""

from pathlib import Path
from ultralytics import YOLO


def get_model_path(model_filename: str = "best.pt") -> Path:
    """
    모델 파일 경로 반환

    Args:
        model_filename: 모델 파일 이름 (기본값: best.pt)

    Returns:
        Path: 모델 파일 경로

    Raises:
        FileNotFoundError: 모델 파일이 존재하지 않을 경우
    """
    # 프로젝트 루트 경로
    project_root = Path(__file__).parent.parent.parent.parent
    model_path = project_root / "models" / "ppe_detection" / "weights" / model_filename

    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    return model_path


def load_model(model_path: str | Path) -> YOLO:
    """
    YOLOv8 모델 로드

    Args:
        model_path: 모델 파일 경로

    Returns:
        YOLO: 로드된 YOLOv8 모델

    Raises:
        Exception: 모델 로드 실패 시
    """
    try:
        model = YOLO(str(model_path))
        return model
    except Exception as e:
        raise Exception(f"YOLOv8 모델 로드 실패: {str(e)}")
