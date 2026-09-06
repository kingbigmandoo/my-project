"""
사전학습 재활용품 탐지 YOLO 모델 다운로드
------------------------------------------------------------------
Roboflow API 키나 직접 학습 없이, HuggingFace Hub 에 공개된
사전학습 YOLOv8 재활용품(쓰레기) 탐지 모델을 내려받아
`models/waste_yolov8.pt` 로 저장합니다.

    python download_model.py

저장이 끝나면 app.py 의 "YOLO 객체 탐지" 기능이 자동으로 이 가중치를 사용합니다.

모델 출처: https://huggingface.co/HrutikAdsare/waste-detection-yolov8
클래스: cardboard, e-waste, glass, medical, metal, organic, paper, plastic
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

REPO_ID = os.getenv("WASTE_MODEL_REPO", "HrutikAdsare/waste-detection-yolov8")
FILENAME = os.getenv("WASTE_MODEL_FILE", "best.pt")

BASE_DIR = Path(__file__).parent
TARGET = BASE_DIR / "models" / "waste_yolov8.pt"

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def main() -> None:
    from huggingface_hub import hf_hub_download

    if TARGET.is_file():
        print(f"이미 존재합니다: {TARGET} ({TARGET.stat().st_size / 1e6:.1f} MB)")
        return

    print(f"다운로드 중: {REPO_ID}/{FILENAME}")
    cached = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

    TARGET.parent.mkdir(exist_ok=True)
    shutil.copy(cached, TARGET)
    print(f"저장 완료 → {TARGET} ({TARGET.stat().st_size / 1e6:.1f} MB)")

    # 간단 검증
    try:
        from ultralytics import YOLO

        model = YOLO(str(TARGET))
        print(f"클래스({len(model.names)}종): {model.names}")
    except Exception as exc:  # noqa: BLE001
        print(f"모델 로드 검증 건너뜀: {exc}")


if __name__ == "__main__":
    main()
