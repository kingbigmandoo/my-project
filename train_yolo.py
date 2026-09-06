from __future__ import annotations

import argparse
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent
TARGET_WEIGHTS = BASE_DIR / "models" / "waste_yolo.pt"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--roboflow-key", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)

    args = parser.parse_args()

    from roboflow import Roboflow
    from ultralytics import YOLO

    # Roboflow Waste Detection V1 다운로드
    rf = Roboflow(api_key=args.roboflow_key)

    project = (
        rf.workspace("projectverba")
        .project("yolo-waste-detection")
    )

    dataset = project.version(1).download("yolov11")

    data_yaml = Path(dataset.location) / "data.yaml"

    print("데이터셋:", data_yaml)

    # 정확도/속도 균형이 좋은 YOLO11s
    model = YOLO("yolo11s.pt")

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(BASE_DIR / "runs"),
        name="waste_yolo11s",
    )

    # 가장 성능 좋은 모델
    best = Path(results.save_dir) / "weights" / "best.pt"

    TARGET_WEIGHTS.parent.mkdir(exist_ok=True)

    shutil.copy(best, TARGET_WEIGHTS)

    print("학습 완료")
    print("모델 저장:", TARGET_WEIGHTS)


if __name__ == "__main__":
    main()