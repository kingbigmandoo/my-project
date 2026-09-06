"""
YOLO 재활용품 객체 탐지 모듈
------------------------------------------------------------------
기존 ViT 이미지 분류(app.py) 기능은 그대로 두고,
"이미지 1장 → 다중 객체 탐지 → Bounding Box → 품목/신뢰도 →
 품목별 개수 → 정상/반려 판정 → 포인트 계산" 파이프라인을 담당합니다.

가중치 우선순위
    1) 환경변수 WASTE_YOLO_MODEL 이 가리키는 .pt 파일
    2) ./models/waste_yolov8.pt  (사전학습 재활용품 탐지 모델
       - HuggingFace: HrutikAdsare/waste-detection-yolov8)
    3) yolo11n.pt  (COCO 사전학습 모델 — 가중치가 없을 때의 데모용 폴백)

사전학습 모델은 `python download_model.py` 로 내려받습니다(Roboflow API 키 불필요).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# -------------------------------------------------------------------
# 경로 / 가중치 설정
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DEFAULT_WEIGHTS = BASE_DIR / "models" / "waste_yolov8.pt"
FALLBACK_WEIGHTS = "yolo11n.pt"  # ultralytics 가 자동 다운로드

# -------------------------------------------------------------------
# Roboflow "YOLO Waste Detection V1" 클래스 (44종)
# -------------------------------------------------------------------
WASTE_CLASSES = [
    "Aerosols", "Aluminum can", "Aluminum caps", "Cardboard", "Cellulose",
    "Ceramic", "Combined plastic", "Container for household chemicals",
    "Disposable tableware", "Electronics", "Foil", "Furniture", "Glass bottle",
    "Iron utensils", "Liquid", "Metal shavings", "Milk bottle", "Organic",
    "Paper bag", "Paper cups", "Paper shavings", "Paper", "Papier mache",
    "Plastic bag", "Plastic bottle", "Plastic can", "Plastic canister",
    "Plastic caps", "Plastic cup", "Plastic shaker", "Plastic shavings",
    "Plastic toys", "Postal packaging", "Printing industry", "Scrap metal",
    "Stretch film", "Tetra pack", "Textile", "Tin", "Unknown plastic", "Wood",
    "Zip plastic bag", "Ramen Cup", "Food Packet",
]

# -------------------------------------------------------------------
# 국내 재활용 분류 체계 매핑 (app.py 의 label_map 과 포인트 정책 일치)
#   플라스틱 10P / 캔 15P / 유리병 20P / 종이류 5P / 골판지 5P / 일반쓰레기 0P(반려)
# -------------------------------------------------------------------
CATEGORY_INFO = {
    "플라스틱": {"label": "플라스틱 (Plastic)", "points": 10, "status": "ACCEPTED"},
    "캔": {"label": "캔/고철 (Metal)", "points": 15, "status": "ACCEPTED"},
    "유리병": {"label": "유리병 (Glass)", "points": 20, "status": "ACCEPTED"},
    "종이류": {"label": "종이류 (Paper)", "points": 5, "status": "ACCEPTED"},
    "골판지": {"label": "골판지/박스 (Cardboard)", "points": 5, "status": "ACCEPTED"},
    "일반쓰레기": {"label": "일반쓰레기/이물질 (Trash)", "points": 0, "status": "REJECTED"},
}

# 탐지 클래스 → 국내 분류
_WASTE_TO_CATEGORY = {
    # 사전학습 모델(HrutikAdsare/waste-detection-yolov8)의 8개 클래스
    "plastic": "플라스틱", "metal": "캔", "glass": "유리병",
    "paper": "종이류", "cardboard": "골판지",
    "organic": "일반쓰레기", "e-waste": "일반쓰레기", "medical": "일반쓰레기",
    # 플라스틱
    "plastic bottle": "플라스틱", "plastic can": "플라스틱",
    "plastic canister": "플라스틱", "plastic cup": "플라스틱",
    "plastic bag": "플라스틱", "plastic caps": "플라스틱",
    "plastic toys": "플라스틱", "plastic shaker": "플라스틱",
    "plastic shavings": "플라스틱", "combined plastic": "플라스틱",
    "unknown plastic": "플라스틱", "zip plastic bag": "플라스틱",
    "stretch film": "플라스틱", "disposable tableware": "플라스틱",
    "milk bottle": "플라스틱", "container for household chemicals": "플라스틱",
    "cellulose": "플라스틱",
    # 캔 / 고철
    "aluminum can": "캔", "aluminium can": "캔",
    "aluminum caps": "캔", "aluminium caps": "캔",
    "foil": "캔", "tin": "캔", "scrap metal": "캔",
    "iron utensils": "캔", "metal shavings": "캔", "aerosols": "캔",
    # 유리병
    "glass bottle": "유리병",
    # 종이류
    "paper": "종이류", "paper bag": "종이류", "paper cups": "종이류",
    "paper shavings": "종이류", "papier mache": "종이류",
    "printing industry": "종이류", "tetra pack": "종이류",
    "postal packaging": "종이류",
    # 골판지
    "cardboard": "골판지",
    # 일반쓰레기 / 이물질 → 반려
    "ceramic": "일반쓰레기", "organic": "일반쓰레기", "liquid": "일반쓰레기",
    "textile": "일반쓰레기", "wood": "일반쓰레기", "furniture": "일반쓰레기",
    "electronics": "일반쓰레기", "food packet": "일반쓰레기",
    "ramen cup": "일반쓰레기",
}

# COCO 사전학습(yolo11n.pt) 폴백 클래스 → 국내 분류 (데모용 근사 매핑)
_COCO_TO_CATEGORY = {
    "bottle": "플라스틱",
    "cup": "플라스틱",
    "wine glass": "유리병",
    "vase": "유리병",
    "book": "종이류",
}


def map_to_category(raw_name: str) -> str:
    """탐지된 원본 클래스명을 국내 재활용 분류로 변환."""
    key = str(raw_name).lower().strip()
    if key in _WASTE_TO_CATEGORY:
        return _WASTE_TO_CATEGORY[key]
    if key in _COCO_TO_CATEGORY:
        return _COCO_TO_CATEGORY[key]
    return "일반쓰레기"


# -------------------------------------------------------------------
# 모델 로더 (프로세스당 1회 로드)
# -------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_yolo_model():
    """(model, kind) 반환. kind 는 'custom' 또는 'fallback'."""
    from ultralytics import YOLO

    candidates = [os.getenv("WASTE_YOLO_MODEL", "").strip(), str(DEFAULT_WEIGHTS)]
    for path in candidates:
        if path and Path(path).is_file():
            return YOLO(path), "custom"
    return YOLO(FALLBACK_WEIGHTS), "fallback"


# -------------------------------------------------------------------
# 추론
# -------------------------------------------------------------------
def run_waste_detection(
    image: Image.Image,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
):
    """이미지에서 재활용품 객체를 탐지.

    Returns
    -------
    detections : list[dict]
        각 원소: raw_name, category, display_name, confidence, points, bbox(x1,y1,x2,y2)
    model_kind : str
        'custom'(직접 학습 가중치) 또는 'fallback'(COCO 사전학습)
    """
    model, model_kind = load_yolo_model()
    results = model.predict(
        source=image, conf=conf, iou=iou, imgsz=imgsz, verbose=False
    )
    res = results[0]
    names = res.names  # dict[int, str] 또는 list[str]

    detections: list[dict] = []
    boxes = getattr(res, "boxes", None)
    if boxes is None:
        return detections, model_kind

    for box in boxes:
        cls_id = int(box.cls[0])
        if isinstance(names, dict):
            raw_name = str(names.get(cls_id, cls_id))
        else:
            raw_name = str(names[cls_id])
        score = float(box.conf[0])
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())

        category = map_to_category(raw_name)
        info = CATEGORY_INFO[category]
        detections.append({
            "raw_name": raw_name,
            "category": category,
            "display_name": info["label"],
            "confidence": score,
            "points": info["points"],
            "bbox": (x1, y1, x2, y2),
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections, model_kind


# -------------------------------------------------------------------
# 품목별 개수 집계 + 정상/반려 판정 + 포인트 계산
# -------------------------------------------------------------------
def summarize_detections(detections: list[dict], threshold: float) -> dict:
    """탐지 결과를 집계해 최종 판정/포인트를 계산."""
    counts: dict[str, int] = {}
    accepted_details: list[dict] = []
    low_conf_items: list[dict] = []
    rejected_items: list[dict] = []
    total_points = 0

    for d in detections:
        if d["category"] == "일반쓰레기":
            rejected_items.append(d)
            continue
        if d["confidence"] < threshold:
            low_conf_items.append(d)
            continue
        counts[d["category"]] = counts.get(d["category"], 0) + 1
        total_points += d["points"]
        accepted_details.append(d)

    if rejected_items:
        status = "REJECTED"
        reason = (
            f"투입 불가 품목(이물질/일반쓰레기) {len(rejected_items)}건 감지 "
            f"→ 오작동 방지를 위해 투입구를 잠급니다."
        )
        counts, accepted_details, total_points = {}, [], 0
    elif not accepted_details:
        status = "REJECTED"
        if low_conf_items:
            reason = (
                f"감지된 모든 품목의 확신도가 기준({threshold:.2f}) 미만입니다 "
                f"→ 오작동 방지 반려."
            )
        else:
            reason = "재활용 가능한 품목이 감지되지 않았습니다."
    else:
        status = "ACCEPTED"
        n_items = sum(counts.values())
        reason = f"{n_items}개 품목 정상 판별 · 수거 완료"

    return {
        "status": status,
        "reason": reason,
        "counts": counts,
        "total_points": total_points,
        "accepted_details": accepted_details,
        "low_conf_items": low_conf_items,
        "rejected_items": rejected_items,
        "num_detections": len(detections),
    }


# -------------------------------------------------------------------
# Bounding Box 시각화
# -------------------------------------------------------------------
def _load_font(size: int):
    for path in (
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\malgunbd.ttf",
        "malgun.ttf",
        "arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_detections(
    image: Image.Image, detections: list[dict], threshold: float
) -> Image.Image:
    """탐지 박스/라벨을 그려 넣은 새 이미지를 반환.

    색상: 초록=정상 수거 / 주황=확신도 부족 / 빨강=이물질(반려)
    """
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    font = _load_font(max(14, img.width // 45))

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        if d["category"] == "일반쓰레기":
            color = (220, 38, 38)
        elif d["confidence"] < threshold:
            color = (234, 140, 20)
        else:
            color = (22, 163, 74)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        short_name = d["display_name"].split(" (")[0]
        label = f"{short_name} {d['confidence'] * 100:.0f}%"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ly = y1 - th - 6
        if ly < 0:
            ly = y1 + 2
        draw.rectangle([x1, ly, x1 + tw + 8, ly + th + 6], fill=color)
        draw.text((x1 + 4, ly + 2), label, fill=(255, 255, 255), font=font)

    return img
