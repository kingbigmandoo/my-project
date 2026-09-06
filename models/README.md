# YOLO 재활용품 탐지 가중치 폴더

`app.py` 의 YOLO 객체 탐지 기능은 아래 순서로 가중치를 찾습니다.

1. 환경변수 `WASTE_YOLO_MODEL` 이 가리키는 `.pt` 파일
2. `models/waste_yolov8.pt` — 사전학습 재활용품 탐지 모델
   (HuggingFace: [HrutikAdsare/waste-detection-yolov8](https://huggingface.co/HrutikAdsare/waste-detection-yolov8))
3. `yolo11n.pt` — 위 두 개가 없을 때 자동 다운로드되는 COCO 사전학습 모델(데모용 폴백)

## waste_yolov8.pt 내려받기 (Roboflow API 키 불필요)

```bash
pip install -r requirements.txt
python ../download_model.py
```

`download_model.py` 가 HuggingFace Hub 에서 사전학습 가중치를 받아
이 폴더에 `waste_yolov8.pt` 로 저장합니다.

탐지 클래스(8종): `cardboard`, `e-waste`, `glass`, `medical`, `metal`,
`organic`, `paper`, `plastic` → 국내 재활용 분류(플라스틱/캔/유리병/종이류/골판지/일반쓰레기)로 매핑됩니다.
