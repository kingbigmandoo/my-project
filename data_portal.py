"""
공공데이터포털(data.go.kr) 연동 모듈
-------------------------------------------------
1) 한국환경공단_탄소중립실천포인트 참여기업 및 항목별 적립단가 정보
2) 환경부_제품별 온실가스 배출량 및 탄소발자국 정보

- API 를 호출해 응답을 받아 CSV 로 저장(fetch_* 함수)
- 저장된 CSV 를 pandas 로 로드(load_* 함수)
- 네트워크/키가 없으면 내장 샘플 데이터로 자동 대체(오프라인 데모 가능)

사용법(터미널):
    python data_portal.py --key "발급받은_서비스키(Decoding)"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import requests

# -------------------------------------------------------------------
# 공통 설정
# -------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CARBON_POINT_CSV = DATA_DIR / "carbon_point_unit_price.csv"
CARBON_FOOTPRINT_CSV = DATA_DIR / "product_carbon_footprint.csv"

# data.go.kr 파일데이터가 자동 변환된 오픈API(odcloud) 기본 엔드포인트.
# 실제 신청한 데이터셋의 URL(uddi 포함)로 교체해서 사용하세요.
CARBON_POINT_URL = os.getenv(
    "CARBON_POINT_API_URL",
    "https://api.odcloud.kr/api/15111384/v1/uddi:carbon-point-unit-price",
)
CARBON_FOOTPRINT_URL = os.getenv(
    "CARBON_FOOTPRINT_API_URL",
    "https://api.odcloud.kr/api/15068999/v1/uddi:product-carbon-footprint",
)

# 서비스키는 코드에 하드코딩하지 말고 환경변수 / Streamlit secrets 로 주입
SERVICE_KEY = os.getenv("DATA_GO_KR_SERVICE_KEY", "")


# -------------------------------------------------------------------
# 내장 샘플 데이터 (API 실패 시 폴백)
# -------------------------------------------------------------------
def _sample_carbon_point() -> pd.DataFrame:
    rows = [
        ("전자영수증 발급", "GS리테일", 100, "건"),
        ("전자영수증 발급", "이마트", 100, "건"),
        ("텀블러/다회용컵 이용", "스타벅스코리아", 300, "회"),
        ("텀블러/다회용컵 이용", "메가엠지씨커피", 300, "회"),
        ("일회용컵 반환", "롯데GRS", 200, "개"),
        ("다회용기 이용(배달)", "요기요", 1000, "건"),
        ("무공해차 대여", "롯데렌탈", 5000, "일"),
        ("친환경제품(리필스테이션)", "아모레퍼시픽", 2000, "회"),
        ("고품질 재활용품 배출", "수퍼빈", 100, "kg"),
        ("폐휴대폰 반납", "SK네트웍스", 1000, "대"),
        ("친환경 숙박", "야놀자", 1000, "박"),
        ("기후행동 실천", "당근마켓", 300, "건"),
    ]
    return pd.DataFrame(
        rows, columns=["실천항목", "참여기업", "적립단가(원)", "단위"]
    )


def _sample_carbon_footprint() -> pd.DataFrame:
    rows = [
        ("제2020-001", "A식품", "생수 2L", "먹는샘물", 0.12, "2020-03-11", "저탄소"),
        ("제2020-014", "B음료", "탄산음료 500ml", "음료", 0.19, "2020-05-20", "일반"),
        ("제2021-052", "C제지", "복사용지 A4 1박스", "종이제품", 6.40, "2021-02-18", "저탄소"),
        ("제2021-133", "D전자", "LED 전구 10W", "조명기기", 2.10, "2021-09-02", "일반"),
        ("제2022-021", "E화학", "주방세제 1L", "세제류", 1.05, "2022-04-14", "저탄소"),
        ("제2022-088", "F섬유", "면 티셔츠", "의류", 3.30, "2022-07-30", "일반"),
        ("제2023-004", "G건자재", "시멘트 40kg", "건축자재", 32.10, "2023-01-19", "일반"),
        ("제2023-076", "H가전", "냉장고 300L급", "가전제품", 410.00, "2023-08-05", "저탄소"),
        ("제2024-012", "I식품", "두부 300g", "가공식품", 0.35, "2024-03-22", "저탄소"),
        ("제2024-045", "J생활", "화장지 30롤", "위생용품", 4.80, "2024-06-10", "일반"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "인증번호",
            "업체명",
            "제품명",
            "제품군",
            "탄소배출량(kgCO2eq)",
            "인증일자",
            "인증구분",
        ],
    )


# -------------------------------------------------------------------
# API 호출 유틸
# -------------------------------------------------------------------
def _request_odcloud(url: str, service_key: str, max_rows: int = 1000) -> list[dict]:
    """odcloud(파일데이터 변환형) 오픈API 페이지네이션 호출."""
    all_rows: list[dict] = []
    page = 1
    per_page = 100
    while len(all_rows) < max_rows:
        params = {
            "page": page,
            "perPage": per_page,
            "serviceKey": service_key,
            "returnType": "JSON",
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()

        # odcloud 형식: {"data": [...], "totalCount": N, ...}
        # 표준 형식: {"response": {"body": {"items": [...]}}}
        if "data" in payload:
            batch = payload.get("data", [])
        else:
            body = payload.get("response", {}).get("body", {})
            items = body.get("items", [])
            batch = items.get("item", []) if isinstance(items, dict) else items

        if not batch:
            break
        all_rows.extend(batch)
        if len(batch) < per_page:
            break
        page += 1
    return all_rows


def _fetch(url: str, service_key: str, fallback: pd.DataFrame, label: str) -> pd.DataFrame:
    if not service_key:
        print(f"[{label}] 서비스키가 없어 샘플 데이터로 대체합니다.")
        return fallback
    try:
        rows = _request_odcloud(url, service_key)
        if not rows:
            print(f"[{label}] 응답이 비어 있어 샘플 데이터로 대체합니다.")
            return fallback
        print(f"[{label}] {len(rows)}건 수신 완료.")
        return pd.DataFrame(rows)
    except Exception as exc:  # noqa: BLE001
        print(f"[{label}] API 호출 실패({exc}) → 샘플 데이터로 대체합니다.")
        return fallback


# -------------------------------------------------------------------
# 공개 함수: 수집 + CSV 저장
# -------------------------------------------------------------------
def fetch_carbon_point(service_key: str = SERVICE_KEY) -> pd.DataFrame:
    df = _fetch(
        CARBON_POINT_URL, service_key, _sample_carbon_point(), "탄소중립실천포인트 적립단가"
    )
    df.to_csv(CARBON_POINT_CSV, index=False, encoding="utf-8-sig")
    print(f"저장: {CARBON_POINT_CSV}")
    return df


def fetch_carbon_footprint(service_key: str = SERVICE_KEY) -> pd.DataFrame:
    df = _fetch(
        CARBON_FOOTPRINT_URL, service_key, _sample_carbon_footprint(), "제품별 탄소발자국"
    )
    df.to_csv(CARBON_FOOTPRINT_CSV, index=False, encoding="utf-8-sig")
    print(f"저장: {CARBON_FOOTPRINT_CSV}")
    return df


# -------------------------------------------------------------------
# 공개 함수: CSV 로드 (없으면 즉시 수집)
# -------------------------------------------------------------------
def load_carbon_point(service_key: str = SERVICE_KEY) -> pd.DataFrame:
    if CARBON_POINT_CSV.exists():
        return pd.read_csv(CARBON_POINT_CSV)
    return fetch_carbon_point(service_key)


def load_carbon_footprint(service_key: str = SERVICE_KEY) -> pd.DataFrame:
    if CARBON_FOOTPRINT_CSV.exists():
        return pd.read_csv(CARBON_FOOTPRINT_CSV)
    return fetch_carbon_footprint(service_key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="공공데이터포털 데이터 수집 → CSV 저장")
    parser.add_argument("--key", default=SERVICE_KEY, help="data.go.kr 서비스키(Decoding)")
    args = parser.parse_args()

    fetch_carbon_point(args.key)
    fetch_carbon_footprint(args.key)
    print("완료.")
