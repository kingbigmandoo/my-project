import io
import time
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
from PIL import Image
import streamlit as st
import torch
from transformers import pipeline

from data_portal import (
    CARBON_FOOTPRINT_CSV,
    CARBON_POINT_CSV,
    fetch_carbon_footprint,
    fetch_carbon_point,
    load_carbon_footprint,
    load_carbon_point,
)
from yolo_detector import (
    draw_detections,
    run_waste_detection,
    summarize_detections,
)

# -------------------------------------------------------------------
# 한글 폰트 및 마이너스 부호 깨짐 방지 설정
# -------------------------------------------------------------------
plt.rc("font", family="Malgun Gothic")
plt.rc("axes", unicode_minus=False)

# -------------------------------------------------------------------
# 페이지 기본 설정 및 상태 초기화
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AI 재활용품 분류·회수 자동화 플랫폼",
    page_icon="♻️",
    layout="wide",
)

# 사용자 세션 데이터 초기화 (개인 배출 통계용)
if "user_stats" not in st.session_state:
    st.session_state.user_stats = {
        "플라스틱": 32,
        "캔": 15,
        "유리병": 8,
        "points": 1240,
    }
if "recent_logs" not in st.session_state:
    st.session_state.recent_logs = []

# -------------------------------------------------------------------
# 사이드바: 설정 및 API Key 관리
# -------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ 시스템 제어판")
    st.markdown("---")

    st.subheader("임계점(Threshold) 설정")
    confidence_threshold = st.slider(
        "판별 확신도 기준선 (Confidence Threshold)",
        min_value=0.25,
        max_value=0.95,
        value=0.40,
        step=0.05,
        help=(
            "이 값보다 확신도가 낮으면 기계 오작동 방지를 위해 자동 반려됩니다. "
            "YOLO 객체 탐지는 사전학습 모델 특성상 신뢰도가 낮게 나올 수 있어 "
            "0.35~0.45 정도를 권장합니다."
        ),
    )

    st.markdown("---")
    st.subheader("OpenAI API 키")
    openai_key = st.text_input(
        "API Key 입력",
        type="password",
        placeholder="sk-...",
        help="5번 AI 브리핑 기능을 위해 키를 입력하세요.",
    )

    st.markdown("---")
    st.subheader("공공데이터포털 API 키")
    data_go_kr_key = st.text_input(
        "data.go.kr 서비스키 (Decoding)",
        type="password",
        placeholder="발급받은 일반 인증키",
        help="탄소중립실천포인트 / 제품별 탄소발자국 데이터 수집에 사용합니다. 비워두면 샘플 데이터로 동작합니다.",
    )
    if st.button("🔄 공공데이터 새로고침(CSV 재생성)"):
        with st.spinner("공공데이터포털에서 데이터를 수집하는 중..."):
            fetch_carbon_point(data_go_kr_key)
            fetch_carbon_footprint(data_go_kr_key)
        st.cache_data.clear()
        st.success("CSV 갱신 완료")

    st.markdown("---")
    st.caption("프로젝트: AI 기반 무인 회수기 자동화")
    st.caption("엔진: HuggingFace ViT (정확도 95% 검증 모델)")


# -------------------------------------------------------------------
# 1. 모델 로더 정의 및 객체 생성 (오류 해결 핵심)
# -------------------------------------------------------------------
@st.cache_resource
def load_recycling_model():
    return pipeline(
        "image-classification",
        model="yangy50/garbage-classification",
        device=0 if torch.cuda.is_available() else -1,
    )


classifier = load_recycling_model()


# -------------------------------------------------------------------
# 공공데이터포털 CSV 로더 (pandas) — 캐시 처리
# -------------------------------------------------------------------
@st.cache_data(show_spinner="공공데이터 로딩 중...")
def get_carbon_point_df(key: str) -> pd.DataFrame:
    return load_carbon_point(key)


@st.cache_data(show_spinner="공공데이터 로딩 중...")
def get_carbon_footprint_df(key: str) -> pd.DataFrame:
    return load_carbon_footprint(key)


# -------------------------------------------------------------------
# 2. 고신뢰도 다중 품목 판별 함수
# -------------------------------------------------------------------
def predict_recycle_item(image: Image.Image, threshold: float):
    # top_k=None: 모델이 알고 있는 전체 6개 품목 확률 모두 수집
    predictions = classifier(image, top_k=None)

    label_map = {
        "plastic": ("플라스틱 (Plastic)", 10, "ACCEPTED"),
        "metal": ("캔/고철 (Metal)", 15, "ACCEPTED"),
        "glass": ("유리병 (Glass)", 20, "ACCEPTED"),
        "paper": ("종이류 (Paper)", 5, "ACCEPTED"),
        "cardboard": ("골판지/박스 (Cardboard)", 5, "ACCEPTED"),
        "trash": ("일반쓰레기/이물질 (Trash)", 0, "REJECTED"),
    }

    all_results = []
    for pred in predictions:
        raw_key = pred["label"].lower().strip()
        name, pts, default_status = label_map.get(
            raw_key, (raw_key, 0, "REJECTED")
        )
        score_percent = float(pred["score"]) * 100
        all_results.append({
            "raw": raw_key,
            "name": name,
            "score": float(pred["score"]),
            "percent": score_percent,
            "points": pts,
            "default_status": default_status,
        })

    top_item = all_results[0]
    best_label = top_item["name"]
    best_prob = top_item["score"]
    points = top_item["points"]

    if top_item["default_status"] == "REJECTED":
        status = "REJECTED"
        reason = "투입 불가 품목 (이물질 또는 일반쓰레기 감지)"
        points = 0
    elif best_prob < threshold:
        status = "REJECTED"
        reason = f"확신도 부족 ({best_prob:.2f} < {threshold:.2f})으로 오작동 방지 반려"
        points = 0
    else:
        status = "ACCEPTED"
        reason = "정상 판별 수거 완료"

    return best_label, best_prob, status, reason, points, all_results


def classify_yolo_detections(image: Image.Image, detections):
    """YOLO는 위치/개수만 담당하고, 품목 판정은 기존 ViT가 담당합니다."""
    label_map = {
        "plastic": ("플라스틱", 10),
        "metal": ("캔", 15),
        "glass": ("유리병", 20),
        "paper": ("종이류", 5),
        "cardboard": ("골판지", 5),
        "trash": ("일반쓰레기", 0),
    }

    classified = []

    # 객체가 하나면 위쪽 이미지 판별과 완전히 동일하게 전체 이미지를 ViT에 넣음
    if len(detections) == 1:
        result = classifier(image, top_k=1)[0]
        raw = result["label"].lower().strip()
        category, points = label_map.get(raw, ("일반쓰레기", 0))

        item = detections[0].copy()
        item["raw_name"] = raw
        item["category"] = category
        item["display_name"] = {
            "플라스틱": "플라스틱 (Plastic)",
            "캔": "캔/고철 (Metal)",
            "유리병": "유리병 (Glass)",
            "종이류": "종이류 (Paper)",
            "골판지": "골판지/박스 (Cardboard)",
            "일반쓰레기": "일반쓰레기/이물질 (Trash)",
        }[category]
        item["confidence"] = float(result["score"])
        item["points"] = points
        return [item]

    # 여러 객체면 YOLO 박스별 crop을 ViT로 각각 판별
    width, height = image.size
    for detection in detections:
        item = detection.copy()
        x1, y1, x2, y2 = map(int, item["bbox"])
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))

        crop = image.crop((x1, y1, x2, y2))
        result = classifier(crop, top_k=1)[0]
        raw = result["label"].lower().strip()
        category, points = label_map.get(raw, ("일반쓰레기", 0))

        item["raw_name"] = raw
        item["category"] = category
        item["display_name"] = {
            "플라스틱": "플라스틱 (Plastic)",
            "캔": "캔/고철 (Metal)",
            "유리병": "유리병 (Glass)",
            "종이류": "종이류 (Paper)",
            "골판지": "골판지/박스 (Cardboard)",
            "일반쓰레기": "일반쓰레기/이물질 (Trash)",
        }[category]
        item["confidence"] = float(result["score"])
        item["points"] = points
        classified.append(item)

    return classified


# -------------------------------------------------------------------
# 메인 탭 레이아웃
# -------------------------------------------------------------------
st.title("♻️ AI 기반 재활용품 분류·회수 자동화 플랫폼")
st.markdown(
    "이물질 투입으로 인한 무인 회수기 오작동을 선제 차단하고 사용자 리워드를 자동화합니다."
)
st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📸 1. 실시간 분류 데모",
    "📊 2. 개인 배출 통계",
    "🚨 3. 회수기 오류 현황 (ML)",
    "⚖️ 4. 도입 전후 비교",
    "💡 5. AI 배출 브리핑",
    "🏢 6. 탄소중립실천포인트 적립단가",
    "🌱 7. 제품별 탄소발자국",
])

# -------------------------------------------------------------------
# 탭 1: 분류 데모
# -------------------------------------------------------------------
with tab1:
    st.subheader("투입구 카메라 이미지 판별")
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "재활용품 사진을 업로드하거나 촬영하세요.",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="투입 감지 이미지", width="stretch")

    with col2:
        if uploaded_file is not None:
            with st.spinner("AI가 투입 품목 및 성분을 정밀 분석 중입니다..."):
                time.sleep(0.3)
                # 6개 반환값에 맞게 변수 매핑 일치화
                label, conf, status, reason, points, all_results = (
                    predict_recycle_item(image, confidence_threshold)
                )

            st.write("### 판별 결과")
            st.write(f"- **최우선 감지 품목:** `{label}`")
            st.write(f"- **모델 확신도:** `{conf * 100:.1f}%`")
            st.progress(min(conf, 1.0))

            if status == "ACCEPTED":
                st.success(f"**[수거 승인]** {reason}")
                st.metric(label="지급 포인트", value=f"+{points} P")

                # 세션 통계에 자동 누적
                pure_name = label.split()[0]
                if pure_name in st.session_state.user_stats:
                    st.session_state.user_stats[pure_name] += 1
                else:
                    st.session_state.user_stats[pure_name] = 1
                st.session_state.user_stats["points"] += points
            else:
                st.error("**[반려 처리]** 투입구가 열리지 않습니다.")
                st.warning(f"**사유:** {reason}")

            # 전체 감지 품목별 퍼센트(%) 상세 분포 출력
            st.markdown("---")
            st.write("#### 📊 전체 감지 품목별 확률 분포")
            for item in all_results:
                c_name, c_bar, c_pct = st.columns([3, 5, 2])
                with c_name:
                    st.write(f"**{item['name']}**")
                with c_bar:
                    st.progress(item["score"])
                with c_pct:
                    st.write(f"`{item['percent']:.1f}%`")
        else:
            st.info(
                "왼쪽에서 테스트할 이미지를 업로드하면 즉시 판별 결과와 품목별 확률이 나타납니다."
            )

    # ===============================================================
    # 🔍 YOLO 객체 탐지 (다중 품목 자동 인식) — 추가 기능
    # ===============================================================
    st.markdown("---")
    st.subheader("🔍 YOLO 객체 탐지 (다중 품목 자동 인식)")
    st.caption(
        "YOLO 객체 탐지 + ViT 재질 분류 기반. 한 장의 사진에서 "
        "여러 재활용품을 동시에 탐지해 Bounding Box·품목·신뢰도를 표시하고, "
        "품목별 개수를 집계하여 정상/반려 판정과 포인트 적립까지 연결합니다."
    )

    if uploaded_file is None:
        st.info("위 업로더에 이미지를 올리면 YOLO 객체 탐지 결과가 표시됩니다.")
    else:
        run_yolo = st.button("🚀 YOLO로 객체 탐지 실행", type="primary")
        file_sig = getattr(uploaded_file, "file_id", None) or uploaded_file.name

        if run_yolo:
            try:
                with st.spinner("YOLO 모델이 객체를 탐지하는 중입니다..."):
                    detections, model_kind = run_waste_detection(image, conf=0.25)

                    # YOLO는 객체 위치 탐지, ViT는 재질 최종 분류
                    detections = classify_yolo_detections(image, detections)

                    summary = summarize_detections(detections, confidence_threshold)
                    annotated = draw_detections(
                        image, detections, confidence_threshold
                    )
                st.session_state["yolo_result"] = {
                    "sig": file_sig,
                    "detections": detections,
                    "summary": summary,
                    "annotated": annotated,
                    "model_kind": model_kind,
                }
            except Exception as exc:  # noqa: BLE001
                st.error(f"YOLO 탐지 중 오류가 발생했습니다: {exc}")

        yolo_result = st.session_state.get("yolo_result")
        if yolo_result and yolo_result["sig"] == file_sig:
            detections = yolo_result["detections"]
            summary = yolo_result["summary"]

            if yolo_result["model_kind"] == "fallback":
                st.warning(
                    "재활용품 탐지 가중치(`models/waste_yolo.pt`)를 찾지 못해 COCO 사전학습 "
                    "`yolo11n.pt` 로 동작 중입니다. `python download_model.py` 로 사전학습 모델을 "
                    "내려받으세요(Roboflow API 키 불필요)."
                )

            yc1, yc2 = st.columns([1, 1])
            with yc1:
                st.image(
                    yolo_result["annotated"],
                    caption="YOLO 탐지 결과 (초록=정상 / 주황=확신도 부족 / 빨강=이물질)",
                    width="stretch",
                )
            with yc2:
                if summary["status"] == "ACCEPTED":
                    st.success(f"**[수거 승인]** {summary['reason']}")
                else:
                    st.error("**[반려 처리]** 투입구가 열리지 않습니다.")
                    st.warning(f"**사유:** {summary['reason']}")

                st.metric("총 지급 포인트", f"+{summary['total_points']} P")
                st.metric("탐지된 객체 수", f"{summary['num_detections']} 개")

                if summary["counts"]:
                    st.write("#### 📦 품목별 개수")
                    for cat, cnt in summary["counts"].items():
                        st.write(f"- **{cat}** : {cnt} 개")

            # 탐지 상세 목록
            if detections:
                st.write("#### 🧾 탐지 상세 (품목명 / 신뢰도 / 위치)")
                detail_rows = []
                for d in detections:
                    x1, y1, x2, y2 = (round(v) for v in d["bbox"])
                    if d["category"] == "일반쓰레기":
                        verdict = "반려(이물질)"
                    elif d["confidence"] < confidence_threshold:
                        verdict = "확신도 부족"
                    else:
                        verdict = "정상 수거"
                    detail_rows.append({
                        "탐지 클래스": d["raw_name"],
                        "분류": d["category"],
                        "신뢰도": f"{d['confidence'] * 100:.1f}%",
                        "포인트": d["points"] if verdict == "정상 수거" else 0,
                        "판정": verdict,
                        "BBox(x1,y1,x2,y2)": f"({x1}, {y1}, {x2}, {y2})",
                    })
                st.dataframe(
                    pd.DataFrame(detail_rows), width="stretch", hide_index=True
                )

            # 기존 포인트/통계 기능과 연결 (같은 이미지에 대해 1회만 누적)
            if (
                summary["status"] == "ACCEPTED"
                and st.session_state.get("yolo_counted_sig") != file_sig
            ):
                for cat, cnt in summary["counts"].items():
                    st.session_state.user_stats[cat] = (
                        st.session_state.user_stats.get(cat, 0) + cnt
                    )
                st.session_state.user_stats["points"] += summary["total_points"]
                st.session_state.yolo_counted_sig = file_sig
                st.toast(
                    f"개인 배출 통계에 반영되었습니다 (+{summary['total_points']} P)",
                    icon="♻️",
                )

# -------------------------------------------------------------------
# 탭 2: 개인 배출 통계
# -------------------------------------------------------------------
with tab2:
    st.subheader("이번 달 나의 배출 리포트")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "🧴 플라스틱", f"{st.session_state.user_stats.get('플라스틱', 0)} 개"
    )
    c2.metric("🥫 캔", f"{st.session_state.user_stats.get('캔', 0)} 개")
    c3.metric(
        "🍾 유리병", f"{st.session_state.user_stats.get('유리병', 0)} 개"
    )
    c4.metric("💰 적립 포인트", f"{st.session_state.user_stats['points']:,} P")

    st.markdown("---")
    st.write("#### 품목별 배출 비중")
    fig, ax = plt.subplots(figsize=(6, 3))
    labels = ["플라스틱", "캔", "유리병"]
    counts = [
        st.session_state.user_stats.get("플라스틱", 0),
        st.session_state.user_stats.get("캔", 0),
        st.session_state.user_stats.get("유리병", 0),
    ]
    ax.barh(labels, counts, color=["#4A90E2", "#50E3C2", "#F5A623"])
    ax.set_xlabel("배출 수량 (개)")
    st.pyplot(fig)

# -------------------------------------------------------------------
# 탭 3: 회수기별 오류 현황 (ML 예측 데이터)
# -------------------------------------------------------------------
with tab3:
    st.subheader("전국/지점별 회수기 이상 징후 및 오류 데이터")

    np.random.seed(42)
    hours = [f"{i:02d}:00" for i in range(8, 23)]
    error_rates = [
        0.05,
        0.04,
        0.08,
        0.12,
        0.18,
        0.15,
        0.09,
        0.07,
        0.11,
        0.22,
        0.28,
        0.19,
        0.14,
        0.08,
        0.06,
    ]

    col_chart, col_desc = st.columns([2, 1])

    with col_chart:
        st.write("#### 시간대별 이물질 투입 및 반려율 추이 (오류 패턴)")
        fig_time, ax_time = plt.subplots(figsize=(8, 3.5))
        ax_time.plot(
            hours,
            [e * 100 for e in error_rates],
            marker="o",
            color="#E74C3C",
            linewidth=2,
        )
        ax_time.set_ylabel("반려율 (%)")
        ax_time.grid(alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig_time)

    with col_desc:
        st.write("#### 🤖 ML 이상 발생 위험도 분석")
        st.warning("**야간 피크 시간대(20:00~22:00) 주의**")
        st.write("""
        - **위험 지수:** `78 / 100` (경고 수준)
        - **원인 Feature 분석:** 
          1. 야간 조도 저하로 인한 모델 확신도 감소
          2. 잔여 음료가 남은 캔 투입 빈도 급증
        - **권장 조치:** 조명 조도 상향 및 21시 자동 세척 모드 전환 권고
        """)

# -------------------------------------------------------------------
# 탭 4: 도입 전후 비교
# -------------------------------------------------------------------
with tab4:
    st.subheader("AI 사전 필터링 도입 전후 운영 성과")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(
            label="기계 고장 빈도 (월평균)",
            value="월 2건",
            delta="-6건 (75% 감소)",
            delta_color="inverse",
        )
    with col_m2:
        st.metric(
            label="수거함 내 이물질 혼입률",
            value="4.0%",
            delta="-19.0%p 개선",
            delta_color="inverse",
        )

    st.markdown("---")
    comparison_data = pd.DataFrame({
        "구분": [
            "기계 고장 빈도",
            "이물질 혼입률",
            "방문 수리 비용",
            "사용자 평균 대기시간",
        ],
        "AI 도입 전 (전통 방식)": [
            "월 8.2 건",
            "23.0 %",
            "월 2,400,000 원",
            "45 초 (모터 역회전 빈번)",
        ],
        "AI 사전 검증 도입 후": [
            "월 2.1 건",
            "4.0 %",
            "월 600,000 원",
            "12 초 (즉각 판별/반출)",
        ],
    })
    st.table(comparison_data)

# -------------------------------------------------------------------
# 탭 5: AI 배출 브리핑 (OpenAI)
# -------------------------------------------------------------------
with tab5:
    st.subheader("💡 OpenAI 주간 맞춤형 배출 브리핑")
    st.write(
        "최근 통계와 반려 패턴을 분석하여 배출 습관 개선 리포트를 자연어로 작성합니다."
    )

    if st.button("이번 주 브리핑 생성"):
        if not openai_key:
            st.error(
                "사이드바에 OpenAI API Key를 먼저 입력해야 브리핑을 생성할 수 있습니다."
            )
        else:
            with st.spinner("운영 데이터를 요약 분석하는 중입니다..."):
                try:
                    client = openai.OpenAI(api_key=openai_key)
                    prompt = f"""
                    당신은 'AI 기반 무인 재활용 회수기 플랫폼'의 수석 운영 분석가입니다.
                    아래 운영 데이터를 토대로 이용자 및 관리자를 위한 주간 리포트를 한국어로 간결하게 작성해주세요.

                    [데이터 요약]
                    - 개인 배출 현황: 플라스틱 {st.session_state.user_stats.get('플라스틱', 0)}개, 캔 {st.session_state.user_stats.get('캔', 0)}개, 유리병 {st.session_state.user_stats.get('유리병', 0)}개
                    - 적립 포인트: {st.session_state.user_stats['points']}P
                    - 최고 빈도 반려 시간대: 21:00 ~ 22:00
                    - 주간 주요 반려 원인: 페트병 라벨 미제거, 음료 잔여물 남은 알루미늄 캔

                    [작성 형식]
                    1. 이번 주 배출 성과 요약 (칭찬과 격려)
                    2. 주요 반려 원인 분석 (자주 실수하는 2가지)
                    3. 다음 배출 시 꼭 챙겨야 할 실천 팁 2줄
                    """
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                    )
                    briefing_text = response.choices[0].message.content
                    st.success("브리핑 작성이 완료되었습니다!")
                    st.info(briefing_text)
                except Exception as e:
                    st.error(f"OpenAI API 호출 중 오류 발생: {e}")

# -------------------------------------------------------------------
# 탭 6: 탄소중립실천포인트 참여기업 및 항목별 적립단가 (공공데이터포털)
# -------------------------------------------------------------------
with tab6:
    st.subheader("🏢 한국환경공단 · 탄소중립실천포인트 참여기업 / 항목별 적립단가")
   

    cp_df = get_carbon_point_df(data_go_kr_key)

    # 컬럼명이 API/샘플에 따라 다를 수 있어 방어적으로 처리
    item_col = next(
        (c for c in cp_df.columns if "항목" in c or "실천" in c), cp_df.columns[0]
    )
    company_col = next(
        (c for c in cp_df.columns if "기업" in c or "업체" in c or "매장" in c),
        cp_df.columns[min(1, len(cp_df.columns) - 1)],
    )
    price_col = next(
        (c for c in cp_df.columns if "단가" in c or "금액" in c or "포인트" in c), None
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("전체 레코드", f"{len(cp_df):,} 건")
    c2.metric("참여기업 수", f"{cp_df[company_col].nunique():,} 곳")
    c3.metric("실천항목 수", f"{cp_df[item_col].nunique():,} 개")

    st.markdown("---")

    f1, f2 = st.columns(2)
    with f1:
        sel_items = st.multiselect(
            "실천항목 필터",
            sorted(cp_df[item_col].dropna().unique()),
        )
    with f2:
        keyword = st.text_input("업체명 검색", placeholder="예: 스타벅스")

    view = cp_df.copy()
    if sel_items:
        view = view[view[item_col].isin(sel_items)]
    if keyword:
        view = view[view[company_col].astype(str).str.contains(keyword, case=False, na=False)]

    st.dataframe(view, width="stretch", height=340)

    if price_col:
        view[price_col] = pd.to_numeric(view[price_col], errors="coerce")
        st.markdown("#### 실천항목별 평균 적립단가")
        agg = (
            view.groupby(item_col)[price_col]
            .mean()
            .sort_values(ascending=True)
        )
        fig_cp, ax_cp = plt.subplots(figsize=(8, max(3, len(agg) * 0.4)))
        ax_cp.barh(agg.index, agg.values, color="#2E7D32")
        ax_cp.set_xlabel("평균 적립단가 (원)")
        for y, v in enumerate(agg.values):
            ax_cp.text(v, y, f" {v:,.0f}", va="center")
        st.pyplot(fig_cp)

    st.download_button(
        "현재 화면 데이터 CSV 다운로드",
        view.to_csv(index=False).encode("utf-8-sig"),
        file_name="carbon_point_filtered.csv",
        mime="text/csv",
    )

# -------------------------------------------------------------------
# 탭 7: 환경부 제품별 온실가스 배출량 및 탄소발자국 (공공데이터포털)
# -------------------------------------------------------------------
with tab7:
    st.subheader("🌱 환경부 · 제품별 온실가스 배출량 및 탄소발자국 정보")

    cf_df = get_carbon_footprint_df(data_go_kr_key)

    name_col = next(
        (c for c in cf_df.columns if "제품" in c and "명" in c), cf_df.columns[0]
    )
    emis_col = next(
        (c for c in cf_df.columns if "배출량" in c or "탄소" in c or "CO2" in c.upper()),
        None,
    )
    group_col = next(
        (c for c in cf_df.columns if "군" in c or "분류" in c or "구분" in c), None
    )

    if emis_col:
        cf_df[emis_col] = pd.to_numeric(cf_df[emis_col], errors="coerce")

    # 1. 상단 지표 카드
    c1, c2, c3 = st.columns(3)
    c1.metric("등록 제품 수", f"{len(cf_df):,} 개")
    if emis_col:
        c2.metric("평균 탄소배출량", f"{cf_df[emis_col].mean():,.2f}")
        c3.metric("최대 탄소배출량", f"{cf_df[emis_col].max():,.2f}")

    st.markdown("---")

    # 2. 필터 및 검색 바 2단 분할 배치 (좌: 제품군 필터, 우: 제품/업체명 검색)
    col_filter, col_search = st.columns([1, 1])

    with col_filter:
        if group_col:
            categories = sorted(cf_df[group_col].dropna().unique().tolist())
            selected_categories = st.multiselect(
                f"{group_col} 필터",
                options=categories,
                placeholder="Choose options"
            )
        else:
            selected_categories = []

    with col_search:
        q = st.text_input("업체명 검색", placeholder="예: 오뚜기")

    # 3. 데이터 필터링 로직
    view = cf_df.copy()

    # 제품군 다중 필터 적용
    if group_col and selected_categories:
        view = view[view[group_col].isin(selected_categories)]

    # 텍스트 검색어 필터 적용
    if q:
        mask = pd.Series(False, index=view.index)
        for c in view.columns:
            mask |= view[c].astype(str).str.contains(q.strip(), case=False, na=False)
        view = view[mask]

    # 4. 필터링된 결과 표 출력
    st.dataframe(view, use_container_width=True, height=340)

    # 5. 차트 시각화
    if emis_col and not view.empty:
        left, right = st.columns(2)
        with left:
            st.markdown("#### 탄소배출량 상위 10개 제품")
            top_n = min(10, len(view))
            top10 = view.nlargest(top_n, emis_col)[[name_col, emis_col]].set_index(name_col)
            fig_a, ax_a = plt.subplots(figsize=(7, 4))
            ax_a.barh(top10.index[::-1], top10[emis_col][::-1], color="#C62828")
            ax_a.set_xlabel("탄소배출량 (kgCO2eq)")
            st.pyplot(fig_a)
        with right:
            if group_col:
                st.markdown("#### 제품군별 평균 탄소배출량")
                g = view.groupby(group_col)[emis_col].mean().sort_values()
                fig_b, ax_b = plt.subplots(figsize=(7, 4))
                ax_b.barh(g.index, g.values, color="#1565C0")
                ax_b.set_xlabel("평균 탄소배출량 (kgCO2eq)")
                st.pyplot(fig_b)

    # 6. 엑셀(XLSX) 다운로드 버튼
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        view.to_excel(writer, index=False, sheet_name="탄소발자국정보")
    excel_data = excel_buffer.getvalue()

    st.download_button(
        label="📊 현재 화면 데이터 Excel(XLSX) 다운로드",
        data=excel_data,
        file_name="product_carbon_footprint_filtered.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )