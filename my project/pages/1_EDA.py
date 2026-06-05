# pages/1_EDA.py — 데이터 들여다보기
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import load_data, CATEGORY_MAP

st.title("📊 EDA — 데이터 살펴보기")

df = load_data()

# ════════════════════════════════════════════════════════════
# 1. 데이터 개요
# ════════════════════════════════════════════════════════════
st.header("1. 데이터 개요")

c1, c2, c3, c4 = st.columns(4)
c1.metric("행 수", f"{len(df):,}")
c2.metric("열 수", f"{df.shape[1]}")
c3.metric("결측 있는 열", f"{int(df.isna().any().sum())}")
c4.metric("바이럴 비율", f"{df['is_viral'].mean()*100:.1f}%")

st.subheader("미리보기 (상위 20행)")
st.dataframe(df.head(20), use_container_width=True)

# ════════════════════════════════════════════════════════════
# 2. 요약 통계
# ════════════════════════════════════════════════════════════
st.header("2. 요약 통계")

num_cols = ["views", "likes", "dislikes", "comment_count", "days_to_trending",
            "like_ratio", "tag_count", "title_len"]
st.dataframe(df[num_cols].describe().T.round(2), use_container_width=True)

# ════════════════════════════════════════════════════════════
# 3. 결측치
# ════════════════════════════════════════════════════════════
st.header("3. 결측치")
na = df.isna().sum()
na = na[na > 0].sort_values(ascending=False)
if len(na) == 0:
    st.success("결측치 없음 ✅")
else:
    st.bar_chart(na)

# ════════════════════════════════════════════════════════════
# 4. 타겟 분포 (바이럴 vs 비바이럴)
# ════════════════════════════════════════════════════════════
st.header("4. 타겟 분포")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("바이럴 레이블 (75th pct 기준)")
    counts = df["is_viral"].value_counts().rename({0: "비바이럴", 1: "바이럴"})
    fig_pie = px.pie(
        values=counts.values,
        names=counts.index,
        color_discrete_sequence=["#636EFA", "#EF553B"],
        hole=0.4,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_b:
    st.subheader("조회수 분포 (log scale)")
    fig_hist = px.histogram(
        df, x="log_views", nbins=60,
        color="is_viral",
        color_discrete_map={0: "#636EFA", 1: "#EF553B"},
        labels={"log_views": "log(1 + views)", "is_viral": "바이럴"},
        barmode="overlay", opacity=0.7,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ════════════════════════════════════════════════════════════
# 5. 카테고리 분석
# ════════════════════════════════════════════════════════════
st.header("5. 카테고리별 분석")

cat_stats = (
    df.groupby("category_name")
    .agg(
        영상수=("video_id", "count"),
        평균조회수=("views", "mean"),
        바이럴비율=("is_viral", "mean"),
    )
    .sort_values("평균조회수", ascending=False)
    .reset_index()
)
cat_stats["평균조회수"] = cat_stats["평균조회수"].round(0).astype(int)
cat_stats["바이럴비율"] = (cat_stats["바이럴비율"] * 100).round(1)

col_c, col_d = st.columns(2)
with col_c:
    fig_cat = px.bar(
        cat_stats, x="평균조회수", y="category_name",
        orientation="h", title="카테고리별 평균 조회수",
        color="평균조회수", color_continuous_scale="Blues",
    )
    fig_cat.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_cat, use_container_width=True)

with col_d:
    fig_viral = px.bar(
        cat_stats.sort_values("바이럴비율", ascending=False),
        x="바이럴비율", y="category_name",
        orientation="h", title="카테고리별 바이럴 비율 (%)",
        color="바이럴비율", color_continuous_scale="Reds",
    )
    fig_viral.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_viral, use_container_width=True)

# ════════════════════════════════════════════════════════════
# 6. 시간 피처 분석
# ════════════════════════════════════════════════════════════
st.header("6. 시간 피처 분석")

col_e, col_f = st.columns(2)

with col_e:
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_stats = (
        df.groupby("published_day_of_week")["is_viral"]
        .mean()
        .reindex(dow_order)
        .reset_index()
    )
    dow_stats.columns = ["요일", "바이럴비율"]
    dow_stats["바이럴비율"] = (dow_stats["바이럴비율"] * 100).round(2)

    fig_dow = px.bar(
        dow_stats, x="요일", y="바이럴비율",
        title="요일별 바이럴 비율 (%)",
        color="바이럴비율", color_continuous_scale="Greens",
    )
    st.plotly_chart(fig_dow, use_container_width=True)

with col_f:
    fig_days = px.histogram(
        df[df["days_to_trending"] <= 30],
        x="days_to_trending", nbins=30,
        color="is_viral",
        color_discrete_map={0: "#636EFA", 1: "#EF553B"},
        labels={"days_to_trending": "업로드→트렌딩 일수", "is_viral": "바이럴"},
        title="트렌딩까지 걸린 일수 분포",
        barmode="overlay", opacity=0.75,
    )
    st.plotly_chart(fig_days, use_container_width=True)

# ════════════════════════════════════════════════════════════
# 7. 상관관계
# ════════════════════════════════════════════════════════════
st.header("7. 수치 피처 상관관계")

corr_cols = ["views", "likes", "dislikes", "comment_count",
             "like_ratio", "tag_count", "title_len", "days_to_trending", "is_viral"]
corr = df[corr_cols].corr()

fig_corr = px.imshow(
    corr, text_auto=".2f",
    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
    title="Pearson Correlation Heatmap",
    aspect="auto",
)
st.plotly_chart(fig_corr, use_container_width=True)

# ════════════════════════════════════════════════════════════
# 8. 국가별 분포
# ════════════════════════════════════════════════════════════
st.header("8. 국가별 분포")

country_stats = (
    df.groupby("publish_country")
    .agg(영상수=("video_id", "count"), 바이럴비율=("is_viral", "mean"))
    .reset_index()
)
country_stats["바이럴비율"] = (country_stats["바이럴비율"] * 100).round(1)

fig_country = px.bar(
    country_stats, x="publish_country", y="영상수",
    color="바이럴비율", color_continuous_scale="Viridis",
    title="국가별 영상 수 및 바이럴 비율",
    labels={"publish_country": "국가", "영상수": "영상 수"},
)
st.plotly_chart(fig_country, use_container_width=True)

# ════════════════════════════════════════════════════════════
# 9. 발견 사실 메모
# ════════════════════════════════════════════════════════════
st.header("9. 내가 발견한 것")
 
st.subheader("🗂️ 데이터 구조")
st.info(
    "- 총 **161,470행**, 18개 원본 컬럼 + 파생 컬럼 추가\n"
    "- 4개국 데이터: **US / GB / CANADA / FRANCE** (각 약 4만 건으로 균등 분포)\n"
    "- 바이럴 기준: 조회수 75th percentile ≈ **133만 뷰** → 바이럴:비바이럴 = **25%:75%** (클래스 불균형 존재)"
)
 
st.subheader("🎬 카테고리 인사이트")
st.success(
    "- **Music** 카테고리가 바이럴 비율 **53.3%** 로 압도적 1위 — 평균 조회수도 약 **822만**으로 최고\n"
    "- **Entertainment**는 전체 영상의 26%를 차지하는 가장 많은 카테고리이지만, 바이럴 비율은 21.7%로 중간 수준\n"
    "- **News & Politics**는 영상 수(11,623개)에 비해 바이럴 비율이 **6.8%** 로 가장 낮음 — 단기 소비성 콘텐츠 특성\n"
    "- **Film & Animation**은 적은 영상 수(9,139개)에도 바이럴 비율 **33.2%** — 예고편·클립의 팬덤 효과"
)
 
st.subheader("📅 시간 피처 인사이트")
st.warning(
    "- **금요일(29.6%)·목요일(28.7%)** 업로드가 바이럴 비율 가장 높음 — 주말 시청 급증 효과\n"
    "- **토요일(18.8%)** 이 가장 낮음 — 이미 주말이라 경쟁 영상이 많고 새 영상 소비가 분산됨\n"
    "- 업로드 시각은 **새벽 4~5시(UTC 기준)** 가 바이럴 비율 최상위 — 실제 현지 저녁 시간대와 일치할 가능성\n"
    "- **트렌딩까지 걸린 일수**: 바이럴 영상 평균 **0.03일(≈40분)** vs 비바이럴 **0.35일** → 바이럴은 업로드 직후 빠르게 반응"
)
 
st.subheader("🔗 상관관계 & 피처 인사이트")
st.info(
    "- **likes**(r=0.41)가 is_viral과 가장 높은 상관관계 — views(r=0.35)보다도 높음\n"
    "- views와 likes 간 상관관계가 매우 높아 **다중공선성 주의** 필요 (둘 다 모델 입력 시 중복 정보)\n"
    "- **title_len**(r=−0.09): 제목이 **짧을수록** 바이럴 확률 소폭 상승 — 바이럴 평균 50자 vs 비바이럴 56자\n"
    "- **tag_count**: 바이럴 영상이 평균 **19.2개** 태그, 비바이럴 **17.5개** — 태그가 많을수록 유리하나 효과 미미\n"
    "- **감탄사(!)·질문(?)·숫자 포함 제목이 오히려 바이럴 비율 낮음** — 어그로성 제목보다 콘텐츠 품질이 핵심"
)
 
st.subheader("🌍 국가별 인사이트")
st.success(
    "- **GB(영국)** 바이럴 비율 **44.1%** 로 4개국 중 1위 — 영어권 + 소규모 크리에이터 생태계 특성\n"
    "- **US** 32.0%, **CANADA** 18.5%, **FRANCE** 6.2% 순\n"
    "- **FRANCE**는 영어 콘텐츠 중심 플랫폼에서 현지 언어 콘텐츠의 글로벌 도달 한계가 반영된 결과"
)
 
st.subheader("⚠️ 모델링 시 주의사항")
st.error(
    "- **likes / comment_count** 는 업로드 이후 집계되는 값 → 업로드 *전* 예측 서비스에 그대로 쓰면 데이터 리크\n"
    "- 바이럴 레이블 25%:75% 불균형 → `class_weight='balanced'` 또는 `scale_pos_weight` 필수\n"
    "- 동일 영상이 여러 국가에서 트렌딩되면 중복 행 존재 가능 → 영상 ID 기준 중복 제거 검토 필요"
)