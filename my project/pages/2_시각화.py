
# pages/2_시각화.py — 2차 작업: 그래프로 인사이트 찾기
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
 
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from src.data_loader import load_data
 
st.set_page_config(page_title="시각화", page_icon="📈", layout="wide")
st.title("📈 시각화 — 그래프로 인사이트 찾기")
 
df = load_data()
 
# 컬럼 분류
num_cols = ["views", "likes", "dislikes", "comment_count",
            "log_views", "like_ratio", "tag_count", "title_len", "days_to_trending"]
cat_cols = ["category_name", "publish_country", "published_day_of_week", "time_frame"]
all_cols = num_cols + cat_cols
 
# days_to_trending NaN 행은 시각화에서만 제외
num_cols_safe = [c for c in num_cols if c in df.columns]
 
# ════════════════════════════════════════════════════════════
# 그래프 1 — 분포
# ════════════════════════════════════════════════════════════
st.header("그래프 1 — 분포")
 
col_pick, color_pick = st.columns([3, 1])
col1 = col_pick.selectbox("볼 컬럼", num_cols_safe, index=num_cols_safe.index("log_views"), key="hist")
color_by_viral = color_pick.checkbox("바이럴 여부로 색 구분", value=True, key="hist_color")
 
fig1 = px.histogram(
    df, x=col1,
    color="is_viral" if color_by_viral else None,
    color_discrete_map={0: "#A7ABE4", 1: "#EF553B"},
    labels={"is_viral": "바이럴"},
    nbins=60, barmode="overlay", opacity=0.75,
    title=f"{col1} 분포",
)
fig1.update_layout(legend_title_text="바이럴")
st.plotly_chart(fig1, use_container_width=True)
st.caption(
    "📌 **해석:** log_views 기준 바이럴(빨강) 중앙값은 **약 15.0**, 비바이럴(파랑)은 **약 12.3**으로 "
    "두 분포가 오른쪽으로 뚜렷이 분리됩니다. 조회수 자체가 바이럴 예측의 가장 강력한 신호이지만, "
    "업로드 *전* 예측에는 사용할 수 없으므로 제목·태그·시간 등 사전 피처에 집중해야 합니다."
)
 
# ════════════════════════════════════════════════════════════
# 그래프 2 — 관계 (Scatter)
# ════════════════════════════════════════════════════════════
st.header("그래프 2 — 관계")
 
c1, c2, c3 = st.columns(3)
x_col = c1.selectbox("X축", num_cols_safe, index=num_cols_safe.index("log_views"), key="x")
y_col = c2.selectbox("Y축", num_cols_safe, index=num_cols_safe.index("like_ratio"), key="y")
sample_n = c3.slider("샘플 수 (속도)", 1000, 20000, 5000, step=1000)
 
df_sample = df.sample(n=min(sample_n, len(df)), random_state=42)
 
fig2 = px.scatter(
    df_sample, x=x_col, y=y_col,
    color="is_viral",
    color_discrete_map={0: "#636EFA", 1: "#EF553B"},
    labels={"is_viral": "바이럴"},
    opacity=0.5,
    title=f"{x_col} vs {y_col}",
)
st.plotly_chart(fig2, use_container_width=True)
st.caption(
    "📌 **해석:** log_views가 높을수록 like_ratio도 대체로 높으며, 바이럴 영상이 우상단에 집중됩니다. "
    "단, like_ratio 상위 25% 영상의 바이럴 비율은 **20.1%** 로 전체 평균(25.0%)보다 오히려 낮습니다. "
    "좋아요 비율이 높다고 무조건 바이럴로 이어지지 않으며, 절대적인 조회수·좋아요 수가 더 중요한 신호입니다."
)
 
# ════════════════════════════════════════════════════════════
# 그래프 3 — 카테고리 × 바이럴 (Box plot)
# ════════════════════════════════════════════════════════════
st.header("그래프 3 — 카테고리별 조회수 분포")
 
cat_order = (
    df.groupby("category_name")["log_views"]
    .median()
    .sort_values(ascending=False)
    .index.tolist()
)
fig3 = px.box(
    df, x="log_views", y="category_name",
    color="is_viral",
    color_discrete_map={0: "#636EFA", 1: "#EF553B"},
    labels={"is_viral": "바이럴", "log_views": "log(1+views)", "category_name": "카테고리"},
    category_orders={"category_name": cat_order},
    title="카테고리별 log(views) 분포 — 바이럴 여부 비교",
)
fig3.update_layout(height=550)
st.plotly_chart(fig3, use_container_width=True)
st.caption(
    "📌 **해석:** **Music** 카테고리는 바이럴 IQR이 비바이럴보다 현저히 오른쪽에 위치하며, "
    "바이럴 비율 **53.3%** 로 전 카테고리 중 압도적 1위입니다. "
    "반면 **News & Politics**는 바이럴/비바이럴 간 IQR 겹침이 크고 바이럴 비율 **6.8%** 로 최하위 — "
    "단기 소비성 뉴스 콘텐츠의 특성을 반영합니다. 카테고리 피처가 모델에 유효한 이유를 시각적으로 확인할 수 있습니다."
)
 
# ════════════════════════════════════════════════════════════
# 그래프 4 — 요일 × 시간대 Heatmap
# ════════════════════════════════════════════════════════════
st.header("그래프 4 — 업로드 시간대 × 요일 바이럴 비율")
 
# time_frame에서 시작 시각(정수)만 추출
df["hour"] = df["time_frame"].str.extract(r"(\d+):").astype(float)
 
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
heat_data = (
    df.groupby(["published_day_of_week", "hour"])["is_viral"]
    .mean()
    .reset_index()
    .rename(columns={"is_viral": "바이럴비율", "published_day_of_week": "요일"})
)
heat_pivot = heat_data.pivot(index="요일", columns="hour", values="바이럴비율").reindex(dow_order)
 
fig4 = px.imshow(
    heat_pivot,
    color_continuous_scale="YlOrRd",
    labels={"color": "바이럴 비율", "x": "업로드 시각 (hour)", "y": "요일"},
    title="요일 × 업로드 시각별 바이럴 비율",
    aspect="auto",
    text_auto=".2f",
)
st.plotly_chart(fig4, use_container_width=True)
st.caption(
    "📌 **해석:** **금요일 새벽 4시(UTC)** 조합이 바이럴 비율 **63.3%** 로 전체 최고점입니다. "
    "금요일·목요일 새벽 4~5시대(UTC)가 집중적으로 높은데, 이는 미국·영국 기준 전날 저녁 업로드와 일치합니다. "
    "반면 **토요일 오전 9~10시(UTC)** 는 1.1~1.2%로 최저 — 주말 낮 시간대는 경쟁이 몰리고 알고리즘 노출이 분산됩니다."
)
 
# ════════════════════════════════════════════════════════════
# 그래프 5 — 태그 수 × 제목 길이 × 바이럴
# ════════════════════════════════════════════════════════════
st.header("그래프 5 — 태그 수 & 제목 길이와 바이럴의 관계")
 
c_l, c_r = st.columns(2)
 
with c_l:
    fig5a = px.box(
        df, x="is_viral", y="tag_count",
        color="is_viral",
        color_discrete_map={0: "#636EFA", 1: "#EF553B"},
        labels={"is_viral": "바이럴", "tag_count": "태그 수"},
        title="바이럴 여부별 태그 수",
    )
    fig5a.update_layout(showlegend=False)
    st.plotly_chart(fig5a, use_container_width=True)
    st.caption(
        "📌 **해석:** 바이럴 영상의 태그 수 중앙값은 **19개**, 비바이럴은 **15개**로 약 4개 차이가 납니다. "
        "차이가 크지는 않지만 태그를 충분히 채우는 습관이 바이럴에 소폭 유리하게 작용합니다."
    )
 
with c_r:
    fig5b = px.box(
        df, x="is_viral", y="title_len",
        color="is_viral",
        color_discrete_map={0: "#636EFA", 1: "#EF553B"},
        labels={"is_viral": "바이럴", "title_len": "제목 길이(글자)"},
        title="바이럴 여부별 제목 길이",
    )
    fig5b.update_layout(showlegend=False)
    st.plotly_chart(fig5b, use_container_width=True)
    st.caption(
        "📌 **해석:** 제목 길이 중앙값이 바이럴 **47자**, 비바이럴 **51자**로 오히려 바이럴이 약간 짧습니다. "
        "IQR 범위도 거의 겹쳐 단독 피처로서의 예측력은 낮습니다. "
        "제목 길이보다는 키워드·감정 표현 등 *내용* 이 더 중요한 신호입니다."
    )