import pandas as pd
import numpy as np
import os
import streamlit as st

DATA_PATH = r"C:\Users\6-112\Desktop\빅데이터분석프로그래밍\bigdata-project-a-pcw\bigdata-project-a-pcw11\my project\data\youtube.csv"

CATEGORY_MAP = {
    1: "Film & Animation",
    2: "Autos & Vehicles",
    10: "Music",
    15: "Pets & Animals",
    17: "Sports",
    18: "Short Movies",
    19: "Travel & Events",
    20: "Gaming",
    21: "Videoblogging",
    22: "People & Blogs",
    23: "Comedy",
    24: "Entertainment",
    25: "News & Politics",
    26: "Howto & Style",
    27: "Education",
    28: "Science & Technology",
    29: "Nonprofits & Activism",
}

@st.cache_data
def load_data(raw: bool = False) -> pd.DataFrame:
    """
    raw=True  → 원본 그대로 반환
    raw=False → 기본 타입 변환 + 파생 컬럼 추가
    """
    df = pd.read_csv(DATA_PATH)

    if raw:
        return df

    # ── 날짜 파싱 ──────────────────────────────────────────
    df["publish_date"] = pd.to_datetime(df["publish_date"], dayfirst=True, errors="coerce")
    df["trending_date"] = pd.to_datetime(
        df["trending_date"], format="%y.%d.%m", errors="coerce"
    )

    # ── 파생 컬럼 ──────────────────────────────────────────
    # 업로드 후 트렌딩까지 걸린 일수
    df["days_to_trending"] = (df["trending_date"] - df["publish_date"]).dt.days.clip(lower=0)

    # 조회수 log 변환 (시각화·모델링용)
    df["log_views"] = np.log1p(df["views"])

    # 카테고리 이름
    df["category_name"] = df["category_id"].map(CATEGORY_MAP).fillna("Unknown")

    # 좋아요 비율 (likes / (likes + dislikes))
    total_reactions = df["likes"] + df["dislikes"]
    df["like_ratio"] = np.where(total_reactions > 0, df["likes"] / total_reactions, np.nan)

    # 태그 개수
    df["tag_count"] = df["tags"].apply(
        lambda x: 0 if pd.isna(x) or x == "[none]" else len(str(x).split("|"))
    )

    # 제목 길이
    df["title_len"] = df["title"].str.len()

    # ── 바이럴 레이블 (75th percentile 기준) ──────────────
    threshold = df["views"].quantile(0.75)
    df["is_viral"] = (df["views"] >= threshold).astype(int)

    return df


def get_numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def get_text_cols(df: pd.DataFrame) -> list[str]:
    return ["title", "tags", "channel_title"]
