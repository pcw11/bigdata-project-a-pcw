# src/features.py — 피처 엔지니어링
import pandas as pd
import numpy as np

# 모델에 사용할 피처 목록 (전역 참조용)
FEATURE_COLS = [
    "log_views_proxy",   # like + comment 기반 proxy (예측 시 views 없음)
    "like_ratio",
    "tag_count",
    "title_len",
    "days_to_trending",
    "category_id",
    "is_weekend",
    "publish_hour",
    "comments_disabled",
    "ratings_disabled",
]

# 예측 입력용 피처 (views 계열 제외 — 실제 서비스에서 사전 입력 가능한 것만)
INPUT_FEATURE_COLS = [
    "like_ratio",
    "tag_count",
    "title_len",
    "category_id",
    "is_weekend",
    "publish_hour",
    "comments_disabled",
    "ratings_disabled",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """data_loader.load_data() 결과를 받아 모델용 피처를 추가한다."""
    df = df.copy()

    # 주말 여부
    weekend = {"Saturday", "Sunday"}
    df["is_weekend"] = df["published_day_of_week"].isin(weekend).astype(int)

    # 업로드 시각 (정수)
    df["publish_hour"] = df["time_frame"].str.extract(r"(\d+):").astype(float)

    # views 없이 사용할 수 있는 engagement proxy (likes + comments 합산 log)
    df["log_views_proxy"] = np.log1p(df["likes"] + df["comment_count"])

    # bool → int
    for col in ["comments_disabled", "ratings_disabled"]:
        df[col] = df[col].astype(int)

    return df


def get_X_y(df: pd.DataFrame, feature_cols: list[str] = FEATURE_COLS):
    df = add_features(df)
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df["is_viral"]
    return X, y
