# pages/3_모델_서비스.py — 3차 작업: 입력 → 예측 → 결과
# 경로 A) 표 데이터 분류 — scikit-learn + XGBoost
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve,
)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

from src.data_loader import load_data
from src.features import add_features, get_X_y, FEATURE_COLS, INPUT_FEATURE_COLS

st.title("🤖 모델 · 서비스")

# ════════════════════════════════════════════════════════════
# 모델 학습 (캐시)
# ════════════════════════════════════════════════════════════
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=200, random_state=42,
                              eval_metric="logloss", verbosity=0),
}


@st.cache_resource(show_spinner="모델 학습 중… (최초 1회)")
def train_all():
    df = load_data()
    X, y = get_X_y(df, FEATURE_COLS)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}
    for name, clf in MODELS.items():
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        prob = clf.predict_proba(Xte)[:, 1]
        results[name] = {
            "model": clf,
            "accuracy": accuracy_score(yte, pred),
            "f1": f1_score(yte, pred, average="weighted"),
            "roc_auc": roc_auc_score(yte, prob),
            "report": classification_report(yte, pred, target_names=["비바이럴", "바이럴"]),
            "cm": confusion_matrix(yte, pred),
            "fpr": roc_curve(yte, prob)[0],
            "tpr": roc_curve(yte, prob)[1],
            "y_test": yte.values,
            "prob": prob,
        }

    # 피처 중요도 (Random Forest)
    rf = results["Random Forest"]["model"]
    feat_imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    return results, Xtr, Xte, ytr, yte, feat_imp


results, Xtr, Xte, ytr, yte, feat_imp = train_all()

# ════════════════════════════════════════════════════════════
# 1. 모델 성능 비교
# ════════════════════════════════════════════════════════════
st.header("1. 모델 성능 비교")

metrics_df = pd.DataFrame(
    {name: {"정확도": r["accuracy"], "F1 (weighted)": r["f1"], "ROC-AUC": r["roc_auc"]}
     for name, r in results.items()}
).T.round(4)
st.dataframe(metrics_df.style.highlight_max(axis=0, color="#c6f5c6"), use_container_width=True)

# 모델 선택
selected = st.selectbox("상세 분석할 모델", list(results.keys()), index=1)
res = results[selected]

c1, c2, c3 = st.columns(3)
c1.metric("정확도", f"{res['accuracy']*100:.2f}%")
c2.metric("F1 (weighted)", f"{res['f1']*100:.2f}%")
c3.metric("ROC-AUC", f"{res['roc_auc']:.4f}")

# ════════════════════════════════════════════════════════════
# 2. Confusion Matrix & ROC Curve
# ════════════════════════════════════════════════════════════
st.header("2. 오차 행렬 & ROC 곡선")

col_cm, col_roc = st.columns(2)

with col_cm:
    cm = res["cm"]
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        x=["예측: 비바이럴", "예측: 바이럴"],
        y=["실제: 비바이럴", "실제: 바이럴"],
        color_continuous_scale="Blues",
        title="Confusion Matrix",
    )
    fig_cm.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_cm, use_container_width=True)

with col_roc:
    fig_roc = go.Figure()
    for name, r in results.items():
        fig_roc.add_trace(go.Scatter(
            x=r["fpr"], y=r["tpr"],
            name=f"{name} (AUC={r['roc_auc']:.3f})",
            mode="lines",
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray"), name="Random"
    ))
    fig_roc.update_layout(
        title="ROC Curve (전체 모델 비교)",
        xaxis_title="FPR", yaxis_title="TPR",
        legend=dict(x=0.55, y=0.1),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

with st.expander("📋 상세 Classification Report"):
    st.text(res["report"])

# ════════════════════════════════════════════════════════════
# 3. 피처 중요도 (Random Forest)
# ════════════════════════════════════════════════════════════
st.header("3. 피처 중요도 (Random Forest)")

fig_imp = px.bar(
    x=feat_imp.values,
    y=feat_imp.index,
    orientation="h",
    labels={"x": "중요도", "y": "피처"},
    color=feat_imp.values,
    color_continuous_scale="Teal",
    title="Feature Importance",
)
fig_imp.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
st.plotly_chart(fig_imp, use_container_width=True)

# ════════════════════════════════════════════════════════════
# 4. 예측해 보기
# ════════════════════════════════════════════════════════════
st.header("4. 바이럴 가능성 예측해 보기")
st.markdown("업로드 전 영상 정보를 입력하면 바이럴 확률을 예측합니다.")

df_ref = load_data()

with st.form("predict_form"):
    col_l, col_r = st.columns(2)

    with col_l:
        category_id = st.selectbox(
            "카테고리",
            options=sorted(df_ref["category_id"].unique()),
            format_func=lambda x: f"{x} — {df_ref[df_ref['category_id']==x]['category_name'].iloc[0]}",
        )
        tag_count = st.slider("태그 수", 0, 50, 10)
        title_len = st.slider("제목 길이 (글자 수)", 10, 100, 40)
        publish_hour = st.slider("업로드 예정 시각 (hour)", 0, 23, 16)

    with col_r:
        like_ratio = st.slider("예상 좋아요 비율 (likes / reactions)", 0.0, 1.0, 0.95, step=0.01)
        is_weekend = st.radio("업로드 요일", ["평일", "주말"], horizontal=True) == "주말"
        comments_disabled = st.checkbox("댓글 비활성화")
        ratings_disabled = st.checkbox("좋아요 비활성화")

    submitted = st.form_submit_button("🔮 바이럴 가능성 예측", use_container_width=True)

if submitted:
    # INPUT_FEATURE_COLS 순서에 맞춰 입력
    user_input = pd.DataFrame([{
        "like_ratio": like_ratio,
        "tag_count": tag_count,
        "title_len": title_len,
        "category_id": category_id,
        "is_weekend": int(is_weekend),
        "publish_hour": publish_hour,
        "comments_disabled": int(comments_disabled),
        "ratings_disabled": int(ratings_disabled),
    }])

    # FEATURE_COLS 기준으로 학습됐으므로 누락 컬럼은 중앙값으로 채움
    df_aug = load_data()
    X_all, _ = get_X_y(df_aug, FEATURE_COLS)
    medians = X_all.median()

    full_input = pd.DataFrame([medians], columns=FEATURE_COLS)
    for col in INPUT_FEATURE_COLS:
        full_input[col] = user_input[col].values[0]

    best_model = results["XGBoost"]["model"]
    prob = best_model.predict_proba(full_input)[0][1]
    pred = int(prob >= 0.5)

    st.divider()
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        if pred == 1:
            st.success(f"### 🔥 바이럴 가능성 높음")
        else:
            st.warning(f"### 📉 바이럴 가능성 낮음")
        st.metric("예측 확률", f"{prob*100:.1f}%")

    with res_col2:
        # 게이지 차트
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            title={"text": "바이럴 확률"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#EF553B" if pred == 1 else "#636EFA"},
                "steps": [
                    {"range": [0, 50], "color": "#e8eaf6"},
                    {"range": [50, 75], "color": "#ffecb3"},
                    {"range": [75, 100], "color": "#ffcdd2"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)