# pages/3_모델_서비스.py — 3차 작업: 입력 → 예측 → 결과
# 경로 A) 표 데이터 분류 — scikit-learn + XGBoost
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve,
)
from xgboost import XGBClassifier

from src.data_loader import load_data, CATEGORY_MAP
from src.features import add_features, get_X_y, FEATURE_COLS, INPUT_FEATURE_COLS

st.title("🤖 모델 · 서비스")

# ════════════════════════════════════════════════════════════
# EDA에서 발견한 클래스 불균형(25:75) 반영
#   - XGBoost: scale_pos_weight = 비바이럴 수 / 바이럴 수 ≈ 3
#   - RandomForest: class_weight="balanced"
#   - LogisticRegression: class_weight="balanced"
# ════════════════════════════════════════════════════════════
MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, random_state=42,
        eval_metric="logloss", verbosity=0,
        scale_pos_weight=3,   # 비바이럴:바이럴 ≈ 3:1 (EDA 섹션 1 확인)
    ),
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
            "prob": prob,
        }

    rf = results["Random Forest"]["model"]
    feat_imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)

    # 조회수 구간 통계 (예상 조회수 범위 출력용)
    view_stats = df.groupby("is_viral")["views"].describe()

    return results, feat_imp, view_stats


results, feat_imp, view_stats = train_all()

# ════════════════════════════════════════════════════════════
# 1. 모델 성능 비교
# ════════════════════════════════════════════════════════════
st.header("1. 모델 성능 비교")
st.caption("EDA에서 확인한 클래스 불균형(25%:75%)을 보정하여 학습했습니다.")

metrics_df = pd.DataFrame(
    {name: {"정확도": r["accuracy"], "F1 (weighted)": r["f1"], "ROC-AUC": r["roc_auc"]}
     for name, r in results.items()}
).T.round(4)
st.dataframe(metrics_df.style.highlight_max(axis=0, color="#c6f5c6"), use_container_width=True)

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
    fig_cm = px.imshow(
        res["cm"], text_auto=True,
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
st.caption(
    "EDA 결과: like_ratio 단독 예측력은 낮으나(상위 25% 바이럴 비율 20.1%), "
    "다른 피처와 결합 시 기여도가 있습니다."
)

fig_imp = px.bar(
    x=feat_imp.values, y=feat_imp.index,
    orientation="h",
    labels={"x": "중요도", "y": "피처"},
    color=feat_imp.values, color_continuous_scale="Teal",
    title="Feature Importance",
)
fig_imp.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
st.plotly_chart(fig_imp, use_container_width=True)

# ════════════════════════════════════════════════════════════
# 4. 바이럴 가능성 예측해 보기
# ════════════════════════════════════════════════════════════
st.header("4. 바이럴 가능성 예측해 보기")
st.markdown("업로드 **전** 영상 메타데이터를 입력하면 AI가 바이럴 가능성과 예상 조회수 범위를 분석합니다.")

df_ref = load_data()

# 카테고리별 바이럴 비율 사전 계산 (피드백용)
cat_viral_rate = df_ref.groupby("category_name")["is_viral"].mean()
# 요일별 바이럴 비율 사전 계산 (피드백용)
dow_viral_rate = df_ref.groupby("published_day_of_week")["is_viral"].mean()

with st.form("predict_form"):
    col_l, col_r = st.columns(2)

    with col_l:
        category_id = st.selectbox(
            "카테고리",
            options=sorted(df_ref["category_id"].unique()),
            format_func=lambda x: f"{x} — {df_ref[df_ref['category_id']==x]['category_name'].iloc[0]}",
        )
        tag_count = st.slider("태그 수", 0, 50, 10,
                              help="EDA: 바이럴 영상 평균 19개, 비바이럴 15개")
        title_len = st.slider("제목 길이 (글자 수)", 10, 100, 40,
                              help="EDA: 바이럴 평균 47자, 비바이럴 51자 — 짧을수록 소폭 유리")
        publish_hour = st.slider("업로드 예정 시각 (UTC hour)", 0, 23, 4,
                                 help="EDA: 새벽 4~5시(UTC)가 바이럴 비율 최고")

    with col_r:
        like_ratio = st.slider("예상 좋아요 비율", 0.0, 1.0, 0.95, step=0.01,
                               help="EDA: 단독 신호로는 약함 — 절대 조회수·카테고리가 더 중요")
        day_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_labels  = ["월요일", "화요일", "수요일", "목요일", "금요일(추천)", "토요일", "일요일"]
        publish_day = st.selectbox("업로드 요일", day_options,
                                   format_func=lambda x: day_labels[day_options.index(x)],
                                   index=4,
                                   help="EDA: 금요일(29.6%)·목요일(28.7%)이 바이럴 비율 최고")
        is_weekend = publish_day in ["Saturday", "Sunday"]
        comments_disabled = st.checkbox("댓글 비활성화")
        ratings_disabled  = st.checkbox("좋아요 비활성화")

    submitted = st.form_submit_button("🔮 바이럴 가능성 예측", use_container_width=True)

if submitted:
    user_input = pd.DataFrame([{
        "like_ratio":          like_ratio,
        "tag_count":           tag_count,
        "title_len":           title_len,
        "category_id":         category_id,
        "is_weekend":          int(is_weekend),
        "publish_hour":        publish_hour,
        "comments_disabled":   int(comments_disabled),
        "ratings_disabled":    int(ratings_disabled),
    }])

    X_all, _ = get_X_y(df_ref, FEATURE_COLS)
    medians = X_all.median()
    full_input = pd.DataFrame([medians], columns=FEATURE_COLS)
    for col in INPUT_FEATURE_COLS:
        full_input[col] = user_input[col].values[0]

    best_model = results["XGBoost"]["model"]
    prob = best_model.predict_proba(full_input)[0][1]
    pred = int(prob >= 0.5)

    st.divider()

    # ── 결과 레이아웃 ──────────────────────────────────────
    r1, r2 = st.columns([1, 2])

    with r1:
        if prob >= 0.75:
            st.success("### 🔥 바이럴 가능성 매우 높음")
        elif prob >= 0.5:
            st.success("### ✅ 바이럴 가능성 높음")
        elif prob >= 0.3:
            st.warning("### ⚠️ 바이럴 가능성 보통")
        else:
            st.error("### 📉 바이럴 가능성 낮음")

        st.metric("바이럴 확률", f"{prob*100:.1f}%")

        # 예상 조회수 범위
        if pred == 1:
            q25 = int(view_stats.loc[1, "25%"])
            q75 = int(view_stats.loc[1, "75%"])
            st.metric("예상 조회수 범위", f"{q25//10000}만 ~ {q75//10000}만 뷰")
        else:
            q25 = int(view_stats.loc[0, "25%"])
            q75 = int(view_stats.loc[0, "75%"])
            st.metric("예상 조회수 범위", f"{q25//10000}만 ~ {q75//10000}만 뷰")

    with r2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            title={"text": "바이럴 확률"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#EF553B" if pred == 1 else "#636EFA"},
                "steps": [
                    {"range": [0,  30], "color": "#e8eaf6"},
                    {"range": [30, 50], "color": "#fff9c4"},
                    {"range": [50, 75], "color": "#ffecb3"},
                    {"range": [75, 100], "color": "#ffcdd2"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75, "value": 50,
                },
            },
        ))
        fig_gauge.update_layout(height=260, margin=dict(t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ════════════════════════════════════════════════════════════
    # 5. 결과 기반 피드백
    # ════════════════════════════════════════════════════════════
    st.subheader("📋 AI 분석 피드백")

    cat_name = df_ref[df_ref["category_id"] == category_id]["category_name"].iloc[0]
    cat_rate = cat_viral_rate.get(cat_name, 0.25)
    dow_rate = dow_viral_rate.get(publish_day, 0.25)

    # ── 종합 한 줄 코멘트 ──
    if prob >= 0.75:
        summary = f"이 영상은 바이럴 확률이 **{prob*100:.1f}%** 로 매우 높습니다. 현재 설정을 유지하세요! 🔥"
    elif prob >= 0.5:
        summary = f"바이럴 확률 **{prob*100:.1f}%** — 가능성이 있습니다. 아래 개선점을 참고하면 더 높일 수 있어요. ✅"
    elif prob >= 0.3:
        summary = f"바이럴 확률 **{prob*100:.1f}%** — 아직 부족합니다. 아래 항목을 개선해보세요. ⚠️"
    else:
        summary = f"바이럴 확률 **{prob*100:.1f}%** — 현재 설정으로는 바이럴이 어렵습니다. 전략 수정이 필요합니다. 📉"

    st.markdown(f"> {summary}")
    st.divider()

    # ── 항목별 피드백 수집 ──
    feedbacks = []

    # 카테고리
    if cat_rate >= 0.4:
        feedbacks.append(("✅", f"**카테고리**: {cat_name}은 바이럴 비율 {cat_rate*100:.1f}%로 매우 유리한 카테고리입니다."))
    elif cat_rate >= 0.25:
        feedbacks.append(("➡️", f"**카테고리**: {cat_name}은 바이럴 비율 {cat_rate*100:.1f}%로 평균 수준입니다."))
    else:
        feedbacks.append(("⚠️", f"**카테고리**: {cat_name}은 바이럴 비율 {cat_rate*100:.1f}%로 불리합니다. 카테고리 변경을 고려해보세요."))

    # 요일
    if dow_rate >= 0.28:
        feedbacks.append(("✅", f"**업로드 요일**: {publish_day}은 바이럴 비율 {dow_rate*100:.1f}%로 좋은 타이밍입니다."))
    elif dow_rate < 0.20:
        feedbacks.append(("⚠️", f"**업로드 요일**: {publish_day}은 바이럴 비율 {dow_rate*100:.1f}%로 낮습니다. 목·금요일(28~29%)을 추천합니다."))
    else:
        feedbacks.append(("➡️", f"**업로드 요일**: {publish_day}은 바이럴 비율 {dow_rate*100:.1f}%로 평균 수준입니다."))

    # 시간대
    if 3 <= publish_hour <= 6:
        feedbacks.append(("✅", f"**업로드 시각**: UTC {publish_hour}시는 바이럴 비율 최상위 시간대입니다."))
    elif 9 <= publish_hour <= 11:
        feedbacks.append(("⚠️", f"**업로드 시각**: UTC {publish_hour}시는 바이럴 비율이 낮습니다. UTC 4~5시를 권장합니다."))
    else:
        feedbacks.append(("➡️", f"**업로드 시각**: UTC {publish_hour}시 — UTC 4~5시 대비 바이럴 효과가 다소 낮을 수 있습니다."))

    # 태그
    if tag_count >= 19:
        feedbacks.append(("✅", f"**태그 수**: {tag_count}개로 바이럴 영상 평균(19개) 이상입니다."))
    else:
        feedbacks.append(("➡️", f"**태그 수**: {tag_count}개 — 19개까지 늘리면 소폭 유리합니다."))

    # 제목 길이
    if title_len <= 50:
        feedbacks.append(("✅", f"**제목 길이**: {title_len}자로 바이럴 평균(47자)에 근접합니다."))
    else:
        feedbacks.append(("⚠️", f"**제목 길이**: {title_len}자 — 바이럴 평균(47자)보다 깁니다. 간결하게 줄여보세요."))

    # 댓글/좋아요
    if comments_disabled or ratings_disabled:
        feedbacks.append(("⚠️", "**댓글/좋아요**: 비활성화 시 알고리즘 engagement 신호가 줄어 바이럴에 불리합니다."))
    else:
        feedbacks.append(("✅", "**댓글/좋아요**: 활성화 상태로 알고리즘에 유리합니다."))

    # ── 그룹별 출력 ──
    good = [(e, m) for e, m in feedbacks if e == "✅"]
    warn = [(e, m) for e, m in feedbacks if e == "⚠️"]
    neut = [(e, m) for e, m in feedbacks if e == "➡️"]

    if good:
        st.markdown("**잘 된 점**")
        for e, m in good:
            st.success(f"{e} {m}")

    if warn:
        st.markdown("**개선이 필요한 점**")
        for e, m in warn:
            st.warning(f"{e} {m}")

    if neut:
        st.markdown("**참고 사항**")
        for e, m in neut:
            st.info(f"{e} {m}")