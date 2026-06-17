# app.py — 프로젝트 진입점
# 실행: streamlit run app.py
import streamlit as st

st.set_page_config(
    page_title="YouTube 바이럴 예측",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

eda     = st.Page("../pages/1_EDA.py",         title="EDA",         icon="📊", default=True)
viz     = st.Page("../pages/2_시각화.py",      title="시각화",       icon="📈")
service = st.Page("../pages/3_모델_서비스.py", title="모델·서비스",  icon="🤖")

pg = st.navigation({
    "프로젝트": [eda, viz, service],
})

st.sidebar.markdown("### 🎬 YouTube 바이럴 예측")
st.sidebar.caption("빅데이터분석프로그래밍 기말 프로젝트")
st.sidebar.markdown("---")
st.sidebar.caption("이름: 박채원 / 학번 : 20241487")

pg.run()
