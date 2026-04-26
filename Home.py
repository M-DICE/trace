import streamlit as st

st.set_page_config(
    page_title="TRACE",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌱 TRACE")
st.subheader("Temporal Rewilding Analysis for Changepoint Estimation")

st.markdown("""
Rewilding requires robust methods to detect
whether an intervention is actually working. This app provides **interactive simulations**
and **statistical analyses** for evaluating the detectability of ecological change after an
intervention has been applied.
""")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    ### Trend Change Detection (AMOC)
    Detect whether a rewilding intervention caused a **change in the trend** (slope) of an
    ecological time series, using the **At Most One Change (AMOC)** offline changepoint method.
    """)
    st.page_link("pages/01_Trend_Change_AMOC.py", label="Trend Change (AMOC)", icon=":material/open_in_new:")

with col2:
    st.markdown("""
    ### Coming next
    - **Distribution Change (AMOC)** — detect shifts in the distribution of an indicator
    - **Trend Change (BOCPD)** — online Bayesian changepoint detection
    - **Distribution Change (BOCPD)** — online variant for distributional shifts
    """)
