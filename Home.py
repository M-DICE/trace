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

    st.markdown("""
    ### Distribution Change Detection (AMOC)
    Detect whether an intervention caused a **shift in the distribution** (mean or variance)
    of an ecological indicator, using distance-based AMOC detection.
    """)
    st.page_link("pages/02_Distribution_Change_AMOC.py", label="Distribution Change (AMOC)", icon=":material/open_in_new:")

with col2:
    st.markdown("""
    ### Trend Change Detection (Forecast)
    Detect whether an intervention caused a **change in trend** by comparing observed data
    against a **pre-intervention forecast**, using Page-CUSUM analysis.
    """)
    st.page_link("pages/03_Trend_Change_Forecast.py", label="Trend Change (Forecast)", icon=":material/open_in_new:")

    st.markdown("""
    ### Coming next
    - **Trend Change (BOCPD)** — online Bayesian changepoint detection
    - **Distribution Change (BOCPD)** — online variant for distributional shifts
    """)

st.divider()

st.markdown("""
    ### Glossary
    Frequently used terms and statistical concepts.
    """)
st.page_link("pages/99_Glossary.py", label="Glossary", icon=":material/open_in_new:")
