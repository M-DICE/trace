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

The analyses come in two flavours: **Trend Change** detection asks whether the *slope* of an
indicator shifted (e.g. a population that starts growing faster), while **Distribution Change**
detection asks whether the *shape* of the indicator's distribution shifted (e.g. a change in
its mean or variability), capturing effects that a trend test alone would miss.
""")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    ### Trend Change Detection (AMOC)
    **Offline** method: scans the *entire* series after the study to locate the single most
    likely slope change (**At Most One Change**).
    """)
    st.page_link(
        "pages/01_Trend_Change_AMOC.py", label="Trend Change (AMOC)", icon=":material/open_in_new:"
    )

    st.markdown("""
    ### Trend Change Detection (Forecast)
    **Online** method: watches the slope unfold against a **pre-intervention forecast** and
    raises an alarm when it diverges (**Page-CUSUM**).
    """)
    st.page_link(
        "pages/03_Trend_Change_Forecast.py",
        label="Trend Change (Forecast)",
        icon=":material/open_in_new:",
    )

    st.markdown("""
    ### Trend Change Detection (BOCPD)
    **Online** method: processes one observation at a time, flagging a slope change as soon as
    the evidence appears, with no pre-computed threshold (**BOCPD**).
    """)
    st.page_link(
        "pages/05_Trend_Change_BOCPD.py",
        label="Trend Change (BOCPD)",
        icon=":material/open_in_new:",
    )

with col2:
    st.markdown("""
       ### Distribution Change Detection (AMOC)
       **Offline** counterpart: scans the *whole* series for the single most likely shift in
       the distribution's shape (mean *or* variance).
       """)
    st.page_link(
        "pages/02_Distribution_Change_AMOC.py",
        label="Distribution Change (AMOC)",
        icon=":material/open_in_new:",
    )

    st.markdown("""
    ### Distribution Change Detection (Forecast)
    **Online** counterpart: monitors the distribution against a **pre-intervention forecast**
    and alarms when its shape diverges (**Page-CUSUM**).
    """)
    st.page_link(
        "pages/04_Distribution_Change_Forecast.py",
        label="Distribution Change (Forecast)",
        icon=":material/open_in_new:",
    )

    st.markdown("""
    ### Distribution Change Detection (BOCPD)
    **Online** counterpart: flags a distribution shift one observation at a time from the
    **Wasserstein distance** series, with no pre-computed threshold (**BOCPD**).
    """)
    st.page_link(
        "pages/06_Distribution_Change_BOCPD.py",
        label="Distribution Change (BOCPD)",
        icon=":material/open_in_new:",
    )

st.divider()

st.markdown("""
    ### Glossary
    Frequently used terms and statistical concepts.
    """)
st.page_link("pages/99_Glossary.py", label="Glossary", icon=":material/open_in_new:")
