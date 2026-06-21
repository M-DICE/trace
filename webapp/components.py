import streamlit as st


def render_run_navigator(
    n_runs: int,
    idx_key: str,
    slider_key: str,
    prev_key: str,
    next_key: str,
) -> int:
    if idx_key not in st.session_state:
        st.session_state[idx_key] = 0
    if slider_key not in st.session_state:
        st.session_state[slider_key] = 1
    run_i = min(st.session_state[idx_key], max(n_runs - 1, 0))
    st.session_state[idx_key] = run_i
    st.session_state[slider_key] = min(st.session_state[slider_key], max(n_runs, 1))

    def _prev():
        new = max(0, st.session_state[idx_key] - 1)
        st.session_state[idx_key] = new
        st.session_state[slider_key] = new + 1

    def _next():
        new = min(n_runs - 1, st.session_state[idx_key] + 1)
        st.session_state[idx_key] = new
        st.session_state[slider_key] = new + 1

    def _on_slider():
        st.session_state[idx_key] = st.session_state[slider_key] - 1

    col1, col2, col3 = st.columns([1, 8, 1])
    with col1:
        st.button("◀", on_click=_prev, key=prev_key, width="stretch")
    with col2:
        st.slider("Run", 1, max(n_runs, 1), key=slider_key, on_change=_on_slider)
    with col3:
        st.button("▶", on_click=_next, key=next_key, width="stretch")

    return st.session_state[idx_key]
