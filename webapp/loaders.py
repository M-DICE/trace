import pickle
from pathlib import Path

import streamlit as st


@st.cache_data(show_spinner=False)
def load_results(path: Path, mtime: float) -> dict | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
