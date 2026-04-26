"""
SimRewilding — Glossary of Statistical Terms
Educational reference page explaining key terms for non-statisticians.
"""

import streamlit as st
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Glossary",
    page_icon="📚",
    layout="wide",
)

# ── Load glossary markdown ──────────────────────────────────────────────────────
glossary_path = Path(__file__).parent / "GLOSSARY_FOR_REWILDING.md"

if not glossary_path.exists():
    st.error(
        f"Glossary file not found at `{glossary_path}`. "
        "Please ensure GLOSSARY_FOR_REWILDING.md exists in the reviews/ directory."
    )
    st.stop()

with open(glossary_path, "r") as f:
    glossary_content = f.read()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📚 Glossary of Terms")
st.caption("A non-technical guide to statistical concepts in rewilding analysis")

# ── Display glossary ──────────────────────────────────────────────────────────
st.markdown(glossary_content)
