from __future__ import annotations

import streamlit as st


def notify(message: str, level: str = "info") -> None:
    level = level.lower()
    if level == "success":
        st.success(message)
    elif level == "error":
        st.error(message)
    else:
        st.info(message)
