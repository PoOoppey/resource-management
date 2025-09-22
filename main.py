from __future__ import annotations

import streamlit as st

from services.data_loader import initialize_session_state


def main() -> None:
    st.set_page_config(page_title="Resource Allocation Dashboard", layout="wide")
    initialize_session_state()
    st.title("Resource Allocation Dashboard")
    st.write(
        "Use the navigation sidebar to manage data, explore coverage dashboards, and evaluate scenarios."
    )
    st.caption("Data is loaded from local JSON fixtures on startup.")


if __name__ == "__main__":
    main()
