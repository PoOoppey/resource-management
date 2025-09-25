from __future__ import annotations

import streamlit as st

from services.data_loader import initialize_session_state
from sections import (
    coverage_dashboard,
    data_management,
    expertise_dashboard,
    scenario_planner,
)

SECTIONS = {
    "Data management": data_management.render,
    "Coverage dashboard": coverage_dashboard.render,
    "Scenario planner": scenario_planner.render,
    "Expertise dashboard": expertise_dashboard.render,
}


def main() -> None:
    st.set_page_config(page_title="Resource Allocation Dashboard", layout="wide")
    initialize_session_state()

    st.sidebar.title("Navigation")
    selected_label = st.sidebar.selectbox(
        "Select a view",
        options=list(SECTIONS.keys()),
        index=0,
        help="Switch between the data management tools, coverage analytics, scenarios, and expertise overview.",
    )
    render_section = SECTIONS[selected_label]
    render_section()


if __name__ == "__main__":
    main()
