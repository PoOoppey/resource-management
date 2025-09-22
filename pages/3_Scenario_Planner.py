from __future__ import annotations

import streamlit as st

from models import Scenario
from services.coverage import compute_theoretical_coverage
from services.data_loader import get_data
from services.scenario import apply_scenario, compare_scenario


def _scenario_select(scenarios: list[Scenario]) -> Scenario | None:
    if not scenarios:
        st.info("No scenarios available")
        return None
    names = {scenario.name: scenario for scenario in scenarios}
    choice = st.selectbox("Scenario", list(names.keys()))
    return names.get(choice)


def main():
    st.title("Scenario Planner")
    data = get_data()

    baseline_coverage = compute_theoretical_coverage(
        data,
        view="process",
        group_by="office",
        unit="hours",
    )

    scenario = _scenario_select(data.get("scenarios", []))
    if scenario is None:
        st.dataframe(baseline_coverage)
        return

    st.subheader("Scenario Adjustments")
    for adjustment in scenario.adjustments:
        st.write(f"**{adjustment.type.value}**: {adjustment.payload}")

    modified_data = apply_scenario(data, scenario)
    scenario_coverage = compute_theoretical_coverage(
        modified_data,
        view="process",
        group_by="office",
        unit="hours",
    )

    comparison = compare_scenario(baseline_coverage, scenario_coverage)

    st.subheader("Comparison")
    st.dataframe(comparison, use_container_width=True)


if __name__ == "__main__":
    main()
