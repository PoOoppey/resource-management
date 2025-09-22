from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from services.coverage import (
    compute_attendance_impact_details,
    compute_live_coverage,
    compute_theoretical_coverage,
    weeks_in_range,
)
from services.data_loader import get_data


VIEW_OPTIONS = {"Theoretical": "theoretical", "Live": "live"}
DISPLAY_OPTIONS = {"Role": "role", "Process": "process"}
GROUP_OPTIONS = {"Office": "office", "Region": "region"}
UNIT_OPTIONS = {"Hours": "hours", "FTE": "fte"}
DISPLAY_MODES = {"Coverage": "coverage", "JIRA": "jira", "Coverage + JIRA": "coverage+jira"}


def _default_date_range() -> tuple[date, date]:
    today = date.today()
    start = today - timedelta(days=today.weekday())
    end = start + timedelta(days=28)
    return start, end


def _style_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df


def main():
    st.title("Dashboard")
    data = get_data()

    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            view = st.radio("View", list(VIEW_OPTIONS.keys()), horizontal=True)
        with col2:
            display = st.radio("Data display", list(DISPLAY_OPTIONS.keys()), horizontal=True)
        with col3:
            group_by = st.radio("Group by", list(GROUP_OPTIONS.keys()), horizontal=True)

        unit = st.radio("Units", list(UNIT_OPTIONS.keys()), horizontal=True)
        search_term = st.text_input("Search")

        display_mode = "coverage"
        date_range = _default_date_range()
        if VIEW_OPTIONS[view] == "live":
            start, end = st.date_input("Date range", value=date_range, help="Only applicable to live view")
            display_mode = st.selectbox("Display mode", list(DISPLAY_MODES.keys()))
            date_range = (start, end)

    unit_value = UNIT_OPTIONS[unit]

    if VIEW_OPTIONS[view] == "theoretical":
        df = compute_theoretical_coverage(
            data,
            view=DISPLAY_OPTIONS[display],
            group_by=GROUP_OPTIONS[group_by],
            unit=unit_value,
        )
    else:
        df = compute_live_coverage(
            data,
            attendance=data.get("attendances", []),
            date_range=date_range,
            display_mode=DISPLAY_MODES.get(display_mode, "coverage"),
            group_by=GROUP_OPTIONS[group_by],
            unit=unit_value,
        )

    if search_term:
        mask = pd.Series([False] * len(df))
        for column in df.columns:
            mask = mask | df[column].astype(str).str.contains(search_term, case=False)
        df = df[mask]

    st.dataframe(_style_dataframe(df), use_container_width=True)

    if VIEW_OPTIONS[view] == "live":
        weeks = weeks_in_range(date_range)
        attendance_details = pd.DataFrame()
        if weeks:
            selected_week = st.selectbox(
                "Week",
                weeks,
                format_func=lambda d: d.strftime("%Y-W%W"),
                help="Select a week to review attendance impact details.",
            )
            attendance_details = compute_attendance_impact_details(
                data,
                attendance=data.get("attendances", []),
                week_start=selected_week,
                group_by=GROUP_OPTIONS[group_by],
                unit=unit_value,
            )

        st.subheader("Attendance impact details")
        if attendance_details.empty:
            st.info("No attendance impact recorded for the selected parameters.")
        else:
            st.dataframe(attendance_details, use_container_width=True)


if __name__ == "__main__":
    main()
