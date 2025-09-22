from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from pandas.io.formats.style import Styler
import streamlit as st

from services.coverage import (
    compute_attendance_impact_details,
    compute_live_coverage,
    compute_theoretical_coverage,
    format_week_label,
    weeks_in_range,
)
from services.data_loader import get_data


VIEW_OPTIONS = {"Theoretical": "theoretical", "Live": "live"}
GROUP_OPTIONS = {"Office": "office", "Region": "region"}
UNIT_OPTIONS = {"Hours": "hours", "FTE": "fte"}
DATA_DISPLAY = {
    "theoretical": {"Role": "role", "Process": "process"},
    "live": {"Coverage": "coverage", "Coverage + JIRA": "coverage+jira", "JIRA": "jira"},
}

STATIC_COLUMN_STYLE = {"background-color": "#f3f0ff", "font-style": "italic"}


def _default_date_range() -> tuple[date, date]:
    today = date.today()
    start = today - timedelta(days=today.weekday())
    end = start + timedelta(days=28)
    return start, end


def _coerce_numeric(value):
    if isinstance(value, str):
        try:
            # extract leading numeric portion (e.g. "123 (4)")
            numeric = value.strip().split(" ")[0].replace(",", "")
            return float(numeric)
        except (ValueError, IndexError):
            return None
    return value


def _colorize_cell(required: float | None, value) -> str:
    value = _coerce_numeric(value)
    if pd.isna(required) or pd.isna(value):
        return ""
    gap = value - required
    if gap < 0:
        return "background-color: #ffd6d6"
    if gap < 0.1 * required:
        return "background-color: #fff4cc"
    return "background-color: #d6f5d6"


def _style_dataframe(
    df: pd.DataFrame,
    view: str,
    display_mode: str,
) -> Styler | pd.DataFrame:
    if df.empty:
        return df

    styler = df.style

    if view == "live":
        reference_columns = [col for col in ["Required", "Theoretical"] if col in df.columns]
        if reference_columns:
            styler = styler.set_properties(subset=reference_columns, **STATIC_COLUMN_STYLE)

    if view == "theoretical":
        coverage_columns = [col for col in ["Coverage"] if col in df.columns]

        def _style_coverage(row: pd.Series) -> pd.Series:
            required = row.get("Required")
            styles = {}
            for column in coverage_columns:
                styles[column] = _colorize_cell(required, row.get(column))
            return pd.Series(styles)

        if coverage_columns:
            styler = styler.apply(_style_coverage, axis=1, subset=coverage_columns)

    if view == "live" and display_mode.lower() not in {"jira"}:
        week_columns = df.attrs.get("week_columns") or [
            col
            for col in df.columns
            if col not in {"Region", "Office", "Process", "Required", "Theoretical"}
            and pd.api.types.is_numeric_dtype(df[col])
        ]

        def _style_weeks(row: pd.Series) -> pd.Series:
            required = row.get("Required")
            styles = {column: _colorize_cell(required, row.get(column)) for column in week_columns}
            return pd.Series(styles)

        if week_columns:
            styler = styler.apply(_style_weeks, axis=1, subset=week_columns)

    return styler


def main():
    st.title("Dashboard")
    data = get_data()

    with st.expander("Filters", expanded=True):
        top_row = st.columns(2)
        with top_row[0]:
            view_label = st.selectbox("View", list(VIEW_OPTIONS.keys()))
        with top_row[1]:
            group_by_label = st.selectbox("Group by", list(GROUP_OPTIONS.keys()))

        second_row = st.columns(2)
        data_display_label = None
        with second_row[0]:
            data_options = DATA_DISPLAY[VIEW_OPTIONS[view_label]]
            data_display_label = st.selectbox("Data display", list(data_options.keys()))
        with second_row[1]:
            unit_label = st.selectbox("Units", list(UNIT_OPTIONS.keys()))

        search_term = st.text_input("Search", placeholder="Filter rows by any column value")

        date_range = _default_date_range()
        if VIEW_OPTIONS[view_label] == "live":
            start, end = st.date_input(
                "Date range",
                value=date_range,
                help="Only applicable to live view",
            )
            date_range = (start, end)

    view = VIEW_OPTIONS[view_label]
    group_by = GROUP_OPTIONS[group_by_label]
    unit_value = UNIT_OPTIONS[unit_label]
    display_value = DATA_DISPLAY[view][data_display_label]

    if view == "theoretical":
        df = compute_theoretical_coverage(
            data,
            view=display_value,
            group_by=group_by,
            unit=unit_value,
        )
    else:
        df = compute_live_coverage(
            data,
            attendance=data.get("attendances", []),
            date_range=date_range,
            display_mode=display_value,
            group_by=group_by,
            unit=unit_value,
        )

    if search_term:
        mask = pd.Series([False] * len(df))
        for column in df.columns:
            mask = mask | df[column].astype(str).str.contains(search_term, case=False)
        df = df[mask]

    styled = _style_dataframe(df, view, display_value)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    if view == "live":
        weeks = weeks_in_range(date_range)
        attendance_details = pd.DataFrame()
        if weeks:
            selected_week = st.selectbox(
                "Week",
                weeks,
                format_func=format_week_label,
                help="Select a week to review attendance impact details.",
            )
            attendance_details = compute_attendance_impact_details(
                data,
                attendance=data.get("attendances", []),
                week_start=selected_week,
                group_by=group_by,
                unit=unit_value,
            )

        st.subheader("Attendance impact details")
        if attendance_details.empty:
            st.info("No attendance impact recorded for the selected parameters.")
        else:
            st.dataframe(attendance_details, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
