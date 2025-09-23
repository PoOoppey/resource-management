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
    required = _coerce_numeric(required)
    if pd.isna(required) or pd.isna(value):
        return ""

    if required == 0:
        if value == 0:
            return ""
        # highlight positive coverage when nothing was required
        return "background-color: #bae6fd"

    gap_ratio = (value - required) / required
    if gap_ratio <= -0.15:
        return "background-color: #fca5a5"
    if gap_ratio < 0:
        return "background-color: #fecaca"
    if gap_ratio < 0.1:
        return "background-color: #fef08a"
    if gap_ratio < 0.25:
        return "background-color: #bbf7d0"
    return "background-color: #86efac"


def _style_dataframe(
    df: pd.DataFrame,
    view: str,
    display_mode: str,
    unit: str,
) -> Styler | pd.DataFrame:
    if df.empty:
        return df

    styler = df.style

    numeric_columns = [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column]) and column not in {"Region", "Office", "Process", "Role"}
    ]

    if numeric_columns:
        decimals = 0
        if display_mode.lower() == "jira":
            decimals = 0
        elif unit.lower() in {"hours", "fte"}:
            decimals = 1
        formatter = {
            column: (lambda value, d=decimals: "" if pd.isna(value) else f"{value:,.{d}f}")
            for column in numeric_columns
        }
        styler = styler.format(formatter)

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

    zero_sensitive_columns = [
        column
        for column in df.columns
        if column not in {"Region", "Office", "Process", "Role"}
    ]

    has_required_column = "Required" in df.columns

    def _style_neutral_zero(row: pd.Series) -> pd.Series:
        required_value = _coerce_numeric(row.get("Required"))
        styles = {column: "" for column in zero_sensitive_columns}
        for column in zero_sensitive_columns:
            value = _coerce_numeric(row.get(column))
            if value == 0 and (required_value == 0 or pd.isna(required_value)):
                styles[column] = "color: #6b7280; font-style: italic"
        return pd.Series(styles)

    if zero_sensitive_columns and has_required_column:
        styler = styler.apply(_style_neutral_zero, axis=1, subset=zero_sensitive_columns)

    return styler


def main():
    st.title("Dashboard")
    data = get_data()

    filter_container = st.container()
    with filter_container:
        st.markdown("### Filters")
        st.divider()
        primary_row = st.columns([1.2, 1.6, 1.0])
        with primary_row[0]:
            view_label = st.selectbox("View", list(VIEW_OPTIONS.keys()))
        with primary_row[1]:
            segmented = st.segmented_control(
                "Group by",
                list(GROUP_OPTIONS.keys()),
                selection_mode="single",
                default=list(GROUP_OPTIONS.keys())[0],
            )
            if isinstance(segmented, (list, tuple, set)):
                segmented = next(iter(segmented), list(GROUP_OPTIONS.keys())[0])
            group_by_label = segmented
        with primary_row[2]:
            unit_label = st.selectbox("Units", list(UNIT_OPTIONS.keys()))

        view = VIEW_OPTIONS[view_label]
        data_options = DATA_DISPLAY[view]

        secondary_row = st.columns([1.6, 1.2, 1.4])
        with secondary_row[0]:
            pill_selection = st.pills(
                "Display",
                list(data_options.keys()),
                selection_mode="single",
                default=list(data_options.keys())[0],
            )
            if isinstance(pill_selection, (list, tuple, set)):
                pill_selection = next(iter(pill_selection), list(data_options.keys())[0])
            data_display_label = pill_selection
        with secondary_row[1]:
            st.caption("Search")
            search_term = st.text_input(
                "Search",
                placeholder="Filter rows by any column value",
                label_visibility="collapsed",
            )

        date_range = _default_date_range()
        with secondary_row[2]:
            if view == "live":
                start, end = st.date_input(
                    "Date range",
                    value=date_range,
                    help="Only applicable to live view",
                )
                date_range = (start, end)
            else:
                st.caption("Date range")
                st.empty()

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

    styled = _style_dataframe(df, view, display_value, unit_value)
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
