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
from utils.styling import coerce_numeric, coverage_style


VIEW_OPTIONS = {"Theoretical": "theoretical", "Live": "live"}
GROUP_OPTIONS = {"Office": "office", "Region": "region"}
UNIT_OPTIONS = {"Hours": "hours", "FTE": "fte"}
DATA_DISPLAY = {
    "theoretical": {"Role": "role", "Process": "process"},
    "live": {"Coverage": "coverage", "JIRA": "jira"},
}

STATIC_COLUMN_STYLE = {
    "background-color": "#ede9fe",
    "font-style": "italic",
    "color": "#111827",
}

st.set_page_config(page_title="Dashboard", layout="wide")


def _default_date_range() -> tuple[date, date]:
    today = date.today()
    start = today - timedelta(days=today.weekday())
    end = start + timedelta(days=28)
    return start, end


def _style_dataframe(
    df: pd.DataFrame,
    view: str,
    display_mode: str,
    unit: str,
) -> Styler | pd.DataFrame:
    if df.empty:
        return df

    styler = df.style

    required_series = df["Required"] if "Required" in df.columns else None

    numeric_columns = [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column]) and column not in {"Region", "Office", "Process", "Role"}
    ]

    style_operations: list[tuple[str, dict]] = []

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
        style_operations.append(("format", {"formatter": formatter}))

    reference_columns = [col for col in ["Required", "Theoretical"] if col in df.columns]
    if reference_columns:
        style_operations.append(
            (
                "set_properties",
                {"subset": reference_columns, **STATIC_COLUMN_STYLE},
            )
        )

    if view == "theoretical":
        coverage_columns = [col for col in ["Coverage"] if col in df.columns]

        def _style_coverage(row: pd.Series) -> pd.Series:
            required = required_series.get(row.name) if required_series is not None else None
            styles = {column: coverage_style(required, row.get(column)) for column in coverage_columns}
            return pd.Series(styles)

        if coverage_columns and required_series is not None:
            style_operations.append(
                ("apply", {"func": _style_coverage, "axis": 1, "subset": coverage_columns})
            )

    if view == "live" and display_mode.lower() not in {"jira"}:
        week_columns = df.attrs.get("week_columns") or [
            col
            for col in df.columns
            if col not in {"Region", "Office", "Process", "Required", "Theoretical"}
            and pd.api.types.is_numeric_dtype(df[col])
        ]

        def _style_weeks(row: pd.Series) -> pd.Series:
            required = required_series.get(row.name) if required_series is not None else None
            styles = {column: coverage_style(required, row.get(column)) for column in week_columns}
            return pd.Series(styles)

        if week_columns and required_series is not None:
            style_operations.append(
                ("apply", {"func": _style_weeks, "axis": 1, "subset": week_columns})
            )

    zero_sensitive_columns = [
        column
        for column in df.columns
        if column not in {"Region", "Office", "Process", "Role"}
    ]

    def _style_neutral_zero(row: pd.Series) -> pd.Series:
        required_value = coerce_numeric(required_series.get(row.name)) if required_series is not None else None
        styles = {column: "" for column in zero_sensitive_columns}
        for column in zero_sensitive_columns:
            value = coerce_numeric(row.get(column))
            if value == 0 and (required_value == 0 or pd.isna(required_value)):
                styles[column] = "color: #6b7280; font-style: italic"
        return pd.Series(styles)

    if zero_sensitive_columns and required_series is not None:
        style_operations.append(
            ("apply", {"func": _style_neutral_zero, "axis": 1, "subset": zero_sensitive_columns})
        )

    for method, kwargs in style_operations:
        styler = getattr(styler, method)(**kwargs)

    return styler


def main():
    st.title("Dashboard")
    data = get_data()

    filter_container = st.container()
    with filter_container:
        primary_row = st.columns([1.2, 1, 1.6])

        with primary_row[0]:
            view_label = st.selectbox("View", list(VIEW_OPTIONS.keys()))

        view = VIEW_OPTIONS[view_label]
        data_options = DATA_DISPLAY[view]
        date_range = _default_date_range()

        with primary_row[1]:
            display_labels = list(data_options.keys())
            if view == "live":
                pill_selection = st.pills(
                    "Display",
                    display_labels,
                    selection_mode="multi",
                    default=[display_labels[0]] if display_labels else None,
                )
                if pill_selection is None:
                    pill_selection = []
                if isinstance(pill_selection, str):
                    pill_selection = [pill_selection]
                elif not isinstance(pill_selection, (list, tuple, set)):
                    pill_selection = [pill_selection]
                selected_display_labels = list(pill_selection)
                if not selected_display_labels and display_labels:
                    selected_display_labels = [display_labels[0]]
            else:
                pill_selection = st.pills(
                    "Display",
                    display_labels,
                    selection_mode="single",
                    default=display_labels[0] if display_labels else None,
                )
                if isinstance(pill_selection, (list, tuple, set)):
                    pill_selection = next(iter(pill_selection), display_labels[0] if display_labels else None)
                selected_display_labels = [pill_selection] if pill_selection else []
            data_display_label = selected_display_labels[0] if selected_display_labels else None

        with primary_row[2]:
            if view == "live":
                start, end = st.date_input(
                    "Date range",
                    value=date_range,
                    help="Only applicable to live view",
                )
                date_range = (start, end)
            else:
                st.empty()

        secondary_row = st.columns([0.8, 1.8, 0.8])
        selected_display_labels: list[str] = []
        with secondary_row[0]:
            segmented = st.segmented_control(
                "Group by",
                list(GROUP_OPTIONS.keys()),
                selection_mode="single",
                default=list(GROUP_OPTIONS.keys())[0],
            )
            if isinstance(segmented, (list, tuple, set)):
                segmented = next(iter(segmented), list(GROUP_OPTIONS.keys())[0])
            group_by_label = segmented
        with secondary_row[1]:
            st.caption("Search")
            search_term = st.text_input(
                "Search",
                placeholder="Filter rows by any column value",
                label_visibility="collapsed",
            )

        with secondary_row[2]:
            unit_selection = st.segmented_control(
                "Units",
                list(UNIT_OPTIONS.keys()),
                selection_mode="single",
                default=list(UNIT_OPTIONS.keys())[0],
            )
            if isinstance(unit_selection, (list, tuple, set)):
                unit_selection = next(
                    iter(unit_selection), list(UNIT_OPTIONS.keys())[0]
                )
            unit_label = unit_selection or list(UNIT_OPTIONS.keys())[0]

    group_by = GROUP_OPTIONS[group_by_label]
    unit_value = UNIT_OPTIONS[unit_label]
    if view == "live":
        selected_values = {
            DATA_DISPLAY[view][label]
            for label in selected_display_labels
            if label in DATA_DISPLAY[view]
        }
        if not selected_values:
            default_value = next(iter(DATA_DISPLAY[view].values()), "coverage")
            default_label = next(iter(DATA_DISPLAY[view].keys()), "Coverage")
            selected_values = {default_value}
            selected_display_labels = [default_label]

        if {"coverage", "jira"}.issubset(selected_values):
            display_value = "coverage+jira"
        else:
            display_value = next(iter(selected_values))
    else:
        fallback_label = next(iter(DATA_DISPLAY[view].keys()), None)
        effective_label = data_display_label or fallback_label
        display_value = DATA_DISPLAY[view].get(effective_label, "process")

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
