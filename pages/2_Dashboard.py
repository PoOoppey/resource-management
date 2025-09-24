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
from services.expertise import build_expertise_dataframe
from utils.styling import coerce_numeric, coverage_style


VIEW_OPTIONS = {"Theoretical": "theoretical", "Live": "live"}
GROUP_OPTIONS = {"Office": "office", "Region": "region"}
UNIT_OPTIONS = {"Hours": "hours", "FTE": "fte"}
DATA_DISPLAY = {
    "theoretical": {"Role": "role", "Process": "process"},
    "live": {"Coverage": "coverage", "JIRA": "jira"},
}

STATIC_COLUMN_STYLE = {"background-color": "#f3f0ff", "font-style": "italic"}


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

    if view == "live":
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

        view = VIEW_OPTIONS[view_label]
        data_options = DATA_DISPLAY[view]

        secondary_row = st.columns([1.6, 1.2, 1.4])
        selected_display_labels: list[str] = []
        with secondary_row[0]:
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

    st.divider()
    st.header("Employee expertise overview")

    expertise_levels = data.get("expertise_levels", [])
    if not expertise_levels:
        st.info(
            "No expertise assignments recorded yet. Add them in Data Management → Expertise."
        )
        return

    filter_row = st.columns([1.2, 1.2, 1.0])
    with filter_row[0]:
        as_of = st.date_input(
            "As of date",
            value=date.today(),
            key="expertise_as_of",
            help="Expertise is considered active when the selected date falls within the assignment window.",
        )
    with filter_row[1]:
        process_labels = sorted({process.name for process in data.get("processes", [])})
        process_filter = st.multiselect(
            "Processes",
            process_labels,
            key="expertise_process_filter",
        )
    with filter_row[2]:
        region_labels = sorted({office.region.value for office in data.get("offices", [])})
        region_filter = st.multiselect(
            "Regions",
            region_labels,
            key="expertise_region_filter",
        )

    level_row = st.columns([1.2, 0.8, 1.0])
    with level_row[0]:
        level_range = st.slider(
            "Level range",
            min_value=1,
            max_value=5,
            value=(1, 5),
            step=1,
            key="expertise_level_range",
        )
    with level_row[1]:
        show_active_only = st.checkbox(
            "Active only",
            value=True,
            key="expertise_active_only",
        )
    with level_row[2]:
        expertise_search = st.text_input(
            "Search",
            key="expertise_search",
            placeholder="Search employee, trigram, process or office",
        )

    expertise_df = build_expertise_dataframe(
        expertise_levels,
        data.get("employees", []),
        data.get("processes", []),
        data.get("offices", []),
        as_of=as_of,
    )

    if expertise_df.empty:
        st.info("No expertise assignments available for the selected criteria.")
        return

    filtered_df = expertise_df.copy()
    min_level, max_level = level_range
    filtered_df = filtered_df[(filtered_df["Level"] >= min_level) & (filtered_df["Level"] <= max_level)]

    if process_filter:
        filtered_df = filtered_df[filtered_df["Process"].isin(process_filter)]

    if region_filter:
        filtered_df = filtered_df[filtered_df["Region"].isin(region_filter)]

    if show_active_only:
        filtered_df = filtered_df[filtered_df["Active"]]

    if expertise_search:
        lowered = expertise_search.lower()
        search_columns = ["Employee", "Trigram", "Process", "Office", "Region"]
        mask = pd.Series(False, index=filtered_df.index)
        for column in search_columns:
            mask = mask | filtered_df[column].fillna("").astype(str).str.lower().str.contains(lowered)
        filtered_df = filtered_df[mask]

    summary_levels = filtered_df.groupby("Level").size()
    summary_columns = st.columns(5)
    for idx, level in enumerate(range(1, 6)):
        count = int(summary_levels.get(level, 0))
        summary_columns[idx].metric(f"Level {level}", f"{count}")

    overview_tabs = st.tabs(["Assignments", "Matrix", "Process summary"])

    if filtered_df.empty:
        for tab in overview_tabs:
            with tab:
                st.info("No expertise assignments match the current filters.")
        return

    formatted_df = filtered_df.copy()

    def _format_date(value: object) -> str:
        if isinstance(value, date):
            return value.isoformat()
        return "—"

    formatted_df["Start Date"] = formatted_df["Start Date"].apply(_format_date)
    formatted_df["End Date"] = formatted_df["End Date"].apply(_format_date)
    formatted_df["Active"] = formatted_df["Active"].map({True: "Yes", False: "No"})

    assignment_columns = [
        "Employee",
        "Trigram",
        "Process",
        "Level",
        "Start Date",
        "End Date",
        "Office",
        "Region",
        "Active",
    ]

    with overview_tabs[0]:
        st.dataframe(
            formatted_df[assignment_columns],
            use_container_width=True,
            hide_index=True,
        )

    labeled_df = filtered_df.copy()
    labeled_df["Employee label"] = labeled_df.apply(
        lambda row: f"{row['Employee']} ({row['Trigram']})" if row.get("Trigram") else row["Employee"],
        axis=1,
    )
    matrix_df = (
        labeled_df.pivot_table(
            index="Employee label",
            columns="Process",
            values="Level",
            aggfunc="max",
        )
        .sort_index()
    )

    with overview_tabs[1]:
        if matrix_df.empty:
            st.info("No expertise assignments match the current filters.")
        else:
            matrix_styler = (
                matrix_df.style.format(lambda value: "" if pd.isna(value) else f"{int(value)}")
                .background_gradient(cmap="GnBu", vmin=1, vmax=5)
            )
            st.dataframe(matrix_styler, use_container_width=True)

    process_summary = (
        filtered_df.groupby("Process")
        .agg(
            Active_assignments=("Active", "sum"),
            Total_assignments=("Active", "count"),
            Average_level=("Level", "mean"),
        )
        .sort_index()
    )
    process_summary["Average_level"] = process_summary["Average_level"].round(2)
    process_summary = process_summary.rename(
        columns={
            "Active_assignments": "Active",
            "Total_assignments": "Total",
            "Average_level": "Avg. level",
        }
    )
    process_summary[["Active", "Total"]] = process_summary[["Active", "Total"]].astype(int)
    process_summary = process_summary.reset_index()

    with overview_tabs[2]:
        st.dataframe(process_summary, use_container_width=True)


if __name__ == "__main__":
    main()
