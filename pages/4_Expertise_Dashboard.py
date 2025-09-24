from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from services.data_loader import get_data
from services.expertise import build_expertise_dataframe

st.set_page_config(page_title="Expertise Dashboard", layout="wide")


def _format_date(value: object) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return "—"


def _render_assignments(filtered_df: pd.DataFrame) -> None:
    formatted_df = filtered_df.copy()
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

    st.dataframe(
        formatted_df[assignment_columns],
        use_container_width=True,
        hide_index=True,
    )


def _render_matrix(filtered_df: pd.DataFrame) -> None:
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

    if matrix_df.empty:
        st.info("No expertise assignments match the current filters.")
        return

    matrix_styler = (
        matrix_df.style.format(lambda value: "" if pd.isna(value) else f"{int(value)}")
        .background_gradient(cmap="GnBu", vmin=1, vmax=5)
    )
    st.dataframe(matrix_styler, use_container_width=True)


def _render_process_summary(filtered_df: pd.DataFrame) -> None:
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

    st.dataframe(process_summary, use_container_width=True)


def main() -> None:
    st.title("Expertise Dashboard")
    data = get_data()

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

    with overview_tabs[0]:
        _render_assignments(filtered_df)

    with overview_tabs[1]:
        _render_matrix(filtered_df)

    with overview_tabs[2]:
        _render_process_summary(filtered_df)


if __name__ == "__main__":
    main()

