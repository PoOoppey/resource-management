from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st
import matplotlib
from services.data_loader import get_data
from services.expertise import build_expertise_dataframe

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

    column_medians = matrix_df.median(axis=0, skipna=True)
    deviation_df = matrix_df.subtract(column_medians, axis="columns")
    max_abs_deviation = deviation_df.abs().max().max()
    if pd.isna(max_abs_deviation) or max_abs_deviation == 0:
        max_abs_deviation = 1.0

    def _style_missing(row: pd.Series) -> list[str]:
        return ["background-color: #f6f6f9" if pd.isna(value) else "" for value in row]

    matrix_styler = (
        matrix_df.style.format(lambda value: "" if pd.isna(value) else f"{int(value)}")
        .background_gradient(
            cmap="PiYG",
            axis=None,
            gmap=deviation_df,
            vmin=-max_abs_deviation,
            vmax=max_abs_deviation,
        )
        .apply(_style_missing, axis=1)
    )
    st.dataframe(matrix_styler, use_container_width=True)
    st.caption("Cell colors show the deviation from the process median expertise level.")

    process_summary = (
        filtered_df.groupby("Process")
        .agg(
            Employees=("Employee UUID", pd.Series.nunique),
            Assignments=("Process", "count"),
            Median_level=("Level", "median"),
        )
        .rename_axis("Process")
        .sort_values("Median_level", ascending=False)
    )
    process_summary["Median_level"] = process_summary["Median_level"].round(2)
    process_summary[["Employees", "Assignments"]] = process_summary[["Employees", "Assignments"]].fillna(0).astype(int)

    def _merge_regions(values: pd.Series) -> str:
        unique_regions = sorted({value for value in values if isinstance(value, str) and value})
        return ", ".join(unique_regions)

    employee_summary = (
        labeled_df.groupby("Employee label")
        .agg(
            Regions=("Region", _merge_regions),
            Processes=("Process UUID", lambda series: series.dropna().nunique()),
            Assignments=("Process", "count"),
            Median_level=("Level", "median"),
        )
        .rename_axis("Employee")
        .sort_values(["Median_level", "Assignments"], ascending=[False, False])
    )
    employee_summary["Median_level"] = employee_summary["Median_level"].round(2)
    employee_summary[["Processes", "Assignments"]] = employee_summary[["Processes", "Assignments"]].fillna(0).astype(int)

    insight_columns = st.columns(2)
    with insight_columns[0]:
        st.markdown("#### Process overview")
        st.dataframe(process_summary, use_container_width=True)
    with insight_columns[1]:
        st.markdown("#### Employee overview")
        st.dataframe(employee_summary, use_container_width=True)


def _render_process_summary(filtered_df: pd.DataFrame) -> None:
    process_summary = (
        filtered_df.groupby("Process")
        .agg(
            Employees=("Employee UUID", pd.Series.nunique),
            Active_assignments=("Active", "sum"),
            Total_assignments=("Active", "count"),
            Median_level=("Level", "median"),
            Average_level=("Level", "mean"),
        )
        .sort_index()
    )
    process_summary["Average_level"] = process_summary["Average_level"].round(2)
    process_summary["Median_level"] = process_summary["Median_level"].round(2)
    process_summary = process_summary.rename(
        columns={
            "Employees": "Employees",
            "Active_assignments": "Active",
            "Total_assignments": "Total",
            "Average_level": "Avg. level",
            "Median_level": "Median level",
        }
    )
    process_summary[["Employees", "Active", "Total"]] = process_summary[["Employees", "Active", "Total"]].fillna(0).astype(int)
    process_summary = process_summary.reset_index()

    st.dataframe(process_summary, use_container_width=True)


def render() -> None:
    st.title("Expertise Dashboard")
    data = get_data()

    expertise_levels = data.get("expertise_levels", [])
    if not expertise_levels:
        st.info(
            "No expertise assignments recorded yet. Add them in Data Management → Expertise."
        )
        return

    employee_lookup = {
        employee.uuid: (
            f"{employee.first_name} {employee.last_name}".strip()
            or employee.trigram
            or employee.uuid
        )
        for employee in data.get("employees", [])
    }
    process_lookup = {process.uuid: process.name for process in data.get("processes", [])}

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

    detail_row = st.columns([1.2, 1.2])
    with detail_row[0]:
        employee_options = [None] + sorted(employee_lookup.keys(), key=lambda item: employee_lookup[item])
        selected_employee_uuid = st.selectbox(
            "Employee",
            options=employee_options,
            format_func=lambda value: "All employees"
            if value is None
            else employee_lookup.get(value, value),
            key="expertise_employee_filter",
        )

    available_process_ids = sorted(
        {
            row.get("Process UUID")
            for _, row in expertise_df.iterrows()
            if pd.notna(row.get("Process UUID"))
            and row.get("Process UUID")
            and (
                not selected_employee_uuid
                or row.get("Employee UUID") == selected_employee_uuid
            )
        }
    )

    process_evolution_key = "expertise_process_evolution"
    valid_process_values = {None, *available_process_ids}
    if (
        process_evolution_key in st.session_state
        and st.session_state[process_evolution_key] not in valid_process_values
    ):
        st.session_state[process_evolution_key] = None

    with detail_row[1]:
        selected_process_uuid = st.selectbox(
            "Process (evolution)",
            options=[None] + available_process_ids,
            format_func=lambda value: "Select a process"
            if value is None
            else process_lookup.get(value, value),
            key=process_evolution_key,
        )

    filtered_df = expertise_df.copy()
    min_level, max_level = level_range
    filtered_df = filtered_df[(filtered_df["Level"] >= min_level) & (filtered_df["Level"] <= max_level)]

    if selected_employee_uuid:
        filtered_df = filtered_df[filtered_df["Employee UUID"] == selected_employee_uuid]

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

    summary_columns = st.columns(4)
    summary_columns[0].metric("Assignments", f"{len(filtered_df)}")
    summary_columns[1].metric(
        "Employees",
        f"{filtered_df['Employee UUID'].nunique()}",
    )
    summary_columns[2].metric(
        "Processes",
        f"{filtered_df['Process UUID'].nunique()}",
    )
    median_level = filtered_df["Level"].median()
    summary_columns[3].metric(
        "Median level",
        "—" if pd.isna(median_level) else f"{median_level:.1f}",
    )

    if selected_employee_uuid and selected_process_uuid:
        employee_label = employee_lookup.get(selected_employee_uuid, "Selected employee")
        process_label = process_lookup.get(selected_process_uuid, "Selected process")
        st.markdown(f"#### Expertise evolution for {employee_label} – {process_label}")

        evolution_df = expertise_df[
            (expertise_df["Employee UUID"] == selected_employee_uuid)
            & (expertise_df["Process UUID"] == selected_process_uuid)
        ].sort_values("Start Date")

        if evolution_df.empty:
            st.info("No expertise history recorded for this combination yet.")
        else:
            timeline_points: list[dict[str, object]] = []
            for _, row in evolution_df.iterrows():
                start_raw = row.get("Start Date")
                end_raw = row.get("End Date")
                level_value = int(row.get("Level", 0))
                start_value = pd.to_datetime(start_raw, errors="coerce")
                end_value = pd.to_datetime(end_raw, errors="coerce") if end_raw else None
                if pd.isna(start_value):
                    continue
                start_date_value = start_value.date()
                end_date_value = end_value.date() if end_value and not pd.isna(end_value) else None

                timeline_points.append({"Date": start_date_value, "Level": level_value})

                if end_date_value:
                    timeline_points.append({"Date": end_date_value, "Level": level_value})
                    timeline_points.append(
                        {"Date": end_date_value + timedelta(days=1), "Level": 0}
                    )
                else:
                    extension_date = as_of if as_of >= start_date_value else start_date_value
                    timeline_points.append({"Date": extension_date, "Level": level_value})
                    timeline_points.append({"Date": extension_date + timedelta(days=1), "Level": 0})

            timeline_df = pd.DataFrame(timeline_points)
            timeline_df = timeline_df.dropna(subset=["Date"])
            if timeline_df.empty:
                st.info("No timeline data available for the selected expertise history.")
            else:
                timeline_df = timeline_df.sort_values("Date")
                timeline_df = timeline_df.drop_duplicates(subset=["Date"], keep="last")
                timeline_df = timeline_df.set_index("Date")
                st.line_chart(timeline_df, use_container_width=True, height=260)

                history_columns = [
                    "Start Date",
                    "End Date",
                    "Level",
                    "Active",
                ]
                history_df = evolution_df[history_columns].copy()
                history_df["Start Date"] = history_df["Start Date"].apply(_format_date)
                history_df["End Date"] = history_df["End Date"].apply(_format_date)
                history_df["Active"] = history_df["Active"].map({True: "Yes", False: "No"})
                st.dataframe(history_df, use_container_width=True, hide_index=True)

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
    import streamlit as st

    st.set_page_config(page_title="Expertise Dashboard", layout="wide")
    render()

