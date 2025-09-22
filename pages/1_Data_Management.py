from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, List

import pandas as pd
import streamlit as st

from models import (
    Allocation,
    App,
    Criticality,
    Employee,
    Process,
    RequiredCoverage,
    Role,
    Scenario,
    SupportAllocation,
    SupportStatus,
)
from services.data_loader import get_data, update_data
from utils.notifications import notify


def _serialize_value(value: Any):
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, list):
        return value
    return value


def _editable_dataframe(items: List) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    rows = []
    for item in items:
        if is_dataclass(item):
            raw = asdict(item)
        else:
            raw = item
        rows.append({key: _serialize_value(value) for key, value in raw.items()})
    return pd.DataFrame(rows)


def render_employees(employees: List[Employee]):
    st.subheader("Employees")
    df = _editable_dataframe(employees)
    edited = st.data_editor(df, num_rows="dynamic")
    if st.button("Save Employees"):
        update_data("employees", [Employee.from_dict(row.to_dict()) for _, row in edited.iterrows()])
        notify("Employees updated", "success")


def render_offices():
    st.subheader("Offices")
    data = get_data()["offices"]
    df = _editable_dataframe(data)
    edited = st.data_editor(df, num_rows="dynamic")
    if st.button("Save Offices"):
        from models import Office

        update_data("offices", [Office.from_dict(row.to_dict()) for _, row in edited.iterrows()])
        notify("Offices updated", "success")


def render_roles(roles: List[Role]):
    st.subheader("Roles")
    df = _editable_dataframe(roles)
    edited = st.data_editor(df, num_rows="dynamic")
    if st.button("Save Roles"):
        update_data("roles", [Role.from_dict(row.to_dict()) for _, row in edited.iterrows()])
        notify("Roles updated", "success")


def render_apps(apps: List[App]):
    st.subheader("Apps")
    df = _editable_dataframe(apps)
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "criticality": st.column_config.SelectboxColumn(
                "Criticality", options=[option.value for option in Criticality]
            ),
        },
    )
    if st.button("Save Apps"):
        update_data("apps", [App.from_dict(row.to_dict()) for _, row in edited.iterrows()])
        notify("Apps updated", "success")


def render_processes(processes: List[Process], apps: List[App]):
    st.subheader("Processes")
    df = _editable_dataframe(processes)
    app_options = [app.uuid for app in apps]
    process_options = [proc.uuid for proc in processes]
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "criticality": st.column_config.SelectboxColumn(
                "Criticality", options=[option.value for option in Criticality]
            ),
            "support_status": st.column_config.SelectboxColumn(
                "Support Status", options=[option.value for option in SupportStatus]
            ),
            "apps_related": st.column_config.MultiSelectColumn(
                "Apps Related",
                options=app_options,
            ),
            "process_related": st.column_config.MultiSelectColumn(
                "Process Related",
                options=process_options,
            ),
        },
    )
    if st.button("Save Processes"):
        update_data("processes", [Process.from_dict(row.to_dict()) for _, row in edited.iterrows()])
        notify("Processes updated", "success")


def render_required_coverage(items: List[RequiredCoverage]):
    st.subheader("Required Coverage")
    df = _editable_dataframe(items)
    edited = st.data_editor(df, num_rows="dynamic")
    if st.button("Save Coverage"):
        update_data("coverage", [RequiredCoverage.from_dict(row.to_dict()) for _, row in edited.iterrows()])
        notify("Required coverage updated", "success")


def render_allocations(allocations: List[Allocation], support_allocations: List[SupportAllocation]):
    st.subheader("Allocations")
    employee_filter = st.selectbox("Select Employee", options=["All"] + [alloc.employee_uuid for alloc in allocations])
    df = _editable_dataframe(allocations)
    if employee_filter != "All":
        df = df[df["employee_uuid"] == employee_filter]
    edited = st.data_editor(df, num_rows="dynamic")
    if st.button("Save Allocations"):
        update_data("allocations", [Allocation.from_dict(row.to_dict()) for _, row in edited.iterrows()])
        notify("Allocations updated", "success")

    st.markdown("### Support Allocations")
    support_df = _editable_dataframe(support_allocations)
    support_edited = st.data_editor(support_df, num_rows="dynamic")
    if st.button("Save Support Allocations"):
        update_data(
            "support_allocations",
            [SupportAllocation.from_dict(row.to_dict()) for _, row in support_edited.iterrows()],
        )
        notify("Support allocations updated", "success")


def render_scenarios(scenarios: List[Scenario]):
    st.subheader("Scenarios")
    df = pd.DataFrame(
        {
            "uuid": [scn.uuid for scn in scenarios],
            "name": [scn.name for scn in scenarios],
            "adjustments": [len(scn.adjustments) for scn in scenarios],
        }
    )
    st.dataframe(df)


def main():
    st.title("Data Management")
    data = get_data()
    tabs = st.tabs(
        [
            "Employees",
            "Offices",
            "Roles",
            "Processes",
            "Apps",
            "Required Coverage",
            "Allocations",
            "Scenarios",
        ]
    )

    with tabs[0]:
        render_employees(data["employees"])
    with tabs[1]:
        render_offices()
    with tabs[2]:
        render_roles(data["roles"])
    with tabs[3]:
        render_processes(data["processes"], data["apps"])
    with tabs[4]:
        render_apps(data["apps"])
    with tabs[5]:
        render_required_coverage(data["coverage"])
    with tabs[6]:
        render_allocations(data["allocations"], data["support_allocations"])
    with tabs[7]:
        render_scenarios(data["scenarios"])


if __name__ == "__main__":
    main()
