from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import streamlit as st

from models import (
    Allocation,
    App,
    Criticality,
    Employee,
    EmployeeExpertise,
    Office,
    Process,
    Region,
    RequiredCoverage,
    Role,
    RoleType,
    SupportAllocation,
    SupportStatus,
)
from services.data_loader import get_data, update_data
from utils.notifications import notify

from uuid import uuid4

st.set_page_config(page_title="Data Management", layout="wide")



def _serialize_value(value: Any):
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
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


def _ensure_dataframe(df: pd.DataFrame, field_order: Sequence[str]) -> pd.DataFrame:
    if not df.empty:
        return df
    return pd.DataFrame(columns=list(field_order))


def _normalize_records(items: Iterable) -> Dict[str, Dict[str, Any]]:
    normalized: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if is_dataclass(item):
            raw = asdict(item)
        else:
            raw = dict(item)
        record = {key: _serialize_value(value) for key, value in raw.items()}
        identifier = str(record.get("uuid", ""))
        if identifier:
            normalized[identifier] = record
    return normalized


def _generate_uuid(prefix: Optional[str] = None) -> str:
    token = uuid4().hex[:8]
    return f"{prefix}-{token}" if prefix else uuid4().hex


def _summarize_changes(
    entity: str,
    original: Dict[str, Dict[str, Any]],
    updated: Dict[str, Dict[str, Any]],
    labeler: Optional[Callable[[Dict[str, Any]], str]] = None,
) -> str:
    added_ids = [key for key in updated.keys() if key not in original]
    removed_ids = [key for key in original.keys() if key not in updated]
    common_ids = [key for key in updated.keys() if key in original]
    changed_ids = [key for key in common_ids if original[key] != updated[key]]

    if not added_ids and not removed_ids and not changed_ids:
        return f"No changes detected for {entity}."

    def describe(ids: List[str], source: Dict[str, Dict[str, Any]]) -> str:
        if not ids:
            return "0"
        labels = []
        for identifier in ids:
            record = source.get(identifier)
            if not record:
                continue
            if labeler:
                labels.append(labeler(record))
            else:
                labels.append(str(record.get("name") or record.get("uuid")))
        label_text = ", ".join(labels)
        return f"{len(ids)} ({label_text})"

    parts = [f"{entity} saved."]
    parts.append(f"Added: {describe(added_ids, updated)}")
    parts.append(f"Updated: {describe(changed_ids, updated)}")
    parts.append(f"Removed: {describe(removed_ids, original)}")
    return "\n".join(parts)


def _smart_editor(
    *,
    items: List,
    model_cls,
    dataset_key: str,
    save_label: str,
    column_labels: Dict[str, str],
    select_options: Optional[Dict[str, Dict[str, str]]] = None,
    number_columns: Optional[Dict[str, Dict[str, Any]]] = None,
    list_columns: Optional[Sequence[str]] = None,
    multiselect_options: Optional[Dict[str, Dict[str, str]]] = None,
    uuid_prefix: Optional[str] = None,
    labeler: Optional[Callable[[Dict[str, Any]], str]] = None,
    extra_fixed_values: Optional[Dict[str, Any]] = None,
    hide_columns: Optional[Sequence[str]] = None,
    key: Optional[str] = None,
    entity_label: Optional[str] = None,
    on_save: Optional[Callable[[List], None]] = None,
    date_columns: Optional[Sequence[str]] = None,
):
    field_order = list(model_cls.__dataclass_fields__.keys())
    df = _editable_dataframe(items)
    df = _ensure_dataframe(df, field_order)

    date_columns = list(date_columns or [])

    if list_columns:
        for column in list_columns:
            if multiselect_options and column in multiselect_options:
                continue
            if column in df.columns:
                df[column] = df[column].apply(
                    lambda value: ", ".join(value) if isinstance(value, list) else (value or "")
                )

    if date_columns:
        for column in date_columns:
            if column in df.columns:
                df[column] = pd.to_datetime(df[column], errors="coerce").dt.date

    column_config: Dict[str, Any] = {}
    column_order: List[str] = []

    hidden_columns = set(hide_columns or [])

    has_uuid = "uuid" in df.columns and "uuid" in field_order
    if has_uuid:
        df = df.set_index("uuid")

    for column in df.columns:
        if column in hidden_columns:
            continue
        label = column_labels.get(column, column.replace("_", " ").title())
        if multiselect_options and column in multiselect_options:
            options = list(multiselect_options[column].keys())

            def _make_formatter(mapping: Dict[str, str]) -> Callable[[Any], str]:
                def _format(values: Any) -> str:
                    if not values:
                        return "—"
                    if isinstance(values, str):
                        values = [values]
                    return ", ".join(mapping.get(value, "—") for value in values)

                return _format

            column_config[column] = st.column_config.MultiselectColumn(
                label,
                options=options,
                format_func=_make_formatter(multiselect_options[column]),
            )
        elif select_options and column in select_options:
            options = list(select_options[column].keys())

            def _make_formatter(mapping: Dict[str, str]) -> Callable[[Any], str]:
                return lambda value: mapping.get(value, "—")

            column_config[column] = st.column_config.SelectboxColumn(
                label,
                options=options,
                format_func=_make_formatter(select_options[column]),
            )
        elif date_columns and column in date_columns:
            column_config[column] = st.column_config.DateColumn(label, format="YYYY-MM-DD")
        elif number_columns and column in number_columns:
            number_config = number_columns[column]
            column_config[column] = st.column_config.NumberColumn(label, **number_config)
        else:
            column_config[column] = st.column_config.TextColumn(label)
        column_order.append(column)

    editor_key = key or f"editor_{dataset_key}"
    edited_df = st.data_editor(
        df,
        hide_index=has_uuid,
        num_rows="dynamic",
        column_config=column_config,
        column_order=column_order,
        use_container_width=True,
        key=editor_key,
    )

    if has_uuid:
        edited_df = edited_df.reset_index().rename(columns={"index": "uuid"})
    else:
        edited_df = edited_df.copy()

    if list_columns:
        for column in list_columns:
            if multiselect_options and column in multiselect_options:
                continue
            if column in edited_df.columns:
                edited_df[column] = (
                    edited_df[column]
                    .fillna("")
                    .apply(
                        lambda value: [item.strip() for item in str(value).split(",") if item.strip()]
                    )
                )

    existing_ids = {str(getattr(item, "uuid")) for item in items if hasattr(item, "uuid")}
    records: List[Dict[str, Any]] = []
    for raw in edited_df.to_dict(orient="records"):
        record = {key: value for key, value in raw.items() if key not in hidden_columns}
        for column, value in list(record.items()):
            if isinstance(value, list):
                record[column] = value
                continue
            if date_columns and column in date_columns:
                if value in ("", None):
                    record[column] = None
                    continue
                if isinstance(value, pd.Timestamp):
                    value = value.to_pydatetime().date()
                if isinstance(value, datetime):
                    value = value.date()
                if isinstance(value, date):
                    record[column] = value.isoformat()
                    continue
                parsed = pd.to_datetime(value, errors="coerce")
                if pd.isna(parsed):
                    record[column] = None
                else:
                    record[column] = parsed.date().isoformat()
                continue
            if pd.isna(value):
                record[column] = None
        if has_uuid:
            identifier = str(record.get("uuid") or "")
            if not identifier or identifier not in existing_ids:
                record["uuid"] = _generate_uuid(uuid_prefix)
            else:
                record["uuid"] = identifier
        if extra_fixed_values:
            record.update(extra_fixed_values)
        records.append(record)

    if st.button(save_label, key=f"save_{dataset_key}"):
        updated_items = [model_cls.from_dict(record) for record in records]
        updated_lookup = _normalize_records(updated_items)
        original_lookup = _normalize_records(items)
        if on_save:
            on_save(updated_items)
        else:
            update_data(dataset_key, updated_items)
        entity_name = entity_label or dataset_key.capitalize()
        message = _summarize_changes(entity_name, original_lookup, updated_lookup, labeler)
        notify(message, "success")

    return records


def render_employees(employees: List[Employee], offices: List[Office]):
    st.subheader("Employees")
    office_options = {office.uuid: office.name for office in offices}
    _smart_editor(
        items=employees,
        model_cls=Employee,
        dataset_key="employees",
        save_label="Save employees",
        column_labels={
            "first_name": "First name",
            "last_name": "Last name",
            "trigram": "Trigram",
            "office_uuid": "Office",
            "working_hours": "Working hours",
        },
        select_options={"office_uuid": office_options},
        number_columns={"working_hours": {"min_value": 0.0, "max_value": 80.0, "step": 1.0}},
        uuid_prefix="emp",
        labeler=lambda record: f"{record.get('first_name', '')} {record.get('last_name', '')}".strip(),
    )


def render_offices(offices: List[Office]):
    st.subheader("Offices")
    region_options = {region.value: region.value for region in Region}
    _smart_editor(
        items=offices,
        model_cls=Office,
        dataset_key="offices",
        save_label="Save offices",
        column_labels={"name": "Name", "region": "Region"},
        select_options={"region": region_options},
        uuid_prefix="office",
        labeler=lambda record: record.get("name", ""),
    )


def render_apps(apps: List[App]):
    st.subheader("Apps")
    criticality_options = {item.value: item.value.title() for item in Criticality}
    _smart_editor(
        items=apps,
        model_cls=App,
        dataset_key="apps",
        save_label="Save apps",
        column_labels={"name": "Name", "criticality": "Criticality"},
        select_options={"criticality": criticality_options},
        uuid_prefix="app",
        labeler=lambda record: record.get("name", ""),
    )


def render_roles(roles: List[Role]):
    st.subheader("Roles")
    role_type_options = {role_type.value: role_type.value.title().replace("_", " ") for role_type in RoleType}
    _smart_editor(
        items=roles,
        model_cls=Role,
        dataset_key="roles",
        save_label="Save roles",
        column_labels={"name": "Name", "type": "Role type"},
        select_options={"type": role_type_options},
        uuid_prefix="role",
        labeler=lambda record: record.get("name", ""),
    )


def render_processes(processes: List[Process], apps: List[App]):
    st.subheader("Processes")
    criticality_options = {item.value: item.value.title() for item in Criticality}
    support_status_options = {item.value: item.value.title().replace("_", " ") for item in SupportStatus}
    app_options = {app.uuid: app.name for app in apps}
    process_options = {process.uuid: process.name for process in processes}
    _smart_editor(
        items=processes,
        model_cls=Process,
        dataset_key="processes",
        save_label="Save processes",
        column_labels={
            "name": "Name",
            "criticality": "Criticality",
            "description": "Description",
            "apps_related": "Apps",
            "process_related": "Related processes",
            "support_status": "Support status",
        },
        select_options={
            "criticality": criticality_options,
            "support_status": support_status_options,
        },
        list_columns=["apps_related", "process_related"],
        multiselect_options={
            "apps_related": app_options,
            "process_related": process_options,
        },
        uuid_prefix="process",
        labeler=lambda record: record.get("name", ""),
    )


def render_required_coverage(
    items: List[RequiredCoverage],
    processes: List[Process],
    offices: List[Office],
):
    st.subheader("Required Coverage")
    process_options = {process.uuid: process.name for process in processes}
    office_options = {office.uuid: office.name for office in offices}
    _smart_editor(
        items=items,
        model_cls=RequiredCoverage,
        dataset_key="coverage",
        save_label="Save coverage",
        column_labels={
            "process_uuid": "Process",
            "office_uuid": "Office",
            "required_hours": "Required hours",
        },
        select_options={
            "process_uuid": process_options,
            "office_uuid": office_options,
        },
        number_columns={"required_hours": {"min_value": 0.0, "step": 1.0}},
        uuid_prefix="coverage",
        labeler=lambda record: process_options.get(record.get("process_uuid"), record.get("uuid", "")),
    )


def render_allocations(
    employees: List[Employee],
    roles: List[Role],
    processes: List[Process],
    allocations: List[Allocation],
    support_allocations: List[SupportAllocation],
):
    st.subheader("Allocations")

    employee_lookup = {employee.uuid: employee for employee in employees}
    role_lookup = {role.uuid: role for role in roles}
    process_lookup = {process.uuid: process for process in processes}

    allocations_by_employee: Dict[str, List[Allocation]] = defaultdict(list)
    for allocation in allocations:
        allocations_by_employee[allocation.employee_uuid].append(allocation)

    support_by_allocation: Dict[str, List[SupportAllocation]] = defaultdict(list)
    for support in support_allocations:
        support_by_allocation[support.allocation_uuid].append(support)

    overview_rows: List[Dict[str, Any]] = []
    for employee in employees:
        employee_allocations = allocations_by_employee.get(employee.uuid, [])
        utilization = sum(allocation.percentage for allocation in employee_allocations)
        role_splits: List[str] = []
        support_details: List[str] = []
        for allocation in employee_allocations:
            role = role_lookup.get(allocation.role_uuid)
            role_name = role.name if role else allocation.role_uuid
            role_splits.append(f"{role_name} ({allocation.percentage * 100:.0f}%)")
            for support in support_by_allocation.get(allocation.uuid, []):
                process = process_lookup.get(support.process_uuid)
                process_name = process.name if process else support.process_uuid
                weight = support.effective_weight(allocation.weight)
                support_details.append(
                    f"{process_name} ({support.percentage * 100:.0f}%x{weight:.1f})"
                )

        overview_rows.append(
            {
                "employee_uuid": employee.uuid,
                "Employee": f"{employee.first_name} {employee.last_name}",
                "Utilization": utilization,
                "Role Split": "; ".join(role_splits) if role_splits else "—",
                "Support Details": "; ".join(support_details) if support_details else "—",
            }
        )

    if overview_rows:
        overview_df = pd.DataFrame(overview_rows).set_index("employee_uuid")
    else:
        overview_df = pd.DataFrame(
            columns=["Employee", "Utilization", "Role Split", "Support Details"]
        )
        overview_df.index.name = "employee_uuid"

    st.caption("Select an employee below to manage role and support allocations.")
    disabled_columns = list(overview_df.columns)
    st.data_editor(
        overview_df,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Employee": st.column_config.TextColumn("Employee"),
            "Utilization": st.column_config.ProgressColumn(
                "Utilization", format="{:.0%}", min_value=0.0, max_value=1.0
            ),
            "Role Split": st.column_config.TextColumn("Roles"),
            "Support Details": st.column_config.TextColumn("Support work"),
        },
        disabled=disabled_columns,
        key="allocation_overview",
    )

    stored_employee_uuid: Optional[str] = st.session_state.get("selected_employee_uuid")
    selected_employee_uuid: Optional[str] = None
    if overview_df is not None and not overview_df.empty:
        selection_state = st.session_state.get("allocation_overview", {})

        def _extract_selected_rows(payload: Dict[str, Any]) -> List[int]:
            if not isinstance(payload, dict):
                return []
            candidates: List[Any] = []
            selection_info = payload.get("selection", {}) if isinstance(payload, dict) else {}
            for key in ("rows", "row_indices"):
                values = selection_info.get(key)
                if values:
                    candidates = values
                    break
            if not candidates:
                fallback = payload.get("selected_rows")
                if fallback:
                    candidates = fallback
            if isinstance(candidates, dict):
                candidates = list(candidates.keys())
            if isinstance(candidates, set):
                candidates = list(candidates)
            normalized: List[int] = []
            for value in candidates:
                if isinstance(value, (int, float)) and not pd.isna(value):
                    normalized.append(int(value))
                elif isinstance(value, str) and value.isdigit():
                    normalized.append(int(value))
            return normalized

        selected_rows = _extract_selected_rows(selection_state)
        if selected_rows:
            row_position = selected_rows[0]
            if 0 <= row_position < len(overview_df.index):
                selected_employee_uuid = overview_df.index[row_position]
                st.session_state["selected_employee_uuid"] = selected_employee_uuid
        elif stored_employee_uuid and stored_employee_uuid in overview_df.index:
            selected_employee_uuid = stored_employee_uuid
        elif stored_employee_uuid and stored_employee_uuid not in overview_df.index:
            st.session_state.pop("selected_employee_uuid", None)
    elif stored_employee_uuid:
        st.session_state.pop("selected_employee_uuid", None)

    st.divider()

    if not selected_employee_uuid:
        st.info("Select an employee in the overview table to manage allocations.")
        return

    employee = employee_lookup[selected_employee_uuid]
    employee_name = f"{employee.first_name} {employee.last_name}".strip()
    st.markdown(f"### {employee_name}")

    employee_allocations = allocations_by_employee.get(selected_employee_uuid, [])
    role_options = {role.uuid: role.name for role in roles}

    def save_employee_allocations(updated_subset: List[Allocation]) -> None:
        remaining_allocations = [
            allocation for allocation in allocations if allocation.employee_uuid != selected_employee_uuid
        ]
        merged_allocations = remaining_allocations + updated_subset
        update_data("allocations", merged_allocations)
        allocations[:] = merged_allocations

        merged_ids = {allocation.uuid for allocation in merged_allocations}
        filtered_support = [
            support
            for support in support_allocations
            if support.allocation_uuid in merged_ids
        ]
        removed_support = len(support_allocations) - len(filtered_support)
        if removed_support:
            update_data("support_allocations", filtered_support)
            support_allocations[:] = filtered_support
            notify(
                f"Removed {removed_support} support allocation(s) linked to deleted roles for {employee_name}.",
                "info",
            )

    allocation_records = _smart_editor(
        items=employee_allocations,
        model_cls=Allocation,
        dataset_key=f"allocations_{selected_employee_uuid}",
        save_label="Save allocations",
        column_labels={
            "role_uuid": "Role",
            "percentage": "Allocation %",
            "weight": "Weight",
        },
        select_options={"role_uuid": role_options},
        number_columns={
            "percentage": {"min_value": 0.0, "max_value": 1.0, "step": 0.05, "format": "%.0f%%"},
            "weight": {"min_value": 0.0, "step": 0.1},
        },
        uuid_prefix="alloc",
        labeler=lambda record: role_options.get(record.get("role_uuid"), record.get("uuid", "")),
        extra_fixed_values={"employee_uuid": selected_employee_uuid},
        hide_columns=["employee_uuid"],
        key=f"alloc_editor_{selected_employee_uuid}",
        entity_label=f"Allocations for {employee_name}",
        on_save=save_employee_allocations,
    )
    employee_allocations = [Allocation.from_dict(record) for record in allocation_records]

    support_role_allocations = [
        allocation
        for allocation in employee_allocations
        if role_lookup.get(allocation.role_uuid) and role_lookup[allocation.role_uuid].type == RoleType.SUPPORT
    ]

    if not support_role_allocations:
        st.info("No support roles allocated for this employee.")
        return

    st.markdown("#### Support allocations")
    support_items = [
        support
        for support in support_allocations
        if support.allocation_uuid in {allocation.uuid for allocation in support_role_allocations}
    ]

    allocation_select_options = {
        allocation.uuid: f"{role_lookup[allocation.role_uuid].name} ({allocation.percentage * 100:.0f}%)"
        for allocation in support_role_allocations
    }
    process_options = {process.uuid: process.name for process in processes}

    def save_support(updated_subset: List[SupportAllocation]) -> None:
        managed_allocation_ids = {allocation.uuid for allocation in support_role_allocations}
        remaining_support = [
            support
            for support in support_allocations
            if support.allocation_uuid not in managed_allocation_ids
        ]
        merged_support = remaining_support + updated_subset
        update_data("support_allocations", merged_support)
        support_allocations[:] = merged_support

    _smart_editor(
        items=support_items,
        model_cls=SupportAllocation,
        dataset_key=f"support_{selected_employee_uuid}",
        save_label="Save support allocations",
        column_labels={
            "allocation_uuid": "Role allocation",
            "process_uuid": "Process",
            "percentage": "Allocation %",
            "weight": "Weight",
        },
        select_options={
            "allocation_uuid": allocation_select_options,
            "process_uuid": process_options,
        },
        number_columns={
            "percentage": {"min_value": 0.0, "max_value": 1.0, "step": 0.05, "format": "%.0f%%"},
            "weight": {"min_value": 0.0, "step": 0.1},
        },
        uuid_prefix="supp",
        labeler=lambda record: process_options.get(record.get("process_uuid"), record.get("uuid", "")),
        key=f"support_editor_{selected_employee_uuid}",
        entity_label=f"Support allocations for {employee_name}",
        on_save=save_support,
    )


def render_expertise(
    expertise: List[EmployeeExpertise],
    employees: List[Employee],
    processes: List[Process],
):
    st.subheader("Employee expertise")

    if not employees or not processes:
        st.info("Add employees and processes to manage expertise records.")
        return

    employee_options = {
        employee.uuid: (
            f"{employee.first_name} {employee.last_name}".strip()
            or employee.trigram
            or employee.uuid
        )
        for employee in employees
    }
    process_options = {process.uuid: process.name for process in processes}

    def _label(record: Dict[str, Any]) -> str:
        employee_label = employee_options.get(record.get("employee_uuid"), "Unknown employee")
        process_label = process_options.get(record.get("process_uuid"), "Unknown process")
        return f"{employee_label} → {process_label}"

    _smart_editor(
        items=expertise,
        model_cls=EmployeeExpertise,
        dataset_key="expertise_levels",
        save_label="Save expertise",
        column_labels={
            "employee_uuid": "Employee",
            "process_uuid": "Process",
            "level": "Level",
            "start_date": "Start date",
            "end_date": "End date",
        },
        select_options={
            "employee_uuid": employee_options,
            "process_uuid": process_options,
        },
        number_columns={
            "level": {"min_value": 1, "max_value": 5, "step": 1, "format": "%d"},
        },
        uuid_prefix="expertise",
        labeler=_label,
        date_columns=["start_date", "end_date"],
    )


def main():
    st.title("Data Management")
    data = get_data()
    tabs = st.tabs(
        [
            "Employees",
            "Offices",
            "Roles",
            "Apps",
            "Processes",
            "Required Coverage",
            "Allocations",
            "Expertise",
        ]
    )

    with tabs[0]:
        render_employees(data["employees"], data["offices"])
    with tabs[1]:
        render_offices(data["offices"])
    with tabs[2]:
        render_roles(data["roles"])
    with tabs[3]:
        render_apps(data["apps"])
    with tabs[4]:
        render_processes(data["processes"], data["apps"])
    with tabs[5]:
        render_required_coverage(data["coverage"], data["processes"], data["offices"])
    with tabs[6]:
        render_allocations(
            data["employees"],
            data["roles"],
            data["processes"],
            data["allocations"],
            data["support_allocations"],
        )
    with tabs[7]:
        render_expertise(
            data["expertise_levels"],
            data["employees"],
            data["processes"],
        )


if __name__ == "__main__":
    main()
