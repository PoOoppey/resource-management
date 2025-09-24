from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import pandas as pd
import streamlit as st

from models import AdjustmentType, Scenario, ScenarioAdjustment
from services.coverage import compute_theoretical_coverage
from services.data_loader import get_data, update_data
from services.scenario import apply_scenario
from utils.relations import (
    FOREIGN_KEY_RELATIONS,
    build_reference_lookup,
    enrich_payload,
)
from utils.styling import coverage_style


DIFF_COLUMN_LABEL = "Diff vs baseline"

DISPLAY_DATASETS: List[Tuple[str, str]] = [
    ("employees", "Employees"),
    ("allocations", "Allocations"),
    ("processes", "Processes"),
    ("coverage", "Required coverage"),
]

NUMERIC_PRECISION = 2


ADJUSTMENT_MAPPING: Dict[str, Tuple[AdjustmentType, AdjustmentType, AdjustmentType]] = {
    "employees": (
        AdjustmentType.ADD_EMPLOYEE,
        AdjustmentType.REMOVE_EMPLOYEE,
        AdjustmentType.UPDATE_EMPLOYEE,
    ),
    "allocations": (
        AdjustmentType.ADD_ALLOCATION,
        AdjustmentType.REMOVE_ALLOCATION,
        AdjustmentType.UPDATE_ALLOCATION,
    ),
    "support_allocations": (
        AdjustmentType.ADD_SUPPORT_ALLOCATION,
        AdjustmentType.REMOVE_SUPPORT_ALLOCATION,
        AdjustmentType.UPDATE_SUPPORT_ALLOCATION,
    ),
    "processes": (
        AdjustmentType.ADD_PROCESS,
        AdjustmentType.REMOVE_PROCESS,
        AdjustmentType.UPDATE_PROCESS,
    ),
    "coverage": (
        AdjustmentType.ADD_REQUIRED_COVERAGE,
        AdjustmentType.REMOVE_REQUIRED_COVERAGE,
        AdjustmentType.UPDATE_REQUIRED_COVERAGE,
    ),
}

UUID_PREFIXES: Dict[str, str] = {
    "employees": "emp",
    "allocations": "alloc",
    "support_allocations": "supp",
    "processes": "proc",
    "coverage": "cov",
}


def _serialize_record(item) -> Dict:
    if is_dataclass(item):
        record = asdict(item)
    elif hasattr(item, "__dict__"):
        record = {**item.__dict__}
    else:
        record = dict(item)
    return record


def _items_to_dataframe(items: Iterable) -> pd.DataFrame:
    records = [_serialize_record(item) for item in items]
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _is_missing(value) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
    except Exception:
        return False
    if isinstance(result, (bool, int)):
        return bool(result)
    if hasattr(result, "all"):
        try:
            return bool(result.all())
        except ValueError:
            return False
    return False


def _normalize_value(value):
    if _is_missing(value):
        return None
    if isinstance(value, float):
        return round(value, 6)
    return value


def _stringify_nullable(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _combined_dataset_frame(
    dataset: str, baseline_data: Dict[str, Iterable], modified_data: Dict[str, Iterable]
) -> pd.DataFrame:
    baseline_df = _items_to_dataframe(baseline_data.get(dataset, [])).copy()
    modified_df = _items_to_dataframe(modified_data.get(dataset, [])).copy()

    frames = [df for df in (modified_df, baseline_df) if not df.empty]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "uuid" in combined.columns:
        combined["uuid"] = combined["uuid"].apply(_stringify_nullable)
        combined = combined.drop_duplicates(subset="uuid", keep="first")
    return combined


def _build_office_labels(
    baseline_data: Dict[str, Iterable], modified_data: Dict[str, Iterable]
) -> Dict[str, str]:
    offices_df = _combined_dataset_frame("offices", baseline_data, modified_data)
    labels: Dict[str, str] = {}
    if offices_df.empty or "uuid" not in offices_df.columns:
        return labels
    for _, row in offices_df.iterrows():
        identifier = _stringify_nullable(row.get("uuid"))
        if not identifier:
            continue
        name = _stringify_nullable(row.get("name"))
        code = _stringify_nullable(row.get("code"))
        labels[identifier] = name or code or identifier
    return labels


def _build_role_labels(
    baseline_data: Dict[str, Iterable], modified_data: Dict[str, Iterable]
) -> Dict[str, str]:
    roles_df = _combined_dataset_frame("roles", baseline_data, modified_data)
    labels: Dict[str, str] = {}
    if roles_df.empty or "uuid" not in roles_df.columns:
        return labels
    for _, row in roles_df.iterrows():
        identifier = _stringify_nullable(row.get("uuid"))
        if not identifier:
            continue
        name = _stringify_nullable(row.get("name"))
        labels[identifier] = name or identifier
    return labels


def _build_process_labels(
    baseline_data: Dict[str, Iterable], modified_data: Dict[str, Iterable]
) -> Dict[str, str]:
    processes_df = _combined_dataset_frame("processes", baseline_data, modified_data)
    labels: Dict[str, str] = {}
    if processes_df.empty or "uuid" not in processes_df.columns:
        return labels
    for _, row in processes_df.iterrows():
        identifier = _stringify_nullable(row.get("uuid"))
        if not identifier:
            continue
        name = _stringify_nullable(row.get("name"))
        key = _stringify_nullable(row.get("key"))
        labels[identifier] = name or key or identifier
    return labels


def _build_employee_labels(
    baseline_data: Dict[str, Iterable],
    modified_data: Dict[str, Iterable],
    office_labels: Dict[str, str],
) -> Dict[str, str]:
    employees_df = _combined_dataset_frame("employees", baseline_data, modified_data)
    labels: Dict[str, str] = {}
    if employees_df.empty or "uuid" not in employees_df.columns:
        return labels
    for _, row in employees_df.iterrows():
        identifier = _stringify_nullable(row.get("uuid"))
        if not identifier:
            continue
        first = _stringify_nullable(row.get("first_name"))
        last = _stringify_nullable(row.get("last_name"))
        full_name = " ".join(part for part in [first, last] if part).strip()
        email = _stringify_nullable(row.get("email"))
        base_label = full_name or email or identifier
        office_uuid = _stringify_nullable(row.get("office_uuid"))
        office_label = office_labels.get(office_uuid, "")
        labels[identifier] = (
            f"{base_label} – {office_label}" if office_label else base_label
        )
    return labels


def _build_allocation_labels(
    baseline_data: Dict[str, Iterable],
    modified_data: Dict[str, Iterable],
    employee_labels: Dict[str, str],
    role_labels: Dict[str, str],
) -> Dict[str, str]:
    allocations_df = _combined_dataset_frame("allocations", baseline_data, modified_data)
    labels: Dict[str, str] = {}
    if allocations_df.empty or "uuid" not in allocations_df.columns:
        return labels
    for _, row in allocations_df.iterrows():
        identifier = _stringify_nullable(row.get("uuid"))
        if not identifier:
            continue
        employee_uuid = _stringify_nullable(row.get("employee_uuid"))
        role_uuid = _stringify_nullable(row.get("role_uuid"))
        employee_label = employee_labels.get(employee_uuid, employee_uuid)
        role_label = role_labels.get(role_uuid, role_uuid)
        pieces = [part for part in [employee_label, role_label] if part]
        base_label = " • ".join(pieces) if pieces else identifier
        percentage = row.get("percentage")
        display_label = base_label
        try:
            percentage_value = float(percentage)
            if not pd.isna(percentage_value):
                display_label = f"{base_label} ({percentage_value * 100:.0f}%)"
        except (TypeError, ValueError):
            pass
        labels[identifier] = display_label or identifier
    return labels


def _build_label_maps(
    baseline_data: Dict[str, Iterable], modified_data: Dict[str, Iterable]
) -> Dict[str, Dict[str, str]]:
    office_labels = _build_office_labels(baseline_data, modified_data)
    role_labels = _build_role_labels(baseline_data, modified_data)
    process_labels = _build_process_labels(baseline_data, modified_data)
    employee_labels = _build_employee_labels(
        baseline_data, modified_data, office_labels
    )
    allocation_labels = _build_allocation_labels(
        baseline_data, modified_data, employee_labels, role_labels
    )
    return {
        "offices": office_labels,
        "roles": role_labels,
        "processes": process_labels,
        "employees": employee_labels,
        "allocations": allocation_labels,
    }


def _make_selectbox_column(
    label: str, options: Dict[str, str]
) -> st.column_config.SelectboxColumn:
    option_values = [""]
    option_values.extend(key for key in options.keys() if key)
    seen: set[str] = set()
    ordered_options: list[str] = []
    for value in option_values:
        if value in seen:
            continue
        seen.add(value)
        ordered_options.append(value)

    def _format(value: str) -> str:
        if not value:
            return "—"
        return options.get(value, value)

    return st.column_config.SelectboxColumn(
        label,
        options=ordered_options,
        format_func=_format,
    )


def _format_value(value) -> str:
    if _is_missing(value):
        return "—"
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value)}"
        return f"{value:.{NUMERIC_PRECISION}f}".rstrip("0").rstrip(".")
    return str(value)


def _format_coverage_delta(delta: float | None) -> str:
    if delta is None or pd.isna(delta):
        return ""
    if abs(delta) < 1e-9:
        return "0"
    if abs(delta) < 1:
        formatted = f"{abs(delta):.2f}".rstrip("0").rstrip(".")
    else:
        formatted = f"{abs(delta):.1f}".rstrip("0").rstrip(".")
    return formatted


def _format_value_with_change(
    scenario_value: float | None, baseline_value: float | None
) -> str:
    scenario_missing = _is_missing(scenario_value)
    baseline_missing = _is_missing(baseline_value)

    if scenario_missing and baseline_missing:
        return "—"

    scenario_numeric = 0.0 if scenario_missing else float(scenario_value)
    baseline_numeric = 0.0 if baseline_missing else float(baseline_value)

    formatted_value = _format_coverage_value(scenario_numeric)
    delta = scenario_numeric - baseline_numeric
    delta_formatted = _format_coverage_delta(delta)

    if not delta_formatted:
        return formatted_value

    if abs(delta) < 1e-9:
        diff_text = "0"
    elif delta > 0:
        diff_text = f"+{delta_formatted}"
    else:
        diff_text = f"-{delta_formatted}"
    return f"{formatted_value} ({diff_text})"


def _summarize_differences(
    *,
    columns: Iterable[str],
    baseline_record: Optional[Dict[str, object]],
    scenario_record: Optional[Dict[str, object]],
) -> str:
    if baseline_record is None and scenario_record is None:
        return "Unchanged"

    if baseline_record is None:
        return "Added"
    if scenario_record is None:
        return "Removed"

    differences: List[str] = []
    for column in columns:
        base_value = baseline_record.get(column)
        scenario_value = scenario_record.get(column)
        if _normalize_value(base_value) != _normalize_value(scenario_value):
            differences.append(
                f"{column}: {_format_value(base_value)} -> {_format_value(scenario_value)}"
            )

    if not differences:
        return "Unchanged"
    return ", ".join(differences)


def _annotate_with_diff(
    baseline_df: pd.DataFrame, scenario_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if baseline_df is None:
        baseline_df = pd.DataFrame()
    if scenario_df is None:
        scenario_df = pd.DataFrame()

    baseline_df = baseline_df.copy()
    scenario_df = scenario_df.copy()

    if "uuid" not in baseline_df.columns and not baseline_df.empty:
        baseline_df.insert(0, "uuid", "")
    if "uuid" not in scenario_df.columns and not scenario_df.empty:
        scenario_df.insert(0, "uuid", "")

    baseline_records = (
        baseline_df.where(pd.notna(baseline_df), None).to_dict(orient="records")
        if not baseline_df.empty
        else []
    )
    scenario_records = (
        scenario_df.where(pd.notna(scenario_df), None).to_dict(orient="records")
        if not scenario_df.empty
        else []
    )

    baseline_map = {
        record.get("uuid", f"baseline-{index}"): record
        for index, record in enumerate(baseline_records)
    }
    scenario_map = {
        record.get("uuid", f"scenario-{index}"): record
        for index, record in enumerate(scenario_records)
    }

    ordered_columns: List[str] = []
    for df in (baseline_df, scenario_df):
        for column in df.columns:
            if column == "uuid":
                continue
            if column not in ordered_columns:
                ordered_columns.append(column)

    annotated_rows: List[Dict[str, object]] = []
    visited_baseline_ids: set[str] = set()

    for index, scenario_record in enumerate(scenario_records):
        identifier = scenario_record.get("uuid", f"scenario-{index}")
        baseline_record = baseline_map.get(identifier)
        if baseline_record:
            visited_baseline_ids.add(identifier)

        row: Dict[str, object] = {**scenario_record}
        row[DIFF_COLUMN_LABEL] = _summarize_differences(
            columns=ordered_columns,
            baseline_record=baseline_record,
            scenario_record=scenario_record,
        )
        annotated_rows.append(row)

    removed_ids = [
        identifier
        for identifier in baseline_map.keys()
        if identifier not in visited_baseline_ids and identifier not in scenario_map
    ]

    removed_records = [baseline_map[identifier] for identifier in removed_ids]

    annotated_df = pd.DataFrame(annotated_rows) if annotated_rows else pd.DataFrame()
    removed_df = pd.DataFrame(removed_records) if removed_records else pd.DataFrame()

    if not annotated_df.empty:
        column_order = list(dict.fromkeys(list(scenario_df.columns) + [DIFF_COLUMN_LABEL]))
        annotated_df = annotated_df.reindex(columns=column_order)

    return annotated_df, removed_df


def _merge_visible_and_hidden_rows(
    visible_df: pd.DataFrame,
    full_df: pd.DataFrame,
    mask: Optional[pd.Series] = None,
) -> pd.DataFrame:
    if mask is None or full_df is None or full_df.empty:
        return visible_df

    if mask.all():
        return visible_df

    hidden_df = full_df.loc[~mask].copy()
    if hidden_df.empty:
        return visible_df

    if "uuid" in hidden_df.columns and "uuid" in visible_df.columns:
        visible_ids = visible_df["uuid"].fillna("").astype(str)
        hidden_df = hidden_df[~hidden_df["uuid"].fillna("").astype(str).isin(visible_ids)]

    merged = pd.concat([visible_df, hidden_df], ignore_index=True, sort=False)
    return merged


def _render_allocation_tab(
    *,
    scenario: Scenario,
    baseline_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    baseline_support_df: pd.DataFrame,
    scenario_support_df: pd.DataFrame,
    baseline_data: Dict[str, Iterable],
    modified_data: Dict[str, Iterable],
    dataset: str,
    support_dataset: str = "support_allocations",
    label_maps: Dict[str, Dict[str, str]],
    show_differences_only: bool,
) -> None:
    employee_labels = label_maps.get("employees", {})
    role_labels = label_maps.get("roles", {})
    process_labels = label_maps.get("processes", {})
    allocation_labels = label_maps.get("allocations", {})

    st.markdown("#### Role allocations overview")

    base_columns = list(
        dict.fromkeys(list(scenario_df.columns) + list(baseline_df.columns))
    )
    if "uuid" not in base_columns:
        base_columns.insert(0, "uuid")

    annotated_df, removed_df = _annotate_with_diff(baseline_df, scenario_df)
    column_order = list(dict.fromkeys(base_columns + [DIFF_COLUMN_LABEL]))

    if annotated_df.empty:
        summary_df = pd.DataFrame(columns=column_order)
    else:
        summary_df = annotated_df.reindex(columns=column_order)
        summary_df[DIFF_COLUMN_LABEL] = summary_df[DIFF_COLUMN_LABEL].fillna("Unchanged")

    if not removed_df.empty:
        removed_summary = removed_df.reindex(columns=column_order).copy()
        removed_summary[DIFF_COLUMN_LABEL] = "Removed"
        summary_df = pd.concat(
            [summary_df, removed_summary], ignore_index=True, sort=False
        )

    for column in ["uuid", "employee_uuid", "role_uuid"]:
        if column in summary_df.columns:
            summary_df[column] = summary_df[column].apply(_stringify_nullable)

    display_df = summary_df.copy()
    if "employee_uuid" in display_df.columns:
        display_df["Employee"] = display_df["employee_uuid"].apply(
            lambda value: employee_labels.get(value, value) if value else "—"
        )
    if "role_uuid" in display_df.columns:
        display_df["Role"] = display_df["role_uuid"].apply(
            lambda value: role_labels.get(value, value) if value else "—"
        )
    if "percentage" in display_df.columns:
        display_df["Allocation %"] = display_df["percentage"].apply(
            lambda value: "—"
            if _is_missing(value)
            else f"{float(value) * 100:.0f}%"
        )
    if "weight" in display_df.columns:
        display_df["Weight"] = display_df["weight"]

    overview_columns = [
        column
        for column in ["Employee", "Role", "Allocation %", "Weight", DIFF_COLUMN_LABEL]
        if column in display_df.columns
    ]

    if show_differences_only and DIFF_COLUMN_LABEL in display_df.columns:
        display_df = display_df[display_df[DIFF_COLUMN_LABEL] != "Unchanged"]

    if display_df.empty:
        st.info("No allocation differences to display for the selected criteria.")
    else:
        st.dataframe(
            display_df[overview_columns],
            use_container_width=True,
            hide_index=True,
        )

    available_employee_ids: set[str] = set()
    for df in (baseline_df, scenario_df):
        if df.empty or "employee_uuid" not in df.columns:
            continue
        ids = df["employee_uuid"].apply(_stringify_nullable)
        available_employee_ids.update(value for value in ids if value)

    if not available_employee_ids:
        st.info("No role allocations available to edit.")
        return

    def _label_for_employee(identifier: str) -> str:
        return employee_labels.get(identifier, identifier)

    ordered_employee_ids = sorted(
        available_employee_ids, key=lambda value: _label_for_employee(value).lower()
    )
    employee_options = {
        _label_for_employee(identifier): identifier
        for identifier in ordered_employee_ids
    }

    selected_label = st.selectbox(
        "Employee", list(employee_options.keys()), key=f"allocation_employee_{scenario.uuid}"
    )
    selected_employee = employee_options[selected_label]

    scenario_allocations = scenario_df.copy()
    baseline_allocations = baseline_df.copy()
    for df in (scenario_allocations, baseline_allocations):
        if df.empty:
            continue
        if "employee_uuid" in df.columns:
            df["employee_uuid"] = df["employee_uuid"].apply(_stringify_nullable)
        if "uuid" in df.columns:
            df["uuid"] = df["uuid"].apply(_stringify_nullable)

    employee_allocations = scenario_allocations.copy()
    if "employee_uuid" in employee_allocations.columns:
        employee_allocations = employee_allocations[
            employee_allocations["employee_uuid"] == selected_employee
        ]
    if employee_allocations.empty:
        employee_allocations = pd.DataFrame(columns=base_columns)
    else:
        employee_allocations = employee_allocations.reindex(columns=base_columns)

    allocation_column_config: Dict[str, Any] = {}
    if "uuid" in employee_allocations.columns:
        allocation_column_config["uuid"] = st.column_config.TextColumn(
            "UUID", disabled=True
        )
    if "employee_uuid" in employee_allocations.columns:
        allocation_column_config["employee_uuid"] = _make_selectbox_column(
            "Employee",
            {selected_employee: _label_for_employee(selected_employee)},
        )
    if "role_uuid" in employee_allocations.columns:
        allocation_column_config["role_uuid"] = _make_selectbox_column(
            "Role", role_labels
        )
    if "percentage" in employee_allocations.columns:
        allocation_column_config["percentage"] = st.column_config.NumberColumn(
            "Allocation %",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            format="%.0f%%",
        )
    if "weight" in employee_allocations.columns:
        allocation_column_config["weight"] = st.column_config.NumberColumn(
            "Weight", min_value=0.0, step=0.1
        )

    st.markdown("##### Allocation details")
    allocation_editor_df = st.data_editor(
        employee_allocations,
        num_rows="dynamic",
        use_container_width=True,
        hide_index="uuid" in employee_allocations.columns,
        column_config=allocation_column_config or None,
        key=f"scenario_allocations_editor_{scenario.uuid}_{selected_employee}",
    )

    removed_allocations = pd.DataFrame()
    if not removed_df.empty:
        removed_allocations = removed_df.copy()
        if "employee_uuid" in removed_allocations.columns:
            removed_allocations["employee_uuid"] = removed_allocations[
                "employee_uuid"
            ].apply(_stringify_nullable)
            removed_allocations = removed_allocations[
                removed_allocations["employee_uuid"] == selected_employee
            ]
        if not removed_allocations.empty:
            preview = removed_allocations.copy()
            if "role_uuid" in preview.columns:
                preview["Role"] = preview["role_uuid"].apply(
                    lambda value: role_labels.get(value, value) if value else "—"
                )
            if "percentage" in preview.columns:
                preview["Allocation %"] = preview["percentage"].apply(
                    lambda value: "—"
                    if _is_missing(value)
                    else f"{float(value) * 100:.0f}%"
                )
            st.caption("Removed allocations in this scenario")
            st.dataframe(
                preview[[col for col in ["Role", "Allocation %"] if col in preview.columns]],
                use_container_width=True,
                hide_index=True,
            )

    if st.button("Save allocation changes", key=f"save_allocations_{scenario.uuid}"):
        edited_df = allocation_editor_df.copy().reindex(columns=base_columns)
        if "employee_uuid" in edited_df.columns:
            edited_df["employee_uuid"] = edited_df["employee_uuid"].apply(
                lambda value: selected_employee
                if not value or not str(value).strip()
                else _stringify_nullable(value)
            )

        other_allocations = scenario_allocations.copy()
        if "employee_uuid" in other_allocations.columns:
            other_allocations = other_allocations[
                other_allocations["employee_uuid"] != selected_employee
            ]
        if not other_allocations.empty:
            other_allocations = other_allocations.reindex(columns=base_columns)

        combined_allocations = pd.concat(
            [other_allocations, edited_df], ignore_index=True, sort=False
        )
        combined_allocations = combined_allocations.reindex(columns=base_columns)

        new_adjustments = _calculate_dataset_adjustments(
            dataset,
            baseline_df,
            combined_allocations,
            baseline_data,
            modified_data,
        )
        _merge_adjustments(scenario, dataset, new_adjustments)
        update_data("scenarios", baseline_data["scenarios"])
        st.success("Scenario allocations saved.")
        st.experimental_rerun()

    st.markdown("#### Support allocations")

    support_base_columns = list(
        dict.fromkeys(list(scenario_support_df.columns) + list(baseline_support_df.columns))
    )
    if "uuid" not in support_base_columns:
        support_base_columns.insert(0, "uuid")
    if "allocation_uuid" not in support_base_columns:
        support_base_columns.insert(0, "allocation_uuid")

    scenario_support = scenario_support_df.copy()
    baseline_support = baseline_support_df.copy()
    for df in (scenario_support, baseline_support):
        if df.empty:
            continue
        for column in ["uuid", "allocation_uuid", "process_uuid"]:
            if column in df.columns:
                df[column] = df[column].apply(_stringify_nullable)

    relevant_allocation_ids: set[str] = set()
    if {"uuid", "employee_uuid"}.issubset(scenario_allocations.columns):
        relevant_allocation_ids.update(
            scenario_allocations.loc[
                scenario_allocations["employee_uuid"] == selected_employee, "uuid"
            ].dropna().astype(str)
        )
    if {"uuid", "employee_uuid"}.issubset(baseline_allocations.columns):
        relevant_allocation_ids.update(
            baseline_allocations.loc[
                baseline_allocations["employee_uuid"] == selected_employee, "uuid"
            ].dropna().astype(str)
        )
    relevant_allocation_ids = {identifier for identifier in relevant_allocation_ids if identifier}

    allocation_option_map = {
        identifier: allocation_labels.get(identifier, identifier)
        for identifier in sorted(relevant_allocation_ids)
    }

    scenario_support_filtered = scenario_support.copy()
    baseline_support_filtered = baseline_support.copy()
    if "allocation_uuid" in scenario_support_filtered.columns:
        scenario_support_filtered = scenario_support_filtered[
            scenario_support_filtered["allocation_uuid"].isin(relevant_allocation_ids)
        ]
    if "allocation_uuid" in baseline_support_filtered.columns:
        baseline_support_filtered = baseline_support_filtered[
            baseline_support_filtered["allocation_uuid"].isin(relevant_allocation_ids)
        ]

    scenario_support_filtered = (
        scenario_support_filtered.reindex(columns=support_base_columns)
        if not scenario_support_filtered.empty
        else pd.DataFrame(columns=support_base_columns)
    )

    support_column_config: Dict[str, Any] = {}
    if "uuid" in scenario_support_filtered.columns:
        support_column_config["uuid"] = st.column_config.TextColumn(
            "UUID", disabled=True
        )
    if "allocation_uuid" in scenario_support_filtered.columns:
        support_column_config["allocation_uuid"] = _make_selectbox_column(
            "Role allocation", allocation_option_map or allocation_labels
        )
    if "process_uuid" in scenario_support_filtered.columns:
        support_column_config["process_uuid"] = _make_selectbox_column(
            "Process", process_labels
        )
    if "percentage" in scenario_support_filtered.columns:
        support_column_config["percentage"] = st.column_config.NumberColumn(
            "Allocation %",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            format="%.0f%%",
        )
    if "weight" in scenario_support_filtered.columns:
        support_column_config["weight"] = st.column_config.NumberColumn(
            "Weight", min_value=0.0, step=0.1
        )

    st.caption("Support allocations linked to the selected employee")
    support_editor_df = st.data_editor(
        scenario_support_filtered,
        num_rows="dynamic",
        use_container_width=True,
        hide_index="uuid" in scenario_support_filtered.columns,
        column_config=support_column_config or None,
        key=f"scenario_support_editor_{scenario.uuid}_{selected_employee}",
    )

    _, support_removed_df = _annotate_with_diff(
        baseline_support_filtered, scenario_support_filtered
    )
    if not support_removed_df.empty:
        support_preview = support_removed_df.copy()
        if "allocation_uuid" in support_preview.columns:
            support_preview["Role allocation"] = support_preview["allocation_uuid"].apply(
                lambda value: allocation_labels.get(value, value) if value else "—"
            )
        if "process_uuid" in support_preview.columns:
            support_preview["Process"] = support_preview["process_uuid"].apply(
                lambda value: process_labels.get(value, value) if value else "—"
            )
        if "percentage" in support_preview.columns:
            support_preview["Allocation %"] = support_preview["percentage"].apply(
                lambda value: "—"
                if _is_missing(value)
                else f"{float(value) * 100:.0f}%"
            )
        st.caption("Removed support allocations in this scenario")
        st.dataframe(
            support_preview[
                [
                    column
                    for column in ["Role allocation", "Process", "Allocation %"]
                    if column in support_preview.columns
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    if st.button(
        "Save support allocation changes",
        key=f"save_support_{scenario.uuid}_{selected_employee}",
    ):
        edited_support_df = support_editor_df.copy().reindex(columns=support_base_columns)
        for column in ["allocation_uuid", "process_uuid"]:
            if column in edited_support_df.columns:
                edited_support_df[column] = edited_support_df[column].apply(
                    _stringify_nullable
                )

        other_support = scenario_support.copy()
        if "allocation_uuid" in other_support.columns:
            other_support = other_support[
                ~other_support["allocation_uuid"].isin(relevant_allocation_ids)
            ]
        if not other_support.empty:
            other_support = other_support.reindex(columns=support_base_columns)

        combined_support = pd.concat(
            [other_support, edited_support_df], ignore_index=True, sort=False
        )
        combined_support = combined_support.reindex(columns=support_base_columns)

        new_adjustments = _calculate_dataset_adjustments(
            support_dataset,
            baseline_support_df,
            combined_support,
            baseline_data,
            modified_data,
        )
        _merge_adjustments(scenario, support_dataset, new_adjustments)
        update_data("scenarios", baseline_data["scenarios"])
        st.success("Scenario support allocations saved.")
        st.experimental_rerun()


def _render_generic_dataset_tab(
    *,
    scenario: Scenario,
    dataset: str,
    label: str,
    baseline_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    baseline_data: Dict[str, Iterable],
    modified_data: Dict[str, Iterable],
    show_differences_only: bool,
    label_maps: Dict[str, Dict[str, str]],
) -> None:
    relations = FOREIGN_KEY_RELATIONS.get(dataset, {})
    annotated_df, removed_df = _annotate_with_diff(baseline_df, scenario_df)

    base_columns = list(dict.fromkeys(list(scenario_df.columns) + list(baseline_df.columns)))
    column_order = list(dict.fromkeys(base_columns + [DIFF_COLUMN_LABEL]))

    if annotated_df.empty:
        full_display_df = pd.DataFrame(columns=column_order)
    else:
        full_display_df = annotated_df.reindex(columns=column_order)
        full_display_df[DIFF_COLUMN_LABEL] = full_display_df[DIFF_COLUMN_LABEL].fillna(
            "Unchanged"
        )

    if not full_display_df.empty:
        for field in relations:
            if field in full_display_df.columns:
                full_display_df[field] = full_display_df[field].apply(_stringify_nullable)

    diff_mask: Optional[pd.Series] = None
    if not full_display_df.empty:
        diff_mask = full_display_df[DIFF_COLUMN_LABEL] != "Unchanged"

    if show_differences_only and diff_mask is not None:
        display_df = full_display_df.loc[diff_mask].copy()
    else:
        display_df = full_display_df.copy()

    if not display_df.empty:
        display_df = display_df.reindex(columns=column_order)

    for field in relations:
        if field in display_df.columns:
            display_df[field] = display_df[field].apply(_stringify_nullable)

    column_config: Dict[str, Any] = {}
    if "uuid" in display_df.columns:
        column_config["uuid"] = st.column_config.TextColumn("UUID", disabled=True)
    for field, (related_dataset, alias) in relations.items():
        options = label_maps.get(related_dataset, {})
        if field in display_df.columns:
            label_text = alias.replace("_", " ").title()
            if options:
                column_config[field] = _make_selectbox_column(label_text, options)
            else:
                column_config.setdefault(field, st.column_config.TextColumn(label_text))
    column_config[DIFF_COLUMN_LABEL] = st.column_config.TextColumn(
        DIFF_COLUMN_LABEL, disabled=True
    )

    editor_df = st.data_editor(
        display_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index="uuid" in display_df.columns,
        column_config=column_config or None,
        key=f"scenario_editor_{scenario.uuid}_{dataset}",
    )

    if not removed_df.empty:
        removed_preview = removed_df.copy()
        for column in removed_preview.columns:
            removed_preview[column] = removed_preview[column].apply(_stringify_nullable)

        labeled_columns: Dict[str, str] = {}
        for field, (related_dataset, alias) in relations.items():
            options = label_maps.get(related_dataset, {})
            if field in removed_preview.columns:
                label_column = alias.replace("_", " ").title()
                labeled_columns[label_column] = field
                removed_preview[label_column] = removed_preview[field].apply(
                    lambda value: options.get(value, value) if value else "—"
                )

        ordered_preview_columns = list(removed_preview.columns)
        for label_column, source_column in labeled_columns.items():
            if label_column in ordered_preview_columns and source_column in ordered_preview_columns:
                ordered_preview_columns.remove(label_column)
                source_index = ordered_preview_columns.index(source_column)
                ordered_preview_columns.insert(source_index + 1, label_column)

        st.caption("Removed entries in this scenario")
        st.dataframe(
            removed_preview[ordered_preview_columns],
            use_container_width=True,
            hide_index="uuid" in removed_preview.columns,
        )

    if st.button(f"Save {label.lower()} changes", key=f"save_{scenario.uuid}_{dataset}"):
        edited_df = editor_df.copy()
        if show_differences_only and diff_mask is not None:
            edited_df = _merge_visible_and_hidden_rows(edited_df, full_display_df, diff_mask)
        if DIFF_COLUMN_LABEL in edited_df.columns:
            edited_df = edited_df.drop(columns=[DIFF_COLUMN_LABEL])

        new_adjustments = _calculate_dataset_adjustments(
            dataset,
            baseline_df,
            edited_df,
            baseline_data,
            modified_data,
        )
        _merge_adjustments(scenario, dataset, new_adjustments)
        update_data("scenarios", baseline_data["scenarios"])
        st.success(f"Scenario adjustments for {label.lower()} saved.")
        st.experimental_rerun()


def _scenario_select(scenarios: List[Scenario]) -> Scenario | None:
    if not scenarios:
        return None
    options = {scenario.name: scenario for scenario in scenarios}
    choice = st.selectbox("Scenario", list(options.keys()))
    return options[choice]


def _format_coverage_value(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    if abs(value) < 1:
        return f"{value:.2f}"
    return f"{value:.1f}"


def _build_coverage_result(
    baseline: pd.DataFrame, scenario: pd.DataFrame
) -> pd.DataFrame:
    baseline = baseline.copy()
    scenario = scenario.copy()

    baseline_keys = [col for col in ["Region", "Office", "Process"] if col in baseline.columns]
    scenario_keys = [col for col in ["Region", "Office", "Process"] if col in scenario.columns]
    merge_keys = [col for col in ["Region", "Office", "Process"] if col in baseline_keys and col in scenario_keys]
    display_columns = [
        col
        for col in ["Region", "Office", "Process"]
        if col in baseline.columns or col in scenario.columns
    ]

    if merge_keys:
        merged = baseline.merge(
            scenario,
            on=merge_keys,
            how="outer",
            suffixes=(" (Baseline)", " (Scenario)"),
        )
    else:
        merged = pd.concat(
            [
                baseline.add_suffix(" (Baseline)") if not baseline.empty else pd.DataFrame(),
                scenario.add_suffix(" (Scenario)") if not scenario.empty else pd.DataFrame(),
            ],
            axis=1,
        )

    rename_map: Dict[str, str] = {}
    if "Required" in merged.columns:
        rename_map["Required"] = "Required (Baseline)"
    if "Coverage" in merged.columns:
        rename_map["Coverage"] = "Coverage (Baseline)"
    merged = merged.rename(columns=rename_map)

    if "Required (Scenario)" not in merged.columns and "Required" in scenario.columns:
        merged = merged.rename(columns={"Required": "Required (Scenario)"})
    if "Coverage (Scenario)" not in merged.columns and "Coverage" in scenario.columns:
        merged = merged.rename(columns={"Coverage": "Coverage (Scenario)"})

    rows: List[Dict[str, object]] = []
    for _, row in merged.iterrows():
        entry: Dict[str, object] = {}
        for column in display_columns:
            if column in row.index:
                entry[column] = row[column]
        entry["Required"] = _format_value_with_change(
            row.get("Required (Scenario)"), row.get("Required (Baseline)")
        )
        entry["Coverage"] = _format_value_with_change(
            row.get("Coverage (Scenario)"), row.get("Coverage (Baseline)")
        )
        rows.append(entry)

    ordered_columns = [*display_columns, "Required", "Coverage"]
    return pd.DataFrame(rows, columns=ordered_columns)


def _style_coverage_result(df: pd.DataFrame) -> pd.io.formats.style.Styler | pd.DataFrame:
    if df.empty:
        return df

    styler = df.style

    def _style_row(row: pd.Series) -> pd.Series:
        styles: Dict[str, str] = {}
        required_value = row.get("Required")
        coverage_value = row.get("Coverage")
        if "Coverage" in df.columns:
            styles["Coverage"] = coverage_style(required_value, coverage_value)
        if "Required" in df.columns:
            styles.setdefault("Required", "font-weight: 600")
        return pd.Series(styles)

    styler = styler.apply(_style_row, axis=1)
    styler = styler.hide(axis="index")
    return styler


def _generate_uuid(prefix: str | None = None) -> str:
    token = uuid4().hex[:8]
    return f"{prefix}-{token}" if prefix else uuid4().hex


def _prepare_records(df: pd.DataFrame) -> Dict[str, Dict]:
    if df.empty:
        return {}
    normalized_df = df.copy()
    if "uuid" not in normalized_df.columns:
        normalized_df.insert(0, "uuid", "")
    normalized_df = normalized_df.where(pd.notna(normalized_df), None)
    records = normalized_df.to_dict(orient="records")
    prepared: Dict[str, Dict] = {}
    for record in records:
        identifier = str(record.get("uuid") or "").strip()
        if not identifier:
            continue
        cleaned: Dict[str, object] = {}
        for key, value in record.items():
            if hasattr(value, "item"):
                try:
                    cleaned[key] = value.item()
                    continue
                except Exception:
                    pass
            cleaned[key] = value
        prepared[identifier] = cleaned
    return prepared


def _calculate_dataset_adjustments(
    dataset: str,
    baseline_df: pd.DataFrame,
    edited_df: pd.DataFrame,
    baseline_data: Dict[str, Iterable],
    modified_data: Dict[str, Iterable],
) -> List[ScenarioAdjustment]:
    if dataset not in ADJUSTMENT_MAPPING:
        return []

    add_type, remove_type, update_type = ADJUSTMENT_MAPPING[dataset]

    reference_lookup: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if dataset in FOREIGN_KEY_RELATIONS:
        reference_lookup = build_reference_lookup(
            baseline_data=baseline_data, modified_data=modified_data
        )

    baseline_records = _prepare_records(baseline_df)
    edited_df = edited_df.copy()
    if "uuid" not in edited_df.columns:
        edited_df.insert(0, "uuid", "")

    prefix = UUID_PREFIXES.get(dataset)
    edited_df["uuid"] = edited_df["uuid"].apply(
        lambda value: value if value and str(value).strip() else _generate_uuid(prefix)
    )
    edited_records = _prepare_records(edited_df)

    baseline_ids = set(baseline_records.keys())
    edited_ids = set(edited_records.keys())

    added_ids = sorted(edited_ids - baseline_ids)
    removed_ids = sorted(baseline_ids - edited_ids)
    potential_updates = baseline_ids & edited_ids

    adjustments: List[ScenarioAdjustment] = []

    for identifier in added_ids:
        payload = {key: value for key, value in edited_records[identifier].items() if key != "uuid"}
        payload["uuid"] = identifier
        payload = enrich_payload(dataset, payload, reference_lookup)
        adjustments.append(
            ScenarioAdjustment(uuid=_generate_uuid("adj"), type=add_type, payload=payload)
        )

    for identifier in removed_ids:
        adjustments.append(
            ScenarioAdjustment(
                uuid=_generate_uuid("adj"),
                type=remove_type,
                payload={"uuid": identifier},
            )
        )

    for identifier in potential_updates:
        baseline_record = baseline_records[identifier]
        edited_record = edited_records[identifier]
        if {
            key: _normalize_value(value)
            for key, value in baseline_record.items()
            if key != "uuid"
        } == {
            key: _normalize_value(value)
            for key, value in edited_record.items()
            if key != "uuid"
        }:
            continue

        payload = {key: value for key, value in edited_record.items() if key != "uuid"}
        payload["uuid"] = identifier
        payload = enrich_payload(dataset, payload, reference_lookup)
        adjustments.append(
            ScenarioAdjustment(uuid=_generate_uuid("adj"), type=update_type, payload=payload)
        )

    return adjustments


def _merge_adjustments(
    scenario: Scenario,
    dataset: str,
    new_adjustments: List[ScenarioAdjustment],
) -> None:
    preserved: List[ScenarioAdjustment] = []
    for adjustment in scenario.adjustments:
        try:
            collection = _collection_for_adjustment(adjustment.type)
        except KeyError:
            preserved.append(adjustment)
            continue
        if collection != dataset:
            preserved.append(adjustment)
    scenario.adjustments = preserved + new_adjustments


def _collection_for_adjustment(adjustment_type: AdjustmentType) -> str:
    for collection, types in ADJUSTMENT_MAPPING.items():
        if adjustment_type in types:
            return collection
    raise KeyError(f"Unknown adjustment type: {adjustment_type}")


def main():
    st.title("Scenario Planner")

    data = get_data()
    scenarios = data.get("scenarios", [])

    if not scenarios:
        st.info("No scenarios available. Create a scenario in the data management page.")
        return

    st.subheader("Scenario selection")
    scenario = _scenario_select(scenarios)

    if scenario is None:
        st.info("Select a scenario to view its impact against the baseline.")
        return

    st.markdown("---")

    baseline_coverage = compute_theoretical_coverage(
        data,
        view="process",
        group_by="office",
        unit="hours",
    )

    modified_data = apply_scenario(data, scenario)
    label_maps = _build_label_maps(data, modified_data)
    scenario_coverage = compute_theoretical_coverage(
        modified_data,
        view="process",
        group_by="office",
        unit="hours",
    )

    st.subheader("Scenario datasets")
    show_differences_only = st.toggle("Show only rows with differences", value=False)

    tabs = st.tabs([label for _, label in DISPLAY_DATASETS])
    for (dataset, label), tab in zip(DISPLAY_DATASETS, tabs):
        with tab:
            baseline_df = _items_to_dataframe(data.get(dataset, [])).copy()
            scenario_df = _items_to_dataframe(modified_data.get(dataset, [])).copy()

            if dataset == "allocations":
                baseline_support_df = _items_to_dataframe(
                    data.get("support_allocations", [])
                ).copy()
                scenario_support_df = _items_to_dataframe(
                    modified_data.get("support_allocations", [])
                ).copy()
                _render_allocation_tab(
                    scenario=scenario,
                    baseline_df=baseline_df,
                    scenario_df=scenario_df,
                    baseline_support_df=baseline_support_df,
                    scenario_support_df=scenario_support_df,
                    baseline_data=data,
                    modified_data=modified_data,
                    dataset=dataset,
                    label_maps=label_maps,
                    show_differences_only=show_differences_only,
                )
            else:
                _render_generic_dataset_tab(
                    scenario=scenario,
                    dataset=dataset,
                    label=label,
                    baseline_df=baseline_df,
                    scenario_df=scenario_df,
                    baseline_data=data,
                    modified_data=modified_data,
                    show_differences_only=show_differences_only,
                    label_maps=label_maps,
                )

    st.subheader("Scenario result")
    coverage_result = _build_coverage_result(baseline_coverage, scenario_coverage)
    styled_coverage = _style_coverage_result(coverage_result)
    st.dataframe(styled_coverage, use_container_width=True)


if __name__ == "__main__":
    main()
