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
from utils.styling import coverage_style


DIFF_COLUMN_LABEL = "Diff vs baseline"

DISPLAY_DATASETS: List[Tuple[str, str]] = [
    ("employees", "Employees"),
    ("allocations", "Allocations"),
    ("support_allocations", "Support allocations"),
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


def _build_allocation_overview(modified_data: Dict[str, Iterable]) -> pd.DataFrame:
    employees_df = _items_to_dataframe(modified_data.get("employees", []))
    allocations_df = _items_to_dataframe(modified_data.get("allocations", []))
    roles_df = _items_to_dataframe(modified_data.get("roles", []))
    processes_df = _items_to_dataframe(modified_data.get("processes", []))
    support_df = _items_to_dataframe(modified_data.get("support_allocations", []))

    role_names: Dict[str, str] = {}
    if not roles_df.empty and {"uuid", "name"}.issubset(roles_df.columns):
        role_names = {
            str(row["uuid"]): str(row["name"]) for _, row in roles_df.iterrows()
        }

    process_names: Dict[str, str] = {}
    if not processes_df.empty and {"uuid", "name"}.issubset(processes_df.columns):
        process_names = {
            str(row["uuid"]): str(row["name"]) for _, row in processes_df.iterrows()
        }

    support_lookup: Dict[str, pd.DataFrame] = {}
    if not support_df.empty and "allocation_uuid" in support_df.columns:
        allocation_ids = support_df["allocation_uuid"].fillna("").astype(str)
        support_df = support_df.assign(allocation_uuid=allocation_ids)
        support_lookup = {
            allocation_uuid: group
            for allocation_uuid, group in support_df.groupby("allocation_uuid")
        }

    overview_rows: List[Dict[str, Any]] = []

    if employees_df.empty or "uuid" not in employees_df.columns:
        return pd.DataFrame(
            columns=["Employee", "Utilization", "Role Split", "Support Details"]
        )

    for _, employee in employees_df.iterrows():
        employee_uuid = str(employee.get("uuid", ""))
        if not employee_uuid:
            continue

        first = str(employee.get("first_name", "") or "").strip()
        last = str(employee.get("last_name", "") or "").strip()
        employee_name = f"{first} {last}".strip() or employee_uuid

        employee_allocations = pd.DataFrame()
        if not allocations_df.empty and "employee_uuid" in allocations_df.columns:
            employee_allocations = allocations_df[
                allocations_df["employee_uuid"].fillna("").astype(str) == employee_uuid
            ]

        utilization = 0.0
        role_splits: List[str] = []
        support_details: List[str] = []

        if not employee_allocations.empty:
            if "percentage" in employee_allocations.columns:
                utilization = float(
                    employee_allocations["percentage"].fillna(0.0).astype(float).sum()
                )

            for _, allocation in employee_allocations.iterrows():
                allocation_uuid = str(allocation.get("uuid", ""))
                role_uuid = str(allocation.get("role_uuid", ""))
                percentage = float(allocation.get("percentage", 0.0) or 0.0)
                role_name = role_names.get(role_uuid, role_uuid or "—")
                role_splits.append(f"{role_name} ({percentage * 100:.0f}%)")

                allocation_weight = allocation.get("weight", 1.0)
                try:
                    allocation_weight_value = float(allocation_weight)
                except (TypeError, ValueError):
                    allocation_weight_value = 1.0

                supports = support_lookup.get(allocation_uuid)
                if supports is None:
                    continue
                for _, support in supports.iterrows():
                    process_uuid = str(support.get("process_uuid", ""))
                    process_name = process_names.get(process_uuid, process_uuid or "—")
                    support_percentage = float(support.get("percentage", 0.0) or 0.0)
                    weight = support.get("weight")
                    try:
                        weight_value = float(weight)
                    except (TypeError, ValueError):
                        weight_value = allocation_weight_value
                    support_details.append(
                        f"{process_name} ({support_percentage * 100:.0f}% · w={weight_value:.2f})"
                    )

        overview_rows.append(
            {
                "employee_uuid": employee_uuid,
                "Employee": employee_name,
                "Utilization": utilization,
                "Role Split": "; ".join(role_splits) if role_splits else "—",
                "Support Details": "; ".join(support_details)
                if support_details
                else "—",
            }
        )

    if overview_rows:
        overview_df = pd.DataFrame(overview_rows).set_index("employee_uuid")
    else:
        overview_df = pd.DataFrame(
            columns=["Employee", "Utilization", "Role Split", "Support Details"]
        )
        overview_df.index.name = "employee_uuid"

    return overview_df


def _render_allocation_overview(
    modified_data: Dict[str, Iterable], scenario_uuid: str
) -> Optional[str]:
    overview_df = _build_allocation_overview(modified_data)

    selection_key = f"allocation_overview_{scenario_uuid}"
    selected_employee_key = f"selected_employee_uuid_{scenario_uuid}"

    st.caption("Select an employee to review scenario allocations and support work.")

    if overview_df.empty:
        st.info("No employees available for the current scenario.")
        st.session_state.pop(selected_employee_key, None)
        return None

    disabled_columns = list(overview_df.columns)
    st.data_editor(
        overview_df,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        disabled=disabled_columns,
        column_config={
            "Employee": st.column_config.TextColumn("Employee"),
            "Utilization": st.column_config.ProgressColumn(
                "Utilization", format="{:.0%}", min_value=0.0, max_value=1.0
            ),
            "Role Split": st.column_config.TextColumn("Roles"),
            "Support Details": st.column_config.TextColumn("Support work"),
        },
        key=selection_key,
    )

    stored_employee_uuid: Optional[str] = st.session_state.get(selected_employee_key)
    selected_employee_uuid: Optional[str] = None

    selection_state = st.session_state.get(selection_key, {})
    selected_rows = selection_state.get("selection", {}).get("rows", [])
    if selected_rows:
        row_position = selected_rows[0]
        if 0 <= row_position < len(overview_df.index):
            selected_employee_uuid = overview_df.index[row_position]
            st.session_state[selected_employee_key] = selected_employee_uuid
    elif stored_employee_uuid and stored_employee_uuid in overview_df.index:
        selected_employee_uuid = stored_employee_uuid
    elif stored_employee_uuid and stored_employee_uuid not in overview_df.index:
        st.session_state.pop(selected_employee_key, None)

    return selected_employee_uuid


def _render_allocation_tab(
    *,
    scenario: Scenario,
    baseline_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    baseline_data: Dict[str, Iterable],
    modified_data: Dict[str, Iterable],
    dataset: str,
) -> None:
    selected_employee_uuid = _render_allocation_overview(modified_data, scenario.uuid)

    st.divider()

    if not selected_employee_uuid:
        st.info("Select an employee in the table above to manage allocations.")
        return

    employees_df = _items_to_dataframe(modified_data.get("employees", []))
    employee_name = selected_employee_uuid
    if not employees_df.empty and "uuid" in employees_df.columns:
        match = employees_df[
            employees_df["uuid"].fillna("").astype(str) == selected_employee_uuid
        ]
        if not match.empty:
            first = str(match.iloc[0].get("first_name", "") or "").strip()
            last = str(match.iloc[0].get("last_name", "") or "").strip()
            full_name = f"{first} {last}".strip()
            if full_name:
                employee_name = full_name

    st.markdown(f"### {employee_name}")
    st.markdown("#### Role allocations")

    roles_df = _items_to_dataframe(modified_data.get("roles", []))
    role_options: Dict[str, str] = {}
    if not roles_df.empty and {"uuid", "name"}.issubset(roles_df.columns):
        role_options = {
            str(row["uuid"]): str(row.get("name", row["uuid"]))
            for _, row in roles_df.iterrows()
        }

    def _filter_by_employee(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "employee_uuid" not in df.columns:
            return pd.DataFrame(columns=df.columns)
        return df[
            df["employee_uuid"].fillna("").astype(str) == selected_employee_uuid
        ]

    baseline_subset = _filter_by_employee(baseline_df.copy())
    scenario_subset = _filter_by_employee(scenario_df.copy())

    annotated_subset, removed_subset = _annotate_with_diff(baseline_subset, scenario_subset)

    base_columns = list(dict.fromkeys(list(scenario_df.columns) + list(baseline_df.columns)))
    if "employee_uuid" not in base_columns:
        base_columns.insert(0, "employee_uuid")
    if "uuid" not in base_columns:
        base_columns.insert(0, "uuid")
    column_order = list(dict.fromkeys(base_columns + [DIFF_COLUMN_LABEL]))

    if annotated_subset.empty:
        display_df = pd.DataFrame(columns=column_order)
    else:
        display_df = annotated_subset.reindex(columns=column_order)

    if "employee_uuid" in display_df.columns:
        display_df["employee_uuid"] = (
            display_df["employee_uuid"].fillna(selected_employee_uuid).replace("", selected_employee_uuid)
        )

    column_config: Dict[str, Any] = {}
    if "uuid" in display_df.columns:
        column_config["uuid"] = st.column_config.TextColumn("UUID", disabled=True)
    if "employee_uuid" in display_df.columns:
        column_config["employee_uuid"] = st.column_config.TextColumn(
            "Employee UUID", disabled=True
        )
    if "role_uuid" in display_df.columns:
        column_config["role_uuid"] = st.column_config.SelectboxColumn(
            "Role",
            options=list(role_options.keys()),
            format_func=lambda value: role_options.get(value, "—"),
        )
    if "percentage" in display_df.columns:
        column_config["percentage"] = st.column_config.NumberColumn(
            "Allocation %",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            format="%.0f%%",
        )
    if "weight" in display_df.columns:
        column_config["weight"] = st.column_config.NumberColumn(
            "Weight", min_value=0.0, step=0.1
        )
    column_config[DIFF_COLUMN_LABEL] = st.column_config.TextColumn(
        DIFF_COLUMN_LABEL, disabled=True
    )

    editor_df = st.data_editor(
        display_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index="uuid" in display_df.columns,
        column_config=column_config or None,
        key=f"scenario_allocations_editor_{scenario.uuid}_{selected_employee_uuid}",
    )

    if not removed_subset.empty:
        removed_ids = removed_subset.get("uuid")
        if removed_ids is not None and not removed_ids.empty:
            removed_labels = ", ".join(str(value) for value in removed_ids.astype(str))
        else:
            removed_labels = f"{len(removed_subset)} record(s)"
        st.caption(f"Removed allocations in scenario: {removed_labels}")

    if st.button(
        "Save allocation changes",
        key=f"save_allocations_{scenario.uuid}",
    ):
        edited_subset = editor_df.copy()
        if DIFF_COLUMN_LABEL in edited_subset.columns:
            edited_subset = edited_subset.drop(columns=[DIFF_COLUMN_LABEL])

        if "employee_uuid" in edited_subset.columns:
            edited_subset["employee_uuid"] = edited_subset["employee_uuid"].apply(
                lambda value: value if value else selected_employee_uuid
            )
        else:
            edited_subset["employee_uuid"] = selected_employee_uuid

        remaining = scenario_df.copy()
        if not remaining.empty and "employee_uuid" in remaining.columns:
            remaining = remaining[
                remaining["employee_uuid"].fillna("").astype(str) != selected_employee_uuid
            ]
        else:
            remaining = pd.DataFrame(columns=scenario_df.columns)

        combined = pd.concat([edited_subset, remaining], ignore_index=True, sort=False)
        combined = combined.reindex(columns=base_columns)

        new_adjustments = _calculate_dataset_adjustments(dataset, baseline_df, combined)
        _merge_adjustments(scenario, dataset, new_adjustments)
        update_data("scenarios", baseline_data["scenarios"])
        st.success("Scenario allocations saved.")
        st.experimental_rerun()


def _render_support_tab(
    *,
    scenario: Scenario,
    baseline_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    baseline_allocations: pd.DataFrame,
    scenario_allocations: pd.DataFrame,
    baseline_data: Dict[str, Iterable],
    modified_data: Dict[str, Iterable],
    dataset: str,
) -> None:
    selected_employee_uuid = _render_allocation_overview(modified_data, scenario.uuid)

    st.divider()

    if not selected_employee_uuid:
        st.info("Select an employee in the table above to review support allocations.")
        return

    roles_df = _items_to_dataframe(modified_data.get("roles", []))
    role_labels: Dict[str, str] = {}
    if not roles_df.empty and {"uuid", "name"}.issubset(roles_df.columns):
        role_labels = {
            str(row["uuid"]): str(row.get("name", row["uuid"]))
            for _, row in roles_df.iterrows()
        }

    processes_df = _items_to_dataframe(modified_data.get("processes", []))
    process_labels: Dict[str, str] = {}
    if not processes_df.empty and {"uuid", "name"}.issubset(processes_df.columns):
        process_labels = {
            str(row["uuid"]): str(row.get("name", row["uuid"]))
            for _, row in processes_df.iterrows()
        }

    def _relevant_allocations(df: pd.DataFrame) -> pd.Series:
        if df.empty or "employee_uuid" not in df.columns:
            return pd.Series(dtype=str)
        return df.loc[
            df["employee_uuid"].fillna("").astype(str) == selected_employee_uuid,
            "uuid",
        ].fillna("").astype(str)

    scenario_alloc_ids = set(_relevant_allocations(scenario_allocations))
    baseline_alloc_ids = set(_relevant_allocations(baseline_allocations))
    relevant_alloc_ids = scenario_alloc_ids | baseline_alloc_ids

    if not relevant_alloc_ids:
        st.info("No support-eligible allocations available for the selected employee.")
        return

    def _filter_support(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "allocation_uuid" not in df.columns:
            return pd.DataFrame(columns=df.columns)
        return df[df["allocation_uuid"].fillna("").astype(str).isin(relevant_alloc_ids)]

    baseline_subset = _filter_support(baseline_df.copy())
    scenario_subset = _filter_support(scenario_df.copy())

    annotated_subset, removed_subset = _annotate_with_diff(baseline_subset, scenario_subset)

    base_columns = list(dict.fromkeys(list(scenario_df.columns) + list(baseline_df.columns)))
    if "uuid" not in base_columns:
        base_columns.insert(0, "uuid")
    if "allocation_uuid" not in base_columns:
        base_columns.insert(0, "allocation_uuid")
    column_order = list(dict.fromkeys(base_columns + [DIFF_COLUMN_LABEL]))

    if annotated_subset.empty:
        display_df = pd.DataFrame(columns=column_order)
    else:
        display_df = annotated_subset.reindex(columns=column_order)

    allocation_options: Dict[str, str] = {}
    if not scenario_allocations.empty and "uuid" in scenario_allocations.columns:
        scenario_allocations = scenario_allocations.assign(
            uuid=scenario_allocations["uuid"].fillna("").astype(str)
        )
        for _, allocation in scenario_allocations.iterrows():
            allocation_uuid = allocation["uuid"]
            if allocation_uuid not in relevant_alloc_ids:
                continue
            role_uuid = str(allocation.get("role_uuid", ""))
            role_name = role_labels.get(role_uuid, role_uuid or "—")
            percentage = float(allocation.get("percentage", 0.0) or 0.0)
            allocation_options[allocation_uuid] = f"{role_name} ({percentage * 100:.0f}%)"

    if not baseline_allocations.empty and "uuid" in baseline_allocations.columns:
        baseline_allocations = baseline_allocations.assign(
            uuid=baseline_allocations["uuid"].fillna("").astype(str)
        )
        for _, allocation in baseline_allocations.iterrows():
            allocation_uuid = allocation["uuid"]
            if allocation_uuid not in relevant_alloc_ids:
                continue
            allocation_options.setdefault(
                allocation_uuid,
                f"{allocation_uuid} (baseline)",
            )

    column_config: Dict[str, Any] = {}
    if "uuid" in display_df.columns:
        column_config["uuid"] = st.column_config.TextColumn("UUID", disabled=True)
    if "allocation_uuid" in display_df.columns:
        column_config["allocation_uuid"] = st.column_config.SelectboxColumn(
            "Role allocation",
            options=list(allocation_options.keys()),
            format_func=lambda value: allocation_options.get(value, "—"),
        )
    if "process_uuid" in display_df.columns:
        column_config["process_uuid"] = st.column_config.SelectboxColumn(
            "Process",
            options=list(process_labels.keys()),
            format_func=lambda value: process_labels.get(value, "—"),
        )
    if "percentage" in display_df.columns:
        column_config["percentage"] = st.column_config.NumberColumn(
            "Allocation %",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            format="%.0f%%",
        )
    if "weight" in display_df.columns:
        column_config["weight"] = st.column_config.NumberColumn(
            "Weight", min_value=0.0, step=0.1
        )
    column_config[DIFF_COLUMN_LABEL] = st.column_config.TextColumn(
        DIFF_COLUMN_LABEL, disabled=True
    )

    editor_df = st.data_editor(
        display_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index="uuid" in display_df.columns,
        column_config=column_config or None,
        key=f"scenario_support_editor_{scenario.uuid}_{selected_employee_uuid}",
    )

    if not removed_subset.empty:
        removed_ids = removed_subset.get("uuid")
        if removed_ids is not None and not removed_ids.empty:
            removed_labels = ", ".join(str(value) for value in removed_ids.astype(str))
        else:
            removed_labels = f"{len(removed_subset)} record(s)"
        st.caption(f"Removed support allocations in scenario: {removed_labels}")

    if st.button(
        "Save support allocation changes",
        key=f"save_support_{scenario.uuid}",
    ):
        edited_subset = editor_df.copy()
        if DIFF_COLUMN_LABEL in edited_subset.columns:
            edited_subset = edited_subset.drop(columns=[DIFF_COLUMN_LABEL])

        remaining = scenario_df.copy()
        if not remaining.empty and "allocation_uuid" in remaining.columns:
            remaining = remaining[
                ~remaining["allocation_uuid"].fillna("").astype(str).isin(relevant_alloc_ids)
            ]
        else:
            remaining = pd.DataFrame(columns=scenario_df.columns)

        combined = pd.concat([edited_subset, remaining], ignore_index=True, sort=False)
        combined = combined.reindex(columns=base_columns)

        new_adjustments = _calculate_dataset_adjustments(dataset, baseline_df, combined)
        _merge_adjustments(scenario, dataset, new_adjustments)
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
    show_differences_only: bool,
) -> None:
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

    diff_mask: Optional[pd.Series] = None
    if not full_display_df.empty:
        diff_mask = full_display_df[DIFF_COLUMN_LABEL] != "Unchanged"

    if show_differences_only and diff_mask is not None:
        display_df = full_display_df.loc[diff_mask].copy()
    else:
        display_df = full_display_df.copy()

    column_config: Dict[str, Any] = {}
    if "uuid" in display_df.columns:
        column_config["uuid"] = st.column_config.TextColumn("UUID", disabled=True)
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
        removed_ids = removed_df.get("uuid")
        if removed_ids is not None and not removed_ids.empty:
            removed_labels = ", ".join(str(value) for value in removed_ids.astype(str))
        else:
            removed_labels = f"{len(removed_df)} record(s)"
        st.caption(f"Removed entries in scenario: {removed_labels}")

    if st.button(f"Save {label.lower()} changes", key=f"save_{scenario.uuid}_{dataset}"):
        edited_df = editor_df.copy()
        if show_differences_only and diff_mask is not None:
            edited_df = _merge_visible_and_hidden_rows(edited_df, full_display_df, diff_mask)
        if DIFF_COLUMN_LABEL in edited_df.columns:
            edited_df = edited_df.drop(columns=[DIFF_COLUMN_LABEL])

        new_adjustments = _calculate_dataset_adjustments(dataset, baseline_df, edited_df)
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
) -> List[ScenarioAdjustment]:
    if dataset not in ADJUSTMENT_MAPPING:
        return []

    add_type, remove_type, update_type = ADJUSTMENT_MAPPING[dataset]

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
                _render_allocation_tab(
                    scenario=scenario,
                    baseline_df=baseline_df,
                    scenario_df=scenario_df,
                    baseline_data=data,
                    modified_data=modified_data,
                    dataset=dataset,
                )
            elif dataset == "support_allocations":
                baseline_allocations = _items_to_dataframe(data.get("allocations", [])).copy()
                scenario_allocations = _items_to_dataframe(
                    modified_data.get("allocations", [])
                ).copy()
                _render_support_tab(
                    scenario=scenario,
                    baseline_df=baseline_df,
                    scenario_df=scenario_df,
                    baseline_allocations=baseline_allocations,
                    scenario_allocations=scenario_allocations,
                    baseline_data=data,
                    modified_data=modified_data,
                    dataset=dataset,
                )
            else:
                _render_generic_dataset_tab(
                    scenario=scenario,
                    dataset=dataset,
                    label=label,
                    baseline_df=baseline_df,
                    scenario_df=scenario_df,
                    baseline_data=data,
                    show_differences_only=show_differences_only,
                )

    st.subheader("Scenario result")
    coverage_result = _build_coverage_result(baseline_coverage, scenario_coverage)
    styled_coverage = _style_coverage_result(coverage_result)
    st.dataframe(styled_coverage, use_container_width=True)


if __name__ == "__main__":
    main()
