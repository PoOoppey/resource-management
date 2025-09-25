from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import pandas as pd
import streamlit as st

from models import (
    AdjustmentType,
    RoleType,
    Scenario,
    ScenarioAdjustment,
    SupportStatus,
)
from services.coverage import compute_theoretical_coverage
from services.data_loader import get_data, update_data
from services.scenario import apply_scenario
from utils.relations import (
    FOREIGN_KEY_RELATIONS,
    build_reference_lookup,
    enrich_payload,
)
from utils.styling import coverage_style

st.set_page_config(page_title="Scenario", layout="wide")


DIFF_COLUMN_LABEL = "Diff vs baseline"

DISPLAY_DATASETS: List[Tuple[str, str]] = [
    ("employees", "Employees"),
    ("allocations", "Allocations"),
    ("processes", "Processes"),
    ("coverage", "Required coverage"),
]

NUMERIC_PRECISION = 2


REQUIRED_COLUMN_STYLE = {
    "background-color": "#ede9fe",
    "font-style": "italic",
    "color": "#111827",
    "font-weight": "600",
}

SUPPORT_STATUS_OPTIONS = {
    item.value: item.value.title().replace("_", " ") for item in SupportStatus
}


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


def _build_app_labels(
    baseline_data: Dict[str, Iterable], modified_data: Dict[str, Iterable]
) -> Dict[str, str]:
    apps_df = _combined_dataset_frame("apps", baseline_data, modified_data)
    labels: Dict[str, str] = {}
    if apps_df.empty or "uuid" not in apps_df.columns:
        return labels
    for _, row in apps_df.iterrows():
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
    app_labels = _build_app_labels(baseline_data, modified_data)
    employee_labels = _build_employee_labels(
        baseline_data, modified_data, office_labels
    )
    allocation_labels = _build_allocation_labels(
        baseline_data, modified_data, employee_labels, role_labels
    )
    return {
        "apps": app_labels,
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


def _make_multiselect_column(
    label: str, options: Dict[str, str]
) -> st.column_config.MultiselectColumn:
    ordered_options = [value for value in options.keys() if value]

    def _format(values: Any) -> str:
        if not values:
            return "—"
        if isinstance(values, str):
            values = [values]
        return ", ".join(options.get(value, value) for value in values)

    return st.column_config.MultiselectColumn(
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
            if column == "percentage":
                base_formatted = _format_percentage_value(base_value)
                scenario_formatted = _format_percentage_value(scenario_value)
                label = "Allocation"
            else:
                base_formatted = _format_value(base_value)
                scenario_formatted = _format_value(scenario_value)
                label = column.replace("_", " ").title()
            differences.append(f"{label}: {base_formatted} -> {scenario_formatted}")

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


def _format_allocation_percentage(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{numeric * 100:.0f}%"


def _format_percentage_value(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{numeric * 100:.0f}%"


def _summarize_role_assignments(
    rows: pd.DataFrame, role_labels: Dict[str, str]
) -> str:
    if rows is None or rows.empty:
        return "—"

    assignments: List[str] = []
    for _, row in rows.iterrows():
        role_identifier = _stringify_nullable(row.get("role_uuid"))
        role_label = role_labels.get(role_identifier, role_identifier or "—")
        percentage = row.get("percentage")
        percentage_text = ""
        if not _is_missing(percentage):
            try:
                percentage_text = f" ({float(percentage) * 100:.0f}%)"
            except (TypeError, ValueError):
                percentage_text = ""
        assignments.append(f"{role_label}{percentage_text}")

    return "; ".join(assignments) if assignments else "—"


def _summarize_support_assignments(
    rows: pd.DataFrame,
    *,
    process_labels: Dict[str, str],
    allocation_lookup: Dict[str, Dict[str, Any]],
    role_labels: Dict[str, str],
) -> str:
    if rows is None or rows.empty:
        return "—"

    summaries: List[str] = []
    for _, row in rows.iterrows():
        allocation_uuid = _stringify_nullable(row.get("allocation_uuid"))
        allocation_info = allocation_lookup.get(allocation_uuid, {})
        role_uuid = _stringify_nullable(allocation_info.get("role_uuid"))
        role_label = role_labels.get(role_uuid, role_uuid or "—")
        process_uuid = _stringify_nullable(row.get("process_uuid"))
        process_label = process_labels.get(process_uuid, process_uuid or "—")

        if role_label and process_label:
            base_label = f"{process_label}"
        else:
            base_label = role_label or process_label or allocation_uuid or "—"

        metrics: List[str] = []
        percentage = row.get("percentage")
        if not _is_missing(percentage):
            try:
                metrics.append(f"{float(percentage) * 100:.0f}%")
            except (TypeError, ValueError):
                pass

        allocation_weight = allocation_info.get("weight")
        support_weight = row.get("weight")
        effective_weight = (
            support_weight
            if not _is_missing(support_weight)
            else allocation_weight
        )
        if not _is_missing(effective_weight):
            try:
                metrics.append(f"×{float(effective_weight):.1f}")
            except (TypeError, ValueError):
                pass

        if metrics:
            summaries.append(f"{base_label} ({' '.join(metrics)})")
        else:
            summaries.append(base_label)

    return "; ".join(summaries) if summaries else "—"


def _summarize_employee_allocations(
    baseline_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    baseline_support_df: pd.DataFrame,
    scenario_support_df: pd.DataFrame,
    employee_labels: Dict[str, str],
    role_labels: Dict[str, str],
    process_labels: Dict[str, str],
    show_differences_only: bool,
    support_role_ids: set[str] | None,
) -> Tuple[pd.DataFrame, List[str]]:
    baseline = baseline_df.copy()
    scenario = scenario_df.copy()
    baseline_support = baseline_support_df.copy()
    scenario_support = scenario_support_df.copy()

    for df in (baseline, scenario):
        if df.empty:
            continue
        if "employee_uuid" in df.columns:
            df["employee_uuid"] = df["employee_uuid"].apply(_stringify_nullable)
        if "role_uuid" in df.columns:
            df["role_uuid"] = df["role_uuid"].apply(_stringify_nullable)
        if "uuid" in df.columns:
            df["uuid"] = df["uuid"].apply(_stringify_nullable)
        if "percentage" in df.columns:
            df["percentage"] = pd.to_numeric(df["percentage"], errors="coerce")

    for df in (baseline_support, scenario_support):
        if df.empty:
            continue
        for column in ["uuid", "allocation_uuid", "process_uuid"]:
            if column in df.columns:
                df[column] = df[column].apply(_stringify_nullable)
        if "percentage" in df.columns:
            df["percentage"] = pd.to_numeric(df["percentage"], errors="coerce")
        if "weight" in df.columns:
            df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    baseline_lookup: Dict[str, Dict[str, Any]] = {}
    scenario_lookup: Dict[str, Dict[str, Any]] = {}
    if not baseline.empty and "uuid" in baseline.columns:
        for _, row in baseline.iterrows():
            identifier = _stringify_nullable(row.get("uuid"))
            if identifier and identifier not in baseline_lookup:
                baseline_lookup[identifier] = row.to_dict()
    if not scenario.empty and "uuid" in scenario.columns:
        for _, row in scenario.iterrows():
            identifier = _stringify_nullable(row.get("uuid"))
            if identifier and identifier not in scenario_lookup:
                scenario_lookup[identifier] = row.to_dict()

    support_role_ids = {value for value in (support_role_ids or set()) if value}

    employee_ids: set[str] = set()
    for source_df in (baseline, scenario):
        if source_df.empty or "employee_uuid" not in source_df.columns:
            continue
        employee_ids.update(
            value for value in source_df["employee_uuid"].dropna().astype(str) if value
        )

    allocation_employee_lookup: Dict[str, str] = {}
    for lookup in (baseline_lookup, scenario_lookup):
        for allocation_id, record in lookup.items():
            employee_id = _stringify_nullable(record.get("employee_uuid"))
            if allocation_id and employee_id:
                allocation_employee_lookup[allocation_id] = employee_id

    for support_df in (baseline_support, scenario_support):
        if support_df.empty or "allocation_uuid" not in support_df.columns:
            continue
        for allocation_id in support_df["allocation_uuid"].dropna():
            identifier = _stringify_nullable(allocation_id)
            employee_id = allocation_employee_lookup.get(identifier)
            if employee_id:
                employee_ids.add(employee_id)

    summary_rows: List[Dict[str, Any]] = []
    employee_order: List[str] = []

    def _collect_allocation_ids(
        rows: pd.DataFrame, *, support_only: bool
    ) -> set[str]:
        if rows.empty or "uuid" not in rows.columns:
            return set()
        subset = rows
        if support_only and support_role_ids and "role_uuid" in rows.columns:
            subset = rows[rows["role_uuid"].isin(support_role_ids)]
        return {
            _stringify_nullable(value)
            for value in subset["uuid"].dropna().astype(str)
            if _stringify_nullable(value)
        }

    def _describe_change(label: str, before: str, after: str) -> str:
        before_display = before if before and before != "—" else "—"
        after_display = after if after and after != "—" else "—"
        return f"{label}: {before_display} → {after_display}"

    for employee_id in sorted(
        employee_ids, key=lambda identifier: employee_labels.get(identifier, identifier).lower()
    ):
        employee_label = employee_labels.get(employee_id, employee_id)

        if "employee_uuid" in baseline.columns:
            baseline_rows = baseline[baseline["employee_uuid"] == employee_id]
        else:
            baseline_rows = pd.DataFrame(columns=baseline.columns)
        if "employee_uuid" in scenario.columns:
            scenario_rows = scenario[scenario["employee_uuid"] == employee_id]
        else:
            scenario_rows = pd.DataFrame(columns=scenario.columns)

        baseline_total = (
            float(baseline_rows["percentage"].sum()) if not baseline_rows.empty else None
        )
        scenario_total = (
            float(scenario_rows["percentage"].sum()) if not scenario_rows.empty else None
        )

        baseline_support_ids = _collect_allocation_ids(
            baseline_rows, support_only=True
        )
        scenario_support_ids = _collect_allocation_ids(
            scenario_rows, support_only=True
        )

        if baseline_support.empty:
            baseline_support_rows = pd.DataFrame(columns=baseline_support.columns)
        else:
            baseline_support_rows = baseline_support[
                baseline_support["allocation_uuid"].isin(baseline_support_ids)
            ]
        if scenario_support.empty:
            scenario_support_rows = pd.DataFrame(columns=scenario_support.columns)
        else:
            scenario_support_rows = scenario_support[
                scenario_support["allocation_uuid"].isin(scenario_support_ids)
            ]

        baseline_roles = _summarize_role_assignments(baseline_rows, role_labels)
        scenario_roles = _summarize_role_assignments(scenario_rows, role_labels)
        baseline_support_summary = _summarize_support_assignments(
            baseline_support_rows,
            process_labels=process_labels,
            allocation_lookup=baseline_lookup,
            role_labels=role_labels,
        )
        scenario_support_summary = _summarize_support_assignments(
            scenario_support_rows,
            process_labels=process_labels,
            allocation_lookup={**baseline_lookup, **scenario_lookup},
            role_labels=role_labels,
        )

        roles_changed = baseline_roles != scenario_roles
        support_changed = baseline_support_summary != scenario_support_summary
        utilisation_changed = False
        if baseline_total is None and scenario_total is not None:
            utilisation_changed = True
        elif baseline_total is not None and scenario_total is None:
            utilisation_changed = True
        elif baseline_total is not None and scenario_total is not None:
            utilisation_changed = abs(baseline_total - scenario_total) > 1e-6

        has_baseline_data = (
            (not baseline_rows.empty) or (not baseline_support_rows.empty)
        )
        has_scenario_data = (
            (not scenario_rows.empty) or (not scenario_support_rows.empty)
        )

        if not has_baseline_data and has_scenario_data:
            status = "Added"
        elif has_baseline_data and not has_scenario_data:
            status = "Removed"
        elif roles_changed or support_changed or utilisation_changed:
            status = "Updated"
        else:
            status = "Unchanged"

        if show_differences_only and status == "Unchanged":
            continue

        details_parts: List[str] = []
        if roles_changed or status in {"Added", "Removed"}:
            details_parts.append(
                _describe_change("Roles", baseline_roles, scenario_roles)
            )
        if (support_changed or status in {"Added", "Removed"}) and (
            baseline_support_summary != "—" or scenario_support_summary != "—"
        ):
            details_parts.append(
                _describe_change(
                    "Support", baseline_support_summary, scenario_support_summary
                )
            )
        if utilisation_changed:
            details_parts.append(
                _describe_change(
                    "Utilisation",
                    _format_allocation_percentage(baseline_total),
                    _format_allocation_percentage(scenario_total),
                )
            )

        detail_text = "; ".join(details_parts) if details_parts else "—"
        diff_summary = status if detail_text == "—" else f"{status}: {detail_text}"

        summary_rows.append(
            {
                "Employee": employee_label or employee_id or "—",
                "Allocations": scenario_roles,
                "Support allocations": scenario_support_summary,
                "Utilisation": scenario_total,
                DIFF_COLUMN_LABEL: diff_summary,
            }
        )
        employee_order.append(employee_id)

    columns = [
        "Employee",
        "Allocations",
        "Support allocations",
        "Utilisation",
        DIFF_COLUMN_LABEL,
    ]

    if not summary_rows:
        return pd.DataFrame(columns=columns), []

    return pd.DataFrame(summary_rows, columns=columns), employee_order


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

    roles_frame = _combined_dataset_frame("roles", baseline_data, modified_data)
    support_role_ids: set[str] = set()
    if not roles_frame.empty and "uuid" in roles_frame.columns:
        roles_frame["uuid"] = roles_frame["uuid"].apply(_stringify_nullable)
        for _, role_row in roles_frame.iterrows():
            identifier = _stringify_nullable(role_row.get("uuid"))
            raw_type = role_row.get("type")
            if hasattr(raw_type, "value"):
                type_value = str(raw_type.value)
            elif isinstance(raw_type, str):
                type_value = raw_type
            else:
                type_value = str(raw_type) if raw_type is not None else ""
            normalized_type = type_value.upper()
            if "." in normalized_type:
                normalized_type = normalized_type.split(".")[-1]
            if identifier and normalized_type == RoleType.SUPPORT.value:
                support_role_ids.add(identifier)

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

    summary_table, employee_order = _summarize_employee_allocations(
        baseline_df,
        scenario_df,
        baseline_support_df,
        scenario_support_df,
        employee_labels,
        role_labels,
        process_labels,
        show_differences_only,
        support_role_ids,
    )

    available_employee_ids: set[str] = set()
    for df in (baseline_df, scenario_df):
        if df.empty or "employee_uuid" not in df.columns:
            continue
        ids = df["employee_uuid"].apply(_stringify_nullable)
        available_employee_ids.update(value for value in ids if value)

    employees_frame = _combined_dataset_frame("employees", baseline_data, modified_data)
    if not employees_frame.empty and "uuid" in employees_frame.columns:
        employees_frame["uuid"] = employees_frame["uuid"].apply(_stringify_nullable)
        available_employee_ids.update(
            value for value in employees_frame["uuid"] if value
        )

    def _label_for_employee(identifier: str) -> str:
        return employee_labels.get(identifier, identifier)

    missing_rows: List[Dict[str, Any]] = []
    for identifier in sorted(
        available_employee_ids, key=lambda value: _label_for_employee(value).lower()
    ):
        if identifier in employee_order:
            continue
        missing_rows.append(
            {
                "Employee": _label_for_employee(identifier),
                "Allocations": "—",
                "Support allocations": "—",
                "Utilisation": None,
                DIFF_COLUMN_LABEL: "Unchanged",
            }
        )
        employee_order.append(identifier)

    if missing_rows:
        filler_df = pd.DataFrame(missing_rows, columns=summary_table.columns)
        summary_table = pd.concat(
            [summary_table, filler_df], ignore_index=True, sort=False
        )

    if summary_table.empty:
        st.info("No allocation data available for the selected criteria.")
        return

    summary_table = summary_table.reset_index(drop=True)
    if "Utilisation" in summary_table.columns:
        summary_table["Utilisation"] = pd.to_numeric(
            summary_table["Utilisation"], errors="coerce"
        )

    overview_key = f"scenario_allocation_overview_{scenario.uuid}"
    selection_key = f"scenario_selected_employee_{scenario.uuid}"

    st.data_editor(
        summary_table,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        disabled=list(summary_table.columns),
        column_config={
            "Utilisation": st.column_config.ProgressColumn(
                "Utilisation", min_value=0.0, max_value=1.0, format="{:.0%}"
            ),
            DIFF_COLUMN_LABEL: st.column_config.TextColumn(
                DIFF_COLUMN_LABEL, disabled=True
            ),
        },
        key=overview_key,
    )

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

    selected_employee: Optional[str] = None
    selection_state = st.session_state.get(overview_key, {})
    selected_rows = _extract_selected_rows(selection_state)
    if selected_rows:
        row_position = selected_rows[0]
        if 0 <= row_position < len(employee_order):
            selected_employee = employee_order[row_position]
            st.session_state[selection_key] = selected_employee

    if selected_employee is None:
        stored_employee = st.session_state.get(selection_key)
        if stored_employee in employee_order:
            selected_employee = stored_employee
        elif employee_order:
            selected_employee = employee_order[0]
            st.session_state[selection_key] = selected_employee

    if not selected_employee:
        st.info("Select an employee in the overview table to manage allocations.")
        return

    scenario_allocations = scenario_df.copy()
    baseline_allocations = baseline_df.copy()
    for df in (scenario_allocations, baseline_allocations):
        if df.empty:
            continue
        if "employee_uuid" in df.columns:
            df["employee_uuid"] = df["employee_uuid"].apply(_stringify_nullable)
        if "uuid" in df.columns:
            df["uuid"] = df["uuid"].apply(_stringify_nullable)
        if "role_uuid" in df.columns:
            df["role_uuid"] = df["role_uuid"].apply(_stringify_nullable)

    employee_allocations = scenario_allocations.copy()
    if "employee_uuid" in employee_allocations.columns:
        employee_allocations = employee_allocations[
            employee_allocations["employee_uuid"] == selected_employee
        ]
    if employee_allocations.empty:
        employee_allocations = pd.DataFrame(columns=base_columns)
    else:
        employee_allocations = employee_allocations.reindex(columns=base_columns)

    allocation_tab, support_tab = st.tabs([
        "Role allocations",
        "Support allocations",
    ])

    with allocation_tab:
        allocation_display_df = employee_allocations.copy()
        if "percentage" in allocation_display_df.columns:
            allocation_display_df["percentage"] = allocation_display_df[
                "percentage"
            ].apply(lambda value: None if _is_missing(value) else float(value) * 100)

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
                max_value=100.0,
                step=5.0,
                format="%.0f%%",
            )
        if "weight" in employee_allocations.columns:
            allocation_column_config["weight"] = st.column_config.NumberColumn(
                "Weight", min_value=0.0, step=0.1
            )

        allocation_editor_df = st.data_editor(
            allocation_display_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
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
                    preview[
                        [col for col in ["Role", "Allocation %"] if col in preview.columns]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

        if st.button(
            "Save allocation changes", key=f"save_allocations_{scenario.uuid}"
        ):
            edited_df = allocation_editor_df.copy().reindex(columns=base_columns)
            if "percentage" in edited_df.columns:
                edited_df["percentage"] = edited_df["percentage"].apply(
                    lambda value: None if _is_missing(value) else float(value) / 100
                )
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

            if {
                "percentage",
                "employee_uuid",
            }.issubset(combined_allocations.columns):
                allocation_totals = (
                    combined_allocations.dropna(subset=["employee_uuid"])
                    .groupby("employee_uuid")["percentage"]
                    .sum(min_count=1)
                )
                if (allocation_totals > 1.00001).any():
                    st.error(
                        "Allocation percentages cannot exceed 100% for an employee."
                    )
                    return

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
            st.rerun()

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
    support_filter = support_role_ids if support_role_ids else None
    if {"uuid", "employee_uuid"}.issubset(scenario_allocations.columns):
        scenario_mask = scenario_allocations["employee_uuid"] == selected_employee
        if support_filter and "role_uuid" in scenario_allocations.columns:
            scenario_mask &= scenario_allocations["role_uuid"].isin(support_role_ids)
        relevant_allocation_ids.update(
            scenario_allocations.loc[scenario_mask, "uuid"].dropna().astype(str)
        )
    if {"uuid", "employee_uuid"}.issubset(baseline_allocations.columns):
        baseline_mask = baseline_allocations["employee_uuid"] == selected_employee
        if support_filter and "role_uuid" in baseline_allocations.columns:
            baseline_mask &= baseline_allocations["role_uuid"].isin(support_role_ids)
        relevant_allocation_ids.update(
            baseline_allocations.loc[baseline_mask, "uuid"].dropna().astype(str)
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

    _, support_removed_df = _annotate_with_diff(
        baseline_support_filtered, scenario_support_filtered
    )

    support_display_df = scenario_support_filtered.copy()
    if support_display_df.empty:
        support_display_df = pd.DataFrame(columns=list(support_base_columns) + ["Status"])
    else:
        support_display_df["Status"] = "Active"
    if "percentage" in support_display_df.columns:
        support_display_df["percentage"] = support_display_df["percentage"].apply(
            lambda value: None if _is_missing(value) else float(value) * 100
        )

    if not support_removed_df.empty:
        removed_display = support_removed_df.reindex(columns=support_base_columns).copy()
        removed_display["Status"] = "Removed"
        if "percentage" in removed_display.columns:
            removed_display["percentage"] = removed_display["percentage"].apply(
                lambda value: None if _is_missing(value) else float(value) * 100
            )
        support_display_df = pd.concat(
            [support_display_df, removed_display], ignore_index=True, sort=False
        )
        ordered_columns = list(support_base_columns) + ["Status"]
        support_display_df = support_display_df.reindex(columns=ordered_columns)
    elif "Status" not in support_display_df.columns:
        support_display_df["Status"] = "Active"

    support_column_config: Dict[str, Any] = {}
    if "uuid" in scenario_support_filtered.columns:
        support_column_config["uuid"] = st.column_config.TextColumn(
            "UUID", disabled=True
        )
    if "allocation_uuid" in scenario_support_filtered.columns:
        support_column_config["allocation_uuid"] = _make_selectbox_column(
            "Role allocation", allocation_option_map
        )
    if "process_uuid" in scenario_support_filtered.columns:
        support_column_config["process_uuid"] = _make_selectbox_column(
            "Process", process_labels
        )
    if "percentage" in scenario_support_filtered.columns:
        support_column_config["percentage"] = st.column_config.NumberColumn(
            "Allocation %",
            min_value=0.0,
            max_value=100.0,
            step=5.0,
            format="%.0f%%",
        )
    if "weight" in scenario_support_filtered.columns:
        support_column_config["weight"] = st.column_config.NumberColumn(
            "Weight", min_value=0.0, step=0.1
        )
    support_column_config["Status"] = st.column_config.TextColumn("Status", disabled=True)

    with support_tab:
        st.caption("Support allocations linked to the selected employee")
        support_editor_df = st.data_editor(
            support_display_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config=support_column_config or None,
            key=f"scenario_support_editor_{scenario.uuid}_{selected_employee}",
        )

        if st.button(
            "Save support allocation changes",
            key=f"save_support_{scenario.uuid}_{selected_employee}",
        ):
            edited_support_df = support_editor_df.copy()
            if "Status" in edited_support_df.columns:
                removed_mask = edited_support_df["Status"] == "Removed"
                if removed_mask.any():
                    edited_support_df = edited_support_df.loc[~removed_mask].copy()
                edited_support_df = edited_support_df.drop(columns=["Status"])
            edited_support_df = edited_support_df.reindex(columns=support_base_columns)
            if "percentage" in edited_support_df.columns:
                edited_support_df["percentage"] = edited_support_df["percentage"].apply(
                    lambda value: None if _is_missing(value) else float(value) / 100
                )
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

            if {"percentage", "allocation_uuid"}.issubset(combined_support.columns):
                support_totals = (
                    combined_support.dropna(subset=["allocation_uuid"])
                    .groupby("allocation_uuid")["percentage"]
                    .sum(min_count=1)
                )
                if (support_totals > 1.00001).any():
                    st.error(
                        "Support allocations cannot exceed 100% for a role allocation."
                    )
                    return

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
            st.rerun()


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

    removed_display_df = pd.DataFrame()
    if not removed_df.empty:
        removed_display_df = removed_df.reindex(columns=column_order).copy()
        removed_display_df[DIFF_COLUMN_LABEL] = "Removed"
        for field in relations:
            if field in removed_display_df.columns:
                removed_display_df[field] = removed_display_df[field].apply(
                    _stringify_nullable
                )
        display_df = pd.concat(
            [display_df, removed_display_df], ignore_index=True, sort=False
        )
        if not display_df.empty:
            display_df = display_df.reindex(columns=column_order)

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
    if dataset == "processes":
        app_options = label_maps.get("apps", {})
        process_options = label_maps.get("processes", {})
        if "support_status" in display_df.columns:
            column_config["support_status"] = _make_selectbox_column(
                "Support status", SUPPORT_STATUS_OPTIONS
            )
        if "apps_related" in display_df.columns:
            column_config["apps_related"] = _make_multiselect_column(
                "Apps", app_options
            )
        if "process_related" in display_df.columns:
            column_config["process_related"] = _make_multiselect_column(
                "Related processes", process_options
            )
    column_config[DIFF_COLUMN_LABEL] = st.column_config.TextColumn(
        DIFF_COLUMN_LABEL, disabled=True
    )

    editor_df = st.data_editor(
        display_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config=column_config or None,
        key=f"scenario_editor_{scenario.uuid}_{dataset}",
    )

    if st.button(f"Save {label.lower()} changes", key=f"save_{scenario.uuid}_{dataset}"):
        edited_df = editor_df.copy()
        if show_differences_only and diff_mask is not None:
            edited_df = _merge_visible_and_hidden_rows(edited_df, full_display_df, diff_mask)
        if DIFF_COLUMN_LABEL in edited_df.columns:
            removed_mask = edited_df[DIFF_COLUMN_LABEL] == "Removed"
            if removed_mask.any():
                edited_df = edited_df.loc[~removed_mask].copy()
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
        st.rerun()


def _render_scenario_creator(
    scenarios: List[Scenario], *, container: Optional[Any] = None, show_header: bool = True
) -> None:
    target = container or st
    if show_header:
        target.subheader("Create a scenario")
    with target.expander("Add a new scenario", expanded=not scenarios):
        raw_name = target.text_input("Scenario name", key="scenario_creation_name")
        new_name = raw_name.strip()
        create_disabled = not new_name

        if target.button(
            "Create scenario",
            key="scenario_creation_button",
            disabled=create_disabled,
        ):
            if any(new_name.lower() == scenario.name.strip().lower() for scenario in scenarios):
                target.error("A scenario with this name already exists.")
                return

            new_scenario = Scenario(
                uuid=_generate_uuid("scenario"),
                name=new_name,
                adjustments=[],
            )
            updated = scenarios + [new_scenario]
            update_data("scenarios", updated)
            target.success(f"Scenario '{new_name}' created.")
            st.rerun()


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

    numeric_columns = [
        "Required (Baseline)",
        "Coverage (Baseline)",
        "Required (Scenario)",
        "Coverage (Scenario)",
    ]
    for column in numeric_columns:
        if column not in merged.columns:
            merged[column] = 0.0
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)

    if display_columns:
        aggregated = (
            merged.groupby(display_columns, dropna=False)[numeric_columns]
            .sum()
            .reset_index()
        )
        aggregated = aggregated.sort_values(by=display_columns).reset_index(drop=True)
    else:
        aggregated = pd.DataFrame([merged[numeric_columns].sum().to_dict()])

    rows: List[Dict[str, object]] = []
    for _, row in aggregated.iterrows():
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
            required_style = "; ".join(
                f"{prop}: {value}" for prop, value in REQUIRED_COLUMN_STYLE.items()
            )
            styles["Required"] = required_style
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
        _render_scenario_creator(scenarios)
        st.info("Create a scenario to begin planning adjustments.")
        return

    selection_column, creation_column = st.columns([3, 2])

    with selection_column:
        st.subheader("Scenario selection")
        scenario = _scenario_select(scenarios)

    with creation_column:
        _render_scenario_creator(
            scenarios, container=creation_column, show_header=False
        )

    if scenario is None:
        st.info("Select a scenario to view its impact against the baseline.")
        return

    st.markdown("---")

    modified_data = apply_scenario(data, scenario)
    label_maps = _build_label_maps(data, modified_data)

    with st.expander("Scenario adjustments", expanded=False):
        st.subheader("Datasets")
        show_differences_only = st.toggle(
            "Show only rows with differences",
            value=False,
            key=f"scenario_diff_only_{scenario.uuid}",
        )

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
    controls = st.columns([1, 1])
    with controls[0]:
        group_choice = st.selectbox(
            "Group by",
            ["Office", "Region"],
            index=0,
            key=f"scenario_result_group_{scenario.uuid}",
        )
    with controls[1]:
        unit_choice = st.selectbox(
            "Unit",
            ["Hours", "FTE"],
            index=0,
            key=f"scenario_result_unit_{scenario.uuid}",
        )

    baseline_coverage = compute_theoretical_coverage(
        data,
        view="process",
        group_by=group_choice.lower(),
        unit=unit_choice.lower(),
    )
    scenario_coverage = compute_theoretical_coverage(
        modified_data,
        view="process",
        group_by=group_choice.lower(),
        unit=unit_choice.lower(),
    )

    coverage_result = _build_coverage_result(baseline_coverage, scenario_coverage)
    styled_coverage = _style_coverage_result(coverage_result)
    st.dataframe(styled_coverage, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
