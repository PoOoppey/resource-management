from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Dict, Iterable, List, Tuple
from uuid import uuid4

import pandas as pd
import streamlit as st

from models import AdjustmentType, Scenario, ScenarioAdjustment
from services.coverage import compute_theoretical_coverage
from services.data_loader import get_data, update_data
from services.scenario import apply_scenario, compare_scenario
from utils.notifications import notify


DATASET_ORDER: Tuple[str, ...] = (
    "employees",
    "allocations",
    "support_allocations",
    "coverage",
)

UUID_PREFIX = {
    "employees": "emp",
    "allocations": "alc",
    "support_allocations": "sup",
    "coverage": "cov",
}

NUMERIC_FIELDS = {
    "employees": ["working_hours"],
    "allocations": ["percentage", "weight"],
    "support_allocations": ["percentage", "weight"],
    "coverage": ["required_hours"],
}

REQUIRED_FIELDS = {
    "employees": [
        "uuid",
        "first_name",
        "last_name",
        "trigram",
        "office_uuid",
        "working_hours",
    ],
    "allocations": ["uuid", "employee_uuid", "role_uuid", "percentage"],
    "support_allocations": ["uuid", "allocation_uuid", "process_uuid", "percentage"],
    "coverage": ["uuid", "process_uuid", "office_uuid", "required_hours"],
}

ADJUSTMENT_TYPES: Dict[str, Dict[str, AdjustmentType]] = {
    "employees": {
        "add": AdjustmentType.ADD_EMPLOYEE,
        "update": AdjustmentType.UPDATE_EMPLOYEE,
        "remove": AdjustmentType.REMOVE_EMPLOYEE,
    },
    "allocations": {
        "add": AdjustmentType.ADD_ALLOCATION,
        "update": AdjustmentType.UPDATE_ALLOCATION,
        "remove": AdjustmentType.REMOVE_ALLOCATION,
    },
    "support_allocations": {
        "add": AdjustmentType.ADD_SUPPORT_ALLOCATION,
        "update": AdjustmentType.UPDATE_SUPPORT_ALLOCATION,
        "remove": AdjustmentType.REMOVE_SUPPORT_ALLOCATION,
    },
    "coverage": {
        "add": AdjustmentType.ADD_REQUIRED_COVERAGE,
        "update": AdjustmentType.UPDATE_REQUIRED_COVERAGE,
        "remove": AdjustmentType.REMOVE_REQUIRED_COVERAGE,
    },
}


def _serialize_record(item) -> Dict:
    if is_dataclass(item):
        record = asdict(item)
    elif hasattr(item, "__dict__"):
        record = {**item.__dict__}
    else:
        record = dict(item)
    for key, value in list(record.items()):
        if isinstance(value, Enum):
            record[key] = value.value
    return record


def _items_to_dataframe(items: Iterable) -> pd.DataFrame:
    records = [_serialize_record(item) for item in items]
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _ensure_uuid_column(dataset: str, df: pd.DataFrame) -> pd.DataFrame:
    if "uuid" not in df.columns:
        df.insert(0, "uuid", "")
    for index, value in df["uuid"].items():
        if pd.isna(value) or not str(value).strip():
            df.at[index, "uuid"] = f"{UUID_PREFIX.get(dataset, 'id')}-{uuid4().hex[:8]}"
    return df


def _coerce_numeric(dataset: str, record: Dict) -> Dict:
    for field in NUMERIC_FIELDS.get(dataset, []):
        if record.get(field) is not None:
            try:
                record[field] = float(record[field])
            except (TypeError, ValueError):
                record[field] = None
    return record


def _records_from_dataframe(dataset: str, df: pd.DataFrame) -> List[Dict]:
    if df is None or df.empty:
        return []
    records: List[Dict] = []
    for raw in df.to_dict(orient="records"):
        record = {}
        for key, value in raw.items():
            if pd.isna(value):
                record[key] = None
            else:
                record[key] = value
        record = _coerce_numeric(dataset, record)
        records.append(record)
    return records


def _normalize_value(value):
    if isinstance(value, float):
        return round(value, 6)
    return value


def _records_equal(left: Dict, right: Dict) -> bool:
    left_clean = {k: _normalize_value(v) for k, v in left.items() if k != "uuid"}
    right_clean = {k: _normalize_value(v) for k, v in right.items() if k != "uuid"}
    keys = set(left_clean) | set(right_clean)
    return all(left_clean.get(key) == right_clean.get(key) for key in keys)


def _has_required_fields(dataset: str, record: Dict) -> Tuple[bool, List[str]]:
    required = REQUIRED_FIELDS.get(dataset, [])
    missing = [field for field in required if not record.get(field)]
    return (not missing, missing)


def _derive_adjustments(
    dataset: str, baseline: List[Dict], updated: List[Dict]
) -> Tuple[List[Dict], List[str]]:
    adjustments: List[Dict] = []
    issues: List[str] = []

    baseline_map = {record["uuid"]: record for record in baseline if record.get("uuid")}
    updated_map = {record["uuid"]: record for record in updated if record.get("uuid")}

    for uuid, record in updated_map.items():
        if uuid not in baseline_map:
            valid, missing = _has_required_fields(dataset, record)
            if not valid:
                issues.append(
                    f"{dataset.title()} entry {uuid} is missing required fields: {', '.join(missing)}"
                )
                continue
            adjustments.append(
                {
                    "uuid": f"adj-{uuid4().hex[:8]}",
                    "type": ADJUSTMENT_TYPES[dataset]["add"],
                    "payload": record,
                }
            )
        else:
            baseline_record = baseline_map[uuid]
            if not _records_equal(baseline_record, record):
                valid, missing = _has_required_fields(dataset, record)
                if not valid:
                    issues.append(
                        f"{dataset.title()} entry {uuid} is missing required fields: {', '.join(missing)}"
                    )
                    continue
                adjustments.append(
                    {
                        "uuid": f"adj-{uuid4().hex[:8]}",
                        "type": ADJUSTMENT_TYPES[dataset]["update"],
                        "payload": record,
                    }
                )

    for uuid in baseline_map:
        if uuid not in updated_map:
            adjustments.append(
                {
                    "uuid": f"adj-{uuid4().hex[:8]}",
                    "type": ADJUSTMENT_TYPES[dataset]["remove"],
                    "payload": {"uuid": uuid},
                }
            )

    return adjustments, issues


def _scenario_select(scenarios: List[Scenario]) -> Scenario | None:
    options = {"Create new scenario": None}
    options.update({scenario.name: scenario for scenario in scenarios})
    choice = st.selectbox(
        "Scenario",
        list(options.keys()),
        key="scenario_select",
    )
    return options[choice]


def _scenario_state_key() -> str:
    return "scenario_editor_state"


def _initialize_state(active_uuid: str | None, scenario: Scenario | None, data: Dict) -> None:
    state = st.session_state.setdefault(
        _scenario_state_key(),
        {
            "active_uuid": None,
            "scenario_name": "",
            "datasets": {},
            "adjustments": [],
            "issues": [],
            "comparison": None,
            "last_signature": None,
        },
    )

    if state.get("active_uuid") == active_uuid:
        return

    datasets = {}
    if scenario is None:
        state["scenario_name"] = ""
        for dataset in DATASET_ORDER:
            datasets[dataset] = _ensure_uuid_column(
                dataset, _items_to_dataframe(data.get(dataset, [])).copy()
            )
    else:
        state["scenario_name"] = scenario.name
        modified = apply_scenario(data, scenario)
        for dataset in DATASET_ORDER:
            datasets[dataset] = _ensure_uuid_column(
                dataset, _items_to_dataframe(modified.get(dataset, [])).copy()
            )

    state["active_uuid"] = scenario.uuid if scenario else None
    state["datasets"] = datasets
    state["adjustments"] = []
    state["issues"] = []
    state["comparison"] = None
    state["last_signature"] = None
    st.session_state["scenario_name_input"] = state["scenario_name"]
    st.session_state["scenario_select"] = scenario.name if scenario else "Create new scenario"


def _build_adjustments(data: Dict) -> Tuple[List[ScenarioAdjustment], List[str]]:
    state = st.session_state[_scenario_state_key()]
    adjustments: List[ScenarioAdjustment] = []
    issues: List[str] = []

    for dataset in DATASET_ORDER:
        baseline_records = _records_from_dataframe(dataset, _items_to_dataframe(data.get(dataset, [])))
        updated_df = state["datasets"].get(dataset, pd.DataFrame())
        updated_records = _records_from_dataframe(dataset, updated_df)
        dataset_adjustments, dataset_issues = _derive_adjustments(
            dataset, baseline_records, updated_records
        )
        issues.extend(dataset_issues)
        for adjustment in dataset_adjustments:
            adjustments.append(
                ScenarioAdjustment(
                    uuid=adjustment["uuid"],
                    type=adjustment["type"],
                    payload=adjustment["payload"],
                )
            )

    return adjustments, issues


def _render_dataset_editor(dataset: str, baseline_df: pd.DataFrame) -> None:
    state = st.session_state[_scenario_state_key()]
    scenario_df = state["datasets"].get(dataset, pd.DataFrame()).copy()
    scenario_df = _ensure_uuid_column(dataset, scenario_df)

    baseline_col, scenario_col = st.columns(2)
    with baseline_col:
        st.caption("Baseline")
        st.dataframe(baseline_df, use_container_width=True)
    with scenario_col:
        st.caption("Scenario")
        edited_df = st.data_editor(
            scenario_df,
            num_rows="dynamic",
            use_container_width=True,
            key=f"editor_{dataset}",
            column_config={
                "uuid": st.column_config.TextColumn("UUID", disabled=True),
            },
        )
        state["datasets"][dataset] = _ensure_uuid_column(dataset, edited_df.copy())


def main():
    st.title("Scenario Planner")

    data = get_data()

    baseline_coverage = compute_theoretical_coverage(
        data,
        view="process",
        group_by="office",
        unit="hours",
    )

    st.subheader("Scenario selection")
    scenarios = data.get("scenarios", [])
    scenario = _scenario_select(scenarios)

    active_uuid = scenario.uuid if scenario else None
    _initialize_state(active_uuid, scenario, data)
    state = st.session_state[_scenario_state_key()]

    state["scenario_name"] = st.text_input(
        "Scenario name",
        value=state.get("scenario_name", ""),
        key="scenario_name_input",
    )

    st.markdown("---")
    st.subheader("Baseline coverage")
    st.dataframe(baseline_coverage, use_container_width=True)

    st.subheader("Scenario data entry")
    for dataset in DATASET_ORDER:
        st.markdown(f"### {dataset.replace('_', ' ').title()}")
        baseline_df = _items_to_dataframe(data.get(dataset, [])).copy()
        baseline_df = _ensure_uuid_column(dataset, baseline_df)
        _render_dataset_editor(dataset, baseline_df)

    adjustments, issues = _build_adjustments(data)
    state["adjustments"] = adjustments
    state["issues"] = issues

    signature_payload = [
        {"uuid": adj.uuid, "type": adj.type.value, "payload": adj.payload}
        for adj in adjustments
    ]
    signature = json.dumps(signature_payload, sort_keys=True)
    if state.get("last_signature") and state.get("last_signature") != signature:
        state["comparison"] = None
    state["last_signature"] = signature

    if issues:
        st.warning("\n".join(issues))

    st.markdown("---")
    st.subheader("Scenario adjustments")
    if adjustments:
        adjustments_df = pd.DataFrame(
            [
                {
                    "uuid": adjustment.uuid,
                    "type": adjustment.type.value,
                    "payload": adjustment.payload,
                }
                for adjustment in adjustments
            ]
        )
        st.dataframe(adjustments_df, use_container_width=True)
    else:
        st.info("No adjustments detected. Edit the scenario tables to create adjustments.")

    recompute = st.button("Recompute scenario", type="primary")
    if recompute and not issues:
        scenario_uuid = state.get("active_uuid") or f"scn-{uuid4().hex[:8]}"
        scenario_name = state.get("scenario_name") or "Untitled scenario"
        scenario_model = Scenario(
            uuid=scenario_uuid,
            name=scenario_name,
            adjustments=adjustments,
        )
        modified_data = apply_scenario(data, scenario_model)
        scenario_coverage = compute_theoretical_coverage(
            modified_data,
            view="process",
            group_by="office",
            unit="hours",
        )
        comparison = compare_scenario(baseline_coverage, scenario_coverage)
        state["comparison"] = comparison
        notify("Scenario recomputed successfully.", level="success")
    elif recompute and issues:
        notify("Please resolve missing information before recomputing.", level="error")

    if state.get("comparison") is not None:
        st.subheader("Coverage comparison")
        st.dataframe(state["comparison"], use_container_width=True)

    if st.button("Save scenario"):
        if issues:
            notify("Cannot save scenario with validation issues.", level="error")
        else:
            scenario_uuid = state.get("active_uuid") or f"scn-{uuid4().hex[:8]}"
            scenario_name = state.get("scenario_name") or "Untitled scenario"
            scenario_model = Scenario(
                uuid=scenario_uuid,
                name=scenario_name,
                adjustments=adjustments,
            )
            updated_scenarios = list(scenarios)
            replaced = False
            for index, existing in enumerate(updated_scenarios):
                if existing.uuid == scenario_uuid:
                    updated_scenarios[index] = scenario_model
                    replaced = True
                    break
            if not replaced:
                updated_scenarios.append(scenario_model)
            update_data("scenarios", updated_scenarios)
            notify("Scenario saved.", level="success")
            state["active_uuid"] = scenario_uuid
            st.session_state["scenario_select"] = scenario_name


if __name__ == "__main__":
    main()
