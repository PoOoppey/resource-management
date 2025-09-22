from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import streamlit as st

from models import (
    Allocation,
    App,
    AttendanceRecord,
    EmployeeExpertise,
    Employee,
    JiraCount,
    Office,
    Process,
    RequiredCoverage,
    Role,
    Scenario,
    SupportAllocation,
)

DATA_FILES: Tuple[Tuple[str, str], ...] = (
    ("employees", "employees.json"),
    ("offices", "offices.json"),
    ("roles", "roles.json"),
    ("processes", "processes.json"),
    ("coverage", "coverage.json"),
    ("allocations", "allocations.json"),
    ("support_allocations", "support_allocations.json"),
    ("attendances", "attendances.json"),
    ("apps", "apps.json"),
    ("scenarios", "scenarios.json"),
    ("expertise_levels", "expertise_levels.json"),
)


MODEL_MAP = {
    "employees": Employee,
    "offices": Office,
    "roles": Role,
    "processes": Process,
    "coverage": RequiredCoverage,
    "allocations": Allocation,
    "support_allocations": SupportAllocation,
    "attendances": AttendanceRecord,
    "apps": App,
    "scenarios": Scenario,
    "expertise_levels": EmployeeExpertise,
}


def _load_json(path: Path):
    import json

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _deserialize_items(key: str, payload: Iterable[Dict]):
    model = MODEL_MAP[key]
    return [model.from_dict(item) for item in payload]


def initialize_session_state(data_dir: Path | None = None) -> None:
    """Load datasets into ``st.session_state`` on first app load."""

    if "data" in st.session_state:
        return

    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / "data"

    datasets: Dict[str, Iterable] = {}
    for key, filename in DATA_FILES:
        raw = _load_json(data_dir / filename)
        datasets[key] = _deserialize_items(key, raw)

    st.session_state.data = datasets
    st.session_state.jira_cache: Dict[str, Iterable[JiraCount]] = {}


def get_data() -> Dict[str, Iterable]:
    initialize_session_state()
    return st.session_state.data


def update_data(key: str, items: Iterable) -> None:
    initialize_session_state()
    st.session_state.data[key] = items


def get_jira_counts(process_uuid: str) -> Iterable[JiraCount]:
    initialize_session_state()
    if process_uuid in st.session_state.jira_cache:
        return st.session_state.jira_cache[process_uuid]

    data_dir = Path(__file__).resolve().parents[1] / "data"
    raw = _load_json(data_dir / "jira.json")
    counts = [JiraCount.from_dict(item) for item in raw if item["process_uuid"] == process_uuid]
    st.session_state.jira_cache[process_uuid] = counts
    return counts
