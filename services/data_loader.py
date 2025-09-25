from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

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


DATA_FILE_MAP = {key: filename for key, filename in DATA_FILES}


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
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _serialize_value(value: Any):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_value(item) for item in value]
    return value


def _serialize_item(item: Any):
    if is_dataclass(item):
        raw = asdict(item)
    elif isinstance(item, dict):
        raw = dict(item)
    elif hasattr(item, "__dict__"):
        raw = {**item.__dict__}
    else:
        return _serialize_value(item)
    return {key: _serialize_value(value) for key, value in raw.items()}


def _deserialize_items(key: str, payload: Iterable[Dict]):
    model = MODEL_MAP[key]
    return [model.from_dict(item) for item in payload]


def _resolve_data_dir(data_dir: Path | None = None) -> Path:
    if data_dir is not None:
        return data_dir
    return Path(__file__).resolve().parents[1] / "data"


def _load_datasets(data_dir: Path) -> Dict[str, Iterable]:
    datasets: Dict[str, Iterable] = {}
    for key, filename in DATA_FILES:
        raw = _load_json(data_dir / filename)
        datasets[key] = _deserialize_items(key, raw)
    return datasets


def initialize_session_state(data_dir: Path | None = None) -> None:
    """Load datasets into ``st.session_state`` on first app load."""

    if "data" in st.session_state:
        return

    refresh_data(data_dir=data_dir)


def get_data() -> Dict[str, Iterable]:
    initialize_session_state()
    return st.session_state.data


def update_data(key: str, items: Iterable) -> None:
    initialize_session_state()
    materialized_items = list(items)
    st.session_state.data[key] = materialized_items

    data_dir = st.session_state.get("data_dir")
    if data_dir is None:
        data_dir = _resolve_data_dir()
        st.session_state.data_dir = data_dir

    filename = DATA_FILE_MAP.get(key)
    if not filename:
        return

    payload = [_serialize_item(item) for item in materialized_items]
    target = data_dir / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def refresh_data(data_dir: Path | None = None) -> None:
    """Force reloading of datasets into ``st.session_state``."""

    resolved_dir = _resolve_data_dir(data_dir)
    st.session_state.data_dir = resolved_dir
    st.session_state.data = _load_datasets(resolved_dir)
    st.session_state.jira_cache = {}


def get_jira_counts(process_uuid: str) -> Iterable[JiraCount]:
    initialize_session_state()
    if process_uuid in st.session_state.jira_cache:
        return st.session_state.jira_cache[process_uuid]

    data_dir = Path(__file__).resolve().parents[1] / "data"
    raw = _load_json(data_dir / "jira.json")
    counts = [JiraCount.from_dict(item) for item in raw if item["process_uuid"] == process_uuid]
    st.session_state.jira_cache[process_uuid] = counts
    return counts
