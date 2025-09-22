from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
from typing import Dict, Iterable, List

import pandas as pd

from models import AdjustmentType, Scenario
from services.data_loader import MODEL_MAP


def apply_scenario(baseline_data: Dict[str, Iterable], scenario: Scenario) -> Dict[str, Iterable]:
    updated = {key: copy.deepcopy(value) for key, value in baseline_data.items()}

    for adjustment in scenario.adjustments:
        payload = adjustment.payload
        collection = updated.get(_collection_key(adjustment.type))
        if collection is None:
            continue

        model_cls = MODEL_MAP.get(_collection_key(adjustment.type))

        if adjustment.type.name.startswith("ADD"):
            if model_cls:
                collection.append(model_cls.from_dict(payload))
        elif adjustment.type.name.startswith("REMOVE"):
            updated[_collection_key(adjustment.type)] = [
                item
                for item in collection
                if _get_uuid(item) != payload.get("uuid")
            ]
        elif adjustment.type.name.startswith("UPDATE"):
            for index, item in enumerate(collection):
                item_uuid = _get_uuid(item)
                if item_uuid == payload.get("uuid"):
                    if model_cls:
                        updated[_collection_key(adjustment.type)][index] = model_cls.from_dict(
                            {**_to_dict(item), **payload}
                        )
                    break

    return updated


def compare_scenario(baseline: pd.DataFrame, scenario: pd.DataFrame) -> pd.DataFrame:
    merged = baseline.merge(
        scenario,
        on=[col for col in ["Region", "Office", "Process"] if col in baseline.columns],
        how="outer",
        suffixes=(" - Baseline", " - Scenario"),
    ).fillna(0)

    allocated_columns = [col for col in merged.columns if "Allocated" in col and "Scenario" in col]
    baseline_columns = [col for col in merged.columns if "Allocated" in col and "Baseline" in col]

    for base_col, scn_col in zip(sorted(baseline_columns), sorted(allocated_columns)):
        diff_col = scn_col.replace("Scenario", "Δ")
        merged[diff_col] = merged[scn_col] - merged[base_col]

    return merged


def _collection_key(adjustment_type: AdjustmentType) -> str:
    mapping = {
        AdjustmentType.ADD_EMPLOYEE: "employees",
        AdjustmentType.REMOVE_EMPLOYEE: "employees",
        AdjustmentType.UPDATE_EMPLOYEE: "employees",
        AdjustmentType.ADD_ALLOCATION: "allocations",
        AdjustmentType.REMOVE_ALLOCATION: "allocations",
        AdjustmentType.UPDATE_ALLOCATION: "allocations",
        AdjustmentType.ADD_SUPPORT_ALLOCATION: "support_allocations",
        AdjustmentType.REMOVE_SUPPORT_ALLOCATION: "support_allocations",
        AdjustmentType.UPDATE_SUPPORT_ALLOCATION: "support_allocations",
        AdjustmentType.ADD_REQUIRED_COVERAGE: "coverage",
        AdjustmentType.REMOVE_REQUIRED_COVERAGE: "coverage",
        AdjustmentType.UPDATE_REQUIRED_COVERAGE: "coverage",
        AdjustmentType.ADD_PROCESS: "processes",
        AdjustmentType.REMOVE_PROCESS: "processes",
        AdjustmentType.UPDATE_PROCESS: "processes",
    }
    return mapping[adjustment_type]


def _get_uuid(item) -> str:
    if hasattr(item, "uuid"):
        return getattr(item, "uuid")
    return item.get("uuid")


def _to_dict(item) -> Dict:
    if is_dataclass(item):
        return asdict(item)
    if hasattr(item, "__dict__"):
        return {**item.__dict__}
    return item
