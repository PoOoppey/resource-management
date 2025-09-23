from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Mapping


FOREIGN_KEY_RELATIONS: Dict[str, Dict[str, tuple[str, str]]] = {
    "employees": {
        "office_uuid": ("offices", "office"),
    },
    "allocations": {
        "employee_uuid": ("employees", "employee"),
        "role_uuid": ("roles", "role"),
    },
    "support_allocations": {
        "allocation_uuid": ("allocations", "allocation"),
        "process_uuid": ("processes", "process"),
    },
    "coverage": {
        "process_uuid": ("processes", "process"),
        "office_uuid": ("offices", "office"),
    },
}


def _to_dict(item: Any) -> Dict[str, Any]:
    if is_dataclass(item):
        return asdict(item)
    if hasattr(item, "__dict__"):
        return {**item.__dict__}
    if isinstance(item, Mapping):
        return dict(item)
    return {}


def build_reference_lookup(
    *, baseline_data: Mapping[str, Iterable], modified_data: Mapping[str, Iterable]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    lookup: Dict[str, Dict[str, Dict[str, Any]]] = {}
    datasets = set(baseline_data.keys()) | set(modified_data.keys())

    for dataset in datasets:
        records: Dict[str, Dict[str, Any]] = {}
        for source in (modified_data.get(dataset), baseline_data.get(dataset)):
            if not source:
                continue
            for item in source:
                record = _to_dict(item)
                identifier = str(record.get("uuid") or "").strip()
                if not identifier:
                    continue
                if identifier not in records:
                    records[identifier] = record
        if records:
            lookup[dataset] = records

    return lookup


def enrich_payload(
    dataset: str,
    payload: Dict[str, Any],
    reference_lookup: Mapping[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    relations = FOREIGN_KEY_RELATIONS.get(dataset)
    if not relations:
        return dict(payload)

    enriched = dict(payload)

    for field, (related_dataset, alias) in relations.items():
        if field not in enriched:
            continue

        related_identifier = enriched.get(field)
        related_record: Dict[str, Any] | None = None
        if related_identifier:
            related_record = reference_lookup.get(related_dataset, {}).get(
                str(related_identifier)
            )

        enriched.pop(field, None)
        if related_record is not None:
            enriched[alias] = dict(related_record)
        elif related_identifier:
            enriched[alias] = {"uuid": str(related_identifier)}
        else:
            enriched[alias] = None

    return enriched


def _extract_uuid(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        raw_uuid = value.get("uuid")
    elif is_dataclass(value):
        raw_uuid = getattr(value, "uuid", None)
    else:
        raw_uuid = getattr(value, "uuid", value)
    if raw_uuid is None:
        return None
    return str(raw_uuid)


def flatten_payload(dataset: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    relations = FOREIGN_KEY_RELATIONS.get(dataset)
    if not relations:
        return dict(payload)

    flattened = dict(payload)

    for field, (_, alias) in relations.items():
        if alias in flattened:
            identifier = _extract_uuid(flattened.pop(alias))
            flattened[field] = identifier

    return flattened
