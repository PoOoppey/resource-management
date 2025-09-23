from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Dict, Iterable, List, Tuple
from uuid import uuid4

import pandas as pd
import streamlit as st

from models import AdjustmentType, Scenario, ScenarioAdjustment
from services.coverage import compute_theoretical_coverage
from services.data_loader import get_data, update_data
from services.scenario import apply_scenario


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


def _diff_table(dataset: str, baseline_df: pd.DataFrame, scenario_df: pd.DataFrame) -> pd.DataFrame:
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

    all_keys = sorted(set(baseline_map) | set(scenario_map))

    columns: List[str] = []
    for df in (baseline_df, scenario_df):
        for column in df.columns:
            if column == "uuid":
                continue
            if column not in columns:
                columns.append(column)

    display_rows: List[Dict] = []

    for key in all_keys:
        baseline_record = baseline_map.get(key)
        scenario_record = scenario_map.get(key)

        display_uuid = None
        if baseline_record and baseline_record.get("uuid"):
            display_uuid = baseline_record.get("uuid")
        elif scenario_record and scenario_record.get("uuid"):
            display_uuid = scenario_record.get("uuid")
        else:
            display_uuid = key

        row: Dict[str, object] = {"UUID": display_uuid}
        differences: List[str] = []

        if baseline_record and not scenario_record:
            diff_label = "Removed"
        elif scenario_record and not baseline_record:
            diff_label = "Added"
        else:
            diff_label = "Unchanged"

        for column in columns:
            base_value = baseline_record.get(column) if baseline_record else None
            scenario_value = scenario_record.get(column) if scenario_record else None

            if scenario_record is None:
                display_value = base_value
            elif baseline_record is None:
                display_value = scenario_value
            else:
                display_value = scenario_value

            row[column] = display_value

            if baseline_record and scenario_record:
                if _normalize_value(base_value) != _normalize_value(scenario_value):
                    differences.append(
                        f"{column}: {_format_value(base_value)} -> {_format_value(scenario_value)}"
                    )
            elif baseline_record and not scenario_record:
                differences.append(f"{column}: {_format_value(base_value)} -> —")
            elif scenario_record and not baseline_record:
                differences.append(f"{column}: — -> {_format_value(scenario_value)}")

        if diff_label in {"Added", "Removed"}:
            row[DIFF_COLUMN_LABEL] = diff_label
        elif differences:
            row[DIFF_COLUMN_LABEL] = ", ".join(differences)
        else:
            row[DIFF_COLUMN_LABEL] = "Unchanged"

        display_rows.append(row)

    ordered_columns = ["UUID", *columns, DIFF_COLUMN_LABEL]

    diff_df = pd.DataFrame(display_rows, columns=ordered_columns)
    if "UUID" in diff_df.columns:
        diff_df = diff_df.sort_values(by="UUID", kind="stable")
    return diff_df


def _build_diff_annotations(
    baseline_df: pd.DataFrame, scenario_df: pd.DataFrame
) -> Tuple[List[str], List[str]]:
    baseline_df = baseline_df.copy()
    scenario_df = scenario_df.copy()

    if not baseline_df.empty and "uuid" not in baseline_df.columns:
        baseline_df.insert(0, "uuid", "")
    if not scenario_df.empty and "uuid" not in scenario_df.columns:
        scenario_df.insert(0, "uuid", "")

    baseline_records = _prepare_records(baseline_df)

    annotations: List[str] = []
    scenario_uuids: List[str] = []

    comparison_columns: List[str] = [
        column for column in scenario_df.columns if column not in {"uuid", DIFF_COLUMN_LABEL}
    ]

    for _, row in scenario_df.iterrows():
        uuid_value = str(row.get("uuid") or "").strip()
        scenario_uuids.append(uuid_value)

        if not uuid_value:
            annotations.append("Added")
            continue

        baseline_record = baseline_records.get(uuid_value)
        if baseline_record is None:
            annotations.append("Added")
            continue

        differences: List[str] = []
        for column in comparison_columns:
            baseline_value = baseline_record.get(column)
            scenario_value = row.get(column)
            if _normalize_value(baseline_value) != _normalize_value(scenario_value):
                differences.append(
                    f"{column}: {_format_value(baseline_value)} -> {_format_value(scenario_value)}"
                )

        if differences:
            annotations.append(", ".join(differences))
        else:
            annotations.append("Unchanged")

    baseline_uuid_set = set(baseline_records.keys())
    scenario_uuid_set = {uuid for uuid in scenario_uuids if uuid}
    removed = sorted(baseline_uuid_set - scenario_uuid_set)
    return annotations, removed


def _scenario_select(scenarios: List[Scenario]) -> Scenario | None:
    if not scenarios:
        return None
    options = {scenario.name: scenario for scenario in scenarios}
    choice = st.selectbox("Scenario", list(options.keys()))
    return options[choice]


def _coerce_numeric(value):
    if isinstance(value, str):
        try:
            numeric = value.strip().split(" ")[0].replace(",", "")
            return float(numeric)
        except (ValueError, IndexError):
            return None
    return value


def _colorize_cell(required: float | None, value) -> str:
    value = _coerce_numeric(value)
    required = _coerce_numeric(required)
    if pd.isna(required) or pd.isna(value):
        return ""

    if required == 0:
        if value == 0:
            return ""
        return "background-color: #bae6fd"

    gap_ratio = (value - required) / required
    if gap_ratio <= -0.15:
        return "background-color: #fca5a5"
    if gap_ratio < 0:
        return "background-color: #fecaca"
    if gap_ratio < 0.1:
        return "background-color: #fef08a"
    if gap_ratio < 0.25:
        return "background-color: #bbf7d0"
    return "background-color: #86efac"


def _format_coverage_value(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    if abs(value) < 1:
        return f"{value:.2f}"
    return f"{value:.1f}"


def _build_coverage_result(
    baseline: pd.DataFrame, scenario: pd.DataFrame
) -> pd.DataFrame:
    common_columns = [col for col in ["Region", "Office", "Process"] if col in baseline.columns]
    merged = baseline.merge(
        scenario,
        on=common_columns,
        how="outer",
        suffixes=(" (Baseline)", " (Scenario)"),
    ).fillna(0.0)

    if "Required (Baseline)" not in merged.columns and "Required" in baseline.columns:
        merged = merged.rename(columns={"Required": "Required (Baseline)"})
    if "Coverage (Baseline)" not in merged.columns and "Coverage" in baseline.columns:
        merged = merged.rename(columns={"Coverage": "Coverage (Baseline)"})
    if "Required (Scenario)" not in merged.columns and "Required" in scenario.columns:
        merged = merged.rename(columns={"Required": "Required (Scenario)"})
    if "Coverage (Scenario)" not in merged.columns and "Coverage" in scenario.columns:
        merged = merged.rename(columns={"Coverage": "Coverage (Scenario)"})

    merged["Required Diff"] = (
        merged.get("Required (Scenario)", 0.0) - merged.get("Required (Baseline)", 0.0)
    )
    merged["Coverage Diff"] = (
        merged.get("Coverage (Scenario)", 0.0) - merged.get("Coverage (Baseline)", 0.0)
    )

    merged = merged.rename(
        columns={"Required (Scenario)": "Required", "Coverage (Scenario)": "Coverage"}
    )

    ordered_columns = [
        *common_columns,
        "Required",
        "Required Diff",
        "Coverage",
        "Coverage Diff",
    ]
    ordered_columns = [col for col in ordered_columns if col in merged.columns]
    return merged[ordered_columns]


def _style_coverage_result(df: pd.DataFrame) -> pd.io.formats.style.Styler | pd.DataFrame:
    if df.empty:
        return df

    numeric_df = df.copy()
    display_df = df.copy()

    def _format_with_diff(value: float | None, diff: float | None) -> str:
        if value is None or pd.isna(value):
            return "—"
        value_str = _format_coverage_value(value)
        if diff is None or pd.isna(diff) or diff == 0:
            return value_str
        diff_str = _format_coverage_value(abs(diff))
        sign = "+" if diff > 0 else "-"
        return f"{value_str} ({sign}{diff_str})"

    if "Required" in display_df.columns and "Required Diff" in display_df.columns:
        display_df["Required"] = display_df.apply(
            lambda row: _format_with_diff(row.get("Required"), row.get("Required Diff")), axis=1
        )
    if "Coverage" in display_df.columns and "Coverage Diff" in display_df.columns:
        display_df["Coverage"] = display_df.apply(
            lambda row: _format_with_diff(row.get("Coverage"), row.get("Coverage Diff")), axis=1
        )

    drop_columns = [column for column in ["Required Diff", "Coverage Diff"] if column in display_df.columns]
    display_df = display_df.drop(columns=drop_columns)

    styler = display_df.style

    def _style_row(row: pd.Series) -> pd.Series:
        styles: Dict[str, str] = {}
        numeric_row = numeric_df.loc[row.name]
        scenario_required = numeric_row.get("Required")
        scenario_coverage = numeric_row.get("Coverage")

        if "Coverage" in display_df.columns:
            styles["Coverage"] = _colorize_cell(scenario_required, scenario_coverage)
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
    edited_df = edited_df.drop(columns=[DIFF_COLUMN_LABEL], errors="ignore")
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

            st.markdown("**Scenario data**")
            annotations, removed_items = _build_diff_annotations(baseline_df, scenario_df)

            scenario_display_df = scenario_df.copy()
            if not scenario_display_df.empty and "uuid" not in scenario_display_df.columns:
                scenario_display_df.insert(0, "uuid", "")
            scenario_display_df[DIFF_COLUMN_LABEL] = annotations

            column_config: Dict[str, object] = {}
            if "uuid" in scenario_display_df.columns:
                column_config["uuid"] = st.column_config.TextColumn("UUID", disabled=True)
            column_config[DIFF_COLUMN_LABEL] = st.column_config.TextColumn(
                DIFF_COLUMN_LABEL, disabled=True
            )

            display_df = scenario_display_df
            if show_differences_only:
                display_df = scenario_display_df[scenario_display_df[DIFF_COLUMN_LABEL] != "Unchanged"].copy()

            editor_result: pd.DataFrame | None = None
            if show_differences_only:
                if display_df.empty:
                    st.info("No changes detected for this dataset.")
                else:
                    st.dataframe(display_df, use_container_width=True)
            else:
                editor_result = st.data_editor(
                    display_df,
                    num_rows="dynamic",
                    use_container_width=True,
                    key=f"scenario_editor_{scenario.uuid}_{dataset}",
                    column_config=column_config or None,
                )

            if removed_items:
                removed_label = ", ".join(removed_items)
                st.caption(f"Removed entries: {removed_label}")

            if not show_differences_only and st.button(
                f"Save {label.lower()} changes", key=f"save_{scenario.uuid}_{dataset}"
            ):
                cleaned_editor_df = editor_result.drop(columns=[DIFF_COLUMN_LABEL], errors="ignore")
                new_adjustments = _calculate_dataset_adjustments(dataset, baseline_df, cleaned_editor_df)
                _merge_adjustments(scenario, dataset, new_adjustments)
                update_data("scenarios", data["scenarios"])
                st.success(f"Scenario adjustments for {label.lower()} saved.")
                st.experimental_rerun()

    st.subheader("Scenario result")
    coverage_result = _build_coverage_result(baseline_coverage, scenario_coverage)
    styled_coverage = _style_coverage_result(coverage_result)
    st.dataframe(styled_coverage, use_container_width=True)


if __name__ == "__main__":
    main()
