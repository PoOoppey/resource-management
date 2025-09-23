from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import streamlit as st

from models import Scenario
from services.coverage import compute_theoretical_coverage
from services.data_loader import get_data
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


def _normalize_value(value):
    if value is None:
        return None
    if value is pd.NA or (hasattr(pd, "isna") and pd.isna(value)):
        return None
    if isinstance(value, float):
        return round(value, 6)
    return value


def _format_value(value) -> str:
    if value is None or value is pd.NA or (hasattr(pd, "isna") and pd.isna(value)):
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

            baseline_column = f"{column} (Baseline)"
            scenario_column = f"{column} (Scenario)"

            row[baseline_column] = base_value
            row[scenario_column] = scenario_value

            if baseline_record and scenario_record:
                if _normalize_value(base_value) != _normalize_value(scenario_value):
                    differences.append(
                        f"{column}: {_format_value(base_value)} -> {_format_value(scenario_value)}"
                    )

        if diff_label in {"Added", "Removed"}:
            row[DIFF_COLUMN_LABEL] = diff_label
        elif differences:
            row[DIFF_COLUMN_LABEL] = ", ".join(differences)
        else:
            row[DIFF_COLUMN_LABEL] = "Unchanged"

        display_rows.append(row)

    ordered_columns = ["UUID"]
    for column in columns:
        ordered_columns.extend([f"{column} (Baseline)", f"{column} (Scenario)"])
    ordered_columns.append(DIFF_COLUMN_LABEL)

    diff_df = pd.DataFrame(display_rows, columns=ordered_columns)
    if "UUID" in diff_df.columns:
        diff_df = diff_df.sort_values(by="UUID", kind="stable")
    return diff_df


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

    merged[DIFF_COLUMN_LABEL] = (
        merged.get("Coverage (Scenario)", 0.0) - merged.get("Coverage (Baseline)", 0.0)
    )

    ordered_columns = [
        *common_columns,
        "Required (Baseline)",
        "Required (Scenario)",
        "Coverage (Baseline)",
        "Coverage (Scenario)",
        DIFF_COLUMN_LABEL,
    ]
    ordered_columns = [col for col in ordered_columns if col in merged.columns]
    return merged[ordered_columns]


def _style_coverage_result(df: pd.DataFrame) -> pd.io.formats.style.Styler | pd.DataFrame:
    if df.empty:
        return df

    styler = df.style.format(
        {col: _format_coverage_value for col in df.columns if col not in {"Region", "Office", "Process"}}
    )

    def _style_row(row: pd.Series) -> pd.Series:
        styles: Dict[str, str] = {}
        baseline_required = row.get("Required (Baseline)")
        baseline_coverage = row.get("Coverage (Baseline)")
        scenario_required = row.get("Required (Scenario)")
        scenario_coverage = row.get("Coverage (Scenario)")

        if "Coverage (Baseline)" in df.columns:
            styles["Coverage (Baseline)"] = _colorize_cell(baseline_required, baseline_coverage)
        if "Coverage (Scenario)" in df.columns:
            styles["Coverage (Scenario)"] = _colorize_cell(scenario_required, scenario_coverage)

        diff_value = row.get(DIFF_COLUMN_LABEL)
        if pd.notna(diff_value):
            if diff_value > 0:
                styles[DIFF_COLUMN_LABEL] = "color: #166534"
            elif diff_value < 0:
                styles[DIFF_COLUMN_LABEL] = "color: #b91c1c"
            else:
                styles[DIFF_COLUMN_LABEL] = "color: #4b5563"
        return pd.Series(styles)

    styler = styler.apply(_style_row, axis=1)
    return styler


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

    with st.expander("Baseline vs scenario details", expanded=True):
        tabs = st.tabs([label for _, label in DISPLAY_DATASETS])
        for (dataset, label), tab in zip(DISPLAY_DATASETS, tabs):
            with tab:
                baseline_df = _items_to_dataframe(data.get(dataset, [])).copy()
                scenario_df = _items_to_dataframe(modified_data.get(dataset, [])).copy()
                diff_table = _diff_table(dataset, baseline_df, scenario_df)

                if show_differences_only:
                    diff_table = diff_table[diff_table[DIFF_COLUMN_LABEL] != "Unchanged"]

                if diff_table.empty:
                    st.info("No changes detected for this dataset.")
                else:
                    st.dataframe(diff_table, use_container_width=True)

    st.subheader("Scenario result")
    coverage_result = _build_coverage_result(baseline_coverage, scenario_coverage)
    styled_coverage = _style_coverage_result(coverage_result)
    st.dataframe(styled_coverage, use_container_width=True)


if __name__ == "__main__":
    main()
