from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from models import (
    Allocation,
    AttendanceRecord,
    Employee,
    Office,
    Process,
    RequiredCoverage,
    Role,
    SupportAllocation,
)

STANDARD_WEEKLY_HOURS = 40


def _build_lookup(items: Iterable, key: str) -> Dict[str, object]:
    return {getattr(item, key): item for item in items}


def _allocation_hours(
    employee: Employee,
    allocation: Allocation,
    support_allocation: SupportAllocation | None = None,
) -> float:
    base_hours = employee.working_hours * allocation.percentage * allocation.weight
    if support_allocation is not None:
        base_hours *= support_allocation.percentage
        base_hours *= support_allocation.effective_weight(allocation.weight)
    return base_hours


def _convert_unit(value: float, unit: str) -> float:
    if unit.lower() == "fte":
        return value / STANDARD_WEEKLY_HOURS
    return value


def compute_theoretical_coverage(data, view: str, group_by: str, unit: str) -> pd.DataFrame:
    employees: List[Employee] = data.get("employees", [])
    offices: List[Office] = data.get("offices", [])
    roles: List[Role] = data.get("roles", [])
    allocations: List[Allocation] = data.get("allocations", [])
    support_allocations: List[SupportAllocation] = data.get("support_allocations", [])
    coverage: List[RequiredCoverage] = data.get("coverage", [])
    processes: List[Process] = data.get("processes", [])

    office_lookup = _build_lookup(offices, "uuid")
    role_lookup = _build_lookup(roles, "uuid")
    process_lookup = _build_lookup(processes, "uuid")

    group_key = "office_uuid" if group_by.lower() == "office" else "region"

    if view.lower() == "role":
        matrix: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        employee_lookup = _build_lookup(employees, "uuid")
        for allocation in allocations:
            employee = employee_lookup.get(allocation.employee_uuid)
            if not employee:
                continue
            office = office_lookup[employee.office_uuid]
            role = role_lookup[allocation.role_uuid]

            group_identifier = office.uuid if group_key == "office_uuid" else office.region.value
            column = role.name
            hours = _convert_unit(_allocation_hours(employee, allocation), unit)
            matrix[group_identifier][column] += hours
            if group_key == "office_uuid":
                matrix[group_identifier]["Office"] = office.name
                matrix[group_identifier]["Region"] = office.region.value
            else:
                matrix[group_identifier]["Region"] = office.region.value
        records = []
        for identifier, values in matrix.items():
            row = {"Region": values.get("Region")}
            if group_key == "office_uuid":
                row["Office"] = values.get("Office")
            row.update({col: val for col, val in values.items() if col not in {"Region", "Office"}})
            records.append(row)
        df = pd.DataFrame(records)
        df = df.fillna(0.0)
        return df.sort_values(by=[col for col in ["Region", "Office"] if col in df.columns])

    # Process view
    required_rows = []
    for cov in coverage:
        office = office_lookup.get(cov.office_uuid)
        process = process_lookup.get(cov.process_uuid)
        if not office or not process:
            continue
        group_identifier = office.uuid if group_key == "office_uuid" else office.region.value
        required_rows.append(
            {
                "Region": office.region.value,
                "Office": office.name,
                "Process": process.name,
                "Group": group_identifier,
                "Required": cov.required_hours,
            }
        )

    required_df = pd.DataFrame(required_rows)

    contribution_rows = []
    support_lookup = defaultdict(list)
    for support in support_allocations:
        support_lookup[support.allocation_uuid].append(support)

    employee_lookup = _build_lookup(employees, "uuid")

    for allocation in allocations:
        employee = employee_lookup.get(allocation.employee_uuid)
        if not employee:
            continue
        office = office_lookup.get(employee.office_uuid)
        if not office:
            continue
        allocation_supports = support_lookup.get(allocation.uuid) or [None]
        for support in allocation_supports:
            if support is None:
                continue
            process = process_lookup.get(support.process_uuid)
            if not process:
                continue
            group_identifier = office.uuid if group_key == "office_uuid" else office.region.value
            contribution_rows.append(
                {
                    "Region": office.region.value,
                    "Office": office.name,
                    "Process": process.name,
                    "Group": group_identifier,
                    "Allocated": _convert_unit(
                        _allocation_hours(employee, allocation, support), unit
                    ),
                }
            )

    allocation_df = pd.DataFrame(contribution_rows)

    merged = required_df.merge(
        allocation_df,
        on=["Group", "Region", "Office", "Process"],
        how="left",
    )
    merged["Allocated"] = merged["Allocated"].fillna(0.0)
    merged["Required"] = merged["Required"].fillna(0.0)

    column_order = ["Region"]
    if group_key == "office_uuid":
        column_order.append("Office")
    column_order.extend(["Process", f"Required ({unit.lower()})", f"Allocated ({unit.lower()})"])

    merged = merged.rename(
        columns={
            "Required": f"Required ({unit.lower()})",
            "Allocated": f"Allocated ({unit.lower()})",
        }
    )

    return merged[column_order].sort_values(by=column_order[:-2])


def _week_range(start: date, end: date) -> List[date]:
    if start > end:
        start, end = end, start
    weeks = []
    cursor = start - timedelta(days=start.weekday())
    while cursor <= end:
        weeks.append(cursor)
        cursor += timedelta(days=7)
    return weeks


def _attendance_factor(attendances: List[AttendanceRecord], employee_uuid: str, week_start: date) -> float:
    week_end = week_start + timedelta(days=6)
    for record in attendances:
        if record.employee_uuid != employee_uuid:
            continue
        overlap = max(0, (min(record.end_date, week_end) - max(record.start_date, week_start)).days + 1)
        if overlap > 0:
            return 0.0
    return 1.0


def compute_live_coverage(
    data,
    attendance: List[AttendanceRecord],
    date_range: Tuple[date, date],
    display_mode: str,
    group_by: str,
    unit: str,
) -> pd.DataFrame:
    coverage_df = compute_theoretical_coverage(data, view="process", group_by=group_by, unit=unit)

    if coverage_df.empty:
        return coverage_df

    weeks = _week_range(*date_range)

    employees: List[Employee] = data.get("employees", [])
    allocations: List[Allocation] = data.get("allocations", [])
    support_allocations: List[SupportAllocation] = data.get("support_allocations", [])
    offices: List[Office] = data.get("offices", [])
    processes: List[Process] = data.get("processes", [])

    office_lookup = _build_lookup(offices, "uuid")
    process_lookup = {proc.uuid: proc for proc in processes}
    employee_lookup = _build_lookup(employees, "uuid")
    support_lookup = defaultdict(list)
    for support in support_allocations:
        support_lookup[support.allocation_uuid].append(support)

    base_records = coverage_df[[col for col in coverage_df.columns if "required" in col.lower()]]
    required_col = base_records.columns[-1]

    results = coverage_df.copy()

    for week_start in weeks:
        weekly_hours = defaultdict(float)
        for allocation in allocations:
            employee = employee_lookup.get(allocation.employee_uuid)
            if not employee:
                continue
            office = office_lookup.get(employee.office_uuid)
            if not office:
                continue
            supports = support_lookup.get(allocation.uuid)
            if not supports:
                continue
            factor = _attendance_factor(attendance, employee.uuid, week_start)
            for support in supports:
                process = process_lookup.get(support.process_uuid)
                if not process:
                    continue
                group = office.uuid if group_by.lower() == "office" else office.region.value
                key = (office.region.value, office.name, process.name, group)
                hours = _convert_unit(
                    _allocation_hours(employee, allocation, support) * factor,
                    unit,
                )
                weekly_hours[key] += hours

        column_label = week_start.strftime("%Y-W%W")
        results[column_label] = 0.0
        for (region, office_name, process_name, group), value in weekly_hours.items():
            mask = results["Region"] == region
            if "Office" in results.columns:
                mask = mask & (results["Office"] == office_name)
            mask = mask & (results["Process"] == process_name)
            results.loc[mask, column_label] = value

    if display_mode.lower() in {"jira", "coverage+jira"}:
        # placeholder: data loader returns aggregated JIRA counts; align by process only
        jira_map: Dict[Tuple[str, str, str], Dict[str, int]] = defaultdict(dict)
        for process in processes:
            from services.data_loader import get_jira_counts  # local import to avoid circular

            counts = get_jira_counts(process.uuid)
            for count in counts:
                label = count.week_start.strftime("%Y-W%W")
                jira_map[(process.uuid, label, "ticket_count")]["tickets"] = count.ticket_count

        for column in results.columns:
            if column.startswith("202") and display_mode.lower() == "jira":
                results[column] = results[column].apply(lambda _: "0")

        for process in processes:
            process_name = process.name
            for week_start in weeks:
                label = week_start.strftime("%Y-W%W")
                tickets = 0
                for count in jira_map.get((process.uuid, label, "ticket_count"), {}).values():
                    tickets += count
                if display_mode.lower() == "jira":
                    results.loc[results["Process"] == process_name, label] = tickets
                elif display_mode.lower() == "coverage+jira":
                    column_values = results.loc[results["Process"] == process_name, label]
                    results.loc[results["Process"] == process_name, label] = column_values.map(
                        lambda v: f"{v:.0f}{'h' if unit.lower() == 'hours' else 'fte'} ({tickets})"
                    )

    if display_mode.lower() == "coverage":
        results[required_col] = results[required_col]
    return results
