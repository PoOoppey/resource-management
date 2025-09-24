from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd

from models import Employee, EmployeeExpertise, Office, Process


def build_expertise_dataframe(
    expertise_levels: Iterable[EmployeeExpertise],
    employees: Iterable[Employee],
    processes: Iterable[Process],
    offices: Iterable[Office],
    *,
    as_of: date | None = None,
) -> pd.DataFrame:
    """Return a normalized DataFrame of expertise assignments."""

    employee_lookup = {employee.uuid: employee for employee in employees}
    process_lookup = {process.uuid: process for process in processes}
    office_lookup = {office.uuid: office for office in offices}

    records: list[dict[str, object]] = []
    for item in expertise_levels:
        employee = employee_lookup.get(item.employee_uuid)
        process = process_lookup.get(item.process_uuid)
        if not employee or not process:
            continue

        office = office_lookup.get(employee.office_uuid)
        employee_name = " ".join(part for part in [employee.first_name, employee.last_name] if part)
        if not employee_name:
            employee_name = employee.trigram or employee.uuid

        start = item.start_date
        end = item.end_date
        is_active = True
        if as_of:
            is_active = start <= as_of and (end is None or end >= as_of)

        records.append(
            {
                "Expertise UUID": item.uuid,
                "Employee UUID": employee.uuid,
                "Process UUID": process.uuid,
                "Employee": employee_name,
                "Trigram": employee.trigram,
                "Process": process.name,
                "Office": office.name if office else None,
                "Region": office.region.value if office else None,
                "Level": int(item.level),
                "Start Date": start,
                "End Date": end,
                "Active": is_active,
            }
        )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return pd.DataFrame(
            columns=
            [
                "Expertise UUID",
                "Employee UUID",
                "Process UUID",
                "Employee",
                "Trigram",
                "Process",
                "Office",
                "Region",
                "Level",
                "Start Date",
                "End Date",
                "Active",
            ]
        )

    df = df.sort_values(["Employee", "Process", "Start Date"], ignore_index=True)
    return df
