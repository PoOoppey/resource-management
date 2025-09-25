from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Dict, List, Optional


class ExpertiseLevelError(ValueError):
    """Raised when an expertise level value is outside the allowed range."""



class Region(str, Enum):
    EMEA = "EMEA"
    AMER = "AMER"
    ASIA = "ASIA"


class RoleType(str, Enum):
    SUPPORT = "SUPPORT"
    PROJECT_MANAGEMENT = "PROJECT_MANAGEMENT"
    MANAGEMENT = "MANAGEMENT"


class SupportStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    TO_BE_DEFINED = "TO_BE_DEFINED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"


class Criticality(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class AttendanceType(str, Enum):
    VACATION = "VACATION"
    BIZTRIP = "BIZTRIP"
    SICK = "SICK"


class AdjustmentType(str, Enum):
    ADD_EMPLOYEE = "ADD_EMPLOYEE"
    REMOVE_EMPLOYEE = "REMOVE_EMPLOYEE"
    UPDATE_EMPLOYEE = "UPDATE_EMPLOYEE"
    ADD_ALLOCATION = "ADD_ALLOCATION"
    REMOVE_ALLOCATION = "REMOVE_ALLOCATION"
    UPDATE_ALLOCATION = "UPDATE_ALLOCATION"
    ADD_SUPPORT_ALLOCATION = "ADD_SUPPORT_ALLOCATION"
    REMOVE_SUPPORT_ALLOCATION = "REMOVE_SUPPORT_ALLOCATION"
    UPDATE_SUPPORT_ALLOCATION = "UPDATE_SUPPORT_ALLOCATION"
    ADD_REQUIRED_COVERAGE = "ADD_REQUIRED_COVERAGE"
    REMOVE_REQUIRED_COVERAGE = "REMOVE_REQUIRED_COVERAGE"
    UPDATE_REQUIRED_COVERAGE = "UPDATE_REQUIRED_COVERAGE"
    ADD_PROCESS = "ADD_PROCESS"
    REMOVE_PROCESS = "REMOVE_PROCESS"
    UPDATE_PROCESS = "UPDATE_PROCESS"


@dataclass
class Employee:
    uuid: str
    first_name: str
    last_name: str
    trigram: str
    office_uuid: str
    working_hours: float

    @classmethod
    def from_dict(cls, raw: Dict) -> "Employee":
        return cls(**raw)


@dataclass
class Office:
    uuid: str
    name: str
    region: Region

    @classmethod
    def from_dict(cls, raw: Dict) -> "Office":
        raw = {**raw, "region": Region(raw["region"])}
        return cls(**raw)


@dataclass
class App:
    uuid: str
    name: str
    criticality: Criticality

    @classmethod
    def from_dict(cls, raw: Dict) -> "App":
        raw = {**raw, "criticality": Criticality(raw["criticality"])}
        return cls(**raw)


@dataclass
class Process:
    uuid: str
    name: str
    criticality: Criticality
    description: str
    apps_related: List[str] = field(default_factory=list)
    process_related: List[str] = field(default_factory=list)
    support_status: SupportStatus = SupportStatus.TO_BE_DEFINED

    @classmethod
    def from_dict(cls, raw: Dict) -> "Process":
        raw = {
            **raw,
            "criticality": Criticality(raw["criticality"]),
            "support_status": SupportStatus(raw["support_status"]),
        }
        return cls(**raw)


@dataclass
class EmployeeExpertise:
    uuid: str
    employee_uuid: str
    process_uuid: str
    level: int
    start_date: date
    end_date: Optional[date] = None

    def __post_init__(self) -> None:
        if not 1 <= int(self.level) <= 5:
            raise ExpertiseLevelError("Expertise level must be between 1 and 5 inclusive.")
        self.level = int(self.level)

    @classmethod
    def from_dict(cls, raw: Dict) -> "EmployeeExpertise":
        end_date = raw.get("end_date")
        return cls(
            uuid=raw["uuid"],
            employee_uuid=raw["employee_uuid"],
            process_uuid=raw["process_uuid"],
            level=int(raw["level"]),
            start_date=date.fromisoformat(raw["start_date"]),
            end_date=date.fromisoformat(end_date) if end_date else None,
        )


@dataclass
class RequiredCoverage:
    uuid: str
    process_uuid: str
    office_uuid: str
    required_hours: float

    @classmethod
    def from_dict(cls, raw: Dict) -> "RequiredCoverage":
        return cls(**raw)


@dataclass
class Role:
    uuid: str
    name: str
    type: RoleType

    @classmethod
    def from_dict(cls, raw: Dict) -> "Role":
        raw = {**raw, "type": RoleType(raw["type"])}
        return cls(**raw)


@dataclass
class Allocation:
    uuid: str
    employee_uuid: str
    role_uuid: str
    percentage: float
    weight: float = 1.0

    @classmethod
    def from_dict(cls, raw: Dict) -> "Allocation":
        return cls(**raw)


@dataclass
class SupportAllocation:
    uuid: str
    allocation_uuid: str
    process_uuid: str
    percentage: float
    weight: Optional[float] = None

    @classmethod
    def from_dict(cls, raw: Dict) -> "SupportAllocation":
        return cls(**raw)

    def effective_weight(self, allocation_weight: float) -> float:
        return self.weight if self.weight is not None else allocation_weight


@dataclass
class AttendanceRecord:
    uuid: str
    employee_uuid: str
    start_date: date
    end_date: date
    type: AttendanceType

    @classmethod
    def from_dict(cls, raw: Dict) -> "AttendanceRecord":
        return cls(
            uuid=raw["uuid"],
            employee_uuid=raw["employee_uuid"],
            start_date=date.fromisoformat(raw["start_date"]),
            end_date=date.fromisoformat(raw["end_date"]),
            type=AttendanceType(raw["type"]),
        )


@dataclass
class JiraCount:
    process_uuid: str
    week_start: date
    ticket_count: int

    @classmethod
    def from_dict(cls, raw: Dict) -> "JiraCount":
        return cls(
            process_uuid=raw["process_uuid"],
            week_start=date.fromisoformat(raw["week_start"]),
            ticket_count=int(raw["ticket_count"]),
        )


@dataclass
class ScenarioAdjustment:
    uuid: str
    type: AdjustmentType
    payload: Dict

    @classmethod
    def from_dict(cls, raw: Dict) -> "ScenarioAdjustment":
        return cls(uuid=raw["uuid"], type=AdjustmentType(raw["type"]), payload=raw["payload"])


@dataclass
class Scenario:
    uuid: str
    name: str
    adjustments: List[ScenarioAdjustment]

    @classmethod
    def from_dict(cls, raw: Dict) -> "Scenario":
        adjustments = [ScenarioAdjustment.from_dict(adj) for adj in raw.get("adjustments", [])]
        return cls(uuid=raw["uuid"], name=raw["name"], adjustments=adjustments)


def asdict_list(items: List) -> List[Dict]:
    return [item.__dict__ for item in items]
