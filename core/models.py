from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


DAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


@dataclass(frozen=True)
class AcademicTask:
    name: str
    day: Optional[str] = None
    sessions: int = 1


@dataclass(frozen=True)
class SportsCommitment:
    activity: str
    day: str


@dataclass(frozen=True)
class ParsedRequest:
    academic_tasks: List[AcademicTask] = field(default_factory=list)
    sports_commitments: List[SportsCommitment] = field(default_factory=list)
    gym_sessions: int = 0
    minimum_sleep_hours: int = 7


@dataclass(frozen=True)
class WorkloadAssessment:
    score: int
    level: str
    note: str


@dataclass
class WeeklyPlan:
    days: Dict[str, List[str]]
    workload: WorkloadAssessment

    def to_dict(self) -> dict:
        return {
            "days": self.days,
            "workload": asdict(self.workload),
        }

    def render(self) -> str:
        lines = ["Weekly plan", "==========="]
        for day in DAYS:
            lines.extend(("", day, "-" * len(day)))
            activities = self.days.get(day) or ["Rest / flexible time"]
            lines.extend(f"- {activity}" for activity in activities)

        lines.extend(
            (
                "",
                f"Workload: {self.workload.level} (score {self.workload.score})",
                self.workload.note,
            )
        )
        return "\n".join(lines)
