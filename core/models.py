import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


DAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


class SchemaError(ValueError):
    """Raised when an LLM response does not match the expected contract."""


def normalise_day(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()[:3]
    return next((day for day in DAYS if day.lower().startswith(candidate)), None)


def unwrap_message(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise SchemaError("LLM response must be a JSON object")

    content = payload.get("content") if payload.get("type") == "message" else payload
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError as exc:
            raise SchemaError("LLM message content is not valid JSON") from exc

    if not isinstance(content, dict):
        raise SchemaError("LLM response content must be a JSON object")
    return content


def _positive_int(value: Any, default: int = 1, maximum: int = 7) -> int:
    try:
        return max(1, min(int(value), maximum))
    except (TypeError, ValueError):
        return default


def _positive_float(value: Any, default: float) -> float:
    try:
        return max(0.25, float(value))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class AcademicTask:
    task: str
    day: Optional[str] = None
    frequency: int = 1
    duration_hours: float = 1.5
    priority: str = "planned"


@dataclass(frozen=True)
class SportsCommitment:
    activity: str
    day: Optional[str] = None
    duration_hours: float = 1.5
    priority: str = "fixed"


@dataclass(frozen=True)
class WorkoutGoal:
    activity: str
    frequency: int
    duration_hours: float = 1.0
    priority: str = "planned"


@dataclass(frozen=True)
class Constraint:
    type: str
    value: str


@dataclass(frozen=True)
class PlanningBrief:
    academic_tasks: List[AcademicTask] = field(default_factory=list)
    sports_commitments: List[SportsCommitment] = field(default_factory=list)
    workout_goals: List[WorkoutGoal] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)

    @classmethod
    def from_llm_response(cls, payload: Any) -> "PlanningBrief":
        content = unwrap_message(payload)

        academic_tasks = [
            AcademicTask(
                task=str(item.get("task") or "Academic task"),
                day=normalise_day(item.get("day")),
                frequency=_positive_int(item.get("frequency")),
                duration_hours=_positive_float(
                    item.get("durationHours"),
                    4.0
                    if any(
                        word in str(item.get("task", "")).lower()
                        for word in ("coursework", "deadline")
                    )
                    else 1.5,
                ),
                priority=str(item.get("priority") or "planned"),
            )
            for item in content.get("academicTasks", [])
            if isinstance(item, dict)
        ]
        sports_commitments = [
            SportsCommitment(
                activity=str(item.get("activity") or item.get("name") or "Sports"),
                day=normalise_day(item.get("day")),
                duration_hours=_positive_float(
                    item.get("durationHours"),
                    2.0 if "match" in str(item.get("activity", "")).lower() else 1.5,
                ),
                priority="fixed",
            )
            for item in content.get("sportsCommitments", [])
            if isinstance(item, dict)
        ]
        workout_goals = [
            WorkoutGoal(
                activity=str(item.get("activity") or "Workout"),
                frequency=_positive_int(item.get("frequency")),
                duration_hours=_positive_float(item.get("durationHours"), 1.0),
                priority="planned",
            )
            for item in content.get("workoutGoals", [])
            if isinstance(item, dict)
        ]
        constraints = [
            Constraint(
                type=str(item.get("type") or "constraint"),
                value=str(item.get("value") or ""),
            )
            for item in content.get("constraints", [])
            if isinstance(item, dict)
        ]

        return cls(
            academic_tasks=academic_tasks,
            sports_commitments=sports_commitments,
            workout_goals=workout_goals,
            constraints=constraints,
        )

    @property
    def gym_sessions(self) -> int:
        return sum(
            goal.frequency
            for goal in self.workout_goals
            if goal.activity.lower() in {"gym", "workout", "gymming"}
        )

    @property
    def minimum_sleep_hours(self) -> float:
        for constraint in self.constraints:
            if "sleep" not in constraint.type.lower():
                continue
            for token in constraint.value.split():
                try:
                    return float(token)
                except ValueError:
                    continue
        return 7.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NutritionPlan:
    daily_calories: int
    extra_fuel_days: List[str]
    extra_calories_amount: int
    meals: List[str]
    daily_notes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ScheduledEvent:
    name: str
    duration_hours: float
    priority: str
    source: str


@dataclass(frozen=True)
class ScheduleDraft:
    days: Dict[str, List[ScheduledEvent]]
    conflicts: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class WorkloadFeatures:
    hours_study: float
    hours_sport: float
    hours_work: float
    num_deadlines: float
    sleep_hours: float

    def as_list(self) -> List[float]:
        return list(asdict(self).values())


@dataclass(frozen=True)
class WorkloadAssessment:
    score: int
    label: str
    note: str
    features: WorkloadFeatures
    daily_load: Dict[str, float] = field(default_factory=dict)
    peak_day: str = ""


@dataclass(frozen=True)
class FinalPlan:
    weekly_plan: Dict[str, str]
    stress_note: str
    writer_mode: str
    workload: Optional[WorkloadAssessment] = None
    conflicts: List[str] = field(default_factory=list)

    @classmethod
    def from_llm_response(cls, payload: Any) -> "FinalPlan":
        content = unwrap_message(payload)
        raw_plan = content.get("weeklyPlan") or content.get("weekly_plan")
        if not isinstance(raw_plan, dict):
            raise SchemaError("WriterAgent response is missing weeklyPlan")

        weekly_plan = {}
        for day in DAYS:
            value = raw_plan.get(day, "Rest")
            if isinstance(value, list):
                value = " / ".join(str(item) for item in value)
            weekly_plan[day] = str(value or "Rest")

        stress_note = str(
            content.get("stressNote") or content.get("stress_note") or ""
        )
        return cls(
            weekly_plan=weekly_plan,
            stress_note=stress_note,
            writer_mode="ollama",
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        lines = ["Weekly plan", "==========="]
        for day in DAYS:
            lines.extend(("", day, "-" * len(day), self.weekly_plan[day]))
        lines.extend(("", f"Workload note: {self.stress_note}"))
        if self.workload:
            lines.append(
                "Daily load: "
                + ", ".join(
                    f"{day} {hours:g}h"
                    for day, hours in self.workload.daily_load.items()
                    if hours
                )
            )
        if self.conflicts:
            lines.append("Scheduling conflicts:")
            lines.extend(f"- {conflict}" for conflict in self.conflicts)
        return "\n".join(lines)
