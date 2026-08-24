from typing import Callable, List

from core.models import PlanningBrief, WorkloadAssessment, WorkloadFeatures
from tools.workload_model import heuristic_predict


WORKLOAD_LABELS = ("light", "moderate", "high", "burnout risk")


class CriticAgent:
    def __init__(self, predictor: Callable[[List[float]], int] = heuristic_predict):
        self.predictor = predictor

    def run(self, brief: PlanningBrief) -> WorkloadAssessment:
        hours_study = 0.0
        deadlines = 0
        for task in brief.academic_tasks:
            lowered = task.task.lower()
            if "study" in lowered:
                hours_study += 1.5 * task.frequency
            if "coursework" in lowered or "deadline" in lowered:
                hours_study += 4.0
                deadlines += 1

        hours_sport = float(brief.gym_sessions)
        for commitment in brief.sports_commitments:
            lowered = commitment.activity.lower()
            hours_sport += 2.0 if "match" in lowered else 1.5

        features = WorkloadFeatures(
            hours_study=hours_study,
            hours_sport=hours_sport,
            hours_work=0.0,
            num_deadlines=float(deadlines),
            sleep_hours=brief.minimum_sleep_hours,
        )
        score = int(self.predictor(features.as_list()))
        score = max(0, min(score, len(WORKLOAD_LABELS) - 1))
        label = WORKLOAD_LABELS[score]
        note = (
            f"Estimated workload is {label}. "
            "Keep fixed commitments in place and protect the requested sleep window."
        )
        return WorkloadAssessment(
            score=score,
            label=label,
            note=note,
            features=features,
        )
