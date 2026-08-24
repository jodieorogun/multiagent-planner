from core.models import ParsedRequest, WorkloadAssessment


class CriticAgent:
    """Estimate planning pressure with a transparent demonstration heuristic."""

    def run(self, request: ParsedRequest) -> WorkloadAssessment:
        academic_sessions = sum(task.sessions for task in request.academic_tasks)
        deadline_count = sum(
            "deadline" in task.name.lower() for task in request.academic_tasks
        )
        score = (
            academic_sessions
            + len(request.sports_commitments)
            + request.gym_sessions
            + deadline_count
        )
        if request.minimum_sleep_hours < 7:
            score += 2

        if score <= 6:
            level = "low"
            note = "The week has room for recovery and unexpected changes."
        elif score <= 10:
            level = "moderate"
            note = "Keep at least one flexible block and protect sleep."
        else:
            level = "high"
            note = "Consider dropping or moving one optional session."

        return WorkloadAssessment(score=score, level=level, note=note)
