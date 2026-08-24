import re

from core.models import AcademicTask, DAYS, ParsedRequest, SportsCommitment


DAY_PATTERN = "|".join(DAYS)


def _find_day(text: str):
    match = re.search(rf"\b({DAY_PATTERN})\b", text, re.IGNORECASE)
    return match.group(1).title() if match else None


class PlannerAgent:
    """Turn a small set of natural-language planning phrases into structured data."""

    def run(self, request: str) -> ParsedRequest:
        academic_tasks = []
        sports_commitments = []

        for segment in re.split(r"[,.;]", request):
            text = segment.strip()
            lowered = text.lower()
            day = _find_day(text)

            if day and "match" in lowered:
                activity_match = re.search(r"\b([a-z]+)\s+match\b", lowered)
                sport = activity_match.group(1).title() if activity_match else "Sports"
                sports_commitments.append(
                    SportsCommitment(activity=f"{sport} match", day=day)
                )

            if day and "training" in lowered:
                sports_commitments.append(
                    SportsCommitment(activity="Training", day=day)
                )

            if "deadline" in lowered:
                name = "Coursework deadline" if "coursework" in lowered else "Deadline"
                academic_tasks.append(AcademicTask(name=name, day=day))

        study_match = re.search(
            r"\b(\d+)\s+(?:evenings?|sessions?)\b.{0,35}\bstud(?:y|ying)\b",
            request,
            re.IGNORECASE,
        )
        if study_match:
            academic_tasks.append(
                AcademicTask(name="Study session", sessions=int(study_match.group(1)))
            )
        elif re.search(r"\bstud(?:y|ying)\b", request, re.IGNORECASE):
            academic_tasks.append(AcademicTask(name="Study session"))

        gym_sessions = self._extract_gym_sessions(request)
        minimum_sleep_hours = self._extract_sleep_hours(request)

        return ParsedRequest(
            academic_tasks=academic_tasks,
            sports_commitments=sports_commitments,
            gym_sessions=gym_sessions,
            minimum_sleep_hours=minimum_sleep_hours,
        )

    @staticmethod
    def _extract_gym_sessions(request: str) -> int:
        patterns = (
            r"\bgym\s+(\d+)\s+times?\b",
            r"\b(\d+)\s+(?:gym|workout)\s+sessions?\b",
        )
        for pattern in patterns:
            match = re.search(pattern, request, re.IGNORECASE)
            if match:
                return min(int(match.group(1)), 7)
        return 0

    @staticmethod
    def _extract_sleep_hours(request: str) -> int:
        patterns = (
            r"\bsleep\b.{0,30}\b(\d+)\s+hours?\b",
            r"\b(\d+)\s+hours?\b.{0,30}\bsleep\b",
        )
        for pattern in patterns:
            match = re.search(pattern, request, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return 7
