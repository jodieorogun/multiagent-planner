from core.models import DAYS, ParsedRequest


class FitnessAgent:
    """Keep fixed sports commitments and place requested gym sessions."""

    GYM_DAY_ORDER = (
        "Monday",
        "Thursday",
        "Saturday",
        "Sunday",
        "Tuesday",
        "Friday",
        "Wednesday",
    )

    def run(self, request: ParsedRequest):
        schedule = {day: [] for day in DAYS}

        for commitment in request.sports_commitments:
            schedule[commitment.day].append(commitment.activity)

        sessions_added = 0
        for day in self.GYM_DAY_ORDER:
            if sessions_added >= request.gym_sessions:
                break
            if any("match" in item.lower() for item in schedule[day]):
                continue
            schedule[day].append("Gym session")
            sessions_added += 1

        return schedule
