class NutritionAgent:
    """Add one conservative, activity-aware nutrition reminder."""

    def run(self, fitness_schedule):
        notes = {}
        for day, activities in fitness_schedule.items():
            if any("match" in activity.lower() for activity in activities):
                notes[day] = "Fuel: add a pre-match snack and stay hydrated"
        return notes
