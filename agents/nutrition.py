from core.models import NutritionPlan, PlanningBrief, ScheduleDraft


class NutritionAgent:
    def run(self, brief: PlanningBrief, schedule: ScheduleDraft) -> NutritionPlan:
        match_days = [
            day
            for day, events in schedule.days.items()
            if any("match" in event.name.lower() for event in events)
        ]
        training_days = [
            day
            for day, events in schedule.days.items()
            if any("training" in event.name.lower() for event in events)
        ]
        daily_notes = {
            day: "Match day: add a carbohydrate-rich snack and hydrate before activity."
            for day in match_days
        }
        daily_notes.update(
            {
                day: "Training day: hydrate and include a protein-rich meal afterward."
                for day in training_days
                if day not in daily_notes
            }
        )
        return NutritionPlan(
            daily_calories=2000 + (300 if match_days else 0),
            extra_fuel_days=match_days,
            extra_calories_amount=300 if match_days else 0,
            meals=[
                "Breakfast: Greek yogurt and granola",
                "Lunch: Chicken, rice and vegetables",
                "Dinner: High-protein pasta",
            ],
            daily_notes=daily_notes,
        )
