from core.models import NutritionPlan


class NutritionAgent:
    def run(self, workout_plan) -> NutritionPlan:
        match_days = [
            day
            for day, activities in workout_plan.items()
            if any("match" in activity.lower() for activity in activities)
        ]
        return NutritionPlan(
            daily_calories=2000,
            extra_fuel_days=match_days,
            extra_calories_amount=300 if match_days else 0,
            meals=[
                "Breakfast: Greek yogurt and granola",
                "Lunch: Chicken, rice and vegetables",
                "Dinner: High-protein pasta",
            ],
        )
