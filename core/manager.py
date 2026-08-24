from typing import Callable, Dict, List

from agents import CriticAgent, FitnessAgent, NutritionAgent, PlannerAgent, WriterAgent


class AgentManager:
    """Coordinate explicit, typed handoffs between the five prototype agents."""

    def __init__(
        self,
        llm: Callable[[str], Dict],
        workload_predictor: Callable[[List[float]], int] = None,
    ):
        self.planner = PlannerAgent(llm)
        self.fitness = FitnessAgent()
        self.nutrition = NutritionAgent()
        self.critic = CriticAgent(workload_predictor) if workload_predictor else CriticAgent()
        self.writer = WriterAgent(llm)

    def process(self, user_request: str):
        brief = self.planner.run(user_request)
        workout_plan = self.fitness.run(brief)
        nutrition = self.nutrition.run(brief, workout_plan)
        workload = self.critic.run(brief, workout_plan)
        return self.writer.run(brief, workout_plan, nutrition, workload)
