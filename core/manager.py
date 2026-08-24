from agents import CriticAgent, FitnessAgent, NutritionAgent, PlannerAgent, WriterAgent


class AgentManager:
    """Run the prototype agents in a visible, deterministic sequence."""

    def __init__(self):
        self.planner = PlannerAgent()
        self.fitness = FitnessAgent()
        self.nutrition = NutritionAgent()
        self.critic = CriticAgent()
        self.writer = WriterAgent()

    def process(self, user_request: str):
        parsed_request = self.planner.run(user_request)
        fitness_schedule = self.fitness.run(parsed_request)
        nutrition_notes = self.nutrition.run(fitness_schedule)
        workload = self.critic.run(parsed_request)
        return self.writer.run(
            parsed_request,
            fitness_schedule,
            nutrition_notes,
            workload,
        )
