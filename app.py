from core.llm import call_llm
from core.agentManager import AgentManager
from agents.plannerAgent import PlannerAgent
from agents.fitnessAgent import FitnessAgent
from agents.nutritionAgent import NutritionAgent
from agents.criticAgent import CriticAgent
from agents.writerAgent import WriterAgent


def llm(prompt: str):
    return call_llm(prompt)


def main():
    plannerAgent = PlannerAgent(name="PlannerAgent", llm=llm)
    fitnessAgent = FitnessAgent(name="FitnessAgent", llm=llm)
    nutritionAgent = NutritionAgent(name="NutritionAgent", llm=llm)
    criticAgent = CriticAgent(name="CriticAgent", llm=llm)
    writerAgent = WriterAgent(name="WriterAgent", llm=llm)

    agents = [
        plannerAgent,
        fitnessAgent,
        nutritionAgent,
        criticAgent,
        writerAgent,
    ]

    agentManager = AgentManager(agents)

    userRequest = (
        "plan my week: I have a lacrosse match on Wednesday, 2 training sessions, 3 evenings of study, a coursework deadline on Friday, I want to gym 4 times and still sleep at least 7 hours."
    )

    finalPlan = agentManager.process(userRequest)
    print("\n=== FINAL WEEKLY PLAN ===\n")
    if isinstance(finalPlan, dict) and "content" in finalPlan:
        print(finalPlan["content"])
    else:
        print(finalPlan)


if __name__ == "__main__":
    main()
