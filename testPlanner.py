from agents.plannerAgent import PlannerAgent
from core.llm import call_llm

a = PlannerAgent(name="PlannerAgent")

# inject actual LLM function
a.llm = call_llm

print(a.run("7 hours study", []))
