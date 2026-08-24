class ScriptedLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def planner_response():
    return {
        "type": "message",
        "content": {
            "academicTasks": [
                {"task": "Study", "day": None, "frequency": 3},
                {"task": "Coursework deadline", "day": "Friday", "frequency": 1},
            ],
            "sportsCommitments": [
                {"activity": "Lacrosse match", "day": "Wednesday"},
                {"activity": "Training", "day": "Tuesday"},
            ],
            "workoutGoals": [{"activity": "Gym", "frequency": 4}],
            "constraints": [{"type": "sleep", "value": "7 hours"}],
        },
    }
