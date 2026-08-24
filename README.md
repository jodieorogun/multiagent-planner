# Multi-Agent Planner

A small, dependency-free Python prototype that turns a short description of a busy week into a seven-day plan.

The goal of this branch is deliberately modest: demonstrate a clear multi-agent workflow that anyone can clone, run, read, and test in a few minutes. The agents use transparent Python rules, so the demo does not require an API key, a local language model, or trained model weights.

## Demo

Run the included example:

```bash
python app.py
```

Or provide your own request:

```bash
python app.py "Lacrosse match on Wednesday, training on Tuesday, gym 3 times and 2 evenings to study"
```

For machine-readable output:

```bash
python app.py "Gym 2 times and sleep 8 hours" --json
```

Python 3.9 or newer is recommended. There are no third-party runtime dependencies.

## Agent pipeline

The agents run in a simple sequence:

1. **PlannerAgent** extracts supported commitments and constraints.
2. **FitnessAgent** preserves fixed sports events and places gym sessions.
3. **NutritionAgent** adds a conservative match-day fuel reminder.
4. **CriticAgent** calculates a transparent workload score.
5. **WriterAgent** merges everything into a readable weekly plan.

`AgentManager` owns the handoff between agents. Each agent has one small responsibility and can be tested independently.

## Supported phrases

This basic parser intentionally supports a narrow set of phrases:

- `match on Wednesday`
- `training on Tuesday`
- `3 evenings to study`
- `coursework deadline on Friday`
- `gym 4 times`
- `sleep at least 7 hours`

It is a prototype, not a general natural-language planner. Keeping the rules explicit makes the behaviour repeatable and the design easy to explain.

## Tests

The test suite uses Python's standard library:

```bash
python -m unittest discover -s tests -v
```

It covers request parsing, the end-to-end agent pipeline, event placement, workload output, and JSON CLI output.

## Project structure

```text
.
├── agents/          # Five focused planning agents
├── core/            # Shared data models and orchestration
├── tests/           # Small unittest suite
├── app.py           # Command-line entry point
└── README.md
```

## Sensible next steps

- Add richer phrase extraction behind the same `PlannerAgent` interface.
- Let users provide unavailable times and preferred study days.
- Replace the heuristic parser with an optional LLM adapter.
- Add persistence or a small web interface only after the core behaviour is stable.
