# Multi-Agent Planner

Multi-Agent Planner is a local Python prototype that turns a natural-language description of a busy week into a structured weekly plan. It passes the request through a sequence of specialist agents, uses a small PyTorch model to estimate workload risk, and asks a local Ollama model to produce the final schedule.

## How it works

The pipeline runs five agents in order:

1. **PlannerAgent** extracts academic tasks, sports commitments, workout goals, and constraints from the request.
2. **FitnessAgent** places fixed sports events and gym sessions across the week.
3. **NutritionAgent** adds a simple meal plan and extra fuel guidance for match days.
4. **CriticAgent** converts the extracted commitments into five workload features and calls the PyTorch workload classifier.
5. **WriterAgent** combines the earlier outputs into a seven-day plan with a short workload note.

`AgentManager` coordinates the pipeline and keeps the outputs from earlier agents available as context for later ones.

## Project structure

```text
.
├── agents/                   # Planner, fitness, nutrition, critic, and writer agents
├── core/
│   ├── agentManager.py       # Agent orchestration and tool-call handling
│   └── llm.py                # Local Ollama integration and JSON parsing
├── tools/
│   ├── trainWorkloadModel.py # Synthetic-data training script
│   └── workloadModel.py      # Model loading and workload prediction
├── app.py                    # End-to-end example
├── testPlanner.py            # PlannerAgent smoke-test script
└── requirements.txt
```

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/) installed and running
- The `qwen2.5-coder:1.5b` Ollama model

## Setup

Create and activate a virtual environment, then install the pinned dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Pull the local language model used by `core/llm.py`:

```bash
ollama pull qwen2.5-coder:1.5b
```

Train the workload classifier once before running the full application:

```bash
python tools/trainWorkloadModel.py
```

This creates `saved_models/workloadNet.pth`. The generated model is intentionally ignored by Git.

## Run the planner

```bash
python app.py
```

The example request is currently defined in `app.py`. Edit `userRequest` there to plan a different week. The application logs each agent's input and output, followed by the final weekly plan.

To exercise only the request-parsing stage, run:

```bash
python testPlanner.py
```

## Workload model

The classifier predicts one of four workload levels:

| Value | Meaning |
| --- | --- |
| `0` | Light workload |
| `1` | Moderate workload |
| `2` | High workload |
| `3` | Burnout risk |

It uses five inputs: study hours, sport hours, work hours, number of deadlines, and sleep hours. The training script generates synthetic examples from a hand-written scoring rule, so the result is a demonstration model rather than a clinically validated assessment.

## Current limitations

- The planner runs a fixed example request; there is not yet a command-line or web interface.
- Agent execution is sequential, despite the multi-agent design.
- Several workload estimates are heuristic, and the classifier is trained on synthetic data.
- Output quality depends on the local model consistently returning the requested JSON shape.
- The Ollama model name and the 20-second response timeout are currently hard-coded in `core/llm.py`.
- There is no automated test suite yet; `testPlanner.py` is a manual smoke test.

## Customisation

- Change `MODEL_NAME` in `core/llm.py` to use another Ollama model.
- Adjust agent prompts and scheduling rules in `agents/`.
- Retrain or replace the workload classifier via `tools/trainWorkloadModel.py`.
- Register additional callable tools in `AgentManager.toolRegistry`.
