# Multi-Agent Planner

A small multi-agent weekly-planning prototype built around a local Ollama model. It keeps the original planner, fitness, nutrition, critic, writer, and PyTorch workload-model ideas while making their handoffs explicit and testable.

## What the prototype demonstrates

1. **PlannerAgent** asks Ollama to turn a free-form request into a validated planning brief.
2. **FitnessAgent** deterministically preserves sports commitments and places gym sessions.
3. **NutritionAgent** adds simple match-day fuel guidance.
4. **CriticAgent** estimates workload with either a transparent heuristic or the optional trained PyTorch model.
5. **WriterAgent** asks Ollama to phrase the final week, then validates that fixed commitments stayed on the correct days.

The LLM is used where language understanding and writing are useful. Scheduling rules and validation stay in Python so the result is predictable.

## Quick start

Requirements:

- Python 3.9 or newer
- [Ollama](https://ollama.com/) installed and running
- The `qwen2.5:3b` model

```bash
ollama pull qwen2.5:3b
python app.py
```

Pass a different request directly:

```bash
python app.py "Plan my week with training on Tuesday, a match on Saturday, 3 study sessions and 2 gym sessions"
```

Use JSON output when integrating with another interface:

```bash
python app.py --json
```

The default prototype has no third-party Python dependency. Set `OLLAMA_MODEL` or pass `--model` to use a different local model. Set `OLLAMA_TIMEOUT_SECONDS` or pass `--timeout` to change the response timeout.

Use `--debug` to print the raw Ollama responses to stderr when adjusting prompts or trying another model.

## Optional neural workload model

The cleaned prototype uses the original workload formula as a reliable default heuristic. The PyTorch experiment remains available:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-ml.txt
python tools/train_workload_model.py
python app.py --neural-workload
```

Training creates `saved_models/workload_net.pth`, which is intentionally ignored by Git.

## Tests

The tests replace Ollama with a scripted fake, so they are fast and work offline:

```bash
python -m unittest discover -s tests -v
```

The suite covers JSON extraction, the planner contract, the full five-agent pipeline, workload prediction, fixed-event validation, and malformed writer-output fallback.

## Small live eval

The evaluation set runs three representative requests through the real Ollama pipeline and checks fixed-day events, gym and study counts, and workload labels:

```bash
python -m evals.run_eval
```

Use `--json` for structured results or `--model NAME` to evaluate another installed Ollama model. The command exits with a non-zero status if any scenario fails, making it suitable for a lightweight local quality gate.

## Project structure

```text
.
├── agents/                    # Planner, fitness, nutrition, critic and writer
├── core/
│   ├── llm.py                 # Ollama client and strict JSON extraction
│   ├── manager.py             # Explicit agent orchestration
│   └── models.py              # Shared response contracts
├── tools/
│   ├── workload_model.py      # Heuristic and optional neural prediction
│   └── train_workload_model.py
├── tests/                     # Offline unit and pipeline tests
├── evals/                     # Live Ollama scenarios and scorer
├── app.py                     # Command-line entry point
├── requirements.txt           # No base Python packages required
└── requirements-ml.txt        # Optional PyTorch dependency
```

## Intentional boundaries

- This is a working prototype, not a calendar application.
- Ollama must be available for request parsing; failures produce clear errors.
- The nutrition guidance is deliberately simple and is not medical advice.
- The neural workload model is trained on synthetic data and should be treated as a demonstration, not a validated risk assessment.
