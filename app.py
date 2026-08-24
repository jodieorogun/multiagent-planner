import argparse
import json

from core import AgentManager, LLMError, OllamaClient, SchemaError
from tools.workload_model import WorkloadModelError, neural_predict


DEFAULT_REQUEST = (
    "Plan my week: I have a lacrosse match on Wednesday, training on Tuesday "
    "evening, 3 evenings to study, a coursework deadline on Friday, I want to "
    "gym 4 times and still sleep at least 7 hours."
)


def build_parser():
    parser = argparse.ArgumentParser(description="Create a multi-agent weekly plan.")
    parser.add_argument("request", nargs="?", default=DEFAULT_REQUEST)
    parser.add_argument("--model", help="Override the OLLAMA_MODEL setting.")
    parser.add_argument("--timeout", type=int, help="Ollama timeout in seconds.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument(
        "--debug", action="store_true", help="Print raw Ollama responses to stderr."
    )
    parser.add_argument(
        "--neural-workload",
        action="store_true",
        help="Use the optional trained PyTorch workload model.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    client = OllamaClient(
        model=args.model, timeout_seconds=args.timeout, debug=args.debug
    )
    predictor = neural_predict if args.neural_workload else None
    manager = AgentManager(client.complete, workload_predictor=predictor)

    try:
        plan = manager.process(args.request)
    except (LLMError, SchemaError, WorkloadModelError) as exc:
        parser.exit(1, f"error: {exc}\n")

    print(json.dumps(plan.to_dict(), indent=2) if args.json else plan.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
