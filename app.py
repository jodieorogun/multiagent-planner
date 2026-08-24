import argparse
import json

from core import AgentManager


DEFAULT_REQUEST = (
    "Plan my week: I have a lacrosse match on Wednesday, training on Tuesday, "
    "3 evenings to study, a coursework deadline on Friday, I want to gym 4 "
    "times and sleep at least 7 hours."
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create a basic weekly plan using a deterministic agent pipeline."
    )
    parser.add_argument(
        "request",
        nargs="?",
        default=DEFAULT_REQUEST,
        help="A short description of your weekly commitments.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of the formatted plan.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    plan = AgentManager().process(args.request)

    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
    else:
        print(plan.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
