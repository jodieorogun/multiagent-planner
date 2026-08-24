import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from core import AgentManager, LLMError, OllamaClient, SchemaError


DEFAULT_CASES_PATH = Path(__file__).with_name("cases.json")


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class CaseResult:
    name: str
    passed: bool
    checks: List[CheckResult]
    writer_mode: str
    error: str = ""


def score_plan(plan, expectations: Dict) -> List[CheckResult]:
    checks = []

    for event in expectations.get("fixed_events", []):
        day = event["day"]
        text = event["text"]
        summary = plan.weekly_plan.get(day, "")
        passed = text.lower() in summary.lower()
        checks.append(
            CheckResult(
                name=f"{text} remains on {day}",
                passed=passed,
                detail=summary,
            )
        )

    combined_plan = "\n".join(plan.weekly_plan.values()).lower()
    for text, expected_count in expectations.get("exact_counts", {}).items():
        actual_count = combined_plan.count(text.lower())
        checks.append(
            CheckResult(
                name=f"{text} count is {expected_count}",
                passed=actual_count == expected_count,
                detail=f"actual={actual_count}",
            )
        )

    expected_workload = expectations.get("workload")
    if expected_workload:
        checks.append(
            CheckResult(
                name=f"workload is {expected_workload}",
                passed=expected_workload.lower() in plan.stress_note.lower(),
                detail=plan.stress_note,
            )
        )
    return checks


def run_cases(cases, manager: AgentManager) -> List[CaseResult]:
    results = []
    for case in cases:
        try:
            plan = manager.process(case["request"])
            checks = score_plan(plan, case["expectations"])
            results.append(
                CaseResult(
                    name=case["name"],
                    passed=all(check.passed for check in checks),
                    checks=checks,
                    writer_mode=plan.writer_mode,
                )
            )
        except (LLMError, SchemaError) as exc:
            results.append(
                CaseResult(
                    name=case["name"],
                    passed=False,
                    checks=[],
                    writer_mode="error",
                    error=str(exc),
                )
            )
    return results


def load_cases(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def print_report(results: List[CaseResult]):
    passed_count = sum(result.passed for result in results)
    for result in results:
        marker = "PASS" if result.passed else "FAIL"
        print(f"[{marker}] {result.name} (writer={result.writer_mode})")
        if result.error:
            print(f"  - ERROR: {result.error}")
        for check in result.checks:
            check_marker = "PASS" if check.passed else "FAIL"
            print(f"  - {check_marker}: {check.name} [{check.detail}]")
    print(f"\nScore: {passed_count}/{len(results)} cases passed")


def build_parser():
    parser = argparse.ArgumentParser(description="Run the live planner evaluation set.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--model", help="Override the OLLAMA_MODEL setting.")
    parser.add_argument("--timeout", type=int, help="Ollama timeout in seconds.")
    parser.add_argument("--json", action="store_true", help="Print JSON results.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    client = OllamaClient(model=args.model, timeout_seconds=args.timeout)
    manager = AgentManager(client.complete)
    results = run_cases(load_cases(args.cases), manager)

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        print_report(results)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
