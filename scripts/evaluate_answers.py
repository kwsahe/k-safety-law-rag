"""Run regression checks for the K-Safety Law RAG chatbot.

This is a lightweight quality gate around the scenarios that previously
regressed: scaffold special education, PPE/scaffold standards, employer
liability, serious-accident penalties, prime-contractor responsibility, and
general education questions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DEFAULT_CASE_FILE = ROOT_DIR / "test" / "evaluation_cases.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "evaluation_reports"


def _contains_all(answer: str, expected: list[str]) -> list[str]:
    return [item for item in expected if item not in answer]


def _contains_any(answer: str, forbidden: list[str]) -> list[str]:
    return [item for item in forbidden if item and item in answer]


def _load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    from rag.chatbot import reset_chat_runtime_state, rag_chat
    from rag.report_payload import build_report_pages
    from rag.schemas import AccidentScenario, ChatRequest

    scenario = None
    if case.get("scenario"):
        scenario = AccidentScenario(**case["scenario"])

    reset_chat_runtime_state(clear_scenario_value=True)
    started = time.time()
    response = rag_chat(
        ChatRequest(
            question=case["question"],
            scenario=scenario,
            mode=str(case.get("mode") or ""),
        )
    )
    elapsed_ms = int((time.time() - started) * 1000)

    missing = _contains_all(response.answer, case.get("expected_contains", []))
    forbidden_hits = _contains_any(response.answer, case.get("forbidden_contains", []))
    max_elapsed_ms = int(case.get("max_elapsed_ms") or 0)
    too_slow = bool(max_elapsed_ms and elapsed_ms > max_elapsed_ms)
    citation_check = response.citation_check or {}
    citation_failed = citation_check.get("status") == "fail"
    report_pages = build_report_pages(response.answer, response.sources) if case.get("mode") == "scenario" else []

    passed = not missing and not forbidden_hits and not too_slow and not citation_failed
    return {
        "id": case["id"],
        "passed": passed,
        "elapsed_ms": elapsed_ms,
        "missing_expected": missing,
        "forbidden_hits": forbidden_hits,
        "too_slow": too_slow,
        "citation_status": citation_check.get("status", ""),
        "citation_summary": citation_check.get("summary", ""),
        "citation_warnings": citation_check.get("warnings", []),
        "missing_required_citations": citation_check.get("missing_required", []),
        "answer": response.answer,
        "source_count": len(response.sources),
        "graph_trace": response.graph_trace or {},
        "report_pages": report_pages,
    }


def _print_summary(results: list[dict[str, Any]]) -> None:
    width = max(len(result["id"]) for result in results) if results else 8
    print("\nEvaluation results")
    print("-" * (width + 48))
    for result in results:
        mark = "PASS" if result["passed"] else "FAIL"
        print(
            f"{mark:4}  {result['id']:<{width}}  "
            f"{result['elapsed_ms']:>6}ms  citation={result['citation_status'] or '-'}"
        )
        if result["missing_expected"]:
            print(f"      missing: {', '.join(result['missing_expected'])}")
        if result["forbidden_hits"]:
            print(f"      forbidden: {', '.join(result['forbidden_hits'])}")
        if result["citation_warnings"]:
            print(f"      citation warnings: {' | '.join(result['citation_warnings'])}")
        if result["missing_required_citations"]:
            labels = [item.get("label", "") for item in result["missing_required_citations"]]
            print(f"      missing required citations: {', '.join(labels)}")
    passed = sum(1 for result in results if result["passed"])
    print("-" * (width + 48))
    print(f"Passed {passed}/{len(results)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run chatbot answer quality checks.")
    parser.add_argument("--case-file", type=Path, default=DEFAULT_CASE_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-id", action="append", help="Run only the selected case id. Can be repeated.")
    parser.add_argument("--no-report", action="store_true", help="Do not write a JSON report file.")
    args = parser.parse_args()

    cases = _load_cases(args.case_file)
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case["id"] in selected]
    if not cases:
        print("No evaluation cases selected.")
        return 2

    try:
        results = [_run_case(case) for case in cases]
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        print(f"Missing dependency: {missing}")
        print("Install project dependencies first, for example: python -m pip install -r requirements.txt")
        return 2
    _print_summary(results)

    if not args.no_report:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report: {report_path}")

    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
