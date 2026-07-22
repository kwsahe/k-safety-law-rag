"""Run the v1 deployment quality gate across five incident categories."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate_answers import DEFAULT_CASE_FILE, _load_cases, _run_case


CATEGORY_CASES = {
    "fall_scaffold": [
        "q1a_scaffold_special_education",
        "q1b_ppe_scaffold_installation",
        "q2b_serious_accident_manager",
        "q2c_prime_contractor",
        "q3_final_report",
    ],
    "machine_entanglement": [
        "machine_q1_special_education",
        "machine_q2_controls",
        "machine_q3_death_penalty",
        "machine_q4_prime_contractor",
        "machine_q5_final_report",
    ],
    "struck_by": [
        "struck_q1_special_education",
        "struck_q2_controls",
        "struck_q3_injury_penalty",
        "struck_q4_prime_contractor",
        "struck_q5_final_report",
    ],
    "masonry_falling": [
        "masonry_q1_special_education_scope",
        "masonry_q2_falling_controls",
        "masonry_q3_death_penalty",
        "masonry_q4_prime_contractor",
        "masonry_q5_final_report",
    ],
    "collapse_excavation": [
        "collapse_q1_excavation_education",
        "collapse_q2_excavation_controls",
        "collapse_q3_injury_penalty",
        "collapse_q4_prime_contractor",
        "collapse_q5_final_report",
    ],
    "fire_welding": [
        "q1_welding_special_education",
        "q2_hot_work_controls",
        "q3_hot_work_executive_liability",
        "q4_hot_work_prime_contractor",
        "q5_hot_work_final_report",
    ],
}

REGRESSION_CASES = [
    "machine_q3_single_injury_not_applicable",
    "machine_q4_inspection_confirmation",
    "machine_q5_direct_single_injury_report",
    "machine_q2_controls_and_inspection",
]

GENERAL_LAW_CASES = [
    "general_q1_osha_purpose",
    "general_q2_employer_worker_duties",
    "general_q3_manager_roles",
    "general_q4_regular_education",
    "general_q5_risk_assessment",
]


def _page_contract_ok(result: dict[str, Any]) -> bool:
    pages = result.get("report_pages") or []
    return [page.get("page") for page in pages] == [7, 12]


def evaluate_gate(case_file: Path) -> dict[str, Any]:
    cases_by_id = {case["id"]: case for case in _load_cases(case_file)}
    missing_cases = [
        case_id
        for case_ids in [*CATEGORY_CASES.values(), REGRESSION_CASES, GENERAL_LAW_CASES]
        for case_id in case_ids
        if case_id not in cases_by_id
    ]
    if missing_cases:
        raise ValueError(f"Missing deployment cases: {', '.join(missing_cases)}")

    results: list[dict[str, Any]] = []
    categories: dict[str, dict[str, Any]] = {}
    for category, case_ids in CATEGORY_CASES.items():
        category_results = [_run_case(cases_by_id[case_id]) for case_id in case_ids]
        results.extend(category_results)
        passed = sum(1 for result in category_results if result["passed"])
        score = round(passed * 100 / len(category_results), 1)
        categories[category] = {
            "score": score,
            "passed": passed,
            "total": len(category_results),
            "gate_passed": score >= 90,
        }

    regression_results = [_run_case(cases_by_id[case_id]) for case_id in REGRESSION_CASES]
    results.extend(regression_results)
    machine_regressions_ok = all(result["passed"] for result in regression_results)
    general_law_results = [_run_case(cases_by_id[case_id]) for case_id in GENERAL_LAW_CASES]
    results.extend(general_law_results)
    general_law_regressions_ok = all(result["passed"] for result in general_law_results)

    scenario_results = [result for result in results if result["id"] not in set(GENERAL_LAW_CASES)]
    all_pages_ok = all(_page_contract_ok(result) for result in scenario_results)
    all_citations_ok = all(result.get("citation_status") != "fail" for result in results)
    all_latency_ok = all(not result.get("too_slow") for result in results)
    all_cases_passed = all(result.get("passed") for result in results)
    category_scores_ok = all(item["gate_passed"] for item in categories.values())
    cache_keys = [str((result.get("graph_trace") or {}).get("cache_key") or "") for result in results]
    cache_isolation_ok = all(cache_keys) and len(cache_keys) == len(set(cache_keys))
    final_report_ids = {case_ids[-1] for case_ids in CATEGORY_CASES.values()}
    final_reports = [result for result in results if result["id"] in final_report_ids]
    worker_subject_ok = all(
        "처벌 주체는 사업주" in result["answer"] and "근로자 개인 처벌로 바로 귀결하지" in result["answer"]
        for result in final_reports
    )

    holdout_case = {
        "id": "v1_holdout_new_prime_contractor",
        "mode": "scenario",
        "question": "이동식 크레인 인양 작업을 수급업체 G사에 도급한 상황에서 원청 (주)새빛물류의 책임이 성립하는가? 두 법령의 근거와 책임 범위 차이를 제시하라.",
        "scenario": {
            "overview": "(주)새빛물류 사업장에서 수급업체 G사의 이동식 크레인 인양물이 낙하해 근로자 1명이 다쳤다.",
            "details": "원청이 작업장소와 크레인 동선을 관리했으며 수급업체 안전보건 역량 평가 이력은 없다.",
            "workers": "G사 근로자 1명이 부상을 입었다.",
        },
        "expected_contains": ["(주)새빛물류", "G사", "산업안전보건법 제64조", "중대재해처벌법 제5조", "시행령 제4조제9호"],
        "forbidden_contains": ["(주)스파일럿건설", "(주)대한정밀", "(주)동해플랜트", "B사", "E사", "F사"],
        "max_elapsed_ms": 60000,
    }
    holdout_result = _run_case(holdout_case)
    holdout_ok = holdout_result["passed"] and _page_contract_ok(holdout_result)

    gates = {
        "all_category_cases_passed": all_cases_passed,
        "category_scores_at_least_90": category_scores_ok,
        "citation_validation": all_citations_ok,
        "latency_under_60_seconds": all_latency_ok,
        "report_page_7_and_12_contract": all_pages_ok,
        "scenario_cache_isolation": cache_isolation_ok,
        "worker_responsibility_separation": worker_subject_ok,
        "unseen_scenario_validation": holdout_ok,
        "machine_single_injury_and_inspection_regressions": machine_regressions_ok,
        "general_law_accuracy_regressions": general_law_regressions_ok,
    }
    return {
        "passed": all(gates.values()),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "categories": categories,
        "gates": gates,
        "results": results,
        "regression_results": regression_results,
        "general_law_results": general_law_results,
        "holdout_result": holdout_result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the K-Safety RAG v1 deployment gate.")
    parser.add_argument("--case-file", type=Path, default=DEFAULT_CASE_FILE)
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "data" / "evaluation_reports")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    report = evaluate_gate(args.case_file)
    for category, item in report["categories"].items():
        mark = "PASS" if item["gate_passed"] else "FAIL"
        print(f"{mark:4} {category:<24} {item['score']:>5.1f} ({item['passed']}/{item['total']})")
    print("\nDeployment gates")
    for name, passed in report["gates"].items():
        print(f"{'PASS' if passed else 'FAIL':4} {name}")

    if not args.no_report:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / f"v1_deployment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nReport: {path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
