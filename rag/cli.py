"""Interactive integrated RAG chatbot."""

from __future__ import annotations

import argparse
import json
import re
import runpy
import sys
import time
from datetime import datetime
from typing import Any
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.chatbot import (  # noqa: E402
    clear_scenario,
    get_scenario,
    rag_chat_stream,
    reset_chat_runtime_state,
    set_scenario,
)
from rag.config import LLM_PROVIDER  # noqa: E402
from rag.schemas import AccidentScenario, ChatRequest, SourceDoc  # noqa: E402

LAW_REFERENCES_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "test_report"
MAX_LAW_REFERENCES = 3
DEFAULT_SCENARIO_PATH = Path(__file__).parent.parent / "scenarios" / "default_accident.py"


def print_separator(char: str = "-", width: int = 70) -> None:
    print(char * width)


def display_page(metadata: dict[str, Any]) -> str:
    page = str(metadata.get("citation_page") or metadata.get("page") or "").strip()
    return page if page and page != "0" else "페이지 정보 없음"


def print_sources(sources) -> None:
    print("\n[참고 근거]")
    for index, doc in enumerate(sources, start=1):
        metadata = doc.metadata
        source_type = metadata.get("source_type", "")
        law_name = metadata.get("law_name", "")
        article = metadata.get("article", "")
        page = display_page(metadata)
        score = metadata.get("score", 0.0)
        label = f"{law_name} {article}".strip()
        print(f"  {index}. [{source_type}] {label} {page} score={score}")


def _clip_text(value: Any, max_length: int) -> str:
    text = "" if value is None else str(value)
    return text[:max_length]


def _normalize_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _short_law_name(law_name: Any) -> str:
    text = "" if law_name is None else str(law_name)
    if "산업안전보건법" in text:
        return "산안법"
    if "중대재해처벌법" in text or "중대재해 처벌" in text:
        return "중처법"
    return text[:8] if text else "법령"


def _full_law_name(law_name: Any) -> str:
    text = "" if law_name is None else str(law_name).strip()
    if not text:
        return "법령"
    return text.replace("_", " ").replace(" (건설현장 한정)", "")


def _clean_display_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip(" -:ㆍ\t")
    return _clip_text(cleaned, 80)


def _infer_violation_summary(source: SourceDoc) -> str:
    metadata = source.metadata
    content = re.sub(r"\s+", "", source.content)
    article = str(metadata.get("article") or "")
    law_name = str(metadata.get("law_name") or "")

    if "산업안전보건기준에관한규칙" in re.sub(r"\s+", "", law_name) and article == "제14조":
        return "낙하물 위험구역 출입통제 및 RED ZONE 관리 미흡"
    if "비계" in content and ("교육내용" in content or "별표5" in content):
        return "비계 작업 특별안전교육 미실시"
    if any(term in content for term in ("추락재해", "안전대", "안전고리", "안전모", "보호구")):
        return "추락방지 조치 및 보호구 착용 관리 미흡"
    if "특별안전교육" in content or "안전보건교육을추가" in content or "별표5" in content:
        return "특별안전교육 미실시"
    if "안전보건관리체계" in content or ("중대재해처벌법" in law_name and article == "제4조"):
        return "안전보건관리체계 구축ㆍ이행 미흡"
    if "위험성평가" in content:
        return "작업 전 위험성평가 미실시"
    if "출입금지" in content or "금지표지" in content:
        return "출입금지 표지 및 출입통제 미흡"
    if "도급" in content or "실질적으로지배" in content:
        return "도급 작업 안전보건 확보의무 미흡"
    if "크레인" in content and any(term in content for term in ("인양", "낙하", "신호", "안전검사")):
        return "크레인 작업 안전조치 미흡"
    if "안전난간" in content:
        return "안전난간 설치 의무 위반"
    if article == "제38조":
        return "작업 전 안전조치 미실시"
    if article == "제8조":
        return "경영책임자 안전보건교육 이수의무 위반"
    if article == "제6조":
        return "중대산업재해 처벌 기준"
    return "법령상 안전보건 의무 위반"


def _extract_item_number_from_source(source: SourceDoc) -> str:
    compact = re.sub(r"\s+", "", source.content)
    if "비계의조립·해체또는변경작업" in compact:
        return "23"
    metadata = source.metadata
    item_number = str(metadata.get("item_number") or "")
    match = re.match(r"\s*(\d+)\.", item_number)
    if match:
        return match.group(1)
    match = re.search(r"\[작업항목\]\s*(\d+)\.", source.content)
    return match.group(1) if match else ""


def _law_item_from_source(source: SourceDoc) -> str:
    metadata = source.metadata
    law_name = _full_law_name(metadata.get("law_name"))
    article = _clip_text(metadata.get("article"), 30).strip()
    annex = _clip_text(metadata.get("annex"), 30).strip()
    item_no = _extract_item_number_from_source(source)

    if article:
        return f"{law_name} {article}".strip()
    if annex:
        return f"{law_name} {annex}".strip()
    if item_no:
        return f"{law_name} 별표 5 제{item_no}호".strip()
    return law_name


def _report_source_priority(source: SourceDoc) -> int:
    metadata = source.metadata
    law_name = str(metadata.get("law_name") or "")
    article = str(metadata.get("article") or "")
    annex = str(metadata.get("annex") or "")
    item_no = _extract_item_number_from_source(source)
    summary = _infer_violation_summary(source)

    if "중대재해처벌법" in law_name and article in {"제2조", "제3조"}:
        return -10
    if item_no:
        return 100
    if annex:
        return 95
    if "산업안전보건기준에 관한 규칙" in law_name and article == "제14조":
        return 98
    if article in {"제38조", "제36조", "제64조", "제62조"}:
        return 90
    if "중대재해처벌법" in law_name and article in {"제4조", "제5조", "제6조제1항", "제6조", "제7조", "제15조"}:
        return 85
    if summary != "법령상 안전보건 의무 위반":
        return 70
    return 10


def _select_report_violation_sources(sources: list[SourceDoc]) -> list[SourceDoc]:
    ranked = [
        source
        for source in sources
        if _report_source_priority(source) > 0
    ]
    ranked.sort(
        key=lambda source: (
            _report_source_priority(source),
            float(source.metadata.get("score") or 0.0),
        ),
        reverse=True,
    )
    selected: list[SourceDoc] = []
    seen_law_items: set[str] = set()
    for source in ranked:
        law_item = _law_item_from_source(source)
        if law_item in seen_law_items:
            continue
        selected.append(source)
        seen_law_items.add(law_item)
        if len(selected) >= MAX_LAW_REFERENCES:
            break
    if len(selected) < MAX_LAW_REFERENCES:
        for source in sources:
            if _report_source_priority(source) <= 0:
                continue
            law_item = _law_item_from_source(source)
            if law_item in seen_law_items:
                continue
            selected.append(source)
            seen_law_items.add(law_item)
            if len(selected) >= MAX_LAW_REFERENCES:
                break
    return selected


def _extract_violation_candidates_from_answer(answer: str) -> list[str]:
    """Extract concise violation phrases from deterministic answer text when available."""
    lines = [line.strip() for line in answer.splitlines()]
    candidates: list[str] = []
    in_violation_block = False

    for line in lines:
        if line.startswith("[") and "위반 조항" in line:
            in_violation_block = True
            continue
        if in_violation_block and line.startswith("[") and "위반 조항" not in line:
            break
        if not in_violation_block:
            continue
        if not line.startswith(("-", "①", "②", "③", "1.", "2.", "3.")):
            continue
        if "근거" in line and "위반" not in line:
            continue
        candidates.append(_clean_display_text(re.sub(r"^[①②③]|\d+\.\s*|-\s*", "", line)))
        if len(candidates) >= MAX_LAW_REFERENCES:
            break

    return candidates


def _answer_contains_any(answer: str, terms: tuple[str, ...]) -> bool:
    compact = re.sub(r"\s+", "", answer)
    return any(term in compact for term in terms)


def _build_violation_judgment(answer: str, items: list[dict[str, str]]) -> dict[str, Any]:
    is_violation_likely = bool(items) or _answer_contains_any(answer, ("위반여부:YES", "위반가능성", "의무위반"))
    return {
        "is_violation_likely": is_violation_likely,
        "basis_count": len(items),
        "summary": (
            "검색된 법령 근거와 사고 사실관계상 안전보건 의무 위반 가능성이 있습니다."
            if is_violation_likely
            else "현재 검색 근거만으로 명확한 법령 위반을 단정하기 어렵습니다."
        ),
    }


def _build_responsibility_judgment(answer: str, sources: list[SourceDoc]) -> dict[str, Any]:
    has_serious = any("중대재해처벌법" in str(source.metadata.get("law_name", "")) for source in sources)
    has_osha = any("산업안전보건법" in str(source.metadata.get("law_name", "")) for source in sources)
    has_red_zone_actor = _answer_contains_any(answer, ("B씨", "REDZONE무단진입", "비계무단이동", "직접원인"))
    c_is_victim = _answer_contains_any(answer, ("C씨는비계위에서고정작업중추락한피해자", "C씨는피해자", "C씨에게사고책임은부여하지"))
    worker_fault = False if c_is_victim else _answer_contains_any(answer, ("근로자과실", "과실참작", "출입금지", "무시", "미착용"))
    employer_exemption = _answer_contains_any(answer, ("자동면책", "면책되지", "면제되지", "면책아님"))

    employer_reasons = []
    if has_osha:
        employer_reasons.append("산업안전보건법상 현장 안전조치ㆍ교육ㆍ관리감독 의무 검토 필요")
    if has_serious:
        employer_reasons.append("중대재해처벌법상 경영책임자의 안전보건관리체계 구축ㆍ이행 의무 검토 필요")
    if _answer_contains_any(answer, ("도급", "하청", "원청", "수급")):
        employer_reasons.append("도급 작업에 대한 실질적 지배ㆍ운영ㆍ관리 여부 검토 필요")
    if not employer_reasons:
        employer_reasons.append("사고 사실관계와 검색 근거를 기준으로 사업주 책임 여부 추가 검토 필요")

    payload = {
        "employer": {
            "responsibility_likely": True,
            "reasons": employer_reasons,
        },
        "worker": {
            "fault_considered": worker_fault,
            "employer_exemption": False if employer_exemption or worker_fault or c_is_victim else None,
            "reason": (
                "C씨는 비계 위 고정 작업 중 추락한 피해자로, 현재 시나리오 기준 사고 책임 없음으로 봅니다."
                if c_is_victim
                else
                "근로자 과실은 참작될 수 있으나 사업주의 법정 안전보건 의무를 자동 면제하지 않습니다."
                if worker_fault
                else "근로자 과실 여부는 사고 경위와 교육ㆍ감독 이행 여부를 함께 보아야 합니다."
            ),
        },
    }
    if has_red_zone_actor:
        payload["direct_cause_actor"] = {
            "actor": "B씨",
            "role": "사고 가해자",
            "act": "CCTV RED ZONE 무단 진입 및 비계 무단 이동",
            "causation": "비계 전도와 C씨 추락의 직접 원인",
        }
        payload["victim"] = {
            "actor": "C씨",
            "role": "피해자",
            "responsibility": "책임 없음",
        }
    return payload


def _build_law_applicability(sources: list[SourceDoc], answer: str) -> dict[str, Any]:
    has_osha = any("산업안전보건법" in str(source.metadata.get("law_name", "")) for source in sources)
    has_serious = any("중대재해처벌법" in str(source.metadata.get("law_name", "")) for source in sources)
    serious_applies = has_serious and not _answer_contains_any(answer, ("중대재해처벌법:미적용", "중대재해처벌법상미적용", "적용되지않"))
    return {
        "occupational_safety_health_act": {
            "checked": True,
            "applicable": has_osha,
            "result": "적용 검토 대상" if has_osha else "관련 근거 없음",
        },
        "serious_accident_punishment_act": {
            "checked": True,
            "applicable": serious_applies,
            "result": "적용 검토 대상" if serious_applies else "미적용 또는 추가 사실 확인 필요",
        },
    }


def _build_final_evaluation(answer: str, legal_items: list[dict[str, str]], responsibility: dict[str, Any]) -> dict[str, Any]:
    key_findings = [f"{item['law_item']} - {item['violation']}" for item in legal_items]
    if responsibility["worker"]["fault_considered"]:
        key_findings.append("근로자 과실은 참작 가능하나 사업주 책임 자동 면책 사유는 아님")
    if not key_findings:
        key_findings.append("검색 근거와 사고 사실관계의 추가 확인 필요")

    risk_level = "높음" if len(legal_items) >= 2 else "보통" if legal_items else "확인 필요"
    return {
        "overall_risk_level": risk_level,
        "summary": (
            "사고 사실관계와 RAG 법령 근거를 종합하면 위반 여부와 책임 판단 모두 추가 조치가 필요한 상태입니다."
            if legal_items
            else "현재 근거만으로 최종 법령 평가를 단정하기 어렵습니다."
        ),
        "key_findings": key_findings[:5],
        "recommended_followups": [
            {"category": "증거확인", "action": "영상ㆍ사진ㆍ현장조사 결과와 RAG 법령 판단 근거 대조"},
            {"category": "위반검토", "action": "산업안전보건법과 중대재해처벌법 적용 여부를 각각 확정"},
            {"category": "책임판단", "action": "사업주ㆍ도급인ㆍ근로자 과실 사유를 분리해 보고서에 반영"},
        ],
    }


class LawReferenceWriter:
    """Build and temporarily store top law_references records."""

    def __init__(
        self,
        report_id: int,
        section: str = "S07",
        output_dir: Path = LAW_REFERENCES_OUTPUT_DIR,
    ) -> None:
        self.report_id = report_id
        self.section = _clip_text(section, 20)
        self.output_dir = output_dir

    def build_page_payload(
        self,
        sources: list[SourceDoc],
        answer: str = "",
        page: int = 7,
    ) -> list[dict[str, Any]]:
        """Build compact page-separated report data.

        Final JSON shape:
        [
          {
            "page": 7,
            "data": {
              "legal_violations": {
                "title": "법적 위반 사항 확인 (3건)",
                "items": [
                  {
                    "law_item": "산안법 제38조",
                    "violation": "작업 전 안전조치 미실시",
                    "marker": "①"
                  }
                ]
              }
            }
          }
        ]
        """
        answer_candidates = _extract_violation_candidates_from_answer(answer)
        items = []
        circled_numbers = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨"]
        report_sources = _select_report_violation_sources(sources)
        for index, source in enumerate(report_sources, start=1):
            summary = answer_candidates[index - 1] if index <= len(answer_candidates) else _infer_violation_summary(source)
            marker = circled_numbers[index - 1] if index <= len(circled_numbers) else f"{index})"
            law_item = _law_item_from_source(source)
            items.append(
                {
                    "marker": marker,
                    "law_item": law_item,
                    "violation": summary,
                }
            )

        title = f"법적 위반 사항 확인 ({len(items)}건)"
        violation_judgment = _build_violation_judgment(answer, items)
        responsibility_judgment = _build_responsibility_judgment(answer, sources)
        law_applicability = _build_law_applicability(sources, answer)
        final_evaluation = _build_final_evaluation(answer, items, responsibility_judgment)
        return [
            {
                "page": page,
                "data": {
                    "legal_violations": {
                        "title": title,
                        "items": items,
                    },
                    "violation_judgment": violation_judgment,
                    "responsibility_judgment": responsibility_judgment,
                    "law_applicability": law_applicability,
                },
            },
            {
                "page": 12,
                "data": {
                    "accident_final_evaluation": final_evaluation,
                },
            }
        ]

    def save_json(self, sources: list[SourceDoc], answer: str = "") -> Path:
        """Save compact page-separated report JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = self.build_page_payload(sources, answer=answer, page=7)

        output_path = self.output_dir / f"law_references_{self.report_id}.json"
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # ask_and_generate_pdf(sources=records)
        return output_path
        
    # def ask_and_generate_pdf(sources: list[dict[str, Any]]) -> None:
    #     try:
    #         choice = input("\nPDF 보고서를 생성하시겠습니까? (y/N): ").strip().lower()
    #     except (KeyboardInterrupt, EOFError):
    #         return

    #     if choice != "y":
    #         return

    #     print("  PDF 생성 기능은 별도 report 프로젝트로 분리되었습니다.")


def print_scenario_status() -> None:
    sc = get_scenario()
    if sc:
        print("\n[현재 시나리오]")
        if sc.overview:
            print(f"  사고 개요: {sc.overview[:60]}{'...' if len(sc.overview)>60 else ''}")
        if sc.details:
            print(f"  사고 경위: {sc.details[:60]}{'...' if len(sc.details)>60 else ''}")
        if sc.workers:
            print(f"  근로자 현황: {sc.workers[:60]}{'...' if len(sc.workers)>60 else ''}")
    else:
        print("\n[현재 시나리오] 없음")


def input_scenario() -> None:
    print("\n사고 시나리오를 입력하세요. (빈칸 Enter = 생략)")
    overview = input("  사고 개요: ").strip()
    details  = input("  사고 경위: ").strip()
    workers  = input("  근로자 현황: ").strip()

    if not any([overview, details, workers]):
        print("  (입력 없음 — 시나리오 변경 안 함)")
        return

    reset_chat_runtime_state(clear_scenario_value=True)
    set_scenario(AccidentScenario(overview=overview, details=details, workers=workers))
    print("  ✓ 시나리오 저장됨")


def load_scenario_file(path: Path) -> AccidentScenario:
    """Load an AccidentScenario from a Python file.

    Supported forms:
    - SCENARIO = {"overview": "...", "details": "...", "workers": "..."}
    - overview = "..."; details = "..."; workers = "..."
    """
    if not path.exists():
        raise FileNotFoundError(f"시나리오 파일을 찾을 수 없습니다: {path}")

    namespace = runpy.run_path(str(path))
    scenario_data = namespace.get("SCENARIO")
    if isinstance(scenario_data, AccidentScenario):
        return scenario_data
    if isinstance(scenario_data, dict):
        return AccidentScenario(
            overview=str(scenario_data.get("overview", "")),
            details=str(scenario_data.get("details", "")),
            workers=str(scenario_data.get("workers", "")),
        )
    return AccidentScenario(
        overview=str(namespace.get("overview", "")),
        details=str(namespace.get("details", "")),
        workers=str(namespace.get("workers", "")),
    )


def print_help() -> None:
    print("  /시나리오  : 사고 시나리오 입력·수정")
    print("  /초기화    : 저장된 시나리오 삭제")
    print("  /상태      : 현재 시나리오 확인")
    print("  /도움말    : 이 도움말")
    print("  exit       : 종료")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive integrated RAG chatbot")
    parser.add_argument(
        "--scenario-file",
        type=str,
        default=None,
        help="Python file containing SCENARIO dict or overview/details/workers variables",
    )
    parser.add_argument(
        "--default-scenario",
        action="store_true",
        help=f"load {DEFAULT_SCENARIO_PATH} explicitly",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print_separator("=")
    print("  Focus-Report Integrated RAG Chat [text + table]")
    print("  /도움말 로 명령어 확인 | 종료: exit 또는 Ctrl+C")
    print_separator("=")

    scenario_file = None
    if args.scenario_file:
        scenario_file = Path(args.scenario_file)
    elif args.default_scenario or DEFAULT_SCENARIO_PATH.exists():
        scenario_file = DEFAULT_SCENARIO_PATH

    if scenario_file:
        try:
            reset_chat_runtime_state(clear_scenario_value=True)
            set_scenario(load_scenario_file(scenario_file))
            print(f"\n시나리오 파일 로드됨: {scenario_file}")
        except Exception as exc:
            print(f"\n시나리오 파일 로드 실패: {exc}")
            return
    else:
        # 시작 시 시나리오 여부 확인
        answer = input("\n사고 시나리오를 먼저 입력하시겠습니까? (y/N): ").strip().lower()
        if answer == "y":
            input_scenario()

    print_scenario_status()
    print_separator()

    while True:
        try:
            question = input("\n질문: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n종료합니다.")
            break

        if not question:
            continue
        if question.lower() == "exit":
            print("종료합니다.")
            break
        if question in ("/시나리오", "/scenario"):
            input_scenario()
            print_scenario_status()
            continue
        if question in ("/초기화", "/clear"):
            reset_chat_runtime_state(clear_scenario_value=True)
            clear_scenario()
            print("  시나리오가 초기화됐습니다.")
            continue
        if question in ("/상태", "/status"):
            print_scenario_status()
            continue
        if question in ("/도움말", "/help"):
            print_help()
            continue

        start = time.time()
        sources = []
        answer_parts: list[str] = []
        print("\n[답변]")
        try:
            for token, srcs in rag_chat_stream(ChatRequest(question=question)):
                print(token, end="", flush=True)
                answer_parts.append(token)
                sources = srcs
        except RuntimeError as exc:
            print(f"\n오류: {exc}")
            if LLM_PROVIDER == "remote_openai":
                print("Colab EXAONE 서버 URL(.env의 LLM_API_BASE)이 열려 있는지 확인하세요.")
            else:
                print("Ollama 서버가 실행 중인지 확인하세요. 예: ollama serve")
            continue

        elapsed_ms = int((time.time() - start) * 1000)
        print_sources(sources)
        report_id = int(datetime.now().strftime("%Y%m%d%H%M%S"))
        law_reference_writer = LawReferenceWriter(report_id=report_id, section="S07")
        references_path = law_reference_writer.save_json(sources, answer="".join(answer_parts))
        print(f"\n법령 참조 JSON 저장(상위 {MAX_LAW_REFERENCES}개): {references_path}")
        print(f"\n응답 시간: {elapsed_ms}ms")
        print_separator()


if __name__ == "__main__":
    main()
