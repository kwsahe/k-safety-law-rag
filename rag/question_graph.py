"""Deterministic question-routing graph used by the RAG pipeline.

The nodes keep legal routing observable and testable. They intentionally avoid
an extra LLM classification call so a temporarily unavailable model cannot
break routing before retrieval starts.
"""

from __future__ import annotations

import hashlib
import re
from typing import TypedDict

from rag.special_education_routing import (
    has_special_education_signal,
    has_welding_work_signal,
)
from rag.hot_work_routing import hot_work_issue_signals, is_hot_work_controls_question
from rag.general_law_routing import classify_general_law_question
from rag.falling_object_routing import (
    is_masonry_falling_controls_question,
    is_masonry_falling_scenario,
    is_masonry_special_education_question,
)
from rag.industrial_fire_routing import classify_lithium_question
from rag.v1_incident_routing import (
    is_machine_controls_question,
    is_machine_controls_inspection_question,
    is_machine_entanglement_scenario,
    is_machine_inspection_question,
    is_struck_by_scenario,
    is_struck_controls_question,
)


QUESTION_SCOPE_GENERAL = "general_law"
QUESTION_SCOPE_SCENARIO = "scenario_judgment"


class QuestionGraphState(TypedDict, total=False):
    question: str
    mode: str
    cache_context: str
    scenario_profile: dict
    scope: str
    is_scenario_specific: bool
    scope_signals: list[str]
    intent: str
    intent_signals: list[str]
    route: str
    required_citations: list[str]
    cache_key: str
    graph_nodes: list[str]


SCENARIO_SCOPE_TERMS = (
    "위사고",
    "해당사고",
    "사고에서",
    "시나리오",
    "위상황",
    "해당상황",
    "C씨",
    "근로자C",
    "(주)스파일럿건설",
)


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _mark_node(state: QuestionGraphState, name: str) -> None:
    state.setdefault("graph_nodes", []).append(name)


def scenario_profile_node(state: QuestionGraphState) -> QuestionGraphState:
    """Record whether a validated, version-matched scenario profile is available."""
    next_state = dict(state)
    profile = next_state.get("scenario_profile") or {}
    if profile:
        _mark_node(next_state, "ScenarioProfileNode")
        next_state["scenario_profile_used"] = True
        next_state["scenario_kind"] = str(profile.get("accident_type", ""))
    return next_state


def question_scope_node(state: QuestionGraphState | str, mode: str = "") -> QuestionGraphState:
    """Classify whether the actual question asks for scenario judgment."""
    if isinstance(state, str):
        question = state
        active_mode = mode
        next_state: QuestionGraphState = {"question": question, "mode": active_mode}
    else:
        question = str(state.get("question", ""))
        active_mode = str(state.get("mode", mode or ""))
        next_state = dict(state)

    _mark_node(next_state, "QuestionScopeNode")
    compact = _compact(question)
    if active_mode == "general":
        next_state.update({"scope": QUESTION_SCOPE_GENERAL, "is_scenario_specific": False, "scope_signals": []})
        return next_state

    signals = [term for term in SCENARIO_SCOPE_TERMS if term in compact]
    if next_state.get("scenario_profile"):
        signals.append("active_scenario_profile")
    next_state.update(
        {
            "scope": QUESTION_SCOPE_SCENARIO if signals else QUESTION_SCOPE_GENERAL,
            "is_scenario_specific": bool(signals),
            "scope_signals": signals,
        }
    )
    return next_state


def intent_classifier_node(state: QuestionGraphState) -> QuestionGraphState:
    """Assign one stable intent so similar legal questions cannot share a route."""
    next_state = dict(state)
    _mark_node(next_state, "IntentClassifierNode")
    compact = _compact(str(next_state.get("question", "")))
    fact_compact = compact + _compact(str(next_state.get("cache_context", "")))

    lithium_kind = classify_lithium_question(compact, fact_compact)
    if lithium_kind:
        next_state.update(
            {
                "intent": lithium_kind,
                "intent_signals": [
                    term
                    for term in ("리튬전지", "위험물", "23명", "다수사망", "도급", "파견", "위험성평가", "전조", "종합평가")
                    if term in fact_compact
                ],
            }
        )
        return next_state

    general_intent = classify_general_law_question(compact)
    if general_intent:
        next_state.update({"intent": general_intent, "intent_signals": [general_intent]})
        return next_state

    if is_masonry_falling_controls_question(compact, fact_compact):
        next_state.update(
            {
                "intent": "masonry_falling_controls",
                "intent_signals": [term for term in ("낙하물방지망", "방호선반", "발끝막이판", "출입통제") if term in compact],
            }
        )
        return next_state

    if is_masonry_special_education_question(compact, fact_compact):
        next_state.update(
            {
                "intent": "masonry_special_education_scope",
                "intent_signals": [term for term in ("조적", "벽돌", "특별교육", "특별안전교육", "교육미실시") if term in fact_compact],
            }
        )
        return next_state

    if is_hot_work_controls_question(compact):
        next_state.update(
            {
                "intent": "hot_work_controls",
                "intent_signals": hot_work_issue_signals(compact),
            }
        )
        return next_state

    if has_welding_work_signal(compact) and has_special_education_signal(compact):
        next_state.update(
            {
                "intent": "welding_special_education",
                "intent_signals": [
                    term
                    for term in ("용접", "가스용접", "아세틸렌", "산소-아세틸렌", "용단", "특별교육", "특별안전교육")
                    if term in compact
                ],
            }
        )
        return next_state

    if any(term in compact for term in ("굴착", "굴착면", "토공")) and any(
        term in compact for term in ("특별교육", "특별안전교육", "교육미실시", "교육내용")
    ):
        next_state.update(
            {
                "intent": "excavation_special_education",
                "intent_signals": [term for term in ("굴착", "굴착면", "특별교육", "특별안전교육") if term in compact],
            }
        )
        return next_state

    if any(term in compact for term in ("굴착", "굴착면", "흙막이", "지보공")) and sum(
        term in compact for term in ("흙막이", "지보공", "기울기", "작업중지", "사전조사")
    ) >= 2:
        next_state.update(
            {
                "intent": "excavation_controls",
                "intent_signals": [term for term in ("굴착", "흙막이", "지보공", "기울기", "작업중지") if term in compact],
            }
        )
        return next_state

    if is_machine_controls_inspection_question(compact):
        next_state.update(
            {
                "intent": "machine_controls_inspection",
                "intent_signals": [term for term in ("프레스", "방호장치", "안전검사", "정비") if term in compact],
            }
        )
        return next_state

    if is_machine_inspection_question(compact):
        next_state.update(
            {
                "intent": "machine_inspection",
                "intent_signals": [term for term in ("프레스", "안전검사", "자율안전확인", "방호장치") if term in compact],
            }
        )
        return next_state

    if is_machine_controls_question(compact):
        next_state.update(
            {
                "intent": "machine_controls",
                "intent_signals": [term for term in ("프레스", "방호장치", "동력차단", "운전정지", "정비", "청소") if term in compact],
            }
        )
        return next_state

    if is_struck_controls_question(compact):
        next_state.update(
            {
                "intent": "struck_controls",
                "intent_signals": [term for term in ("크레인", "인양", "낙하", "와이어로프", "해지장치", "출입금지") if term in compact],
            }
        )
        return next_state

    if is_machine_entanglement_scenario(compact) and has_special_education_signal(compact):
        next_state.update(
            {
                "intent": "machine_special_education",
                "intent_signals": [term for term in ("프레스", "끼임", "협착", "특별교육", "특별안전교육") if term in compact],
            }
        )
        return next_state

    if is_struck_by_scenario(compact) and has_special_education_signal(compact):
        next_state.update(
            {
                "intent": "struck_special_education",
                "intent_signals": [term for term in ("크레인", "인양", "낙하", "특별교육", "특별안전교육") if term in compact],
            }
        )
        return next_state

    rules: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
        ("comprehensive_report", ("종합평가", "최종보고서", "사고원인분석", "책임주체별"), ()),
        ("prime_contractor_liability", ("도급", "원청", "협력업체", "수급인"), ()),
        ("executive_liability", ("대표이사", "경영책임자"), ("중대재해처벌법", "처벌수위", "의무위반")),
        ("employer_liability", ("사업주",), ("책임", "과태료", "처벌수위")),
        ("ppe_scaffold_standards", ("보호구", "안전모", "안전대", "설치기준"), ("비계", "작업발판")),
        ("scaffold_special_education", ("비계", "작업발판", "이동식비계"), ("특별교육", "특별안전교육", "교육미실시", "교육내용")),
        ("education_comparison", ("안전보건교육",), ("특별안전교육", "특별교육", "차이")),
    ]

    intent = "scenario_general" if next_state.get("scope") == QUESTION_SCOPE_SCENARIO else "general_law"
    signals: list[str] = []
    for candidate, primary, secondary in rules:
        primary_hits = [term for term in primary if term in compact]
        secondary_hits = [term for term in secondary if term in compact]
        if primary_hits and (not secondary or secondary_hits):
            intent = candidate
            signals = primary_hits + secondary_hits
            break

    next_state.update({"intent": intent, "intent_signals": signals})
    return next_state


INTENT_CITATIONS: dict[str, list[str]] = {
    "lithium_hazard_training": ["산업안전보건법 제29조", "산업안전보건기준에 관한 규칙 제225조", "제232조"],
    "lithium_mass_fatality_sentencing": ["중대재해처벌법 제2조", "제6조", "제7조", "제15조"],
    "lithium_contract_dispatch": ["산업안전보건법 제63조", "제64조", "중대재해처벌법 제5조", "중대재해처벌법 시행령 제4조제9호"],
    "lithium_risk_warning": ["산업안전보건법 제36조", "제51조", "중대재해처벌법 제4조", "중대재해처벌법 시행령 제4조제3호"],
    "lithium_comprehensive": ["산업안전보건법 제29조", "제36조", "제51조", "중대재해처벌법 제4조", "제6조", "제7조"],
    "masonry_special_education_scope": ["산업안전보건법 제29조", "산업안전보건법 시행규칙 별표 5"],
    "masonry_falling_controls": [
        "산업안전보건기준에 관한 규칙 제13조",
        "제14조",
        "제20조",
    ],
    "general_law_purpose": ["산업안전보건법 제1조"],
    "general_law_basic_duties": ["산업안전보건법 제5조", "제6조", "제51조", "제57조"],
    "general_law_manager_roles": ["산업안전보건법 제15조", "제62조"],
    "general_law_regular_education": ["산업안전보건법 제29조", "산업안전보건법 시행규칙 별표 4", "별표 5"],
    "general_law_risk_assessment": ["산업안전보건법 제36조"],
    "hot_work_controls": [
        "산업안전보건기준에 관한 규칙 제241조",
        "산업안전보건기준에 관한 규칙 제232조",
        "산업안전보건기준에 관한 규칙 제32조",
    ],
    "welding_special_education": ["산업안전보건법 시행규칙 별표 5 제2호"],
    "excavation_special_education": ["산업안전보건법 시행규칙 별표 5 제19호"],
    "excavation_controls": [
        "산업안전보건기준에 관한 규칙 제338조",
        "제339조",
        "제340조",
        "제347조",
        "산업안전보건법 제51조",
    ],
    "machine_special_education": ["산업안전보건법 시행규칙 별표 5 제1호라목 제11호"],
    "machine_controls": [
        "산업안전보건기준에 관한 규칙 제87조",
        "제88조",
        "제89조",
        "제92조",
        "제93조",
        "제103조",
        "제104조",
    ],
    "machine_inspection": [
        "산업안전보건법 제80조",
        "산업안전보건법 제89조",
        "산업안전보건법 제93조",
        "산업안전보건기준에 관한 규칙 제36조",
        "산업안전보건기준에 관한 규칙 제93조",
    ],
    "machine_controls_inspection": [
        "산업안전보건법 제80조제3항",
        "산업안전보건법 제93조",
        "산업안전보건기준에 관한 규칙 제87조",
        "제88조",
        "제89조",
        "제92조",
        "제93조",
        "제103조",
        "제104조",
    ],
    "struck_special_education": ["산업안전보건법 시행규칙 별표 5 제1호라목 제14호"],
    "struck_controls": [
        "산업안전보건기준에 관한 규칙 제14조",
        "제133조",
        "제134조",
        "제146조",
        "제149조",
    ],
    "scaffold_special_education": ["산업안전보건법 시행규칙 별표 5 제23호"],
    "ppe_scaffold_standards": ["산업안전보건기준에 관한 규칙 제14조", "제32조", "제42조", "제56조~제62조"],
    "employer_liability": ["산업안전보건법 시행령 별표 35"],
    "executive_liability": ["중대재해처벌법 제2조", "제3조", "제6조", "제7조", "제15조"],
    "prime_contractor_liability": ["산업안전보건법 제64조", "중대재해처벌법 제5조", "중대재해처벌법 시행령 제4조제9호"],
    "education_comparison": ["산업안전보건법 제29조", "산업안전보건법 시행규칙 별표 5"],
}


def retrieval_plan_node(state: QuestionGraphState) -> QuestionGraphState:
    next_state = dict(state)
    _mark_node(next_state, "RetrievalPlanNode")
    intent = str(next_state.get("intent", "general_law"))
    direct_intents = set(INTENT_CITATIONS) | {"comprehensive_report"}
    next_state.update(
        {
            "route": "direct_candidate" if intent in direct_intents else "rag_llm",
            "required_citations": list(INTENT_CITATIONS.get(intent, [])),
        }
    )
    return next_state


def cache_guard_node(state: QuestionGraphState) -> QuestionGraphState:
    """Build a collision-resistant key from the complete normalized question."""
    next_state = dict(state)
    _mark_node(next_state, "CacheGuardNode")
    raw = "\n".join(
        (
            str(next_state.get("mode", "")),
            str(next_state.get("scope", "")),
            str(next_state.get("intent", "")),
            str(next_state.get("question", "")).strip(),
            str(next_state.get("cache_context", "")).strip(),
        )
    )
    next_state["cache_key"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return next_state


def run_question_graph(
    question: str,
    *,
    mode: str = "",
    cache_context: str = "",
    scenario_profile: dict | None = None,
) -> QuestionGraphState:
    state: QuestionGraphState = {
        "question": question,
        "mode": mode,
        "cache_context": cache_context,
        "scenario_profile": scenario_profile or {},
        "graph_nodes": [],
    }
    state = scenario_profile_node(state)
    state = question_scope_node(state)
    state = intent_classifier_node(state)
    state = retrieval_plan_node(state)
    state = cache_guard_node(state)
    return state


def public_graph_trace(state: QuestionGraphState) -> dict[str, object]:
    """Return diagnostics safe to store in the administrator payload."""
    route = str(state.get("route", "rag_llm"))
    return {
        "scope": state.get("scope", QUESTION_SCOPE_GENERAL),
        "intent": state.get("intent", "general_law"),
        "route": route,
        "execution": "deterministic_legal_rule" if route == "direct_candidate" else "rag_llm_generation",
        "cache_used": False,
        "signals": list(state.get("intent_signals", [])),
        "required_citations": list(state.get("required_citations", [])),
        "cache_key": state.get("cache_key", ""),
        "nodes": list(state.get("graph_nodes", [])),
        "scenario_profile_used": bool(state.get("scenario_profile_used", False)),
        "scenario_kind": state.get("scenario_kind", ""),
    }


def question_graph_mermaid() -> str:
    return """flowchart TD
    A[Input question] --> P[ScenarioProfileNode]
    P --> B[QuestionScopeNode]
    B --> C[IntentClassifierNode]
    C --> D[RetrievalPlanNode]
    D --> E[CacheGuardNode]
    E --> F{route}
    F -->|direct_candidate| G[Deterministic legal answer]
    F -->|rag_llm| H[Text and Table RAG plus EXAONE]
    G --> I[CitationValidatorNode]
    H --> I
"""
