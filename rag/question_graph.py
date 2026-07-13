"""Deterministic question-routing graph used by the RAG pipeline.

The nodes keep legal routing observable and testable. They intentionally avoid
an extra LLM classification call so a temporarily unavailable model cannot
break routing before retrieval starts.
"""

from __future__ import annotations

import hashlib
import re
from typing import TypedDict


QUESTION_SCOPE_GENERAL = "general_law"
QUESTION_SCOPE_SCENARIO = "scenario_judgment"


class QuestionGraphState(TypedDict, total=False):
    question: str
    mode: str
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
        )
    )
    next_state["cache_key"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return next_state


def run_question_graph(question: str, *, mode: str = "") -> QuestionGraphState:
    state: QuestionGraphState = {"question": question, "mode": mode, "graph_nodes": []}
    state = question_scope_node(state)
    state = intent_classifier_node(state)
    state = retrieval_plan_node(state)
    state = cache_guard_node(state)
    return state


def public_graph_trace(state: QuestionGraphState) -> dict[str, object]:
    """Return diagnostics safe to store in the administrator payload."""
    return {
        "scope": state.get("scope", QUESTION_SCOPE_GENERAL),
        "intent": state.get("intent", "general_law"),
        "route": state.get("route", "rag_llm"),
        "signals": list(state.get("intent_signals", [])),
        "required_citations": list(state.get("required_citations", [])),
        "cache_key": state.get("cache_key", ""),
        "nodes": list(state.get("graph_nodes", [])),
    }


def question_graph_mermaid() -> str:
    return """flowchart TD
    A[Input question] --> B[QuestionScopeNode]
    B --> C[IntentClassifierNode]
    C --> D[RetrievalPlanNode]
    D --> E[CacheGuardNode]
    E --> F{route}
    F -->|direct_candidate| G[Deterministic legal answer]
    F -->|rag_llm| H[Text and Table RAG plus EXAONE]
    G --> I[CitationValidatorNode]
    H --> I
"""
