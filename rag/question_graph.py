"""Question routing graph primitives.

This module is intentionally dependency-light. The node functions are shaped so
they can be wrapped by LangGraph StateGraph later, while remaining runnable in
the current environment where langgraph is not installed.
"""

from __future__ import annotations

import re
from typing import Any, TypedDict


QUESTION_SCOPE_GENERAL = "general_law"
QUESTION_SCOPE_SCENARIO = "scenario_judgment"


class QuestionGraphState(TypedDict, total=False):
    question: str
    mode: str
    scope: str
    is_scenario_specific: bool
    scope_signals: list[str]


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


def question_scope_node(state: QuestionGraphState | str, mode: str = "") -> QuestionGraphState:
    """Classify whether the user's actual question asks for scenario judgment.

    The classifier must inspect only the user's question text. In scenario mode,
    saved accident facts may be appended later to the retrieval query, but that
    should not turn a simple legal question into an accident-specific judgment.
    """
    if isinstance(state, str):
        question = state
        active_mode = mode
        next_state: QuestionGraphState = {"question": question, "mode": active_mode}
    else:
        question = str(state.get("question", ""))
        active_mode = str(state.get("mode", mode or ""))
        next_state = dict(state)

    compact = re.sub(r"\s+", "", question)
    if active_mode == "general":
        next_state.update(
            {
                "scope": QUESTION_SCOPE_GENERAL,
                "is_scenario_specific": False,
                "scope_signals": [],
            }
        )
        return next_state

    signals = [term for term in SCENARIO_SCOPE_TERMS if term in compact]
    is_scenario_specific = bool(signals)
    next_state.update(
        {
            "scope": QUESTION_SCOPE_SCENARIO if is_scenario_specific else QUESTION_SCOPE_GENERAL,
            "is_scenario_specific": is_scenario_specific,
            "scope_signals": signals,
        }
    )
    return next_state


def run_question_graph(question: str, *, mode: str = "") -> QuestionGraphState:
    """Run the current routing graph.

    Today the graph contains only QuestionScopeNode. Additional intent,
    citation-validation, and cache-guard nodes can be appended here without
    changing callers.
    """
    state: QuestionGraphState = {"question": question, "mode": mode}
    state = question_scope_node(state)
    return state


def question_graph_mermaid() -> str:
    return """flowchart TD
    A[Input question] --> B[QuestionScopeNode]
    B --> C{scope}
    C -->|general_law| D[General law routing]
    C -->|scenario_judgment| E[Scenario judgment routing]
"""
