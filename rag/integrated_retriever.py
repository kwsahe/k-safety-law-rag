"""Integrated retrieval across text-law chunks and extracted table chunks."""

from __future__ import annotations

from typing import Literal

from rag.config import RAG_TOP_K
from rag.retriever import retrieve as retrieve_text
from rag.vector_store import get_vector_store
from rag.schemas import SourceDoc
from rag.table_retriever import (
    find_neighbor_table_chunks,
    search_table_chunks,
    search_table_chunks_lexical,
)
from rag.table_vector_store import get_table_vector_store

SourceType = Literal["text", "table"]


def retrieve_integrated(
    query: str,
    text_top_k: int = RAG_TOP_K,
    table_top_k: int = RAG_TOP_K,
    table_first: bool = True,
) -> list[SourceDoc]:
    """Search both text-law and table collections and return one source list."""
    table_sources = _retrieve_tables(query, table_top_k)
    text_sources = _retrieve_texts(query, text_top_k)
    if (
        _is_serious_accident_act_query(query)
        or _is_penalty_query(query)
        or _is_prevention_query(query)
        or _is_focused_excavation_query(query)
        or _is_responsibility_query(query)
    ):
        sources = sorted(
            table_sources + text_sources,
            key=lambda doc: (
                _source_priority(query, doc),
                float(doc.metadata.get("score", 0.0)),
            ),
            reverse=True,
        )
    else:
        sources = table_sources + text_sources if table_first else text_sources + table_sources
    return _with_global_ranks(sources)


_SIHAENGGYUCHIK_BONUS = 0.07  # 시행규칙이 시행령보다 우선 검색되도록 점수 보정
_SPECIAL_EDUCATION_BONUS = 0.2


def _is_sihaenggyuchik(metadata: dict) -> bool:
    """시행규칙 문서 여부 판별 (시행령과 구분)."""
    name = str(metadata.get("law_name", "") or metadata.get("source", "") or metadata.get("pdf_file", ""))
    return "시행규칙" in name


def _retrieve_texts(query: str, top_k: int) -> list[SourceDoc]:
    docs = _serious_accident_act_text_supplements(query)
    docs.extend(_osha_text_supplements(query))
    docs.extend(_penalty_text_supplements(query))
    docs.extend(_prevention_text_supplements(query))
    docs.extend(retrieve_text(query, top_k=top_k))
    result: list[SourceDoc] = []
    seen: set[tuple[str, str, str]] = set()
    for doc in docs:
        key = (
            str(doc.metadata.get("source", "")),
            str(doc.metadata.get("page", "")),
            str(doc.metadata.get("article", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        score = float(doc.metadata.get("score", 0.0))
        score = _adjust_score(query, doc.content, doc.metadata, score)
        result.append(
            SourceDoc(
                content=doc.content,
                metadata={
                    **doc.metadata,
                    "score": round(score, 4),
                    "source_type": "text",
                },
            )
        )
    result.sort(key=lambda d: float(d.metadata.get("score", 0.0)), reverse=True)
    return result


def _serious_accident_act_text_supplements(query: str) -> list[SourceDoc]:
    """Add core 중대재해처벌법 sources that vector search often ranks below 산안법 chunks."""
    if not _is_serious_accident_act_query(query):
        return []

    results = get_vector_store().collection.get(include=["documents", "metadatas"])
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    wanted = _serious_accident_wanted_sources(query)

    docs: list[SourceDoc] = []
    used_notes: set[str] = set()
    for note, score, issue, article, annex, citation_page, needles in wanted:
        for content, metadata in zip(documents, metadatas):
            text = str(content)
            compact_text = "".join(text.split())
            law_name = str((metadata or {}).get("law_name", "") or (metadata or {}).get("source", ""))
            if "중대재해처벌법" not in law_name:
                continue
            if note in used_notes:
                continue
            if not all(needle in compact_text for needle in needles):
                continue
            page = str((metadata or {}).get("page", ""))
            if note in {"osha_contract_62", "osha_contract_64"} and page in {"", "0"}:
                continue
            extra = {
                **(metadata or {}),
                "score": score,
                "source_type": "text",
                "issue": issue,
                "retrieval_note": note,
                "serious_accident_act": True,
            }
            if article:
                extra["article"] = article
            if annex:
                extra["annex"] = annex
            if citation_page:
                extra["citation_page"] = citation_page
            docs.append(SourceDoc(content=text, metadata=extra))
            used_notes.add(note)
            break
    return docs


def _serious_accident_wanted_sources(query: str) -> list[tuple[str, float, str, str, str, str, tuple[str, ...]]]:
    compact_query = "".join(query.split())
    wants_scope = any(term in compact_query for term in ("중대산업재해", "해당여부", "해당하는가", "사망"))
    wants_duty = any(term in compact_query for term in ("경영책임자", "대표이사", "안전보건관리체계", "위반한의무", "구체적으로나열", "위반조항", "처벌주체"))
    wants_contract = any(term in compact_query for term in ("도급", "하청", "원청", "수급", "위탁", "실질적지배"))
    wants_penalty = any(term in compact_query for term in ("처벌", "처벌수위", "징역", "벌금", "법인", "손해배상"))
    wants_aggravation = any(term in compact_query for term in ("가중처벌", "가중", "이전위반", "위반이력", "재범", "5년이내"))
    wants_training_penalty = any(term in compact_query for term in ("안전보건교육", "교육의무", "과태료", "1차", "2차", "3차"))

    wanted: list[tuple[str, float, str, str, str, str, tuple[str, ...]]] = []
    if wants_scope or not wanted:
        wanted.extend(
            [
                ("serious_definition", 0.98, "중대산업재해 해당 여부", "제2조", "", "", ("제2조", "사망자가1명이상")),
                ("serious_scope", 0.97, "상시 근로자 5명 이상 적용범위", "제3조", "", "", ("제3조", "상시근로자가5명미만")),
            ]
        )
    if wants_duty:
        wanted.extend(
            [
                ("serious_duty_law", 0.98, "경영책임자의 안전 및 보건 확보의무", "제4조", "", "", ("제4조", "안전및보건확보의무")),
                ("serious_duty_system", 0.97, "안전보건관리체계 구축 및 반기 점검", "제4조", "", "", ("제4조", "반기1회이상")),
                ("serious_duty_education_check", 0.96, "안전보건 관계 법령 및 교육 이행 점검", "제4조", "", "", ("제4조", "교육실시여부")),
            ]
        )
    if wants_contract:
        wanted.append(
            ("serious_contract_duty", 0.98, "도급ㆍ용역ㆍ위탁 관계 책임", "제5조", "", "", ("제5조", "실질적으로지배"))
        )
    if wants_penalty:
        wanted.extend(
            [
                ("serious_manager_penalty", 0.98, "경영책임자 형사처벌", "제6조", "", "", ("제6조", "1년이상의징역")),
                ("serious_entity_penalty", 0.97, "법인 양벌규정", "제7조", "", "", ("제7조", "50억원이하의벌금")),
                ("serious_damage", 0.95, "징벌적 손해배상", "제15조", "", "", ("제15조", "5배를넘지아니하는범위")),
            ]
        )
    if wants_aggravation:
        wanted.append(
            ("serious_aggravation", 0.99, "5년 이내 재범 가중처벌", "제6조제3항", "", "", ("5년", "2분의1"))
        )
    if wants_training_penalty:
        wanted.extend(
            [
                ("serious_manager_training_law", 0.98, "경영책임자 안전보건교육 수강 의무", "제8조", "", "", ("제8조", "안전보건교육의수강")),
                ("serious_manager_training_hours", 0.97, "경영책임자 안전보건교육 20시간", "제6조", "", "", ("제6조", "20시간")),
                ("serious_training_penalty", 0.98, "경영책임자 안전보건교육 미이행 과태료", "제7조", "별표 4", "15", ("별표4", "과태료의부과기준")),
            ]
        )
    return wanted


def _osha_text_supplements(query: str) -> list[SourceDoc]:
    """Add core 산업안전보건법 sources for dual-law accident analysis."""
    if not _is_dual_law_query(query) and not _is_scaffold_fall_query(query) and not _is_responsibility_query(query):
        return []

    results = get_vector_store().collection.get(include=["documents", "metadatas"])
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    wanted: list[tuple[str, float, str, str, str, tuple[str, ...]]] = [
        ("osha_risk_assessment_36", 0.97, "산업안전보건법 위험성평가 실시 의무", "제36조", "", ("제36조", "위험성평가")),
        ("osha_article_38", 0.97, "산업안전보건법 안전조치 의무", "제38조", "", ("제38조", "안전조치")),
        ("osha_scaffold_training", 0.96, "비계 조립ㆍ해체 특별교육", "", "별표 5 제23호", ("비계의조립", "추락재해방지")),
        ("osha_contract_62", 0.95, "안전보건총괄책임자", "제62조", "", ("제62조", "안전보건총괄책임자")),
        ("osha_contract_64", 0.95, "도급에 따른 산업재해 예방조치", "제64조", "", ("제64조", "도급에따른산업재해예방조치")),
        ("osha_place_11", 0.94, "도급인이 지배ㆍ관리하는 장소", "제11조", "", ("제11조", "안전난간의설치가필요한장소")),
        ("osha_total_manager_53", 0.93, "안전보건총괄책임자의 직무", "제53조", "", ("제53조", "법제64조에따른도급시산업재해예방조치")),
    ]

    docs: list[SourceDoc] = []
    used_notes: set[str] = set()
    for note, score, issue, article, annex, needles in wanted:
        for content, metadata in zip(documents, metadatas):
            text = str(content)
            compact_text = "".join(text.split())
            law_name = str((metadata or {}).get("law_name", "") or (metadata or {}).get("source", ""))
            if "산업안전보건법" not in law_name:
                continue
            if article in {"제36조", "제38조", "제62조", "제64조"} and any(term in law_name for term in ("시행규칙", "시행령")):
                continue
            if article == "제11조" and "시행령" not in law_name:
                continue
            if note in used_notes:
                continue
            if not all(needle in compact_text for needle in needles):
                continue
            extra = {
                **(metadata or {}),
                "score": score,
                "source_type": "text",
                "issue": issue,
                "retrieval_note": note,
                "osha_supplement": True,
            }
            if article:
                extra["article"] = article
            if annex:
                extra["annex"] = annex
            docs.append(SourceDoc(content=text, metadata=extra))
            used_notes.add(note)
            break
    return docs


def _penalty_text_supplements(query: str) -> list[SourceDoc]:
    """Add exact 과태료/별표35 rows that vector search often misses."""
    if not _is_penalty_query(query):
        return []

    docs: list[SourceDoc] = []
    results = get_vector_store().collection.get(include=["documents", "metadatas"])
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    for content, metadata in zip(documents, metadatas):
        text = str(content)
        compact_text = "".join(text.split())
        law_name = str((metadata or {}).get("law_name", ""))
        if (
            "시행령" in law_name
            and "법제29조제3항" in compact_text
            and "교육대상근로자1명당50100150" in compact_text
        ):
            docs.append(
                SourceDoc(
                    content=text,
                    metadata={
                        **(metadata or {}),
                        "article": "",
                        "score": 0.98,
                        "source_type": "text",
                        "annex": "별표 35",
                        "citation_page": "130~143",
                        "violation_article": "법 제29조제3항",
                        "retrieval_note": "exact_penalty_supplement",
                    },
                )
            )
            break
    return docs


def _prevention_text_supplements(query: str) -> list[SourceDoc]:
    """Add legal-duty sources needed for recurrence-prevention answers."""
    if not _is_prevention_query(query):
        return []

    results = get_vector_store().collection.get(include=["documents", "metadatas"])
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    wanted: list[tuple[str, float, str, str]] = [
        ("산업재해발생은폐금지및보고등", 0.98, "산업재해 발생 보고 및 재발방지 계획", "report_duty"),
        ("유해위험방지계획서의이행", 0.97, "유해위험방지계획서 이행 확인", "hazard_plan_followup"),
        ("유해위험방지계획서", 0.96, "유해위험방지계획서 작성ㆍ비치ㆍ변경 검토", "hazard_plan"),
        ("산업재해조사표", 0.95, "산업재해조사표 작성ㆍ제출", "accident_investigation_form"),
        ("도급인의안전조치및보건조치", 0.94, "도급인의 안전조치 및 보건조치", "contractor_safety_duty"),
    ]

    docs: list[SourceDoc] = []
    used_notes: set[str] = set()
    for needle, score, issue, note in wanted:
        for content, metadata in zip(documents, metadatas):
            text = str(content)
            compact_text = "".join(text.split())
            page = str((metadata or {}).get("page", ""))
            if needle not in compact_text or note in used_notes:
                continue
            if note in {"hazard_plan", "contractor_safety_duty", "accident_investigation_form"} and page in {"", "0"}:
                continue
            extra_metadata = {}
            if note == "accident_investigation_form":
                extra_metadata = {"annex": "별지 제30호서식", "citation_page": "49~50"}
            docs.append(
                SourceDoc(
                    content=text,
                    metadata={
                        **(metadata or {}),
                        **extra_metadata,
                        "score": score,
                        "source_type": "text",
                        "issue": issue,
                        "retrieval_note": note,
                    },
                )
            )
            used_notes.add(note)
            break
    return docs


def _retrieve_tables(query: str, top_k: int) -> list[SourceDoc]:
    hits = search_table_chunks(query, n_results=top_k)
    hits.extend(search_table_chunks_lexical(query, n_results=top_k))
    hits.extend(_neighbor_hits(hits))

    result: list[SourceDoc] = []
    seen: set[tuple[str, str, str, str]] = set()
    for hit in hits:
        metadata = hit["metadata"]
        key = (
            str(metadata.get("source", "")),
            str(metadata.get("page", "")),
            str(metadata.get("table_index", "")),
            str(metadata.get("row_index", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        score = _adjust_score(query, str(hit["text"]), metadata, float(hit["score"]))
        result.append(
            SourceDoc(
                content=str(hit["text"]),
                metadata={
                    **metadata,
                    "score": round(score, 4),
                    "source_type": "table",
                },
            )
        )
    result.extend(_prevention_table_supplements(query))
    result = _dedupe_source_docs(result)
    result.sort(key=lambda d: float(d.metadata.get("score", 0.0)), reverse=True)
    return result


def _prevention_table_supplements(query: str) -> list[SourceDoc]:
    """Add table-form safety inspection/certification sources for prevention questions."""
    if not _is_prevention_query(query):
        return []

    results = get_table_vector_store().collection.get(include=["documents", "metadatas"])
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    wanted: list[tuple[tuple[str, ...], float, dict[str, str]]] = [
        (
            ("안전검사합격증명서",),
            0.97,
            {"annex": "별표 16", "citation_page": "115~116", "issue": "크레인 안전검사 합격증명 확인"},
        ),
        (
            ("종류:크레인", "표시부호:C"),
            0.96,
            {"annex": "별표 16", "citation_page": "115~116", "issue": "크레인 안전검사 표시 확인"},
        ),
        (
            ("안전인증대상:크레인",),
            0.95,
            {"issue": "크레인 안전인증 대상 확인"},
        ),
        (
            ("안전검사대상:크레인",),
            0.95,
            {"issue": "크레인 안전검사 대상 확인"},
        ),
        (
            ("굴착면의높이가2미터이상", "지반굴착"),
            0.94,
            {"annex": "별표 5 제19호", "issue": "지반 굴착 특별안전교육"},
        ),
        (
            ("출입금지",),
            0.94,
            {"annex": "별표 6 제1호", "citation_page": "94~96", "issue": "출입금지 표지ㆍ출입통제"},
        ),
        (
            ("크레인을사용", "작업"),
            0.93,
            {"annex": "별표 5 제14호", "issue": "크레인 사용 작업 특별안전교육"},
        ),
    ]

    docs: list[SourceDoc] = []
    for needles, score, extra_metadata in wanted:
        for content, metadata in zip(documents, metadatas):
            text = str(content)
            compact_text = "".join(text.split())
            if not all(needle in compact_text for needle in needles):
                continue
            docs.append(
                SourceDoc(
                    content=text,
                    metadata={
                        **(metadata or {}),
                        **extra_metadata,
                        "score": score,
                        "source_type": "table",
                        "retrieval_note": "prevention_table_supplement",
                    },
                )
            )
            break
    return docs


def _dedupe_source_docs(docs: list[SourceDoc]) -> list[SourceDoc]:
    result: list[SourceDoc] = []
    seen: set[tuple[str, str, str, str]] = set()
    for doc in docs:
        metadata = doc.metadata
        key = (
            str(metadata.get("source", "") or metadata.get("pdf_file", "")),
            str(metadata.get("page", "")),
            str(metadata.get("table_index", "")),
            str(metadata.get("row_index", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(doc)
    return result


def _adjust_score(query: str, content: str, metadata: dict, score: float) -> float:
    """Query-aware score shaping for safety-law judgment questions."""
    score = normalize_score(score)
    if _is_sihaenggyuchik(metadata):
        score += _SIHAENGGYUCHIK_BONUS

    compact_query = "".join(query.split())
    compact_content = "".join(content.split())
    law_name = str(metadata.get("law_name", "") or metadata.get("source", "") or metadata.get("pdf_file", ""))

    if _is_serious_accident_act_query(query):
        if "중대재해처벌법" in law_name:
            score += 0.25
        if any(term in compact_content for term in ("제2조", "제3조", "제4조", "제5조", "제6조", "제7조", "제8조", "제15조", "별표4")):
            score += 0.12
        if "산업안전보건법" in law_name and (
            "법제29조제3항" in compact_content
            or "교육대상근로자1명당50100150" in compact_content
            or "[작업항목]" in compact_content
        ):
            score -= 0.35
    if _is_dual_law_query(query):
        if "산업안전보건법" in law_name and any(
            term in compact_content
            for term in ("제38조", "제62조", "제64조", "제11조", "비계의조립", "안전난간", "추락")
        ):
            score += 0.22

    excavation_query = any(term in compact_query for term in ("굴착", "지반굴착", "굴착면", "토사붕괴"))
    education_query = any(term in compact_query for term in ("특별교육", "교육", "교육내용", "미이수", "미실시"))
    if excavation_query and education_query:
        if "[작업항목]19." in compact_content or "굴착면의높이가2미터이상인지반굴착작업" in compact_content:
            score += _SPECIAL_EDUCATION_BONUS
        if is_education_time_table(query, content, metadata):
            score -= 0.35
        elif "별표4" in compact_content and "교육시간" in compact_content:
            score -= 0.2
        if "시행령" in law_name and "제29조" in content:
            score -= 0.12

    if education_query:
        crane_query = any(term in compact_query for term in ("크레인", "인양", "양중"))
        steel_query = any(term in compact_query for term in ("철골", "골조", "금속제", "금속", "15층", "고층"))
        tunnel_query = "터널" in compact_query
        rock_query = any(term in compact_query for term in ("암석", "발파", "폭발물"))

        if crane_query and ("[작업항목]14." in compact_content or "크레인을사용하는작업" in compact_content):
            score += 0.16
        if steel_query and ("[작업항목]27." in compact_content or "건축물의골조" in compact_content):
            score += 0.16
        if "[작업항목]21." in compact_content and not tunnel_query:
            score -= 0.18
        if "[작업항목]22." in compact_content and not rock_query:
            score -= 0.35

    if _is_penalty_query(query):
        if (
            ("별표35" in compact_content or "법제29조제3항" in compact_content)
            and "교육대상근로자1명당50100150" in compact_content
        ):
            score += 0.25
        if "별표26" in compact_content or "행정처분기준" in compact_content:
            score += 0.1

    signage_query = any(term in compact_query for term in ("출입금지", "표지", "표지판", "금지표지", "안전보건표지"))
    if signage_query:
        if any(term in compact_content for term in ("출입금지", "금지표지", "안전보건표지", "별표6")):
            score += 0.2
        if "[작업항목]" in compact_content and "별표5" not in compact_content:
            score -= 0.12

    return normalize_score(score)


def _source_priority(query: str, doc: SourceDoc) -> int:
    """Break score ties with query-critical exact supplements first."""
    compact_query = "".join(query.split())
    compact_content = "".join(doc.content.split())
    metadata = doc.metadata
    if _is_serious_accident_act_query(query):
        if metadata.get("serious_accident_act"):
            return 6
        if metadata.get("osha_supplement") and _is_dual_law_query(query):
            return 6
        law_name = str(metadata.get("law_name", "") or metadata.get("source", "") or metadata.get("pdf_file", ""))
        if "중대재해처벌법" in law_name:
            return 4
        if _is_dual_law_query(query) and "산업안전보건법" in law_name:
            return 4
        if "산업안전보건법" in law_name and (
            "별표35" in compact_content
            or "법제29조제3항" in compact_content
            or "[작업항목]" in compact_content
        ):
            return -3
    if _is_penalty_query(query):
        if metadata.get("annex") == "별표 35" or (
            "법제29조제3항" in compact_content and "교육대상근로자1명당50100150" in compact_content
        ):
            return 4
        if "별표26" in compact_content or "행정처분기준" in compact_content:
            return 2
    if _is_prevention_query(query):
        if metadata.get("issue"):
            return 3
    if _is_focused_excavation_query(query):
        if metadata.get("annex") == "별표 5 제19호" or (
            "굴착면의높이가2미터" in compact_content and "지반굴착" in compact_content
        ):
            return 4
        if any(term in compact_content for term in ("[작업항목]27.", "[작업항목]39.", "타워크레인")):
            return -2
    if "굴착" in compact_query and "크레인" not in compact_query and "골조" not in compact_query:
        if any(term in compact_content for term in ("[작업항목]27.", "[작업항목]39.", "타워크레인")):
            return -1
    return 0


def normalize_score(score: float) -> float:
    """Clamp adjusted relevance scores into cosine-like 0.0~1.0 range."""
    return round(max(0.0, min(float(score), 0.98)), 4)


def _is_penalty_query(query: str) -> bool:
    compact_query = "".join(query.split())
    return any(term in compact_query for term in ("과태료", "1차", "2차", "3차", "처분수위", "처벌수위", "행정처분", "금액"))


def _is_serious_accident_act_query(query: str) -> bool:
    compact_query = "".join(query.split())
    return any(
        term in compact_query
        for term in (
            "중대재해처벌법",
            "중대산업재해",
            "경영책임자",
            "대표이사",
            "안전보건관리체계",
            "실질적지배",
            "원청",
            "하청",
            "도급",
        )
    ) or (
        "법인" in compact_query
        and any(term in compact_query for term in ("처벌", "벌금", "손해배상"))
    )


def _is_dual_law_query(query: str) -> bool:
    compact_query = "".join(query.split())
    return "산업안전보건법" in compact_query and "중대재해처벌법" in compact_query


def _is_scaffold_fall_query(query: str) -> bool:
    compact_query = "".join(query.split())
    return any(term in compact_query for term in ("비계", "추락", "안전난간")) and any(
        term in compact_query for term in ("책임", "위반", "처벌", "적용", "안전조치")
    )


def _is_responsibility_query(query: str) -> bool:
    compact_query = "".join(query.split())
    return any(term in compact_query for term in ("책임여부", "책임은", "책임이", "책임판단")) and any(
        term in compact_query for term in ("시공사", "사업주", "근로자", "원청", "도급", "하청")
    )


def _is_prevention_query(query: str) -> bool:
    compact_query = "".join(query.split())
    return any(
        term in compact_query
        for term in ("재발방지", "즉시취해야", "조치를제시", "조치를알려", "법령의무기준", "법적근거")
    )


def _is_focused_excavation_query(query: str) -> bool:
    compact_query = "".join(query.split())
    return (
        any(term in compact_query for term in ("굴착작업관련", "굴착작업", "지반굴착", "굴착면"))
        and any(term in compact_query for term in ("중심", "미실시", "미이수", "위반"))
        and "모든특별교육" not in compact_query
    )


def is_education_time_table(query: str, content: str, metadata: dict) -> bool:
    """별표 4 교육시간표는 교육내용 질문의 핵심 근거가 아니다."""
    compact_query = "".join(query.split())
    compact_content = "".join(content.split())
    asks_content = any(term in compact_query for term in ("교육내용", "특별교육", "미이수", "미실시", "위반"))
    asks_time = any(term in compact_query for term in ("교육시간", "몇시간", "시간은", "시간기준"))
    if not asks_content or asks_time:
        return False

    page = str(metadata.get("page", ""))
    return (
        "별표4" in compact_content
        or "교육과정별교육시간" in compact_content
        or "교육시간" in compact_content
        or page in {"75", "76", "77"}
    )


def _with_global_ranks(sources: list[SourceDoc]) -> list[SourceDoc]:
    ranked: list[SourceDoc] = []
    for index, doc in enumerate(sources, start=1):
        ranked.append(
            SourceDoc(
                content=doc.content,
                metadata={
                    **doc.metadata,
                    "retrieval_rank": index,
                    "evidence_level": evidence_level(index),
                },
            )
        )
    return ranked


def evidence_level(rank: int) -> str:
    if rank <= 3:
        return "PRIMARY"
    if rank <= 6:
        return "SECONDARY"
    return "BACKGROUND"


def _neighbor_hits(hits: list[dict]) -> list[dict]:
    """Attach adjacent table rows so merged-row subcategories stay together."""
    neighbors: list[dict] = []
    for hit in hits:
        metadata = hit.get("metadata") or {}
        text = str(hit.get("text", ""))
        if not _needs_neighbor_rows(text):
            continue
        for neighbor in find_neighbor_table_chunks(metadata, before=0, after=3):
            if _is_relevant_neighbor_row(str(neighbor.get("text", ""))):
                neighbors.append(neighbor)
    return neighbors


def _needs_neighbor_rows(text: str) -> bool:
    return any(
        signal in text
        for signal in (
            "유해인자:",
            "TWA_",
        )
    )


def _is_relevant_neighbor_row(text: str) -> bool:
    """Keep merged-row continuations without flooding sources with unrelated rows."""
    if "유해인자:" in text:
        return False
    return "col_1:" in text or "세부구분:" in text


def split_sources(sources: list[SourceDoc]) -> tuple[list[SourceDoc], list[SourceDoc]]:
    """Split integrated sources into table and text buckets."""
    table_docs = [doc for doc in sources if doc.metadata.get("source_type") == "table"]
    text_docs = [doc for doc in sources if doc.metadata.get("source_type") == "text"]
    return table_docs, text_docs
