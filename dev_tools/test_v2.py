"""test_DestrictionArea.md Q1·Q2 검증."""
import sys, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

from rag.chatbot import rag_chat, set_scenario
from rag.schemas import AccidentScenario, ChatRequest

set_scenario(AccidentScenario(
    overview="아파트 신축 건설현장(지상 15층, 지하 2층) 지하 2층 굴착 공사구역 사고. 굴착면 높이 약 4미터.",
    details="굴착 작업 중 크레인 인양 작업 병행. 출입금지 표지 있었으나 일용직 근로자 A씨가 진입, 낙하 철제 자재에 부상.",
    workers="일용직(당일 처음 현장 투입), 특별안전교육 미이수, 출입금지 표지 인지 후 무시하고 진입."
))

SEP = "=" * 65

queries = [
    ("Q1 — 기본 위반 판단",
     "위 사고에서 사업주가 위반한 법령 조항은 무엇인가? 굴착 작업 관련 특별안전교육 미실시를 중심으로 판단하라."),
    ("Q2 — 복합 위반 판단",
     "크레인 인양 작업과 굴착 작업이 동시에 진행되고 있었다. 사업주가 위반했을 가능성이 있는 모든 특별교육 관련 조항을 나열하라."),
]

for label, q in queries:
    print(f"\n{SEP}\n[{label}]\n질문: {q}\n{SEP}", flush=True)
    resp = rag_chat(ChatRequest(question=q))
    print(f"\n[답변]\n{resp.answer}", flush=True)
    print(f"\n[소스 상위 5개]", flush=True)
    for i, s in enumerate(resp.sources[:5], 1):
        m = s.metadata
        law = m.get("law_name", m.get("source", ""))
        print(f"  [{i}] {law} p.{m.get('page','')} score={m.get('score','')} "
              f"item={str(m.get('item_number',''))[:25]}", flush=True)

print(f"\n{SEP}\n완료", flush=True)
