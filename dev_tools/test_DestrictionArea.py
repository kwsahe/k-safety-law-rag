"""test_DestrictionArea.md Q1~Q5 전체 검증."""
import sys, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

from rag.chatbot import rag_chat, set_scenario
from rag.schemas import AccidentScenario, ChatRequest

set_scenario(AccidentScenario(
    overview="아파트 신축 건설현장(지상 15층, 지하 2층) 지하 2층 굴착 공사구역. 굴착면 높이 약 4미터. 2024년 8월 오전 10시경.",
    details="굴착 작업 중 크레인 인양 작업 병행. 출입금지 표지 설치되어 있었으나 일용직 근로자 A씨가 공구를 가져오기 위해 진입. 크레인 인양 중 철제 자재 낙하, A씨 부상.",
    workers="일용직(당일 처음 현장 투입), 특별안전교육 미이수, 출입금지 표지 인지 후 무시하고 진입."
))

SEP = "=" * 70

queries = [
    ("Q1", "기본 위반 판단 (표 추출 품질 검증)",
     "위 사고에서 사업주가 위반한 법령 조항은 무엇인가? 굴착 작업 관련 특별안전교육 미실시를 중심으로 판단하라."),
    ("Q2", "복합 위반 판단 (다중 조항 교차)",
     "위 사고에서 크레인 인양 작업과 굴착 작업이 동시에 진행되고 있었다. 사업주가 위반했을 가능성이 있는 모든 특별교육 관련 조항을 나열하라."),
    ("Q3", "안전보건표지 설치 의무 위반 판단 (검색 랭킹 검증)",
     "사고 현장에 '출입금지' 표지는 설치되어 있었으나 근로자가 무시하고 진입했다. 표지가 설치되어 있었음에도 사업주에게 추가적인 책임이 있는가?"),
    ("Q4", "행정처분 수위 판단 (처벌 기준 조항 검증)",
     "위 사고에서 특별안전교육 미실시에 대한 행정처분 수위는? 1차·2차·3차 위반 기준으로 구분하여 알려줘."),
    ("Q5", "재발방지 조치 도출 (종합 판단)",
     "위 사고를 바탕으로 사업주가 즉시 취해야 할 재발방지 조치를 법령 의무 기준으로 제시하라. 각 조치의 법적 근거도 함께 명시하라."),
]

for qid, title, question in queries:
    print(f"\n{SEP}", flush=True)
    print(f"[{qid}] {title}", flush=True)
    print(f"질문: {question}", flush=True)
    print(SEP, flush=True)
    resp = rag_chat(ChatRequest(question=question))
    print(f"\n[답변]\n{resp.answer}", flush=True)
    print(f"\n[검색 소스 상위 5개]", flush=True)
    for i, s in enumerate(resp.sources[:5], 1):
        m = s.metadata
        law  = m.get("law_name", m.get("source", ""))
        page = m.get("page", "")
        score = m.get("score", "")
        item = str(m.get("item_number", ""))[:35]
        print(f"  [{i}] {law} p.{page}  score={score}  item={item}", flush=True)

print(f"\n{SEP}\n전체 완료", flush=True)
