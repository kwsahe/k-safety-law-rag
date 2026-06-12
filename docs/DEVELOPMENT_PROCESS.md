# K-Safety Law RAG 개발 과정 정리

## 1. 프로젝트 목표

`K-Safety Law RAG`는 건설현장 사고 시나리오를 입력하면 산업안전보건법과 중대재해처벌법 관련 조항을 검색하고, 법령 위반 여부와 책임 판단을 자연어로 설명하는 법령 RAG 시스템이다.

초기 목표는 단순한 PDF 검색이 아니었다. 실제 사고 시나리오에서 다음 질문에 답하는 것이 목표였다.

- 어떤 법령이 적용되는가?
- 어떤 조항을 위반했는가?
- 사업주, 도급인, 근로자 책임은 어떻게 구분되는가?
- 특별안전교육, 출입통제, 위험성평가, 안전보건관리체계 같은 의무가 사고와 어떻게 연결되는가?
- 답변에 실제 법령 근거와 페이지를 함께 제시할 수 있는가?

최종적으로는 대화형 CLI에서 사고 시나리오를 입력하고 법령 근거 기반 답변을 받을 수 있는 구조로 정리했다.

---

## 2. 전체 아키텍처

현재 구조는 다음과 같다.

```text
사용자 질문 + 사고 시나리오
        ↓
텍스트 법령 검색 + 표 법령 검색
        ↓
통합 검색 결과 정렬
        ↓
질문 유형 라우팅
        ├─ 결정형 direct route
        └─ LLM 판단 route
        ↓
Qwen2.5-14B 원격 LLM 응답
        ↓
답변 + 참고 근거 출력
```

핵심 구성 요소:

| 구성 | 역할 |
|------|------|
| `rag/ingest.py` | 법령 PDF 텍스트 추출 및 임베딩 |
| `rag/table_extraction.py` | `pdfplumber` 기반 표 추출 |
| `rag/table_retriever.py` | 표 데이터 임베딩/검색 |
| `rag/integrated_retriever.py` | 텍스트+표 검색 결과 통합 |
| `rag/chatbot.py` | 프롬프트 구성, 라우팅, LLM 호출, 결정형 답변 |
| `rag/cli.py` | 대화형 CLI 본체 |
| `cli.py` | 개인 프로젝트용 실행 진입점 |
| `scripts/test_chat.py` | 이전 실행 경로 호환 래퍼 |

---

## 3. 개발 초기 문제

### 3.1 텍스트 RAG만으로는 별표/표 질의가 약했다

산업안전보건법 시행규칙에는 중요한 기준이 표 형태로 들어 있다. 예를 들어 노출기준, 특별교육 대상 작업, 과태료 기준, 출입금지 표지 기준 등이 표로 되어 있다.

초기에는 PDF 텍스트만 청킹해서 임베딩했기 때문에 다음 문제가 있었다.

- 병합 셀이 깨짐
- 표의 행과 열 관계가 무너짐
- 수치 기준이 같은 행의 물질명과 연결되지 않음
- `TWA`, `STEL`, `mg/㎥`, `ppm` 같은 열 기준을 LLM이 혼동

예시 실패:

```text
질문: 카드뮴(Cadmium) 및 그 화합물의 mg/㎥ 허용기준은?
정답: 0.01 mg/㎥, 호흡성 분진은 0.002 mg/㎥
실패: 제공된 근거에서 확인할 수 없음
```

해결:

- `pdfplumber`로 표를 별도 추출
- row 단위 청킹 도입
- 표 전용 ChromaDB(`chroma_db_tables`) 분리
- 텍스트 DB와 표 DB를 통합 검색

현재 청크 수:

```text
텍스트 청크: 335개
표 청크: 940개
총 청크: 1,275개
```

---

## 4. 프롬프트 조정 과정

### 4.1 기본 시스템 프롬프트

초기 프롬프트는 “검색 결과를 보고 답하라” 수준이었다. 이 방식은 법령 RAG에서 부족했다. LLM이 익숙한 조항을 기억으로 끌어오거나, 검색 결과에 없는 금액을 만들어내는 일이 있었다.

그래서 시스템 프롬프트에 다음 규칙을 추가했다.

```text
1. 검색 결과 안의 내용만 근거로 사용할 것.
2. 조항 번호를 스스로 생성하지 말 것.
3. 금액은 검색 결과에 명시된 경우에만 인용할 것.
4. PRIMARY 근거를 우선 사용할 것.
5. 검색 순위가 낮아도 질문 키워드와 직접 관련된 근거는 보조로 활용할 것.
```

이후 법령 판단의 기본 안정성이 올라갔다.

### 4.2 표 질의용 프롬프트

노출기준 질문에서는 단위와 열 해석이 중요했다.

추가한 규칙:

```text
- TWA는 시간가중평균값이다.
- STEL은 단시간 노출값이다.
- ppm 기준을 물으면 ppm 값을 답한다.
- 같은 행에 STEL 값이 있으면 함께 답한다.
```

효과:

- 벤젠 TWA 질문에서 `0.5 ppm`, STEL `2.5 ppm`을 함께 답하도록 개선
- 단위 혼동이 줄어듦

### 4.3 위반 판단용 프롬프트

사고 시나리오 질문에서는 답변 형식이 흔들렸다. 그래서 위반 판단용 형식을 고정했다.

```text
위반 여부: YES / NO / 판단불가

[위반 조항]
- 법령명, 별표/조항, 페이지
- 해당 이유

[관련 교육 내용 / 조치 기준]
○ 검색된 교육내용 항목
```

효과:

- 특별교육 대상 작업의 조항과 이유를 분리 출력
- LLM이 시행령 제29조 같은 일반 의무 조항만 반복하는 문제 감소

### 4.4 중대재해처벌법용 프롬프트

중대재해처벌법 질문에서 산업안전보건법 별표가 섞여 나오는 문제가 있었다. 그래서 중대재해처벌법 전용 규칙을 추가했다.

```text
- 중대산업재해 해당 여부는 중대재해처벌법 제2조제2호 및 제3조를 먼저 검토
- 경영책임자 의무는 제4조 및 시행령 제4조를 검토
- 도급 책임은 제5조의 실질적 지배ㆍ운영ㆍ관리 요건을 명시
- 처벌은 제6조, 제7조, 제15조를 구분
```

효과:

- 사망 사고와 부상 사고의 적용 기준을 구분
- 5명 미만 사업장 제외 조건을 별도 판단
- 산업안전보건법과 중대재해처벌법의 책임 주체를 구분

---

## 5. 라우팅 개선 과정

LLM 프롬프트만으로는 부족한 부분이 있었다. 특히 법령 문항은 답이 정해진 구조라, LLM에게 전부 맡기면 흔들렸다. 그래서 질문 유형별 direct route를 만들었다.

### 5.1 특별교육 라우팅

문제:

```text
질문: 굴착 작업 관련 특별교육 의무는?
실패: 별표5 제19호 외에 제27호, 제39호 등이 섞임
```

원인:

- 사고 시나리오에 크레인, 굴착, 골조 같은 단서가 함께 있으면 LLM이 여러 작업항목을 섞음

해결:

- 질문이 특정 작업 하나를 물으면 해당 작업 라우팅으로 제한
- 굴착은 별표5 제19호
- 크레인은 별표5 제14호
- 비계 조립ㆍ해체ㆍ변경은 별표5 제23호로 보정

특히 비계 작업은 초기 구현에서 `제26호`로 잘못 매핑되었다. 이후 다음을 수정했다.

- `test_scenario_prompt.py` 계열 보강 로직
- `rag/chatbot.py`의 비계 direct route
- `rag/integrated_retriever.py`의 보강 라벨
- 보고서 payload 샘플과 문서

최종 기준:

```text
산업안전보건법 시행규칙 별표 5 제23호
비계의 조립ㆍ해체 또는 변경 작업
```

### 5.2 출입금지 표지 라우팅

문제:

```text
질문: 출입금지 표지가 있었는데 근로자가 무시하고 진입했다. 사업주 책임이 있는가?
실패: 특별교육 조항 목록만 출력
```

원인:

- “출입금지” 질문이 특별교육 질문으로 잘못 라우팅됨
- LLM 호출을 우회하고 기존 특별교육 direct answer가 나옴

해결:

- 표지/출입금지/안전보건표지 키워드를 별도 라우팅
- 별표 6 제1호를 보조 근거로 포함
- “표지 설치 = 면책 아님”을 명시

개선 후:

```text
표지 설치는 일부 의무 이행 사정이지만,
물리적 차단, 교육, 감독, 위험구역 통제가 미흡하면 추가 책임 가능성이 있음.
```

### 5.3 과태료/행정처분 라우팅

문제:

```text
질문: 특별안전교육 미실시 과태료 1차·2차·3차는?
실패: 시행령 별표35가 검색되지 않았는데 LLM이 금액을 기억으로 출력하고, 근거 페이지를 틀림
```

해결:

- 시행규칙 별표26과 시행령 별표35의 역할을 분리
- 별표35 청크가 검색되면 50/100/150만원으로 해석
- 근거가 없으면 금액을 만들지 않도록 제한

결과:

```text
1차: 50만원
2차: 100만원
3차 이상: 150만원
기준: 교육대상 근로자 1명당
```

### 5.4 중대재해처벌법 동시 검색 라우팅

문제:

- 산업안전보건법만 출력하고 중대재해처벌법을 누락
- 중대재해처벌법만 출력하고 산업안전보건법을 누락
- 질문별로 이전 답변이 재출력되는 것처럼 보이는 라우팅 오류

해결:

- 법령 구분 질문은 항상 두 법령을 모두 검토
- 적용되지 않는 경우도 “미적용 사유” 출력
- 부상자 수, 치료 기간, 사망 여부, 근로자 수 조건을 직접 판단

대표 개선:

```text
부상자 1명, 8개월 치료 → 중대재해처벌법상 중대산업재해 아님
부상자 2명, 6개월 이상 치료 → 중대산업재해 해당
상시 근로자 5명 미만 → 중대재해처벌법 적용 제외
```

---

## 6. 모델 비교와 선택

테스트 중 다음 모델을 비교했다.

### 6.1 EXAONE 3.5 7.8B

장점:

- 한국어 문장이 자연스러움
- 긴 종합 판단에서 비교적 안정적

단점:

- 짧은 법령 특정 질문에서 조항을 과하게 넓힘
- “추정됩니다” 같은 불확신 표현이 자주 붙음
- 응답 시간이 길어짐

### 6.2 Qwen2.5 14B

장점:

- 조항 특정이 더 정확한 경우가 많음
- Q1/Q4 같은 단순 법령 판단에서 안정적
- 과태료 금액을 더 확신 있게 출력

단점:

- 긴 context에서 검색 원문을 그대로 출력하는 문제가 발생
- 12page 종합평가처럼 긴 프롬프트에서는 가끔 프롬프트 구조가 무너짐

최종 선택:

```text
Qwen/Qwen2.5-14B-Instruct
```

선택 이유:

- CLI 질의응답에서는 조항 특정 정확도가 중요
- 긴 보고서 문단 생성은 프롬프트와 context 축소로 보완 가능
- Colab Pro에서 실행 가능

---

## 7. 테스트하며 발견한 대표 실패와 해결

### 7.1 Hybrid search score 정규화 문제

증상:

```text
score=1.07
score=3.0
```

cosine similarity는 일반적으로 1.0을 넘지 않는데, BM25 보정 후 score가 과도하게 커졌다.

영향:

- 관련 없는 교육시간표가 상위에 올라옴
- 별표5 교육내용 대신 별표4 교육시간을 LLM이 사용

해결:

- score clamp 적용
- 표/텍스트 결과의 상한 관리
- 질문과 관련 없는 교육시간표 필터링

### 7.2 LLM이 상위 근거를 무시하고 하위 근거 사용

증상:

- 별표5 p.82가 1~3위인데 LLM이 시행령 제29조를 근거로 답함

해결:

- `[근거 우선순위]` 섹션 추가
- PRIMARY/BACKGROUND 구분
- “낮은 순위의 익숙한 조항을 쓰지 말라”는 규칙 추가

### 7.3 12page 종합평가에서 검색 원문 출력

증상:

```text
[PRIMARY TEXT RANK 1 ...]
제2조(정의) ...
```

LLM이 답변 대신 검색 context를 그대로 출력했다.

해결:

- 12page 생성 시 전체 RAG context를 다시 넘기지 않음
- 정리된 위반사항 + 시나리오만 넘김
- `[검색 결과]`, `[PRIMARY]` 같은 문자열 후처리
- 짧은 fallback 프롬프트 추가

### 7.4 비계 특별교육 제26호 오류

증상:

```text
산업안전보건법 시행규칙 별표 5 제26호
```

정답:

```text
산업안전보건법 시행규칙 별표 5 제23호
```

해결:

- 비계 조립ㆍ해체ㆍ변경 작업을 제23호로 정규화
- source metadata나 예전 청크가 제26호를 갖고 있어도 출력 단계에서 제23호로 보정
- 문서/샘플/검색 보강 라벨 모두 수정

---

## 8. 현재 부족한 점

### 8.1 법령 DB 원문 품질 의존성

PDF 추출 품질이 좋지 않으면 RAG가 틀린 근거를 올린다. 특히 별표/표는 병합 셀, 줄바꿈, 페이지 경계에 취약하다.

개선 방향:

- 표별 golden QA 추가
- 별표별 수동 검증 CSV 생성
- 청크 단위에 `annex`, `item_no`, `row_header` 같은 metadata 보강

### 8.2 direct route가 많아지고 있음

정확도를 올리기 위해 direct route를 많이 넣었다. 이 방식은 효과적이지만, 장기적으로는 유지보수 부담이 생긴다.

개선 방향:

- 라우팅 규칙을 코드에서 YAML/JSON으로 분리
- 작업항목 번호 매핑 테이블화
- 질문 유형 classifier 분리

### 8.3 LLM context 안정성

Qwen2.5-14B는 짧은 판단에는 강하지만 긴 context에서 원문을 출력하는 현상이 있었다.

개선 방향:

- context 최대 청크 수 제한
- 근거 summary를 먼저 생성한 뒤 최종 답변 생성
- 최종 답변 전용 prompt를 더 짧게 유지

### 8.4 영상 자체는 읽지 않음

`VIDEO_FILE`은 문자열 메타데이터일 뿐, 현재 Qwen은 영상을 직접 읽지 않는다.

현재 구조:

```text
영상 분석 결과 또는 사람이 작성한 사고 시나리오
        ↓
SCENARIO 텍스트
        ↓
RAG/LLM 판단
```

개선 방향:

- 영상팀/CV 모델의 탐지 결과 JSON을 받아 SCENARIO에 자동 반영
- 프레임 캡션, 객체 탐지, 위험구역 침범 이벤트를 text evidence로 변환

### 8.5 법률 판단의 최종성

이 시스템은 법률 검토 보조 도구다. 최종 법적 판단은 실제 수사자료, 현장조사, 계약관계, 교육기록, 작업지시서 등을 함께 봐야 한다.

---

## 9. 현재 완성도 점수

개인 프로젝트 기준으로 평가하면 다음과 같다.

| 항목 | 점수 | 평가 |
|------|-----:|------|
| RAG 검색 구조 | 88/100 | 텍스트+표 통합 검색은 안정적이나 PDF 추출 품질 의존성이 있음 |
| 법령 조항 특정 | 85/100 | 주요 시나리오는 잘 맞지만 일부 별표 번호는 보정 로직 필요 |
| 프롬프트 안정성 | 80/100 | 짧은 질의는 안정적, 긴 context에서는 후처리 필요 |
| CLI 사용성 | 82/100 | 대화형 사용 가능, UI는 단순함 |
| 포트폴리오 완성도 | 90/100 | 구조와 문제 해결 과정 설명이 명확함 |
| 실무 적용 가능성 | 75/100 | 보조 도구로는 가능, 최종 판단 자동화는 추가 검증 필요 |

종합 점수:

```text
85 / 100
```

이 점수의 의미:

- 개인 포트폴리오 프로젝트로는 충분히 강함
- 법령 RAG의 어려운 부분인 표 추출, 법령 구분, 사고 시나리오 판단을 실제로 다뤘음
- 다만 실무 서비스 수준으로 가려면 더 많은 golden test, 법령 metadata 정제, 라우팅 설정화가 필요함

---

## 10. 핵심 코드 수정 내용

아래는 개발하면서 실제로 손댄 핵심 코드 흐름이다. 전체 코드를 그대로 나열하기보다, 문제를 해결한 부분 중심으로 정리했다.

### 10.1 ChatRequest에 direct answer 옵션 추가

문제:

`rag_chat()`은 질문이 특정 라우팅에 걸리면 LLM을 호출하지 않고 결정형 답변을 반환했다. 일반 CLI에서는 유용하지만, “LLM이 실제로 판단했는지” 검증해야 하는 보고서용 테스트에서는 문제가 됐다.

해결:

`ChatRequest`에 `use_direct_answers` 옵션을 추가했다.

```python
# rag/schemas.py
class ChatRequest(BaseModel):
    """사용자의 질문을 받는 데이터 규격"""
    question: str
    scenario: Optional[AccidentScenario] = None
    use_direct_answers: bool = True
```

그리고 `rag_chat()`에서 이 값이 `False`면 direct route를 타지 않게 했다.

```python
# rag/chatbot.py
if request.use_direct_answers:
    direct_answer = direct_answer_from_sources(
        request.question,
        sources,
        retrieval_query,
    )
    if direct_answer:
        return ChatResponse(
            answer=direct_answer,
            sources=direct_answer_sources(
                request.question,
                sources,
                retrieval_query,
            ),
        )
```

효과:

- 일반 CLI는 기존처럼 빠른 결정형 답변 사용
- 보고서 검증 스크립트는 LLM 판단을 강제 가능

---

### 10.2 시나리오 기반 retrieval query 구성

문제:

사용자 질문만 검색하면 사고 시나리오의 단서가 검색에 반영되지 않았다. 예를 들어 질문은 “책임 여부는?”인데, 실제 검색에는 “굴착”, “크레인”, “특별안전교육 미실시” 같은 사고 단서가 빠질 수 있었다.

해결:

질문과 시나리오 텍스트를 합쳐 검색 query로 사용했다.

```python
# rag/chatbot.py
def build_retrieval_query(
    question: str,
    scenario: AccidentScenario | None,
) -> str:
    """Combine user question with stored accident facts for retrieval only."""
    if not scenario:
        return question

    scenario_text = format_scenario(scenario)
    if not scenario_text:
        return question

    return f"{question}\n\n{scenario_text}"
```

효과:

- 질문이 짧아도 사고 사실관계가 검색에 반영됨
- “책임 여부” 같은 추상 질문에서도 사고 유형 관련 법령이 검색됨

---

### 10.3 context 청크 수 제한

문제:

LLM에 너무 많은 검색 결과를 넘기면 Qwen이 프롬프트 구조를 잃고 `[검색 결과]` 원문을 그대로 출력하는 문제가 있었다.

해결:

LLM context에 들어가는 source 수를 제한했다.

```python
# rag/chatbot.py
MAX_CONTEXT_SOURCES = 10

def select_context_sources(
    sources: list[SourceDoc],
    question: str = "",
) -> list[SourceDoc]:
    """Keep the LLM context compact while preserving ranked source order."""
    ...
    selected = table_docs[:RAG_CONTEXT_TABLE_K] + text_docs[:RAG_CONTEXT_TEXT_K]
    ...
    return selected[:MAX_CONTEXT_SOURCES]
```

효과:

- context 과부하 감소
- 응답 시간이 줄어듦
- LLM이 검색 원문을 그대로 출력하는 현상이 줄어듦

---

### 10.4 질문 관련 보조 근거 추가

문제:

출입금지 표지 질문에서 별표6이 검색되더라도 4~6위로 밀려 LLM이 무시하는 일이 있었다.

해결:

질문 키워드에 따라 하위 검색 결과라도 직접 관련된 근거를 context에 추가했다.

```python
# rag/chatbot.py
def find_question_relevant_supplemental_sources(
    question: str,
    sources: list[SourceDoc],
) -> list[SourceDoc]:
    compact_question = re.sub(r"\s+", "", question)
    supplemental: list[SourceDoc] = []

    if any(term in compact_question for term in (
        "출입금지",
        "표지",
        "표지판",
        "금지표지",
        "안전보건표지",
    )):
        for source in sources[:8]:
            compact_content = re.sub(r"\s+", "", source.content)
            page = str(source.metadata.get("page", ""))
            if (
                "출입금지" in compact_content
                or "금지표지" in compact_content
                or "안전보건표지" in compact_content
                or "별표6" in compact_content
                or page in {"94", "95", "96"}
            ):
                supplemental.append(source)
                if len(supplemental) >= 2:
                    break

    return supplemental
```

효과:

- 표지/출입금지 질문에서 별표6 근거가 답변에 반영됨
- “표지 설치 = 면책”으로 오판하는 문제 감소

---

### 10.5 비계 특별교육 제23호 정규화

문제:

초기 청크나 라우팅에서 비계 작업이 `별표 5 제26호`로 출력되는 문제가 있었다.

정답:

```text
산업안전보건법 시행규칙 별표 5 제23호
비계의 조립ㆍ해체 또는 변경 작업
```

해결:

비계 조립ㆍ해체ㆍ변경 작업이면 source metadata가 무엇이든 출력 번호를 `23`으로 정규화했다.

```python
# rag/chatbot.py
def extract_item_number(source: SourceDoc) -> str:
    compact = re.sub(r"\s+", "", source.content)
    if "비계의조립·해체또는변경작업" in compact:
        return "23"

    item_number = str(source.metadata.get("item_number") or "")
    match = re.match(r"\s*(\d+)\.", item_number)
    if match:
        return match.group(1)

    match = re.search(r"\[작업항목\]\s*(\d+)\.", source.content)
    return match.group(1) if match else ""
```

그리고 비계 direct answer의 기본값도 `23`으로 수정했다.

```python
# rag/chatbot.py
item_no = extract_item_number(source) or "23"
```

검색 보강 라벨도 수정했다.

```python
# rag/integrated_retriever.py
(
    "osha_scaffold_training",
    0.96,
    "비계 조립ㆍ해체 특별교육",
    "",
    "별표 5 제23호",
    ("비계의조립", "추락재해방지"),
)
```

효과:

- 예전 청크가 잘못된 번호를 갖고 있어도 출력 단계에서 보정
- 보고서와 CLI 답변 모두 제23호로 통일

---

### 10.6 굴착/크레인 특별교육 누락 보강

문제:

사고 시나리오에 “굴착면 4m”, “크레인 인양”, “특별안전교육 미실시”가 명시되어 있어도 별표5 제19호와 제14호가 출력되지 않는 경우가 있었다.

해결:

보고서용 runner에서는 시나리오 텍스트에서 확정 가능한 위반사항을 보강했다.

```python
# report_scenario_runner.py
if (
    "특별안전교육" in compact
    and "미실시" in compact
    and any(term in compact for term in ("크레인", "인양", "양중"))
):
    add(
        "산업안전보건법 시행규칙 별표 5 제14호",
        "크레인 인양 작업 특별안전교육 미실시",
    )

if (
    "특별안전교육" in compact
    and "미실시" in compact
    and any(term in compact for term in ("굴착", "지반굴착", "굴착면"))
):
    add(
        "산업안전보건법 시행규칙 별표 5 제19호",
        "굴착면 높이 2미터 이상 지반 굴착 작업 특별안전교육 미실시",
    )
```

효과:

- 시나리오에 명확히 적힌 사실관계는 검색 실패와 무관하게 보고서 위반사항에 포함
- 굴착+크레인 복합 사고에서 핵심 특별교육 조항 누락 감소

---

### 10.7 12page 종합평가 context 축소

문제:

12page 종합평가를 생성할 때 전체 RAG context를 다시 넘기면 Qwen이 검색 원문을 출력하는 경우가 있었다.

해결:

12page 생성에는 전체 검색 결과를 넘기지 않고, 정리된 위반사항과 시나리오만 넘겼다.

```python
# report_scenario_runner.py
violations = "\n".join(
    f"- {item['law_item']} - {item['violation']}"
    for item in violation_items
)

messages = [
    {
        "role": "system",
        "content": (
            "당신은 산업안전 사고 보고서의 평가 책임자입니다. "
            "출력은 한국어 한 문단만 작성합니다. "
            "검색 결과 원문, 프롬프트, 목록, 마크다운 제목을 절대 출력하지 않습니다. "
            "법 조항 목록을 반복하지 말고 사고 원인과 책임 판단을 종합 의견으로 연결합니다."
        ),
    },
    {
        "role": "user",
        "content": (
            "/no_think\n\n"
            f"[사고 시나리오]\n{scenario_text(scenario)}\n\n"
            f"[확인된 주요 위반사항]\n{violations}\n\n"
            "[요청]\n보고서 12페이지의 '평가 책임자 종합 의견'을 작성하라."
        ),
    },
]
```

그리고 검색 원문이 섞이면 잘라냈다.

```python
# report_scenario_runner.py
def clean_final_evaluation(text: str) -> str:
    cleaned = strip_thinking(text).strip()
    for marker in (
        "[검색 결과]",
        "[PRIMARY",
        "[표 검색 결과]",
        "[텍스트 법령 검색 결과]",
        "[근거 우선순위]",
    ):
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0].strip()
    ...
    return cleaned
```

효과:

- 12page가 7page 위반 목록을 반복하지 않고 종합 의견 문단으로 생성됨
- Qwen의 context echo 문제 완화

---

### 10.8 Colab LLM 호출 구조

원격 LLM은 OpenAI-compatible API 형식으로 호출했다.

```python
# rag/chatbot.py
def make_remote_openai_payload(
    messages: list[dict[str, str]],
    options: dict,
    *,
    stream: bool,
) -> dict:
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": options.get("temperature", 0.1),
        "stream": stream,
    }
    if "num_predict" in options:
        payload["max_tokens"] = options["num_predict"]
    return payload
```

```python
# rag/chatbot.py
def post_remote_openai(payload: dict) -> dict:
    req = request.Request(
        remote_openai_url(),
        data=json.dumps(payload).encode("utf-8"),
        headers=remote_openai_headers(),
        method="POST",
    )
    with request.urlopen(req, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))
```

효과:

- 로컬에는 RAG/ChromaDB만 두고, 무거운 LLM은 Colab GPU에서 실행
- VSCode/로컬 CLI에서 Colab 모델을 API처럼 호출 가능

---

### 10.9 개인 프로젝트용 진입점 분리

팀 프로젝트에서 쓰던 대화형 CLI 본체를 `rag/cli.py`로 옮기고, 개인 프로젝트에서 바로 실행하기 쉽게 `cli.py`를 얇은 진입점으로 유지했다.

```python
# cli.py
"""Entry point for K-Safety Law RAG."""

from rag.cli import main


if __name__ == "__main__":
    main()
```

효과:

- 개인 프로젝트에서는 `python cli.py`만 기억하면 됨
- 기존 구현은 유지하면서 진입점만 깔끔하게 분리

---

## 11. 면접/포트폴리오에서 설명할 핵심 문장

```text
처음에는 법령 PDF를 단순 임베딩했지만, 산업안전보건법의 핵심 기준이 별표와 표에 많아 텍스트 RAG만으로는 정확도가 부족했습니다. 그래서 pdfplumber 기반 Table RAG를 별도로 만들고, 텍스트 검색과 표 검색을 통합했습니다. 이후 LLM이 조항을 혼동하는 문제를 해결하기 위해 질문 유형별 라우팅과 시스템 프롬프트 제약을 추가했고, 사고 시나리오에 대해 산업안전보건법과 중대재해처벌법을 구분해 판단하도록 개선했습니다.
```

짧은 버전:

```text
법령 PDF와 표 데이터를 함께 임베딩하고, 사고 시나리오에 맞는 조항을 RAG로 검색한 뒤 LLM이 위반 여부와 책임 주체를 판단하는 CLI 기반 산업안전 법령 RAG 시스템입니다.
```
