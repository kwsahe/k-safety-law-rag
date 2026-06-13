# K-Safety Law RAG

건설현장 사고 시나리오를 입력하면 산업안전보건법과 중대재해처벌법 관련 조항을 검색하고, 법령 위반 여부와 책임 판단을 자연어로 설명하는 CLI 기반 법령 RAG 시스템입니다.

단순 PDF 검색이 아니라, 사고 사실관계에 맞는 법령 조항과 별표/표 기준을 함께 찾고, 근거 페이지와 함께 답변하는 것을 목표로 합니다.

---

## 주요 기능

- 건설현장 사고 시나리오 기반 법령 질의응답
- 산업안전보건법과 중대재해처벌법 동시 검색 및 구분 판단
- 텍스트 법령 RAG와 별표/표 전용 Table RAG 통합 검색
- `BAAI/bge-m3` 임베딩과 ChromaDB 로컬 벡터 DB 사용
- Colab GPU에서 실행한 OpenAI-compatible LLM 호출
- 질문 유형별 direct route로 특별교육, 출입금지 표지, 과태료 등 정형 법령 질의 보강
- 답변 근거 source와 페이지 출력
- 대화형 CLI에서 사고 시나리오 입력, 초기화, 상태 확인 지원
- HTML 챗봇 UI, 사용자별 대화 이력 DB, 관리자/일반 사용자 출력 분리 지원

---

## 왜 Table RAG를 분리했나

산업안전보건법의 핵심 기준은 조문 본문뿐 아니라 별표와 표에 많이 들어 있습니다. 예를 들어 특별안전교육 시간, 교육 대상 작업, 유해물질 노출기준, 과태료 기준은 표 구조 안에 있어 일반 텍스트 청킹만으로는 검색 정확도가 떨어졌습니다.

그래서 이 프로젝트는 법령 PDF 본문과 표 데이터를 분리해 임베딩합니다.

```text
사용자 질문 + 사고 시나리오
        |
        v
텍스트 법령 검색 + 표 법령 검색
        |
        v
통합 검색 결과 정렬
        |
        v
질문 유형 판단
        |-- direct route: 정형 법령 답변
        `-- LLM route: Qwen2.5-14B 원격 LLM 판단
        |
        v
답변 + 법령 근거 출력
```

---

## 프로젝트 구조

```text
K-Safety Law RAG/
├── cli.py                         # 실행 진입점
├── web_app.py                     # HTML 챗봇 UI 서버
├── rag/
│   ├── cli.py                     # 대화형 CLI 본체
│   ├── chatbot.py                 # 프롬프트, 라우팅, LLM 호출, 답변 생성
│   ├── integrated_retriever.py    # 텍스트+표 검색 결과 통합
│   ├── ingest.py                  # 법령 PDF 텍스트 추출 및 임베딩
│   ├── table_extraction.py        # pdfplumber 기반 표 추출
│   ├── table_retriever.py         # 표 데이터 임베딩/검색
│   └── schemas.py                 # 시나리오/채팅 데이터 모델
├── scripts/
│   ├── test_chat.py               # 이전 실행 경로 호환 래퍼
│   ├── test_chat_cpu.py           # CPU 옵션 실행
│   ├── run_ingest.py              # 텍스트 법령 재임베딩 래퍼
│   └── reingest_tables.py         # 표 법령 재임베딩 래퍼
├── scenarios/                     # 사고 시나리오 예시
├── data/laws/                     # 법령 PDF
├── chroma_db/                     # 텍스트 법령 벡터 DB
├── chroma_db_tables/              # 표 법령 벡터 DB
├── web/static/                    # 웹 UI 정적 파일
├── notebooks/                     # Colab LLM 서버 노트북
├── docs/DEVELOPMENT_PROCESS.md    # 개발 과정 상세 기록
├── requirements.txt
└── .env.example
```

---

## 실행 방법

### 1. 가상환경 활성화

```cmd
conda activate p311_ragreport
cd /d "C:\K-Safety Law RAG"
```

### 2. 환경 변수 설정

`.env.example`을 `.env`로 복사한 뒤 Colab ngrok URL을 입력합니다.

```env
LLM_PROVIDER=remote_openai
LLM_MODEL=Qwen/Qwen2.5-14B-Instruct
LLM_API_BASE=https://YOUR_NGROK_URL/v1
LLM_API_KEY=dummy
```

### 3. Colab LLM 서버 실행

권장 노트북:

```text
notebooks/RAG_qwen25_14b_colab_pro_server.ipynb
```

Colab 서버가 뜨면 출력된 `/v1` 주소를 `.env`의 `LLM_API_BASE`에 넣습니다. 로컬에는 RAG와 ChromaDB만 두고, 무거운 LLM은 Colab GPU에서 OpenAI-compatible API처럼 호출합니다.

### 4. CLI 실행

```cmd
python cli.py
```

시나리오 파일을 지정해 실행할 수도 있습니다.

```cmd
python cli.py --scenario-file scenarios\default_accident.py
```

기존 경로 호환을 위해 아래 명령도 동작하지만, 앞으로는 `python cli.py` 사용을 권장합니다.

```cmd
python scripts\test_chat.py
```

### 5. 웹 UI 실행

```cmd
python web_app.py --host 127.0.0.1 --port 8000
```

브라우저에서 `http://127.0.0.1:8000`을 엽니다.

첫 실행 시 관리자 계정이 자동 생성됩니다.

```text
admin / admin1234
```

웹 UI는 `data/chatbot_ui.sqlite3`에 계정, 세션, 시나리오, 상담 대화 이력을 저장합니다. 관리자 계정은 CLI의 참고 근거, score, 응답시간을 포함한 전체 출력을 볼 수 있고, 일반 계정은 score와 응답시간 등 내부 진단 정보를 숨긴 답변을 봅니다. 상담과 채팅 삭제는 화면에서 숨김 처리되며, 삭제 시점의 원본 스냅샷은 DB의 `deletion_logs`에 보존됩니다.

---

## CLI 명령어

| 명령어 | 설명 |
|---|---|
| `/시나리오` | 사고 시나리오 입력 또는 수정 |
| `/초기화` | 저장된 시나리오 제거 |
| `/상태` | 현재 시나리오 확인 |
| `exit` | 종료 |

---

## 법령 DB 재생성

벡터 DB가 포함되어 있으면 바로 실행할 수 있습니다. 법령 PDF를 바꾸거나 추출 방식을 바꾼 경우에만 재임베딩합니다.

```cmd
python -m rag.ingest --reset
python -m rag.table_retriever --ingest --reset --strategy row
```

래퍼 스크립트:

```cmd
python scripts\run_ingest.py --reset
python scripts\reingest_tables.py
```

---

## 검증 명령

```cmd
python -m compileall cli.py rag scripts scenarios
```

가상환경 기준 확인:

```cmd
conda run -n p311_ragreport python cli.py --help
```

---

## 예시 질문

```text
이 사고에서 산업안전보건법과 중대재해처벌법이 동시에 적용되는가?
각 법령별 위반 조항과 처벌 주체를 구분해서 알려줘.
```

```text
비계 작업 특별안전교육 미실시는 어떤 법령 조항 위반인가?
```

```text
출입금지 표지가 있었는데 근로자가 무단 진입했다면 사업주 책임은 면제되는가?
```

---

## 개발 과정에서 해결한 핵심 문제

- 텍스트 RAG만으로는 별표/표 질의가 약해 Table RAG를 별도로 구성했습니다.
- LLM이 검색 결과에 없는 조항이나 금액을 기억으로 출력하는 문제를 줄이기 위해 시스템 프롬프트와 질문 유형별 direct route를 추가했습니다.
- 사고 시나리오 단서가 검색 query에 빠지지 않도록 사용자 질문과 시나리오 텍스트를 함께 retrieval query로 사용합니다.
- 긴 검색 context에서 LLM이 원문을 그대로 출력하는 문제를 줄이기 위해 LLM에 넘기는 source 수를 제한했습니다.
- 산업안전보건법과 중대재해처벌법을 동시에 검색하되, 답변에서는 법령별 의무와 책임 주체를 분리하도록 조정했습니다.

---

## 현재 한계

- PDF 추출 품질에 영향을 받습니다. 병합 셀, 줄바꿈, 페이지 경계가 있는 별표/표는 추가 보정이 필요할 수 있습니다.
- direct route가 늘어나면 유지보수 비용이 커질 수 있습니다.
- 긴 사고 시나리오와 많은 검색 결과를 함께 넘기면 LLM 응답 안정성이 떨어질 수 있어 context 제한과 후처리가 필요합니다.
- 영상 자체를 직접 분석하지 않습니다. 영상 분석 결과나 사람이 작성한 사고 시나리오를 입력으로 사용합니다.
- 법률 판단 보조 도구이며, 최종 법률 판단을 자동화하는 시스템은 아닙니다.

---

## 포트폴리오 요약

법령 PDF와 표 데이터를 함께 임베딩하고, 사고 시나리오에 맞는 조항을 RAG로 검색한 뒤 LLM이 위반 여부와 책임 주체를 판단하는 CLI 기반 산업안전 법령 RAG 시스템입니다. 텍스트 RAG만으로 부족했던 별표/표 검색 문제를 Table RAG로 보완했고, 질문 유형별 라우팅과 프롬프트 제약으로 법령 조항 혼동을 줄였습니다.
