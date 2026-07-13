# Devlog

## 2026-07-13

### Question Graph / Quality Guard
- `QuestionScopeNode` 뒤에 `IntentClassifierNode`, `RetrievalPlanNode`, `CacheGuardNode`를 추가했다.
- 질문 전체 텍스트, 모드, 범위, 의도를 포함한 해시를 생성해 유사 키워드 질문이 같은 캐시 키로 충돌하지 않도록 했다.
- 관리자 CLI 출력에 질문 범위, 의도, 실행 경로, 캐시 키, 실행 노드를 표시하도록 했다.
- 답변의 법령 조항과 참고 근거를 대조하는 Citation Validator를 연결하고 반복 출처 혼동을 탐지하도록 했다.

### Admin Quality / Feedback
- 관리자 전용 품질 대시보드를 추가했다.
- 사용자 수, 활성 상담, 저장 답변, 평균 응답시간, 출처 검증 상태, 질문 의도 분포, 최근 사용자 평가를 확인할 수 있다.
- EXAONE API, SQLite DB, Text/Table Vector DB 상태를 확인하는 운영 진단 API를 추가했다.
- 일반 사용자와 관리자 모두 답변에 `도움됨` 또는 `개선 필요` 평가를 남길 수 있도록 `answer_feedback` 테이블과 API를 추가했다.
- 일반 계정에서는 CLI, score, 모델명, 응답시간, Citation Validator, 질문 그래프 진단을 계속 숨긴다.

### Runtime / Frontend QA
- 정형 직접 답변에서 BGE-M3를 불필요하게 불러오지 않도록 통합 검색 모듈을 지연 로드한다.
- 관리자 품질 패널과 채팅 본문을 불투명 흰색 작업면으로 정리하고 모바일 입력창 스크롤바를 숨겼다.
- Q1-A, Q1-B, Q2-A, Q2-B, Q2-C, Q3, 일반 교육 비교 회귀 테스트 7건을 모두 통과했다.
- 1440x900 데스크톱과 390x844 모바일에서 가로 넘침, 사이드바, 입력창, 관리자 패널을 확인했다.
- 최종 화면은 `screenshots/final/`에 저장했다.

## 2026-06-13

### Web UI
- Figma 참고 스타일 기반의 챗봇 UI를 파스텔 블루/화이트 글래스 톤으로 재구성했다.
- 상담 목록에 `...` 메뉴를 추가하고 `이름 수정`, `채팅 삭제` 기능을 통합했다.
- 채팅 삭제는 화면에서만 숨기는 soft delete로 처리하고, DB의 `deletion_logs`에 삭제 시점 스냅샷을 남기도록 했다.
- 개별 말풍선 삭제 버튼은 제거하고, 메시지에는 `입력 시간`/`출력 시간`과 `복사` 버튼만 표시하도록 정리했다.
- 웹 UI의 별도 법령 참조 JSON 저장을 제거하고 SQLite DB payload를 기준 저장소로 정리했다.
- 모델 연결 실패 시 SweetAlert 알림을 띄우도록 했다.

### RAG Routing
- EXAONE 7.8B 대응을 위해 비계 특별안전교육 질문을 별표 5 제23호로 직접 라우팅하도록 추가했다.
- 비계 특별교육 청크가 검색 결과에 없을 때도 p.83 근거를 fallback으로 구성하도록 했다.
- 과도한 비계 라우팅을 수정해 `특별교육/특별안전교육/교육내용/미실시/미이수` 의도가 있을 때만 제23호 라우팅이 작동하도록 좁혔다.
- 보호구 미착용 및 비계 설치 기준 위반 질문은 별표 5 제23호가 아니라 제32조, 제42조, 제56조~제62조, 제14조 항목으로 직접 답변하도록 분리했다.

### EXAONE Notebook
- Colab에서 Google Drive에 업로드한 EXAONE-3.5-7.8B-Instruct 모델 경로를 사용할 수 있도록 notebook을 보완했다.
- Transformers/EXAONE `create_causal_mask` 호환 패치와 `attention_mask` 처리, 직접 모델 테스트 셀을 정리했다.

### Verification
- `python -m compileall web_app.py`
- `python -m compileall rag\chatbot.py rag\integrated_retriever.py rag\table_retriever.py`
- `node --check web\static\app.js`

## 2026-06-15

### Web UI
- 전체 프론트 구조를 ChatGPT형 레이아웃으로 재작성했습니다.
  - 좌측 사이드바, 모드별 대화 목록, 중앙 채팅 스트림, 하단 고정 입력창 구조로 정리했습니다.
  - 기존 파스텔 블루/네이비 색상 스타일은 유지했습니다.
- 진입 화면을 추가했습니다.
  - `localhost:8200` 진입 시 챗봇 소개 화면이 먼저 표시됩니다.
  - 자동 타이머 대신 `다음` 버튼으로 로그인 화면으로 이동하도록 변경했습니다.
- 새 상담 화면을 개선했습니다.
  - `채팅을 시작해보세요!` 안내 문구를 표시합니다.
  - 시나리오 모드 / 일반 모드 설명과 예시 질문 버튼을 제공합니다.
- 채팅 목록을 `시나리오 모드` / `일반 모드` 섹션으로 분리했습니다.
- 관리자/일반 계정 출력 정책을 명확히 했습니다.
  - 관리자 계정은 답변 아래 CLI 전체 출력, 모델명, 응답시간, 상세 근거를 확인합니다.
  - 일반 계정은 CLI/score/raw source 없이 답변만 확인합니다.
- 기본 실행 포트를 `8200`으로 통일했습니다.

### RAG Routing
- `QuestionScopeNode`를 추가했습니다.
  - 단순 법령 질문은 `general_law`로 분류합니다.
  - 사고/시나리오 특정 질문은 `scenario_judgment`로 분류합니다.
  - 일반 모드에서는 질문에 `위 사고` 표현이 있어도 시나리오 판단형으로 강제 이동하지 않도록 했습니다.
- `rag/question_graph.py`를 추가해 LangGraph 전환을 위한 그래프형 라우팅 기반을 마련했습니다.
  - 현재는 외부 `langgraph` 의존성 없이 동작하는 lightweight graph module입니다.
  - 추후 `IntentClassifierNode`, `CitationValidatorNode`, `CacheGuardNode`를 붙일 수 있는 형태로 구성했습니다.
- 비계 특별안전교육 질문 분기를 개선했습니다.
  - 단순 질문: 조건부 법령 설명으로 응답합니다.
  - `위 사고에서...` 형태의 시나리오 질문: 기존처럼 `위반 여부: YES` 판단형으로 응답합니다.
- 안전보건교육과 특별안전교육 차이 질문은 산업안전보건법 제29조 및 시행규칙 별표 5 중심의 직접 답변으로 분리했습니다.
- EXAONE이 `PRIMARY TEXT`, `BACKGROUND [table]`, `score=...` 같은 내부 RAG 라벨을 답변에 복사하지 않도록 프롬프트와 후처리를 보강했습니다.

### Backend / DB
- 새 상담 생성 API의 `ts` 미정의 문제를 수정했습니다.
- `web_app.py` 기본 포트를 `8200`으로 변경했습니다.
- 계정별 DB 관리 구조를 유지했습니다.
  - `users`, `sessions`, `scenarios`, `conversations`, `messages`, `deletion_logs`는 `user_id` 또는 `conversation_id` 기준으로 분리됩니다.
  - 관리자/일반 계정은 같은 SQLite DB 안에서 role과 user_id로 분리 관리합니다.

### Verification
- `node --check web\static\app.js`
- `python -m compileall web_app.py`
- `python -m compileall rag\question_graph.py rag\chatbot.py web_app.py`
- `QuestionScopeNode` 직접 테스트
  - 단순 비계 특별교육 질문 → `general_law`
  - `위 사고에서...` 질문 → `scenario_judgment`
  - 일반 모드의 `위 사고에서...` 질문 → `general_law`
- `8200` 서버 재시작 및 `/api/chat` 분기 확인
