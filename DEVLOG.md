# Devlog

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
