# COMMANDS.md

`K-Safety Law RAG` 폴더 기준 명령어입니다.

---

## 환경 활성화

```cmd
conda activate p311_ragreport
cd /d "C:\K-Safety Law RAG"
```

---

## LLM 설정 확인

```cmd
python -c "from rag.config import LLM_PROVIDER, LLM_MODEL, LLM_API_BASE; print(LLM_PROVIDER); print(LLM_MODEL); print(bool(LLM_API_BASE.strip()))"
```

---

## CLI 실행

```cmd
python cli.py
```

기존 스크립트 직접 실행:

```cmd
python cli.py
```

CPU 옵션:

```cmd
python scripts\test_chat_cpu.py
```

---

## 웹 UI 실행

관리자 계정은 첫 실행 시 자동 생성됩니다. 초기 비밀번호는 환경변수로 지정하세요.

```cmd
set WEB_ADMIN_USERNAME=admin
set WEB_ADMIN_PASSWORD=충분히-긴-초기-비밀번호
set WEB_ALLOW_REGISTRATION=false
set WEB_ALLOWED_HOSTS=localhost,127.0.0.1
python web_app.py --host 127.0.0.1 --port 8200
```

`WEB_ADMIN_PASSWORD`를 지정하지 않으면 안전한 임시 비밀번호가 생성되어 최초 실행 콘솔에 한 번만 표시됩니다.
HTTPS 리버스 프록시 뒤에서 실행할 때는 `WEB_SECURE_COOKIES=true`를 추가합니다.

웹 UI 데이터베이스:

```text
data/chatbot_ui.sqlite3
```

---

## 임베딩 재생성

```cmd
python -m rag.ingest --reset
python -m rag.table_retriever --ingest --reset --strategy row
```

wrapper:

```cmd
python scripts\run_ingest.py --reset
python scripts\reingest_tables.py
```

---

## 검색 확인

```cmd
python -m rag.table_retriever --query "비계 특별안전교육" --top-k 5
python -m rag.chatbot --question "비계 작업 특별안전교육 미실시는 어떤 조항 위반인가?" --show-sources
```

---

## 기본 검증

```cmd
python -c "import ast,pathlib,runpy; files=['cli.py','rag/cli.py','scripts/test_chat.py','rag/chatbot.py','scenarios/default_accident.py']; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8')) for p in files]; assert 'SCENARIO' in runpy.run_path('scenarios/default_accident.py'); print('OK')"
```
