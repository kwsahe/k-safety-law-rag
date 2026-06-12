"""Web UI for K-Safety Law RAG.

Run:
    python web_app.py
    python web_app.py --host 127.0.0.1 --port 8200
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import mimetypes
import os
import secrets
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT_DIR = Path(__file__).parent.resolve()
STATIC_DIR = ROOT_DIR / "web" / "static"
DB_PATH = ROOT_DIR / "data" / "chatbot_ui.sqlite3"
KST = timezone(timedelta(hours=9))
SESSION_DAYS = 7
DEFAULT_ADMIN_USERNAME = os.getenv("WEB_ADMIN_USERNAME", "admin")
DEFAULT_ADMIN_PASSWORD = os.getenv("WEB_ADMIN_PASSWORD", "admin1234")


def now_iso() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def json_dumps(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return f"pbkdf2_sha256${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_hex, digest_hex = stored.split("$", 2)
    except ValueError:
        return False
    if scheme != "pbkdf2_sha256":
        return False
    expected = hash_password(password, bytes.fromhex(salt_hex)).split("$", 2)[2]
    return hmac.compare_digest(expected, digest_hex)


def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db() -> None:
    with get_db() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin', 'user')),
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS scenarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
                overview TEXT NOT NULL DEFAULT '',
                details TEXT NOT NULL DEFAULT '',
                workers TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                public_payload TEXT,
                admin_payload TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        row = con.execute("SELECT id FROM users WHERE username = ?", (DEFAULT_ADMIN_USERNAME,)).fetchone()
        if row is None:
            con.execute(
                "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, 'admin', ?)",
                (DEFAULT_ADMIN_USERNAME, hash_password(DEFAULT_ADMIN_PASSWORD), now_iso()),
            )


def row_to_user(row: sqlite3.Row) -> dict[str, Any]:
    return {"id": row["id"], "username": row["username"], "role": row["role"]}


def display_page(metadata: dict[str, Any]) -> str:
    page = str(metadata.get("citation_page") or metadata.get("page") or "").strip()
    return page if page and page != "0" else "페이지 정보 없음"


def source_to_public(source: Any) -> dict[str, Any]:
    metadata = source.metadata
    law_name = str(metadata.get("law_name") or "").replace("_", " ")
    article = str(metadata.get("article") or metadata.get("annex") or "").strip()
    return {
        "source_type": metadata.get("source_type", ""),
        "law_name": law_name,
        "article": article,
        "page": display_page(metadata),
    }


def source_to_admin(source: Any) -> dict[str, Any]:
    payload = source_to_public(source)
    payload["score"] = source.metadata.get("score", 0.0)
    payload["metadata"] = source.metadata
    payload["content"] = source.content
    return payload


def build_cli_output(answer: str, sources: list[Any], references_path: str, elapsed_ms: int, model_name: str) -> str:
    lines = ["[답변]", answer, "", "[참고 근거]"]
    for index, doc in enumerate(sources, start=1):
        metadata = doc.metadata
        source_type = metadata.get("source_type", "")
        law_name = metadata.get("law_name", "")
        article = metadata.get("article", "")
        page = display_page(metadata)
        score = metadata.get("score", 0.0)
        label = f"{law_name} {article}".strip()
        lines.append(f"  {index}. [{source_type}] {label} {page} score={score}")
    lines.extend(["", f"법령 참조 JSON 저장(상위 3개): {references_path}", "", f"모델명: {model_name}", f"응답 시간: {elapsed_ms}ms"])
    return "\n".join(lines)


class WebAppHandler(BaseHTTPRequestHandler):
    server_version = "KSafetyLawRAG/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write("%s - %s\n" % (self.address_string(), format % args))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/":
            self.serve_file(STATIC_DIR / "index.html")
            return
        if path.startswith("/static/"):
            self.serve_file(STATIC_DIR / path.removeprefix("/static/"))
            return
        if path == "/api/me":
            user = self.current_user()
            self.send_json({"user": user})
            return
        if path == "/api/conversations":
            self.require_user(self.handle_list_conversations)
            return
        if path.startswith("/api/conversations/"):
            self.require_user(lambda user: self.handle_get_conversation(user, path))
            return
        if path == "/api/scenario":
            self.require_user(self.handle_get_scenario)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/api/login":
            self.handle_login()
            return
        if path == "/api/register":
            self.handle_register()
            return
        if path == "/api/logout":
            self.handle_logout()
            return
        if path == "/api/conversations":
            self.require_user(self.handle_create_conversation)
            return
        if path == "/api/chat":
            self.require_user(self.handle_chat)
            return
        if path == "/api/scenario":
            self.require_user(self.handle_save_scenario)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def serve_file(self, path: Path) -> None:
        try:
            resolved = path.resolve()
            resolved.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not resolved.exists() or not resolved.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        data = resolved.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json_dumps(data)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def set_session_cookie(self, token: str, expires_at: datetime) -> None:
        cookie = SimpleCookie()
        cookie["ksafety_session"] = token
        cookie["ksafety_session"]["path"] = "/"
        cookie["ksafety_session"]["httponly"] = True
        cookie["ksafety_session"]["samesite"] = "Lax"
        cookie["ksafety_session"]["expires"] = expires_at.strftime("%a, %d %b %Y %H:%M:%S GMT")
        self.send_header("Set-Cookie", cookie.output(header="").strip())

    def clear_session_cookie(self) -> None:
        cookie = SimpleCookie()
        cookie["ksafety_session"] = ""
        cookie["ksafety_session"]["path"] = "/"
        cookie["ksafety_session"]["expires"] = "Thu, 01 Jan 1970 00:00:00 GMT"
        self.send_header("Set-Cookie", cookie.output(header="").strip())

    def current_user(self) -> dict[str, Any] | None:
        cookie_header = self.headers.get("Cookie", "")
        cookie = SimpleCookie(cookie_header)
        morsel = cookie.get("ksafety_session")
        if morsel is None or not morsel.value:
            return None
        with get_db() as con:
            row = con.execute(
                """
                SELECT users.id, users.username, users.role
                FROM sessions
                JOIN users ON users.id = sessions.user_id
                WHERE sessions.token = ? AND sessions.expires_at > ?
                """,
                (morsel.value, now_iso()),
            ).fetchone()
        return row_to_user(row) if row else None

    def require_user(self, handler: Any) -> None:
        user = self.current_user()
        if user is None:
            self.send_json({"error": "로그인이 필요합니다."}, HTTPStatus.UNAUTHORIZED)
            return
        handler(user)

    def handle_login(self) -> None:
        data = self.read_json()
        username = str(data.get("username", "")).strip()
        password = str(data.get("password", ""))
        with get_db() as con:
            row = con.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            if row is None or not verify_password(password, row["password_hash"]):
                self.send_json({"error": "아이디 또는 비밀번호가 올바르지 않습니다."}, HTTPStatus.UNAUTHORIZED)
                return
            token = secrets.token_urlsafe(32)
            expires = datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS)
            con.execute(
                "INSERT INTO sessions (token, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
                (token, row["id"], expires.isoformat(timespec="seconds"), now_iso()),
            )
        body = json_dumps({"user": row_to_user(row)})
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.set_session_cookie(token, expires)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_register(self) -> None:
        data = self.read_json()
        username = str(data.get("username", "")).strip()
        password = str(data.get("password", ""))
        if len(username) < 3 or len(password) < 6:
            self.send_json({"error": "아이디는 3자 이상, 비밀번호는 6자 이상이어야 합니다."}, HTTPStatus.BAD_REQUEST)
            return
        with get_db() as con:
            try:
                con.execute(
                    "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, 'user', ?)",
                    (username, hash_password(password), now_iso()),
                )
            except sqlite3.IntegrityError:
                self.send_json({"error": "이미 존재하는 아이디입니다."}, HTTPStatus.CONFLICT)
                return
        self.send_json({"ok": True})

    def handle_logout(self) -> None:
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        morsel = cookie.get("ksafety_session")
        if morsel is not None:
            with get_db() as con:
                con.execute("DELETE FROM sessions WHERE token = ?", (morsel.value,))
        body = json_dumps({"ok": True})
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.clear_session_cookie()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_list_conversations(self, user: dict[str, Any]) -> None:
        with get_db() as con:
            rows = con.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
                (user["id"],),
            ).fetchall()
        self.send_json({"conversations": [dict(row) for row in rows]})

    def handle_create_conversation(self, user: dict[str, Any]) -> None:
        title = str(self.read_json().get("title", "")).strip() or "새 상담"
        ts = now_iso()
        with get_db() as con:
            cur = con.execute(
                "INSERT INTO conversations (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (user["id"], title[:80], ts, ts),
            )
        self.send_json({"conversation": {"id": cur.lastrowid, "title": title[:80], "created_at": ts, "updated_at": ts}})

    def handle_get_conversation(self, user: dict[str, Any], path: str) -> None:
        try:
            conversation_id = int(path.rsplit("/", 1)[1])
        except ValueError:
            self.send_json({"error": "잘못된 상담 ID입니다."}, HTTPStatus.BAD_REQUEST)
            return
        with get_db() as con:
            conv = con.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user["id"]),
            ).fetchone()
            if conv is None:
                self.send_json({"error": "상담을 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
                return
            messages = con.execute(
                "SELECT id, role, content, public_payload, admin_payload, created_at FROM messages WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        payload_messages = []
        for msg in messages:
            payload = json.loads(msg["admin_payload"] if user["role"] == "admin" and msg["admin_payload"] else msg["public_payload"] or "{}")
            payload_messages.append(
                {
                    "id": msg["id"],
                    "role": msg["role"],
                    "content": msg["content"],
                    "payload": payload,
                    "created_at": msg["created_at"],
                }
            )
        self.send_json({"conversation": dict(conv), "messages": payload_messages})

    def handle_get_scenario(self, user: dict[str, Any]) -> None:
        with get_db() as con:
            row = con.execute("SELECT overview, details, workers, updated_at FROM scenarios WHERE user_id = ?", (user["id"],)).fetchone()
        self.send_json({"scenario": dict(row) if row else {"overview": "", "details": "", "workers": "", "updated_at": ""}})

    def handle_save_scenario(self, user: dict[str, Any]) -> None:
        data = self.read_json()
        overview = str(data.get("overview", ""))
        details = str(data.get("details", ""))
        workers = str(data.get("workers", ""))
        ts = now_iso()
        with get_db() as con:
            con.execute(
                """
                INSERT INTO scenarios (user_id, overview, details, workers, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    overview = excluded.overview,
                    details = excluded.details,
                    workers = excluded.workers,
                    updated_at = excluded.updated_at
                """,
                (user["id"], overview, details, workers, ts),
            )
        self.send_json({"scenario": {"overview": overview, "details": details, "workers": workers, "updated_at": ts}})

    def handle_chat(self, user: dict[str, Any]) -> None:
        data = self.read_json()
        question = str(data.get("question", "")).strip()
        conversation_id = int(data.get("conversation_id") or 0)
        if not question:
            self.send_json({"error": "질문을 입력하세요."}, HTTPStatus.BAD_REQUEST)
            return

        with get_db() as con:
            conv = con.execute(
                "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user["id"]),
            ).fetchone()
            if conv is None:
                ts = now_iso()
                cur = con.execute(
                    "INSERT INTO conversations (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (user["id"], question[:40] or "새 상담", ts, ts),
                )
                conversation_id = cur.lastrowid
            scenario_row = con.execute(
                "SELECT overview, details, workers FROM scenarios WHERE user_id = ?",
                (user["id"],),
            ).fetchone()

        try:
            from rag.chatbot import rag_chat
            from rag.cli import LawReferenceWriter
            from rag.config import LLM_MODEL
            from rag.schemas import AccidentScenario, ChatRequest
        except Exception as exc:
            self.send_json(
                {
                    "error": (
                        "RAG 의존성을 불러오지 못했습니다. requirements.txt 설치 또는 CLI 실행 환경에서 "
                        f"서버를 실행하세요. 상세: {exc}"
                    )
                },
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        scenario = None
        if scenario_row and any(scenario_row[key] for key in ("overview", "details", "workers")):
            scenario = AccidentScenario(**dict(scenario_row))

        started = time.time()
        try:
            response = rag_chat(ChatRequest(question=question, scenario=scenario))
        except Exception as exc:
            self.send_json({"error": f"챗봇 응답 생성 실패: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        elapsed_ms = int((time.time() - started) * 1000)
        report_id = int(datetime.now().strftime("%Y%m%d%H%M%S"))
        references_path = str(LawReferenceWriter(report_id=report_id, section="WEB").save_json(response.sources, answer=response.answer))
        public_payload = {
            "answer": response.answer,
            "sources": [source_to_public(source) for source in response.sources],
        }
        admin_payload = {
            **public_payload,
            "sources": [source_to_admin(source) for source in response.sources],
            "model_name": LLM_MODEL,
            "elapsed_ms": elapsed_ms,
            "references_path": references_path,
            "cli_output": build_cli_output(response.answer, response.sources, references_path, elapsed_ms, LLM_MODEL),
        }
        ts = now_iso()
        with get_db() as con:
            con.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, 'user', ?, ?)",
                (conversation_id, question, ts),
            )
            cur = con.execute(
                """
                INSERT INTO messages (conversation_id, role, content, public_payload, admin_payload, created_at)
                VALUES (?, 'assistant', ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    response.answer,
                    json.dumps(public_payload, ensure_ascii=False),
                    json.dumps(admin_payload, ensure_ascii=False),
                    now_iso(),
                ),
            )
            con.execute("UPDATE conversations SET updated_at = ?, title = CASE WHEN title = '새 상담' THEN ? ELSE title END WHERE id = ?", (now_iso(), question[:40], conversation_id))
        message = {
            "id": cur.lastrowid,
            "role": "assistant",
            "content": response.answer,
            "payload": admin_payload if user["role"] == "admin" else public_payload,
            "created_at": ts,
        }
        self.send_json({"conversation_id": conversation_id, "message": message})


def main() -> None:
    parser = argparse.ArgumentParser(description="K-Safety Law RAG Web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    init_db()
    server = ThreadingHTTPServer((args.host, args.port), WebAppHandler)
    print(f"K-Safety Law RAG Web UI: http://{args.host}:{args.port}")
    print(f"Admin account: {DEFAULT_ADMIN_USERNAME} / {DEFAULT_ADMIN_PASSWORD}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버를 종료합니다.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
