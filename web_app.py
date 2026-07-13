"""Web UI for 건설현장 중대재해-산업안전 법령 상담 챗봇.

Run:
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
import urllib.error
import urllib.request
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
                mode TEXT NOT NULL DEFAULT 'scenario' CHECK(mode IN ('scenario', 'general')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT,
                deleted_by INTEGER
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                public_payload TEXT,
                admin_payload TEXT,
                created_at TEXT NOT NULL,
                deleted_at TEXT,
                deleted_by INTEGER
            );

            CREATE TABLE IF NOT EXISTS deletion_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_type TEXT NOT NULL CHECK(target_type IN ('conversation', 'message')),
                target_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                snapshot_json TEXT NOT NULL,
                deleted_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS answer_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                rating TEXT NOT NULL CHECK(rating IN ('helpful', 'needs_improvement')),
                comment TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(message_id, user_id)
            );
            """
        )
        columns = {row["name"] for row in con.execute("PRAGMA table_info(conversations)").fetchall()}
        if "mode" not in columns:
            con.execute("ALTER TABLE conversations ADD COLUMN mode TEXT NOT NULL DEFAULT 'scenario'")
        if "deleted_at" not in columns:
            con.execute("ALTER TABLE conversations ADD COLUMN deleted_at TEXT")
        if "deleted_by" not in columns:
            con.execute("ALTER TABLE conversations ADD COLUMN deleted_by INTEGER")
        message_columns = {row["name"] for row in con.execute("PRAGMA table_info(messages)").fetchall()}
        if "deleted_at" not in message_columns:
            con.execute("ALTER TABLE messages ADD COLUMN deleted_at TEXT")
        if "deleted_by" not in message_columns:
            con.execute("ALTER TABLE messages ADD COLUMN deleted_by INTEGER")
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


def build_cli_output(
    answer: str,
    sources: list[Any],
    elapsed_ms: int,
    model_name: str,
    question: str = "",
    citation_check: dict[str, Any] | None = None,
    graph_trace: dict[str, Any] | None = None,
) -> str:
    lines = ["[질문]", question, "", "[답변]", answer, "", "[참고 근거]"] if question else ["[답변]", answer, "", "[참고 근거]"]
    for index, doc in enumerate(sources, start=1):
        metadata = doc.metadata
        source_type = metadata.get("source_type", "")
        law_name = metadata.get("law_name", "")
        article = metadata.get("article", "")
        page = display_page(metadata)
        score = metadata.get("score", 0.0)
        label = f"{law_name} {article}".strip()
        lines.append(f"  {index}. [{source_type}] {label} {page} score={score}")
    if citation_check:
        lines.extend(
            [
                "",
                "[출처 검증]",
                f"- 상태: {str(citation_check.get('status', '')).upper()}",
                f"- 요약: {citation_check.get('summary', '')}",
            ]
        )
        for warning in citation_check.get("warnings", []):
            lines.append(f"- 경고: {warning}")
        for missing in citation_check.get("missing_required", []):
            lines.append(f"- 필수 근거 누락: {missing.get('label', '')} ({missing.get('law_name', '')} {missing.get('ref', '')})")
        for unsupported in citation_check.get("unsupported", []):
            lines.append(f"- 미검증 출처: {unsupported.get('label', '')}")
    if graph_trace:
        lines.extend(
            [
                "",
                "[질문 그래프]",
                f"- 범위: {graph_trace.get('scope', '')}",
                f"- 의도: {graph_trace.get('intent', '')}",
                f"- 경로: {graph_trace.get('route', '')}",
                f"- 캐시 키: {graph_trace.get('cache_key', '')}",
                f"- 실행 노드: {' -> '.join(graph_trace.get('nodes', []))}",
            ]
        )
    lines.extend(["", f"모델명: {model_name}", f"응답 시간: {elapsed_ms}ms"])
    return "\n".join(lines)


class WebAppHandler(BaseHTTPRequestHandler):
    server_version = "KSafetyLawRAG/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write("%s - %s\n" % (self.address_string(), format % args))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path in {"/", "/general"}:
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
        if path == "/api/admin/dashboard":
            self.require_admin(self.handle_admin_dashboard)
            return
        if path == "/api/admin/health":
            self.require_admin(self.handle_admin_health)
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
        if path.startswith("/api/messages/") and path.endswith("/feedback"):
            self.require_user(lambda user: self.handle_feedback(user, path))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_PATCH(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path.startswith("/api/conversations/"):
            self.require_user(lambda user: self.handle_update_conversation(user, path))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path.startswith("/api/messages/"):
            self.require_user(lambda user: self.handle_delete_message(user, path))
            return
        if path.startswith("/api/conversations/"):
            self.require_user(lambda user: self.handle_delete_conversation(user, path))
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
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
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

    def require_admin(self, handler: Any) -> None:
        user = self.current_user()
        if user is None:
            self.send_json({"error": "로그인이 필요합니다."}, HTTPStatus.UNAUTHORIZED)
            return
        if user.get("role") != "admin":
            self.send_json({"error": "관리자 권한이 필요합니다."}, HTTPStatus.FORBIDDEN)
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
                """
                SELECT id, title, mode, created_at, updated_at
                FROM conversations
                WHERE user_id = ? AND deleted_at IS NULL
                ORDER BY updated_at DESC
                """,
                (user["id"],),
            ).fetchall()
        self.send_json({"conversations": [dict(row) for row in rows]})

    def handle_create_conversation(self, user: dict[str, Any]) -> None:
        data = self.read_json()
        title = str(data.get("title", "")).strip() or "새 상담"
        mode = str(data.get("mode", "scenario")).strip()
        if mode not in {"scenario", "general"}:
            mode = "scenario"
        ts = now_iso()
        with get_db() as con:
            cur = con.execute(
                "INSERT INTO conversations (user_id, title, mode, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (user["id"], title[:80], mode, ts, ts),
            )
            conversation_id = cur.lastrowid
        self.send_json({"conversation": {"id": conversation_id, "title": title[:80], "mode": mode, "created_at": ts, "updated_at": ts}})

    def handle_get_conversation(self, user: dict[str, Any], path: str) -> None:
        conversation_id = self.parse_conversation_id(path)
        if conversation_id is None:
            self.send_json({"error": "잘못된 상담 ID입니다."}, HTTPStatus.BAD_REQUEST)
            return
        with get_db() as con:
            conv = con.execute(
                """
                SELECT id, title, mode, created_at, updated_at
                FROM conversations
                WHERE id = ? AND user_id = ? AND deleted_at IS NULL
                """,
                (conversation_id, user["id"]),
            ).fetchone()
            if conv is None:
                self.send_json({"error": "상담을 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
                return
            messages = con.execute(
                """
                SELECT m.id, m.role, m.content, m.public_payload, m.admin_payload, m.created_at,
                       f.rating AS feedback_rating, f.comment AS feedback_comment
                FROM messages m
                LEFT JOIN answer_feedback f ON f.message_id = m.id AND f.user_id = ?
                WHERE m.conversation_id = ? AND m.deleted_at IS NULL
                ORDER BY m.id
                """,
                (user["id"], conversation_id),
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
                    "feedback": {
                        "rating": msg["feedback_rating"],
                        "comment": msg["feedback_comment"] or "",
                    } if msg["feedback_rating"] else None,
                    "created_at": msg["created_at"],
                }
            )
        self.send_json({"conversation": dict(conv), "messages": payload_messages})

    def parse_conversation_id(self, path: str) -> int | None:
        try:
            return int(path.rstrip("/").rsplit("/", 1)[1])
        except (ValueError, IndexError):
            return None

    def parse_message_id(self, path: str) -> int | None:
        try:
            return int(path.rstrip("/").rsplit("/", 1)[1])
        except (ValueError, IndexError):
            return None

    def parse_feedback_message_id(self, path: str) -> int | None:
        parts = [part for part in path.split("/") if part]
        try:
            return int(parts[2]) if len(parts) == 4 and parts[3] == "feedback" else None
        except (ValueError, IndexError):
            return None

    def handle_feedback(self, user: dict[str, Any], path: str) -> None:
        message_id = self.parse_feedback_message_id(path)
        if message_id is None:
            self.send_json({"error": "잘못된 메시지 ID입니다."}, HTTPStatus.BAD_REQUEST)
            return
        data = self.read_json()
        rating = str(data.get("rating", "")).strip()
        comment = str(data.get("comment", "")).strip()[:500]
        if rating not in {"helpful", "needs_improvement"}:
            self.send_json({"error": "올바른 평가를 선택하세요."}, HTTPStatus.BAD_REQUEST)
            return
        ts = now_iso()
        with get_db() as con:
            message = con.execute(
                """
                SELECT m.id
                FROM messages m
                JOIN conversations c ON c.id = m.conversation_id
                WHERE m.id = ? AND m.role = 'assistant' AND c.user_id = ?
                  AND m.deleted_at IS NULL AND c.deleted_at IS NULL
                """,
                (message_id, user["id"]),
            ).fetchone()
            if message is None:
                self.send_json({"error": "평가할 답변을 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
                return
            con.execute(
                """
                INSERT INTO answer_feedback (message_id, user_id, rating, comment, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id, user_id) DO UPDATE SET
                    rating = excluded.rating,
                    comment = excluded.comment,
                    updated_at = excluded.updated_at
                """,
                (message_id, user["id"], rating, comment, ts, ts),
            )
        self.send_json({"feedback": {"message_id": message_id, "rating": rating, "comment": comment, "updated_at": ts}})

    def handle_admin_dashboard(self, user: dict[str, Any]) -> None:
        del user
        with get_db() as con:
            totals = {
                "users": con.execute("SELECT COUNT(*) FROM users").fetchone()[0],
                "conversations": con.execute("SELECT COUNT(*) FROM conversations WHERE deleted_at IS NULL").fetchone()[0],
                "answers": con.execute("SELECT COUNT(*) FROM messages WHERE role = 'assistant' AND deleted_at IS NULL").fetchone()[0],
            }
            rows = con.execute(
                "SELECT admin_payload, created_at FROM messages WHERE role = 'assistant' AND deleted_at IS NULL ORDER BY id DESC LIMIT 500"
            ).fetchall()
            feedback_rows = con.execute(
                """
                SELECT f.rating, f.comment, f.updated_at, u.username, m.content, c.title
                FROM answer_feedback f
                JOIN users u ON u.id = f.user_id
                JOIN messages m ON m.id = f.message_id
                JOIN conversations c ON c.id = m.conversation_id
                ORDER BY f.updated_at DESC LIMIT 8
                """
            ).fetchall()
            feedback_count_rows = con.execute(
                "SELECT rating, COUNT(*) AS count FROM answer_feedback GROUP BY rating"
            ).fetchall()

        citation_counts = {"pass": 0, "warn": 0, "fail": 0, "unknown": 0}
        elapsed_values: list[int] = []
        intent_counts: dict[str, int] = {}
        model_name = ""
        for row in rows:
            try:
                payload = json.loads(row["admin_payload"] or "{}")
            except json.JSONDecodeError:
                payload = {}
            status = str((payload.get("citation_check") or {}).get("status") or "unknown")
            citation_counts[status if status in citation_counts else "unknown"] += 1
            elapsed = payload.get("elapsed_ms")
            if isinstance(elapsed, (int, float)):
                elapsed_values.append(int(elapsed))
            intent = str((payload.get("graph_trace") or {}).get("intent") or "unknown")
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            model_name = model_name or str(payload.get("model_name") or "")

        feedback_counts = {"helpful": 0, "needs_improvement": 0}
        for row in feedback_count_rows:
            feedback_counts[row["rating"]] = row["count"]
        self.send_json(
            {
                "totals": totals,
                "quality": {
                    "citation": citation_counts,
                    "average_elapsed_ms": round(sum(elapsed_values) / len(elapsed_values)) if elapsed_values else 0,
                    "intent_counts": intent_counts,
                    "model_name": model_name,
                },
                "feedback": {
                    "counts": feedback_counts,
                    "recent": [dict(row) for row in feedback_rows],
                },
            }
        )

    def handle_admin_health(self, user: dict[str, Any]) -> None:
        del user
        from rag.config import CHROMA_PATH, LLM_API_BASE, LLM_API_KEY, LLM_MODEL, LLM_PROVIDER, TABLE_CHROMA_PATH

        started = time.time()
        model_connected = False
        detail = "모델 API 주소가 설정되지 않았습니다."
        if LLM_API_BASE:
            headers = {"ngrok-skip-browser-warning": "true"}
            if LLM_API_KEY:
                headers["Authorization"] = f"Bearer {LLM_API_KEY}"
            request = urllib.request.Request(f"{LLM_API_BASE.rstrip('/')}/models", headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=5) as response:
                    model_connected = 200 <= response.status < 300
                    detail = "EXAONE 모델 API에 연결되었습니다." if model_connected else f"모델 API 응답 코드: {response.status}"
            except urllib.error.HTTPError as exc:
                if exc.code in {404, 405}:
                    probe_body = json.dumps(
                        {
                            "model": LLM_MODEL,
                            "messages": [{"role": "user", "content": "연결 확인"}],
                            "max_tokens": 1,
                            "temperature": 0,
                        },
                        ensure_ascii=False,
                    ).encode("utf-8")
                    probe_headers = {**headers, "Content-Type": "application/json"}
                    probe = urllib.request.Request(
                        f"{LLM_API_BASE.rstrip('/')}/chat/completions",
                        data=probe_body,
                        headers=probe_headers,
                        method="POST",
                    )
                    try:
                        with urllib.request.urlopen(probe, timeout=15) as response:
                            model_connected = 200 <= response.status < 300
                            detail = "EXAONE 채팅 API에 연결되었습니다." if model_connected else f"모델 API 응답 코드: {response.status}"
                    except (urllib.error.URLError, TimeoutError, OSError) as probe_exc:
                        detail = f"모델 API 연결 실패: {probe_exc}"
                else:
                    detail = f"모델 API 연결 실패: HTTP {exc.code}"
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                detail = f"모델 API 연결 실패: {exc}"

        db_ok = False
        try:
            with get_db() as con:
                con.execute("SELECT 1").fetchone()
            db_ok = True
        except sqlite3.Error:
            db_ok = False
        self.send_json(
            {
                "app": "ok",
                "database": "ok" if db_ok else "error",
                "vector_db": "ok" if CHROMA_PATH.exists() and TABLE_CHROMA_PATH.exists() else "warning",
                "model": {
                    "provider": LLM_PROVIDER,
                    "name": LLM_MODEL,
                    "configured": bool(LLM_API_BASE),
                    "connected": model_connected,
                    "latency_ms": int((time.time() - started) * 1000),
                    "detail": detail,
                },
            }
        )

    def handle_update_conversation(self, user: dict[str, Any], path: str) -> None:
        conversation_id = self.parse_conversation_id(path)
        if conversation_id is None:
            self.send_json({"error": "잘못된 상담 ID입니다."}, HTTPStatus.BAD_REQUEST)
            return
        data = self.read_json()
        title = str(data.get("title", "")).strip()
        if not title:
            self.send_json({"error": "상담 이름을 입력하세요."}, HTTPStatus.BAD_REQUEST)
            return
        with get_db() as con:
            cur = con.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND user_id = ? AND deleted_at IS NULL",
                (title[:80], now_iso(), conversation_id, user["id"]),
            )
            if cur.rowcount == 0:
                self.send_json({"error": "상담을 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
                return
            row = con.execute(
                """
                SELECT id, title, mode, created_at, updated_at
                FROM conversations
                WHERE id = ? AND user_id = ? AND deleted_at IS NULL
                """,
                (conversation_id, user["id"]),
            ).fetchone()
        self.send_json({"conversation": dict(row)})

    def handle_delete_conversation(self, user: dict[str, Any], path: str) -> None:
        conversation_id = self.parse_conversation_id(path)
        if conversation_id is None:
            self.send_json({"error": "잘못된 상담 ID입니다."}, HTTPStatus.BAD_REQUEST)
            return
        with get_db() as con:
            conv = con.execute(
                """
                SELECT id, user_id, title, mode, created_at, updated_at
                FROM conversations
                WHERE id = ? AND user_id = ? AND deleted_at IS NULL
                """,
                (conversation_id, user["id"]),
            ).fetchone()
            if conv is None:
                self.send_json({"error": "상담을 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
                return
            messages = con.execute(
                """
                SELECT id, role, content, public_payload, admin_payload, created_at
                FROM messages
                WHERE conversation_id = ? AND deleted_at IS NULL
                ORDER BY id
                """,
                (conversation_id,),
            ).fetchall()
            deleted_at = now_iso()
            snapshot = {
                "conversation": dict(conv),
                "messages": [dict(row) for row in messages],
            }
            con.execute(
                """
                INSERT INTO deletion_logs (target_type, target_id, user_id, snapshot_json, deleted_at)
                VALUES ('conversation', ?, ?, ?, ?)
                """,
                (conversation_id, user["id"], json.dumps(snapshot, ensure_ascii=False), deleted_at),
            )
            con.execute(
                "UPDATE conversations SET deleted_at = ?, deleted_by = ?, updated_at = ? WHERE id = ?",
                (deleted_at, user["id"], deleted_at, conversation_id),
            )
            con.execute(
                "UPDATE messages SET deleted_at = ?, deleted_by = ? WHERE conversation_id = ? AND deleted_at IS NULL",
                (deleted_at, user["id"], conversation_id),
            )
        self.send_json({"ok": True, "deleted_id": conversation_id})

    def handle_delete_message(self, user: dict[str, Any], path: str) -> None:
        message_id = self.parse_message_id(path)
        if message_id is None:
            self.send_json({"error": "잘못된 채팅 ID입니다."}, HTTPStatus.BAD_REQUEST)
            return
        with get_db() as con:
            msg = con.execute(
                """
                SELECT
                    m.id, m.conversation_id, m.role, m.content, m.public_payload, m.admin_payload, m.created_at,
                    c.user_id, c.deleted_at AS conversation_deleted_at
                FROM messages m
                JOIN conversations c ON c.id = m.conversation_id
                WHERE m.id = ? AND c.user_id = ? AND m.deleted_at IS NULL AND c.deleted_at IS NULL
                """,
                (message_id, user["id"]),
            ).fetchone()
            if msg is None:
                self.send_json({"error": "채팅을 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
                return
            deleted_at = now_iso()
            snapshot = {"message": dict(msg)}
            con.execute(
                """
                INSERT INTO deletion_logs (target_type, target_id, user_id, snapshot_json, deleted_at)
                VALUES ('message', ?, ?, ?, ?)
                """,
                (message_id, user["id"], json.dumps(snapshot, ensure_ascii=False), deleted_at),
            )
            con.execute(
                "UPDATE messages SET deleted_at = ?, deleted_by = ? WHERE id = ?",
                (deleted_at, user["id"], message_id),
            )
            con.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (deleted_at, msg["conversation_id"]),
            )
        self.send_json({"ok": True, "deleted_id": message_id})

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
        requested_mode = str(data.get("mode", "")).strip()
        if requested_mode not in {"scenario", "general"}:
            requested_mode = ""
        if not question:
            self.send_json({"error": "질문을 입력하세요."}, HTTPStatus.BAD_REQUEST)
            return

        with get_db() as con:
            conv = con.execute(
                "SELECT id, mode FROM conversations WHERE id = ? AND user_id = ? AND deleted_at IS NULL",
                (conversation_id, user["id"]),
            ).fetchone()
            if conv is None:
                ts = now_iso()
                mode = requested_mode or "scenario"
                cur = con.execute(
                    "INSERT INTO conversations (user_id, title, mode, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (user["id"], question[:40] or "새 상담", mode, ts, ts),
                )
                conversation_id = cur.lastrowid
            else:
                mode = str(conv["mode"] or "scenario")
            scenario_row = con.execute(
                "SELECT overview, details, workers FROM scenarios WHERE user_id = ?",
                (user["id"],),
            ).fetchone()

        try:
            from rag.citation_validator import validate_answer_citations
            from rag.chatbot import build_retrieval_query
            from rag.chatbot import direct_answer_from_sources
            from rag.chatbot import direct_answer_sources
            from rag.chatbot import rag_chat
            from rag.chatbot import reset_chat_runtime_state
            from rag.config import LLM_MODEL
            from rag.question_graph import public_graph_trace, run_question_graph
            from rag.schemas import AccidentScenario, ChatRequest, ChatResponse
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
        if mode == "scenario" and scenario_row and any(scenario_row[key] for key in ("overview", "details", "workers")):
            scenario = AccidentScenario(**dict(scenario_row))

        user_ts = now_iso()
        started = time.time()
        try:
            reset_chat_runtime_state(clear_scenario_value=True)
            retrieval_query = build_retrieval_query(question, scenario)
            graph_state = run_question_graph(question, mode=mode)
            direct_answer = direct_answer_from_sources(question, [], retrieval_query, mode=mode)
            if direct_answer:
                response = ChatResponse(
                    answer=direct_answer,
                    sources=direct_answer_sources(question, [], retrieval_query),
                    graph_trace=public_graph_trace(graph_state),
                )
            else:
                response = rag_chat(ChatRequest(question=question, scenario=scenario, mode=mode))
        except Exception as exc:
            self.send_json({"error": f"챗봇 응답 생성 실패: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        assistant_ts = now_iso()
        elapsed_ms = int((time.time() - started) * 1000)
        citation_check = response.citation_check or validate_answer_citations(response.answer, response.sources)
        graph_trace = response.graph_trace or public_graph_trace(graph_state)
        public_payload = {
            "answer": response.answer,
            "mode": mode,
            "request_question": question,
            "sources": [source_to_public(source) for source in response.sources],
        }
        admin_payload = {
            **public_payload,
            "sources": [source_to_admin(source) for source in response.sources],
            "model_name": LLM_MODEL,
            "mode": mode,
            "elapsed_ms": elapsed_ms,
            "citation_check": citation_check,
            "graph_trace": graph_trace,
            "cli_output": build_cli_output(
                response.answer,
                response.sources,
                elapsed_ms,
                LLM_MODEL,
                question,
                citation_check,
                graph_trace,
            ),
        }
        ts = now_iso()
        with get_db() as con:
            user_cur = con.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, 'user', ?, ?)",
                (conversation_id, question, user_ts),
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
                    assistant_ts,
                ),
            )
            con.execute("UPDATE conversations SET updated_at = ?, title = CASE WHEN title = '새 상담' THEN ? ELSE title END WHERE id = ?", (assistant_ts, question[:40], conversation_id))
        message = {
            "id": cur.lastrowid,
            "role": "assistant",
            "content": response.answer,
            "payload": admin_payload if user["role"] == "admin" else public_payload,
            "created_at": assistant_ts,
        }
        user_message = {
            "id": user_cur.lastrowid,
            "role": "user",
            "content": question,
            "created_at": user_ts,
        }
        self.send_json({"conversation_id": conversation_id, "mode": mode, "user_message": user_message, "message": message})


def main() -> None:
    parser = argparse.ArgumentParser(description="건설현장 중대재해-산업안전 법령 상담 챗봇 Web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8200)
    args = parser.parse_args()
    init_db()
    server = ThreadingHTTPServer((args.host, args.port), WebAppHandler)
    print(f"건설현장 중대재해-산업안전 법령 상담 챗봇 Web UI: http://{args.host}:{args.port}")
    print(f"Admin account: {DEFAULT_ADMIN_USERNAME} / {DEFAULT_ADMIN_PASSWORD}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버를 종료합니다.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
