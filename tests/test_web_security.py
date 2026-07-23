from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import web_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "DB_PATH", tmp_path / "chatbot_ui.sqlite3")
    monkeypatch.setattr(web_app, "ALLOW_REGISTRATION", True)
    web_app._LOGIN_ATTEMPTS.clear()
    with TestClient(web_app.app) as test_client:
        yield test_client


def register_and_login(client: TestClient, username: str = "normal-user") -> tuple[dict, str]:
    registered = client.post(
        "/api/register",
        json={"username": username, "password": "long-password-123"},
    )
    assert registered.status_code == 200
    logged_in = client.post(
        "/api/login",
        json={"username": username, "password": "long-password-123"},
    )
    assert logged_in.status_code == 200
    set_cookie = logged_in.headers["set-cookie"].lower()
    assert "httponly" in set_cookie
    assert "samesite=lax" in set_cookie
    payload = logged_in.json()
    return payload["user"], payload["csrf_token"]


def test_health_and_security_headers(client: TestClient):
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_session_cookie_and_csrf_protection(client: TestClient):
    _, csrf_token = register_and_login(client)

    cookie = client.cookies.get("ksafety_session")
    assert cookie
    without_csrf = client.post("/api/conversations", json={"title": "차단 대상"})
    assert without_csrf.status_code == 403

    created = client.post(
        "/api/conversations",
        json={"title": "보안 테스트", "mode": "general"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert created.status_code == 200
    assert created.json()["conversation"]["title"] == "보안 테스트"

    me = client.get("/api/me")
    assert me.status_code == 200
    assert me.json()["csrf_token"] == csrf_token


def test_conversations_are_isolated_by_account(client: TestClient):
    _, first_csrf = register_and_login(client, "first-user")
    created = client.post(
        "/api/conversations",
        json={"title": "첫 사용자 상담"},
        headers={"X-CSRF-Token": first_csrf},
    )
    conversation_id = created.json()["conversation"]["id"]
    client.post("/api/logout", headers={"X-CSRF-Token": first_csrf})

    _, _ = register_and_login(client, "second-user")
    response = client.get(f"/api/conversations/{conversation_id}")

    assert response.status_code == 404


def test_registration_can_be_disabled(client: TestClient, monkeypatch):
    monkeypatch.setattr(web_app, "ALLOW_REGISTRATION", False)

    response = client.post(
        "/api/register",
        json={"username": "blocked-user", "password": "long-password-123"},
    )

    assert response.status_code == 403
    assert "비활성화" in response.json()["error"]


def test_login_rate_limit(client: TestClient, monkeypatch):
    register_and_login(client, "rate-user")
    csrf_token = client.get("/api/me").json()["csrf_token"]
    client.post("/api/logout", headers={"X-CSRF-Token": csrf_token})
    monkeypatch.setattr(web_app, "LOGIN_ATTEMPT_LIMIT", 2)

    for _ in range(2):
        response = client.post(
            "/api/login",
            json={"username": "rate-user", "password": "wrong-password"},
        )
        assert response.status_code == 401

    limited = client.post(
        "/api/login",
        json={"username": "rate-user", "password": "wrong-password"},
    )
    assert limited.status_code == 429
