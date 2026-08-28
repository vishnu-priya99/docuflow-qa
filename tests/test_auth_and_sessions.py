"""login/session creation, session isolation, session deletion."""
from __future__ import annotations

import uuid

import pytest

from tests.conftest import ask, upload_file

pytestmark = pytest.mark.asyncio


async def test_login_creates_user(client, user_id):
    response = await client.post("/api/auth/login", json={"user_id": user_id})
    assert response.status_code == 200
    assert response.json() == {"user_id": user_id}

    # Logging in again with the same id is idempotent.
    response = await client.post("/api/auth/login", json={"user_id": user_id})
    assert response.status_code == 200


async def test_login_rejects_empty_user_id(client):
    response = await client.post("/api/auth/login", json={"user_id": "   "})
    assert response.status_code in (400, 422)


async def test_session_creation_and_listing(client, auth_headers):
    r1 = await client.post("/api/sessions", json={"title": "Chat 1"}, headers=auth_headers)
    r2 = await client.post("/api/sessions", json={"title": "Chat 2"}, headers=auth_headers)
    assert r1.status_code == 201 and r2.status_code == 201
    assert r1.json()["session_id"] != r2.json()["session_id"]

    listing = await client.get("/api/sessions", headers=auth_headers)
    ids = {s["session_id"] for s in listing.json()["sessions"]}
    assert {r1.json()["session_id"], r2.json()["session_id"]} <= ids


async def test_missing_auth_header_is_rejected(client):
    response = await client.get("/api/sessions")
    assert response.status_code == 401


async def test_session_isolation_between_users(client):
    user_a = f"user_{uuid.uuid4().hex[:8]}"
    user_b = f"user_{uuid.uuid4().hex[:8]}"
    await client.post("/api/auth/login", json={"user_id": user_a})
    await client.post("/api/auth/login", json={"user_id": user_b})

    headers_a = {"X-User-Id": user_a}
    headers_b = {"X-User-Id": user_b}

    created = await client.post("/api/sessions", json={"title": "A's chat"}, headers=headers_a)
    session_id = created.json()["session_id"]

    # B cannot see A's session at all.
    listing_b = await client.get("/api/sessions", headers=headers_b)
    assert session_id not in {s["session_id"] for s in listing_b.json()["sessions"]}

    # B cannot fetch, chat in, or delete A's session directly by id.
    assert (await client.get(f"/api/sessions/{session_id}", headers=headers_b)).status_code == 404
    assert (
        await client.post(f"/api/sessions/{session_id}/chat", json={"question": "hi"}, headers=headers_b)
    ).status_code == 404
    assert (await client.delete(f"/api/sessions/{session_id}", headers=headers_b)).status_code == 404

    # A can still access it.
    assert (await client.get(f"/api/sessions/{session_id}", headers=headers_a)).status_code == 200


async def test_deleting_session_cascades_everything(client):
    user = f"user_{uuid.uuid4().hex[:8]}"
    await client.post("/api/auth/login", json={"user_id": user})
    headers = {"X-User-Id": user}

    created = await client.post("/api/sessions", json={"title": "to delete"}, headers=headers)
    session_id = created.json()["session_id"]

    await upload_file(
        client, session_id=session_id, headers=headers, filename="a.txt", content=b"Some content about apples."
    )
    await ask(client, session_id=session_id, headers=headers, question="What is this about?")

    files_before = await client.get(f"/api/sessions/{session_id}/files", headers=headers)
    assert len(files_before.json()["files"]) == 1
    messages_before = await client.get(f"/api/sessions/{session_id}/messages", headers=headers)
    assert len(messages_before.json()["messages"]) == 2  # user + assistant

    delete_resp = await client.delete(f"/api/sessions/{session_id}", headers=headers)
    assert delete_resp.status_code == 204

    # The session and everything scoped to it is gone.
    assert (await client.get(f"/api/sessions/{session_id}", headers=headers)).status_code == 404
    assert (await client.get(f"/api/sessions/{session_id}/files", headers=headers)).status_code == 404
    assert (await client.get(f"/api/sessions/{session_id}/messages", headers=headers)).status_code == 404

    # Deleting again is a no-op 404, not an error.
    assert (await client.delete(f"/api/sessions/{session_id}", headers=headers)).status_code == 404
