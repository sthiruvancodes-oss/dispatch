"""POST /voice is the front door: only Twilio, and only allowed callers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from twilio.request_validator import RequestValidator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main  # noqa: E402
from voice_agent import call_registry  # noqa: E402

AUTH_TOKEN = "test-auth-token"
BASE = "https://dispatch.example.com"
ALLOWED = "+15005550006"
BLOCKED = "+15550000000"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", AUTH_TOKEN)
    monkeypatch.setenv("PUBLIC_BASE_URL", BASE)
    monkeypatch.setenv("ALLOWED_CALLERS", ALLOWED)
    yield
    for sid in list(call_registry._sids):
        call_registry.release(sid)


@pytest.fixture
def client():
    return TestClient(main.app)


def _signed(form: dict[str, str]) -> dict[str, str]:
    signature = RequestValidator(AUTH_TOKEN).compute_signature(f"{BASE}/voice", form)
    return {"X-Twilio-Signature": signature}


def test_rejects_missing_signature(client):
    res = client.post("/voice", data={"From": ALLOWED, "CallSid": "CA-1"})
    assert res.status_code == 403


def test_rejects_forged_signature(client):
    res = client.post(
        "/voice",
        data={"From": ALLOWED, "CallSid": "CA-1"},
        headers={"X-Twilio-Signature": "not-a-real-signature"},
    )
    assert res.status_code == 403


def test_rejects_when_auth_token_unset(client, monkeypatch):
    """No token means we can't verify anything, so nothing is trusted."""
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
    form = {"From": ALLOWED, "CallSid": "CA-1"}
    res = client.post("/voice", data=form, headers=_signed(form))
    assert res.status_code == 403


def test_rejects_caller_not_on_allowlist(client):
    form = {"From": BLOCKED, "CallSid": "CA-blocked"}
    res = client.post("/voice", data=form, headers=_signed(form))
    assert res.status_code == 200
    assert "<Hangup/>" in res.text
    assert "<Stream" not in res.text
    assert not call_registry.is_authorized("CA-blocked")


def test_empty_allowlist_rejects_everyone(client, monkeypatch):
    """Deny by default: an unset allowlist must not mean 'allow anyone'."""
    monkeypatch.setenv("ALLOWED_CALLERS", "")
    form = {"From": ALLOWED, "CallSid": "CA-2"}
    res = client.post("/voice", data=form, headers=_signed(form))
    assert "<Hangup/>" in res.text
    assert not call_registry.is_authorized("CA-2")


def test_allowed_caller_gets_a_media_stream(client):
    form = {"From": ALLOWED, "CallSid": "CA-ok"}
    res = client.post("/voice", data=form, headers=_signed(form))
    assert res.status_code == 200
    # <Connect>, not <Start>: audio has to flow back to the caller.
    assert "<Connect>" in res.text
    assert '<Stream url="wss://dispatch.example.com/voice/stream" />' in res.text
    assert call_registry.is_authorized("CA-ok")


def test_stream_socket_rejects_unauthorized_call(client):
    """The WebSocket upgrade carries no signature, so it checks the CallSid."""
    from voice_agent import VoiceAgentSession

    session = VoiceAgentSession(twilio_ws=None, runner=main.runner)
    assert not call_registry.is_authorized("CA-never-authorized")
    assert session.stream_sid is None
