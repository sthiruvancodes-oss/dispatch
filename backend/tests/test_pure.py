"""Tests for the pure helpers — no browser, no network, no API keys."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_runner import (  # noqa: E402
    _for_speech,
    _preferred_sites_hint,
    _redact,
    _similar_speech,
    _task_text,
)
from voice_agent import CallRegistry, is_backchannel  # noqa: E402

# --- narration cleanup -------------------------------------------------------


def test_for_speech_strips_urls_and_markdown():
    out = _for_speech("Opening **https://example.com/thing** now")
    assert "http" not in out
    assert "*" not in out
    assert out.endswith(".")


def test_for_speech_capitalises_and_terminates():
    assert _for_speech("checking the page") == "Checking the page."


def test_for_speech_redacts_secrets():
    assert "hunter2" not in _for_speech("typed hunter2", extra=["hunter2"])


def test_for_speech_empty_input():
    assert _for_speech("   ") == ""


def test_for_speech_truncates_long_lines():
    assert len(_for_speech("word " * 200)) <= 181


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Opening the page", "Opening the page"),
        ("Opening the page", "Opening the page now"),
    ],
)
def test_similar_speech_catches_repeats(a, b):
    assert _similar_speech(a, b)


def test_similar_speech_allows_new_lines():
    assert not _similar_speech("Opening the page", "Reading the results")


def test_redact_leaves_short_secrets_alone():
    # A 2-char secret would corrupt ordinary text, so it's skipped by design.
    assert _redact("an ordinary line", extra=["an"]) == "an ordinary line"


# --- site preference ---------------------------------------------------------


def test_preferred_sites_hint_empty_by_default(monkeypatch):
    monkeypatch.delenv("PREFERRED_SITES", raising=False)
    assert _preferred_sites_hint() == ""


def test_preferred_sites_hint_lists_sites(monkeypatch):
    monkeypatch.setenv("PREFERRED_SITES", "example.com, other.org")
    hint = _preferred_sites_hint()
    assert "example.com" in hint and "other.org" in hint


def test_task_text_passes_instruction_through_untouched(monkeypatch):
    monkeypatch.delenv("PREFERRED_SITES", raising=False)
    task = _task_text("find a bike on craigslist", set(), "web")
    assert task.startswith("find a bike on craigslist")
    assert "Marketplace" not in task


def test_task_text_mentions_credentials_only_when_present(monkeypatch):
    monkeypatch.delenv("PREFERRED_SITES", raising=False)
    assert "x_user" not in _task_text("do a thing", set(), "web")
    assert "x_user" in _task_text("do a thing", {"x_user", "x_pass"}, "web")


def test_task_text_phone_asks_for_a_spoken_summary(monkeypatch):
    monkeypatch.delenv("PREFERRED_SITES", raising=False)
    assert "spoken summary" in _task_text("do a thing", set(), "phone")


# --- credential scoping ------------------------------------------------------


def _reload_creds(monkeypatch, **env):
    for key in ("LOGIN_DOMAIN", "LOGIN_USERNAME", "LOGIN_PASSWORD", "ALLOWED_DOMAINS"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    from agent_runner import _build_sensitive_data

    return _build_sensitive_data()


def test_credentials_withheld_without_login_domain(monkeypatch):
    assert _reload_creds(
        monkeypatch,
        LOGIN_USERNAME="me",
        LOGIN_PASSWORD="pw",
        ALLOWED_DOMAINS="example.com",
    ) == {}


def test_credentials_withheld_without_allowed_domains(monkeypatch):
    assert _reload_creds(
        monkeypatch,
        LOGIN_USERNAME="me",
        LOGIN_PASSWORD="pw",
        LOGIN_DOMAIN="https://*.example.com",
    ) == {}


def test_credentials_are_domain_scoped(monkeypatch):
    data = _reload_creds(
        monkeypatch,
        LOGIN_USERNAME="me",
        LOGIN_PASSWORD="pw",
        LOGIN_DOMAIN="https://*.example.com",
        ALLOWED_DOMAINS="*.example.com",
    )
    assert data == {"https://*.example.com": {"x_user": "me", "x_pass": "pw"}}


# --- voice -------------------------------------------------------------------


@pytest.mark.parametrize("text", ["mm-hm", "uh huh", "yeah", "okay!", "Got it.", "sure"])
def test_backchannel_detected(text):
    assert is_backchannel(text)


@pytest.mark.parametrize(
    "text",
    [
        "yeah search for a bike",
        "okay now open the other tab",
        "stop",
        "find me a couch under two hundred",
    ],
)
def test_real_instructions_are_not_backchannel(text):
    assert not is_backchannel(text)


def test_call_registry_denies_unknown_sid():
    registry = CallRegistry()
    assert not registry.is_authorized("CA-unknown")
    assert not registry.is_authorized("")


def test_call_registry_round_trip():
    registry = CallRegistry()
    registry.authorize("CA-1")
    assert registry.is_authorized("CA-1")
    registry.release("CA-1")
    assert not registry.is_authorized("CA-1")


def test_call_registry_evicts_oldest():
    registry = CallRegistry(max_entries=2)
    for sid in ("CA-1", "CA-2", "CA-3"):
        registry.authorize(sid)
    assert not registry.is_authorized("CA-1")
    assert registry.is_authorized("CA-3")
