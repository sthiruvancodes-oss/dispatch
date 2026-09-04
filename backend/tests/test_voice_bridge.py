"""Exercise the Twilio <-> Deepgram bridge with fake sockets.

This covers our side of the wire: audio forwarding, barge-in, function dispatch,
narration and stream authorization. It does NOT verify Deepgram's contract —
that the Settings message is accepted, that Flux behaves as configured, or that
audio actually sounds right. Only a real call proves those.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import voice_agent  # noqa: E402
from agent_runner import TaskSession  # noqa: E402
from voice_agent import VoiceAgentSession, build_settings, call_registry  # noqa: E402


class FakeTwilio:
    """Stands in for the FastAPI WebSocket Twilio connects to.

    on_exhausted="idle" mimics a live call waiting for more audio. Pump tests use
    "raise" instead, so an implementation that fails to stop when it should fails
    the test immediately rather than blocking until the suite times out.
    """

    def __init__(
        self, inbound: list[str] | None = None, on_exhausted: str = "raise"
    ) -> None:
        self.sent: list[str] = []
        self._inbound = list(inbound or [])
        self._on_exhausted = on_exhausted

    async def receive_text(self) -> str:
        if not self._inbound:
            if self._on_exhausted == "idle":
                await asyncio.sleep(3600)
            raise AssertionError("read past the end of the script; pump did not stop")
        return self._inbound.pop(0)

    async def send_text(self, text: str) -> None:
        self.sent.append(text)

    def events(self) -> list[dict]:
        return [json.loads(s) for s in self.sent]


class FakeDeepgram:
    """Stands in for the Deepgram Voice Agent socket."""

    def __init__(self, inbound: list | None = None) -> None:
        self.sent: list = []
        self._inbound = list(inbound or [])

    async def send(self, data) -> None:
        self.sent.append(data)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._inbound:
            raise StopAsyncIteration
        await asyncio.sleep(0)
        return self._inbound.pop(0)

    def json_sent(self) -> list[dict]:
        return [json.loads(s) for s in self.sent if isinstance(s, str)]


class FakeConnect:
    def __init__(self, dg: FakeDeepgram) -> None:
        self.dg = dg

    async def __aenter__(self):
        return self.dg

    async def __aexit__(self, *_):
        return False


class FakeRunner:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.stopped = 0
        self.session: TaskSession | None = None

    async def start_task(self, instruction: str, source: str = "web") -> TaskSession:
        self.started.append(instruction)
        self.session = TaskSession(instruction=instruction, source=source)
        return self.session

    async def stop_current(self) -> TaskSession | None:
        self.stopped += 1
        return self.session

    def current_running(self):
        return None


@pytest.fixture(autouse=True)
def _clear_registry():
    yield
    for sid in list(call_registry._sids):
        call_registry.release(sid)


@pytest.fixture(autouse=True)
def _no_network_narration(monkeypatch):
    """Narration rewriting calls OpenAI; keep the suite offline and deterministic.

    Tests that care about the rewriter itself override this.
    """
    async def passthrough(self, line):
        return line

    monkeypatch.setattr(VoiceAgentSession, "_humanize", passthrough)


# --- Settings ----------------------------------------------------------------


def test_settings_matches_twilio_audio_and_configured_providers(monkeypatch):
    for key in ("VOICE_LLM_MODEL", "VOICE_LLM_TEMPERATURE", "VOICE_LLM_ENDPOINT"):
        monkeypatch.delenv(key, raising=False)

    settings = build_settings()
    assert settings["type"] == "Settings"

    # Matching Twilio's native format exactly means no resampling in the loop.
    assert settings["audio"]["input"] == {"encoding": "mulaw", "sample_rate": 8000}
    assert settings["audio"]["output"]["encoding"] == "mulaw"
    assert settings["audio"]["output"]["sample_rate"] == 8000
    assert settings["audio"]["output"]["container"] == "none"

    agent = settings["agent"]
    assert agent["listen"]["provider"]["type"] == "deepgram"
    assert agent["speak"]["provider"]["type"] == "deepgram"

    think = agent["think"]["provider"]
    assert think["type"] == "open_ai"
    assert think["model"] == "gpt-4.1"
    assert think["temperature"] == 0.7

    assert {f["name"] for f in agent["think"]["functions"]} == {
        "start_browser_task",
        "stop_browser_task",
    }


def test_settings_carries_turn_detection_knobs(monkeypatch):
    monkeypatch.setenv("VOICE_EOT_THRESHOLD", "0.85")
    monkeypatch.setenv("VOICE_EOT_TIMEOUT_MS", "2500")
    listen = build_settings()["agent"]["listen"]["provider"]
    assert listen["eot_threshold"] == 0.85
    assert listen["eot_timeout_ms"] == 2500


def test_settings_uses_custom_endpoint_when_model_is_not_hosted(monkeypatch):
    monkeypatch.setenv("VOICE_LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    think = build_settings()["agent"]["think"]["provider"]
    assert think["endpoint"]["url"].endswith("/chat/completions")
    assert think["endpoint"]["headers"]["authorization"] == "Bearer sk-test"


# --- audio in ----------------------------------------------------------------


async def test_media_frames_reach_deepgram_as_raw_bytes():
    call_registry.authorize("CA-ok")
    dg = FakeDeepgram()
    twilio = FakeTwilio(
        [
            json.dumps(
                {"event": "start", "start": {"streamSid": "MZ-1", "callSid": "CA-ok"}}
            ),
            json.dumps(
                {"event": "media", "media": {"payload": base64.b64encode(b"\xff\xfe").decode()}}
            ),
            json.dumps({"event": "stop"}),
        ]
    )
    session = VoiceAgentSession(twilio, FakeRunner())
    session.dg = dg

    await session._pump_twilio()

    assert session.stream_sid == "MZ-1"
    assert b"\xff\xfe" in dg.sent


async def test_unauthorized_callsid_is_refused_and_forwards_nothing():
    dg = FakeDeepgram()
    twilio = FakeTwilio(
        [
            json.dumps(
                {"event": "start", "start": {"streamSid": "MZ-1", "callSid": "CA-forged"}}
            ),
            json.dumps(
                {"event": "media", "media": {"payload": base64.b64encode(b"\x01").decode()}}
            ),
        ]
    )
    session = VoiceAgentSession(twilio, FakeRunner())
    session.dg = dg

    await session._pump_twilio()

    assert session._closing.is_set()
    assert dg.sent == []


# --- audio out ---------------------------------------------------------------


async def test_deepgram_audio_is_wrapped_into_twilio_media_frames():
    twilio = FakeTwilio()
    session = VoiceAgentSession(twilio, FakeRunner())
    session.stream_sid = "MZ-1"
    session.dg = FakeDeepgram([b"\xaa\xbb"])

    await session._pump_deepgram()

    frame = twilio.events()[0]
    assert frame["event"] == "media"
    assert frame["streamSid"] == "MZ-1"
    assert base64.b64decode(frame["media"]["payload"]) == b"\xaa\xbb"


# --- barge-in ----------------------------------------------------------------


async def test_user_started_speaking_flushes_queued_audio():
    """Stopping isn't enough — Twilio's buffer has to be cleared too."""
    twilio = FakeTwilio()
    session = VoiceAgentSession(twilio, FakeRunner())
    session.stream_sid = "MZ-1"

    await session._handle_deepgram_json({"type": "UserStartedSpeaking"})

    assert twilio.events() == [{"event": "clear", "streamSid": "MZ-1"}]


# --- function calling --------------------------------------------------------


async def test_start_browser_task_runs_and_answers_deepgram():
    twilio = FakeTwilio()
    runner = FakeRunner()
    session = VoiceAgentSession(twilio, runner)
    session.dg = FakeDeepgram()

    await session._handle_deepgram_json(
        {
            "type": "FunctionCallRequest",
            "functions": [
                {
                    "id": "fc-1",
                    "name": "start_browser_task",
                    "client_side": True,
                    "arguments": json.dumps({"instruction": "open hacker news"}),
                }
            ],
        }
    )

    assert runner.started == ["open hacker news"]
    reply = session.dg.json_sent()[-1]
    assert reply["type"] == "FunctionCallResponse"
    assert reply["id"] == "fc-1"
    assert reply["name"] == "start_browser_task"

    await session.aclose()


async def test_stop_browser_task_stops_the_run():
    session = VoiceAgentSession(FakeTwilio(), FakeRunner())
    session.dg = FakeDeepgram()

    await session._handle_deepgram_json(
        {
            "type": "FunctionCallRequest",
            "functions": [
                {"id": "fc-2", "name": "stop_browser_task", "client_side": True, "arguments": "{}"}
            ],
        }
    )

    assert session.runner.stopped == 1
    assert session.dg.json_sent()[-1]["id"] == "fc-2"


async def test_malformed_arguments_still_get_a_response():
    """A missing reply would hang the conversation, so answer even on bad JSON."""
    session = VoiceAgentSession(FakeTwilio(), FakeRunner())
    session.dg = FakeDeepgram()

    await session._handle_deepgram_json(
        {
            "type": "FunctionCallRequest",
            "functions": [
                {
                    "id": "fc-3",
                    "name": "start_browser_task",
                    "client_side": True,
                    "arguments": "{not json",
                }
            ],
        }
    )

    assert session.dg.json_sent()[-1]["type"] == "FunctionCallResponse"
    await session.aclose()


async def test_server_side_functions_are_left_to_deepgram():
    session = VoiceAgentSession(FakeTwilio(), FakeRunner())
    session.dg = FakeDeepgram()

    await session._handle_deepgram_json(
        {
            "type": "FunctionCallRequest",
            "functions": [
                {"id": "fc-4", "name": "something", "client_side": False, "arguments": "{}"}
            ],
        }
    )

    assert session.dg.sent == []


# --- narration ---------------------------------------------------------------


async def test_step_progress_is_injected_as_the_agent_speaking():
    session = VoiceAgentSession(FakeTwilio(), FakeRunner())
    session.dg = FakeDeepgram()
    task_session = TaskSession(instruction="open hacker news")

    narrating = asyncio.create_task(session._narrate(task_session))
    await asyncio.sleep(0.01)  # let it subscribe

    await task_session.emit({"type": "speak", "text": "Opening the page."})
    await asyncio.sleep(0.01)
    await task_session.emit({"type": "done", "result": "Found five stories."})
    await asyncio.wait_for(narrating, timeout=2)

    injected = [m for m in session.dg.json_sent() if m["type"] == "InjectAgentMessage"]
    assert injected[0]["message"] == "Opening the page."
    # Progress queues behind the caller rather than talking over them.
    assert injected[0]["behavior"] == "queue"
    assert injected[-1]["message"] == "Found five stories."


# --- lifecycle ---------------------------------------------------------------


async def test_settings_is_the_first_thing_sent(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    dg = FakeDeepgram()
    monkeypatch.setattr(voice_agent.websockets, "connect", lambda *a, **k: FakeConnect(dg))

    session = VoiceAgentSession(FakeTwilio(on_exhausted="idle"), FakeRunner())
    await asyncio.wait_for(session.run(), timeout=2)

    assert dg.json_sent()[0]["type"] == "Settings"


async def test_run_bails_without_a_deepgram_key(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    called = False

    def _connect(*_a, **_k):
        nonlocal called
        called = True
        raise AssertionError("should not connect without a key")

    monkeypatch.setattr(voice_agent.websockets, "connect", _connect)
    await VoiceAgentSession(FakeTwilio(on_exhausted="idle"), FakeRunner()).run()
    assert not called


async def test_hangup_stops_a_running_browser_task():
    runner = FakeRunner()
    session = VoiceAgentSession(FakeTwilio(), runner)
    running = TaskSession(instruction="x")
    runner.session = running
    runner.current_running = lambda: running

    await session.aclose()

    assert runner.stopped == 1


async def test_final_result_is_cleaned_before_it_is_spoken():
    """Regression: a live run returned a result containing a URL and newlines.

    Injected raw, TTS reads "https://example.com" out character by character.
    """
    session = VoiceAgentSession(FakeTwilio(), FakeRunner())
    session.dg = FakeDeepgram()
    task_session = TaskSession(instruction="x")

    narrating = asyncio.create_task(session._narrate(task_session))
    await asyncio.sleep(0.01)
    await task_session.emit(
        {
            "type": "done",
            "result": "The exact heading text on https://example.com is:\n\nExample Domain",
        }
    )
    await asyncio.wait_for(narrating, timeout=2)

    spoken = session.dg.json_sent()[-1]["message"]
    assert "http" not in spoken
    assert "example.com" not in spoken
    assert "\n" not in spoken
    assert "Example Domain" in spoken


async def test_long_final_result_is_not_clipped_to_a_step_line():
    """Final summaries get more room than a one-clause progress update."""
    session = VoiceAgentSession(FakeTwilio(), FakeRunner())
    session.dg = FakeDeepgram()
    task_session = TaskSession(instruction="x")

    narrating = asyncio.create_task(session._narrate(task_session))
    await asyncio.sleep(0.01)
    await task_session.emit({"type": "done", "result": "story number. " * 30})
    await asyncio.wait_for(narrating, timeout=2)

    assert len(session.dg.json_sent()[-1]["message"]) > 200


@pytest.mark.parametrize("behavior", ["queue", "interrupt"])
async def test_valid_inject_behaviors_pass_through(behavior):
    session = VoiceAgentSession(FakeTwilio(), FakeRunner())
    session.dg = FakeDeepgram()
    await session._inject("hello", behavior=behavior)
    assert session.dg.json_sent()[-1]["behavior"] == behavior


@pytest.mark.parametrize("behavior", ["wait_for_silence", "immediate", "", "finish_current"])
async def test_unknown_inject_behavior_is_clamped(behavior):
    """Regression: 'wait_for_silence' was rejected as UNPARSABLE_CLIENT_MESSAGE,
    which tore down the Deepgram session and dropped a live call."""
    session = VoiceAgentSession(FakeTwilio(), FakeRunner())
    session.dg = FakeDeepgram()
    await session._inject("hello", behavior=behavior)
    sent = session.dg.json_sent()[-1]
    assert sent["behavior"] in voice_agent.INJECT_BEHAVIORS


async def test_done_narration_uses_a_valid_behavior():
    session = VoiceAgentSession(FakeTwilio(), FakeRunner())
    session.dg = FakeDeepgram()
    task_session = TaskSession(instruction="x")
    narrating = asyncio.create_task(session._narrate(task_session))
    await asyncio.sleep(0.01)
    await task_session.emit({"type": "done", "result": "All finished."})
    await asyncio.wait_for(narrating, timeout=2)
    assert session.dg.json_sent()[-1]["behavior"] in voice_agent.INJECT_BEHAVIORS


# --- narration rewriting -----------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "Prepare brief spoken summary to relay to the user",
        "Compile the final answer for the user",
        "Report back the results",
    ],
)
def test_bookkeeping_goals_are_recognised(line):
    """Heard on a real call: 'prepare brief spoken summary to relay to the user'."""
    assert voice_agent.looks_like_bookkeeping(line)


@pytest.mark.parametrize(
    "line",
    ["Navigate to the Swatch homepage", "Click the accept cookies button"],
)
def test_real_progress_is_not_bookkeeping(line):
    assert not voice_agent.looks_like_bookkeeping(line)


async def test_humanize_drops_bookkeeping_when_offline(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    session = VoiceAgentSession(FakeTwilio(), FakeRunner())
    monkeypatch.undo()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = await VoiceAgentSession._humanize(
        session, "Prepare brief spoken summary to relay to the user"
    )
    assert out == ""


async def test_humanize_keeps_real_progress_when_offline(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    session = VoiceAgentSession(FakeTwilio(), FakeRunner())
    out = await VoiceAgentSession._humanize(session, "Navigate to the Swatch homepage")
    assert out == "Navigate to the Swatch homepage"


async def test_skipped_narration_is_never_injected(monkeypatch):
    session = VoiceAgentSession(FakeTwilio(), FakeRunner())
    session.dg = FakeDeepgram()

    async def skip_everything(self, line):
        return ""

    monkeypatch.setattr(VoiceAgentSession, "_humanize", skip_everything)
    task_session = TaskSession(instruction="x")
    narrating = asyncio.create_task(session._narrate(task_session))
    await asyncio.sleep(0.01)
    await task_session.emit({"type": "speak", "text": "internal bookkeeping"})
    await asyncio.sleep(0.05)
    await task_session.emit({"type": "done", "result": "All done."})
    await asyncio.wait_for(narrating, timeout=2)

    injected = [m["message"] for m in session.dg.json_sent()]
    assert "internal bookkeeping" not in injected
