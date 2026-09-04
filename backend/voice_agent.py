"""Bridge Twilio Media Streams to the Deepgram Voice Agent API.

    caller <--mulaw 8k--> Twilio <--ws--> us <--ws--> Deepgram (STT + LLM + TTS)

Deepgram owns the whole speech pipeline: Flux for transcription and turn
detection, an OpenAI model for the conversation, and Deepgram TTS for the voice.
We own two things it can't do — starting the browser agent (a client-side
function call) and narrating what that agent is doing (InjectAgentMessage).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from typing import Any

import websockets

from agent_runner import AgentRunner, TaskSession, _for_speech

logger = logging.getLogger("dispatch.voice")

DEEPGRAM_AGENT_URL = "wss://agent.deepgram.com/v1/agent/converse"

# Twilio's native telephony format. Matching it exactly on both sides means no
# resampling anywhere in the loop.
TWILIO_ENCODING = "mulaw"
TWILIO_SAMPLE_RATE = 8000

# Verified against the live API: anything else is rejected with
# UNPARSABLE_CLIENT_MESSAGE, which kills the whole session and drops the call.
INJECT_BEHAVIORS = frozenset({"queue", "interrupt"})
DEFAULT_INJECT_BEHAVIOR = "queue"

NARRATION_MODEL = os.getenv("VOICE_NARRATION_MODEL", "gpt-4.1-mini")
NARRATION_TIMEOUT_S = float(os.getenv("VOICE_NARRATION_TIMEOUT_S", "2.5"))

NARRATION_PROMPT = """
Rewrite one line of a browser agent's internal notes as something a person would
say out loud on a phone call, while working.

Rules:
- One short sentence. Contractions. Present tense.
- Say what is happening, never what "should be prepared" or "relayed to the user".
- Never mention goals, steps, tasks, summaries, or the user themselves.
- No URLs, no markdown, no quotes.
- If the note is purely internal bookkeeping with nothing a caller would care
  about, reply with exactly: SKIP

Examples:
  "Prepare brief spoken summary to relay to the user" -> SKIP
  "Navigate to the Swatch homepage" -> Pulling up Swatch now.
  "Extract the top three product names from the listing grid" -> Reading through the products.
  "Click the accept cookies button" -> Getting the cookie banner out of the way.
""".strip()

# Internal bookkeeping that should never be spoken, if the rewriter is
# unavailable and we fall back to the raw goal.
_META_GOAL_RE = re.compile(
    r"\b(relay|summar|report back|inform the user|prepare|compile|"
    r"final answer|task is complete|call done|output)\b",
    re.I,
)

SYSTEM_PROMPT = """
You are the voice of Dispatch. You take a spoken request and drive a real Chrome
window to do it, narrating as you go.

How to talk: like a competent person on the phone. Short sentences. Contractions.
No lists, no markdown, no URLs, no step numbers. Never spell out a web address.

When the caller describes something to do in a browser, call start_browser_task
with their request in plain words. Don't ask for permission first, and don't
repeat their whole request back to them — just say what you're about to do and
start.

If they only greeted you or said something unclear, talk to them normally and
wait. Don't start a task on a guess.

While a task is running you'll receive progress updates to read out. If the
caller changes their mind mid-run, call stop_browser_task.
""".strip()

# Short acknowledgements that should never be mistaken for a new instruction
# while the agent is narrating. Turn detection handles most of this; the filter
# catches what slips through.
# Matched against letters only, so "mm-hm", "mm hm" and "mmhm" all collapse to
# the same token and punctuation can't smuggle an acknowledgement past the filter.
_BACKCHANNEL_RE = re.compile(
    r"^(?:"
    r"[mh]+|"
    r"uhhuh|ahha|aha|"
    r"yeah|yep|yup|yes|ya|"
    r"ok|okay|kay|alright|"
    r"right|sure|gotit|cool|nice|wow|oh|isee|soundsgood|thanks|thankyou"
    r")$",
    re.I,
)


def looks_like_bookkeeping(text: str) -> bool:
    """Internal agent chatter a caller should never hear."""
    return bool(_META_GOAL_RE.search(text))


def is_backchannel(text: str) -> bool:
    letters = re.sub(r"[^a-z]", "", text.lower())
    return bool(letters) and bool(_BACKCHANNEL_RE.match(letters))


class CallRegistry:
    """CallSids that came through an authorized POST /voice.

    The Media Streams WebSocket upgrade carries no Twilio signature, so without
    this the caller allowlist would be bypassable by connecting to the stream
    socket directly.
    """

    def __init__(self, max_entries: int = 64) -> None:
        self._sids: list[str] = []
        self._max = max_entries

    def authorize(self, call_sid: str) -> None:
        if not call_sid:
            return
        if call_sid in self._sids:
            return
        self._sids.append(call_sid)
        if len(self._sids) > self._max:
            self._sids.pop(0)

    def is_authorized(self, call_sid: str) -> bool:
        return bool(call_sid) and call_sid in self._sids

    def release(self, call_sid: str) -> None:
        if call_sid in self._sids:
            self._sids.remove(call_sid)


call_registry = CallRegistry()


FUNCTIONS: list[dict[str, Any]] = [
    {
        "name": "start_browser_task",
        "description": (
            "Drive the browser to do what the caller asked. Use this as soon as "
            "they describe a concrete task. Not for greetings or small talk."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": (
                        "The caller's request in plain words, including any site "
                        "they named and any limits like a price cap."
                    ),
                }
            },
            "required": ["instruction"],
        },
    },
    {
        "name": "stop_browser_task",
        "description": "Stop the run in progress, when the caller asks you to stop or start over.",
        "parameters": {"type": "object", "properties": {}},
    },
]


def _listen_provider() -> dict[str, Any]:
    """Flux does transcription and turn detection in one model.

    eot_threshold is the confidence needed to call the caller's turn over. Raise
    it if the agent cuts people off; lower it if replies feel sluggish.
    """
    return {
        "type": "deepgram",
        "model": os.getenv("VOICE_STT_MODEL", "flux-general-en"),
        "eot_threshold": float(os.getenv("VOICE_EOT_THRESHOLD", "0.7")),
        "eager_eot_threshold": float(os.getenv("VOICE_EAGER_EOT_THRESHOLD", "0.5")),
        # Deepgram's 5s default suits open-ended chat. Dispatch instructions are
        # short and directive, so cut the worst-case wait.
        "eot_timeout_ms": int(os.getenv("VOICE_EOT_TIMEOUT_MS", "3000")),
    }


def _think_provider() -> dict[str, Any]:
    """OpenAI via Deepgram's hosted integration, or a custom endpoint.

    Deepgram keeps an allowlist of hosted models. If VOICE_LLM_MODEL isn't on it,
    set VOICE_LLM_ENDPOINT to an OpenAI-compatible URL and the same model works
    through the custom-endpoint path instead.
    """
    provider: dict[str, Any] = {
        "type": "open_ai",
        "model": os.getenv("VOICE_LLM_MODEL", "gpt-4.1"),
        "temperature": float(os.getenv("VOICE_LLM_TEMPERATURE", "0.7")),
    }
    endpoint = (os.getenv("VOICE_LLM_ENDPOINT") or "").strip()
    if endpoint:
        headers = {}
        key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if key:
            headers["authorization"] = f"Bearer {key}"
        provider["endpoint"] = {"url": endpoint, "headers": headers}
    return provider


def _speak_provider() -> dict[str, Any]:
    """v2 is the Flux voice family, v1 is Aura. Worth trying both on a real call."""
    return {
        "type": "deepgram",
        "version": os.getenv("VOICE_TTS_VERSION", "v2"),
        "model": os.getenv("VOICE_TTS_MODEL", "flux-kit-en"),
    }


def build_settings() -> dict[str, Any]:
    return {
        "type": "Settings",
        "audio": {
            "input": {
                "encoding": TWILIO_ENCODING,
                "sample_rate": TWILIO_SAMPLE_RATE,
            },
            "output": {
                "encoding": TWILIO_ENCODING,
                "sample_rate": TWILIO_SAMPLE_RATE,
                "container": "none",
            },
        },
        "agent": {
            "language": "en",
            "greeting": os.getenv(
                "VOICE_GREETING", "Hey — what should I do in the browser?"
            ),
            "listen": {"provider": _listen_provider()},
            "think": {
                "provider": _think_provider(),
                "prompt": SYSTEM_PROMPT,
                "functions": FUNCTIONS,
            },
            "speak": {"provider": _speak_provider()},
        },
    }


class VoiceAgentSession:
    """One phone call: pumps audio both ways and runs the browser agent."""

    def __init__(self, twilio_ws: Any, runner: AgentRunner) -> None:
        self.twilio_ws = twilio_ws
        self.runner = runner
        self.stream_sid: str | None = None
        self.call_sid: str | None = None
        self.dg: Any = None
        self.task_session: TaskSession | None = None
        self._narration_task: asyncio.Task | None = None
        self._openai: Any = None
        self._closing = asyncio.Event()

    # ---- outbound to Twilio -------------------------------------------------

    async def _send_audio(self, payload: bytes) -> None:
        if not self.stream_sid:
            return
        await self.twilio_ws.send_text(
            json.dumps(
                {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": base64.b64encode(payload).decode("ascii")},
                }
            )
        )

    async def _clear_audio(self) -> None:
        """Drop whatever Twilio has buffered but not yet played.

        Without this the agent finishes its interrupted sentence a beat after the
        caller starts talking, which is the clearest tell that it's a machine.
        """
        if not self.stream_sid:
            return
        await self.twilio_ws.send_text(
            json.dumps({"event": "clear", "streamSid": self.stream_sid})
        )

    # ---- outbound to Deepgram ----------------------------------------------

    async def _inject(
        self, message: str, behavior: str = DEFAULT_INJECT_BEHAVIOR
    ) -> None:
        """Put words in the agent's mouth without a caller turn.

        Narration queues so progress updates wait their turn rather than talking
        over the caller. An unknown behavior is clamped rather than sent: Deepgram
        drops the session on an unparsable message, which ends the call.
        """
        if self.dg is None or not message:
            return
        if behavior not in INJECT_BEHAVIORS:
            logger.warning("Unknown inject behavior %r; using %s", behavior,
                           DEFAULT_INJECT_BEHAVIOR)
            behavior = DEFAULT_INJECT_BEHAVIOR
        try:
            await self.dg.send(
                json.dumps(
                    {
                        "type": "InjectAgentMessage",
                        "message": message,
                        "behavior": behavior,
                    }
                )
            )
        except Exception:
            logger.debug("InjectAgentMessage failed", exc_info=True)

    async def _function_response(self, call_id: str, name: str, content: str) -> None:
        if self.dg is None:
            return
        await self.dg.send(
            json.dumps(
                {
                    "type": "FunctionCallResponse",
                    "id": call_id,
                    "name": name,
                    "content": content,
                    "client_side": True,
                }
            )
        )

    # ---- browser agent ------------------------------------------------------

    async def _start_task(self, instruction: str) -> str:
        try:
            session = await self.runner.start_task(instruction, source="phone")
        except RuntimeError:
            return "A task is already running. Ask if they want to stop it first."
        except ValueError as exc:
            logger.warning("Could not start task: %s", exc)
            return "The browser agent isn't configured. Apologise and end the call."

        self.task_session = session
        self._narration_task = asyncio.create_task(self._narrate(session))
        return "Started. Narrate progress as updates arrive."

    async def _stop_task(self) -> str:
        try:
            await self.runner.stop_current()
        except RuntimeError:
            return "Nothing was running."
        return "Stopped."

    async def _humanize(self, line: str) -> str:
        """Turn an internal agent goal into something worth hearing.

        Falls back to the raw line on any failure or delay — narration must never
        block on this, and a slightly stiff sentence beats silence.
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            return "" if looks_like_bookkeeping(line) else line

        if self._openai is None:
            key = (os.getenv("OPENAI_API_KEY") or "").strip()
            if not key:
                return "" if looks_like_bookkeeping(line) else line
            self._openai = AsyncOpenAI(api_key=key)

        try:
            resp = await asyncio.wait_for(
                self._openai.chat.completions.create(
                    model=NARRATION_MODEL,
                    messages=[
                        {"role": "system", "content": NARRATION_PROMPT},
                        {"role": "user", "content": line},
                    ],
                    max_tokens=40,
                    temperature=0.7,
                ),
                timeout=NARRATION_TIMEOUT_S,
            )
        except Exception:
            logger.debug("Narration rewrite failed; using raw goal", exc_info=True)
            return "" if looks_like_bookkeeping(line) else line

        out = (resp.choices[0].message.content or "").strip()
        if not out or out.upper().startswith("SKIP"):
            return ""
        return out

    async def _narrate(self, session: TaskSession) -> None:
        """Push progress the moment a step lands, instead of polling for it."""
        queue = session.subscribe()
        try:
            while not self._closing.is_set():
                message = await queue.get()
                kind = message.get("type")
                if kind == "speak" and message.get("text"):
                    spoken = await self._humanize(str(message["text"]))
                    if spoken:
                        await self._inject(spoken)
                elif kind == "done":
                    # The raw result carries URLs and markdown, which a TTS voice
                    # spells out character by character. Clean it like any other
                    # spoken line before it reaches the caller.
                    result = _for_speech(
                        str(message.get("result") or ""),
                        extra=session.secrets,
                        limit=600,
                    )
                    await self._inject(result or "That's done.")
                    break
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Narration loop failed")
        finally:
            session.unsubscribe(queue)

    # ---- inbound from Deepgram ---------------------------------------------

    async def _handle_deepgram_json(self, message: dict[str, Any]) -> None:
        kind = message.get("type")

        if kind == "UserStartedSpeaking":
            await self._clear_audio()
            return

        if kind == "ConversationText":
            role = message.get("role")
            content = str(message.get("content") or "")
            if role == "user" and is_backchannel(content):
                logger.debug("Ignoring backchannel: %s", content)
            return

        if kind == "FunctionCallRequest":
            for call in message.get("functions", []):
                if not call.get("client_side", True):
                    continue
                name = call.get("name") or ""
                call_id = call.get("id") or ""
                raw_args = call.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    logger.warning("Bad function arguments: %r", raw_args)
                    args = {}

                if name == "start_browser_task":
                    content = await self._start_task(str(args.get("instruction") or ""))
                elif name == "stop_browser_task":
                    content = await self._stop_task()
                else:
                    content = f"Unknown function {name}."
                await self._function_response(call_id, name, content)
            return

        if kind == "Error":
            logger.error("Deepgram error: %s", message)

    async def _pump_deepgram(self) -> None:
        async for raw in self.dg:
            if self._closing.is_set():
                break
            if isinstance(raw, bytes):
                await self._send_audio(raw)
                continue
            try:
                await self._handle_deepgram_json(json.loads(raw))
            except json.JSONDecodeError:
                logger.debug("Non-JSON text frame from Deepgram: %r", raw[:200])

    # ---- inbound from Twilio ------------------------------------------------

    async def _pump_twilio(self) -> None:
        while not self._closing.is_set():
            raw = await self.twilio_ws.receive_text()
            message = json.loads(raw)
            event = message.get("event")

            if event == "start":
                start = message.get("start", {})
                self.stream_sid = start.get("streamSid")
                self.call_sid = start.get("callSid")
                if not call_registry.is_authorized(self.call_sid or ""):
                    logger.warning(
                        "Rejecting unauthorized stream for call %s", self.call_sid
                    )
                    self._closing.set()
                    return
                logger.info("Media stream open for call %s", self.call_sid)

            elif event == "media":
                if self.dg is not None:
                    await self.dg.send(base64.b64decode(message["media"]["payload"]))

            elif event == "stop":
                logger.info("Media stream closed for call %s", self.call_sid)
                self._closing.set()
                return

    # ---- lifecycle ----------------------------------------------------------

    async def run(self) -> None:
        api_key = (os.getenv("DEEPGRAM_API_KEY") or "").strip()
        if not api_key:
            logger.error("DEEPGRAM_API_KEY is not set; dropping the call")
            return

        async with websockets.connect(
            DEEPGRAM_AGENT_URL,
            additional_headers={"Authorization": f"Token {api_key}"},
        ) as dg:
            self.dg = dg
            await dg.send(json.dumps(build_settings()))

            pumps = [
                asyncio.create_task(self._pump_twilio()),
                asyncio.create_task(self._pump_deepgram()),
            ]
            try:
                done, pending = await asyncio.wait(
                    pumps, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                # Await the cancellations so they don't outlive the call.
                await asyncio.gather(*pending, return_exceptions=True)
                for task in done:
                    exc = task.exception()
                    if exc is not None:
                        raise exc
            finally:
                await self.aclose()

    async def aclose(self) -> None:
        self._closing.set()
        if self._narration_task is not None and not self._narration_task.done():
            self._narration_task.cancel()
        if self.call_sid:
            call_registry.release(self.call_sid)
        # A call that hangs up mid-run should not leave Chrome driving itself.
        if self.runner.current_running() is not None:
            try:
                await self.runner.stop_current()
            except Exception:
                logger.debug("stop_current on hangup failed", exc_info=True)
