"""One browser-use session, with frames pushed to WebSocket clients."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from browser_use import Agent, Browser, ChatOpenAI

logger = logging.getLogger("dispatch.agent")

VIEWPORT = {"width": 1280, "height": 720}
FRAME_INTERVAL_S = 0.5
MAX_STEPS = 60
MAX_RUN_SECONDS = int(os.getenv("MAX_RUN_SECONDS", "600"))
STEP_TIMEOUT_S = int(os.getenv("STEP_TIMEOUT_S", "120"))
PROFILE_DIR = Path(__file__).resolve().parent / ".browser-profile"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _allowed_domains() -> list[str]:
    """Domains the agent may navigate to. Empty means unrestricted."""
    raw = os.getenv("ALLOWED_DOMAINS", "")
    return [d.strip() for d in raw.split(",") if d.strip()]


def _playwright_chrome_paths() -> list[Path]:
    """Chrome for Testing builds cached by `playwright install chromium`.

    browser-use only finds these through its own installer. Falling back to the
    cache means a machine without Chrome still runs instead of dying at launch.
    """
    roots = [
        Path.home() / "Library/Caches/ms-playwright",   # macOS
        Path.home() / ".cache/ms-playwright",           # Linux
    ]
    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for chromium in sorted(root.glob("chromium-*"), reverse=True):
            app = "Google Chrome for Testing"
            found.extend(
                chromium.glob(f"chrome-*/{app}.app/Contents/MacOS/{app}")
            )
            found.extend(chromium.glob("chrome-*/chrome"))
    return found


def _system_chrome_path() -> Path | None:
    candidates = [
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path("/usr/bin/google-chrome"),
        Path("/usr/bin/google-chrome-stable"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        *_playwright_chrome_paths(),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _make_browser() -> Browser:
    """Prefer real Chrome. Bundled Chromium gets blocked on a lot of sites."""
    # Highlighting draws overlays into the page. Great for the live view, but it
    # mutates the DOM the vision model reads, so it's a flag rather than a constant.
    highlight = _env_flag("HIGHLIGHT_ELEMENTS", True)
    allowed = _allowed_domains()

    cdp = (os.getenv("CHROME_CDP_URL") or "").strip()
    if cdp:
        logger.info("Attaching to existing Chrome via %s", cdp)
        cdp_kwargs: dict[str, Any] = {
            "cdp_url": cdp,
            "is_local": True,
            "highlight_elements": highlight,
        }
        if allowed:
            cdp_kwargs["allowed_domains"] = allowed
        return Browser(**cdp_kwargs)

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "headless": False,
        "viewport": VIEWPORT,
        "window_size": VIEWPORT,
        "keep_alive": True,
        "user_data_dir": str(PROFILE_DIR),
        "highlight_elements": highlight,
        "wait_between_actions": 0.6,
        "ignore_default_args": ["--enable-automation"],
        "args": [
            "--disable-blink-features=AutomationControlled",
            "--disable-infobars",
            "--no-first-run",
            "--no-default-browser-check",
        ],
    }
    if allowed:
        kwargs["allowed_domains"] = allowed
        logger.info("Browser restricted to: %s", ", ".join(allowed))
    chrome = _system_chrome_path()
    if chrome is not None:
        kwargs["executable_path"] = str(chrome)
        kwargs["channel"] = "chrome"
        logger.info("Launching system Chrome at %s", chrome)
    else:
        logger.warning("System Chrome not found; falling back to bundled Chromium")
    return Browser(**kwargs)


INTERACT_INSTRUCTIONS = """
You control a real Chrome window. Do the thing the user asked. Don't just look at the page.

You can:
- open the site they named and close cookie banners
- log in if a login form shows up
- fill username, email, password, and next/continue
- open a new message, type what they asked, and hit Send
- check that it actually sent (or that you're logged in) before you stop

Do what they asked. Go to the site they named. If they didn't name one, pick whichever site actually
answers their question, and say which one you picked.
If they just said hi or didn't give a real task, don't browse. Stop and say you're waiting.

Never type the strings "x_user" or "x_pass" unless this task actually gave those placeholders.
Don't invent passwords. If you hit a login wall and have no credentials, stay on that page.
Someone may log in in this same window. Then keep going. Don't quit just because of a login screen.
If a click fails, try another button or link before you give up.
""".strip()

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_PLACEHOLDER_RE = re.compile(r"\b(x_user|x_pass)\b", re.I)

TaskStatus = Literal["running", "done", "error"]
TaskSource = Literal["web", "phone"]


def _ensure_virtual_display() -> Any | None:
    """Headed Chrome needs a display. On Linux with no DISPLAY, start Xvfb."""
    if sys.platform == "darwin" or sys.platform.startswith("win"):
        return None
    if os.environ.get("DISPLAY"):
        return None
    try:
        from pyvirtualdisplay import Display

        display = Display(visible=False, size=(VIEWPORT["width"], VIEWPORT["height"]))
        display.start()
        logger.info("Started Xvfb virtual display")
        return display
    except Exception:
        logger.warning(
            "No DISPLAY and pyvirtualdisplay/Xvfb unavailable; browser may fail to launch"
        )
        return None


def _format_step(agent: Agent) -> str:
    parts: list[str] = []
    try:
        thoughts = agent.history.model_thoughts()
        if thoughts:
            last = thoughts[-1]
            text = (
                getattr(last, "next_goal", None)
                or getattr(last, "thinking", None)
                or getattr(last, "evaluation_previous_goal", None)
                or str(last)
            )
            if text:
                parts.append(str(text).strip())
    except Exception:
        logger.debug("Could not read model_thoughts", exc_info=True)
    try:
        actions = agent.history.model_actions()
        if actions:
            parts.append(f"action: {actions[-1]}")
    except Exception:
        logger.debug("Could not read model_actions", exc_info=True)
    try:
        urls = agent.history.urls()
        if urls and urls[-1]:
            parts.append(f"url: {urls[-1]}")
    except Exception:
        logger.debug("Could not read urls", exc_info=True)
    text = " / ".join(parts) if parts else "Step completed"
    return _redact(text)


def _redact(text: str, extra: list[str] | None = None) -> str:
    secrets = [
        os.getenv("LOGIN_PASSWORD") or "",
        os.getenv("LOGIN_USERNAME") or "",
        *(extra or []),
    ]
    out = text
    for secret in secrets:
        if secret and len(secret) >= 3:
            out = out.replace(secret, "***")
    return out


def _for_speech(text: str, extra: list[str] | None = None, limit: int = 180) -> str:
    """Strip anything that sounds wrong read aloud: URLs, markdown, secrets."""
    cleaned = _redact(str(text), extra=extra)
    cleaned = _URL_RE.sub("", cleaned)
    cleaned = _PLACEHOLDER_RE.sub("", cleaned)
    cleaned = re.sub(r"[*_`#>\-]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,;:")
    if not cleaned:
        return ""
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 3].rsplit(" ", 1)[0]
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned[0].upper() + cleaned[1:]


def _similar_speech(a: str, b: str) -> bool:
    na = re.sub(r"[^a-z0-9 ]+", "", a.lower()).strip()
    nb = re.sub(r"[^a-z0-9 ]+", "", b.lower()).strip()
    if not na or not nb:
        return False
    return na == nb or na in nb or nb in na


def _speakable_step(agent: Agent) -> str:
    try:
        thoughts = agent.history.model_thoughts()
        if thoughts:
            last = thoughts[-1]
            text = getattr(last, "next_goal", None) or getattr(last, "memory", None)
            if text:
                return _for_speech(text)
    except Exception:
        pass
    return ""


def _credentials() -> dict[str, str]:
    """Placeholder -> real value, straight from the environment."""
    user = (os.getenv("LOGIN_USERNAME") or "").strip()
    pw = (os.getenv("LOGIN_PASSWORD") or "").strip()
    data: dict[str, str] = {}
    if user:
        data["x_user"] = user
    if pw:
        data["x_pass"] = pw
    return data


def _build_sensitive_data() -> dict[str, dict[str, str]]:
    """Credentials scoped to one domain pattern, so they can't be typed into any site.

    browser-use requires allowed_domains whenever sensitive_data is set. If either
    LOGIN_DOMAIN or ALLOWED_DOMAINS is missing we drop the credentials rather than
    hand them to an unrestricted browser.
    """
    creds = _credentials()
    if not creds:
        return {}

    domain = (os.getenv("LOGIN_DOMAIN") or "").strip()
    if not domain:
        logger.warning(
            "LOGIN_USERNAME/LOGIN_PASSWORD are set but LOGIN_DOMAIN is empty. "
            "Credentials withheld — set LOGIN_DOMAIN (e.g. https://*.example.com)."
        )
        return {}
    if not _allowed_domains():
        logger.warning(
            "Credentials configured but ALLOWED_DOMAINS is empty. Credentials "
            "withheld — an unrestricted browser must not carry logins."
        )
        return {}
    return {domain: creds}


def _preferred_sites() -> list[str]:
    """Optional PREFERRED_SITES nudge. Unset means the instruction decides."""
    raw = os.getenv("PREFERRED_SITES", "")
    return [site.strip() for site in raw.split(",") if site.strip()]


def _preferred_sites_hint() -> str:
    sites = _preferred_sites()
    if not sites:
        return ""
    listed = ", ".join(sites)
    return (
        f"If the task doesn't name a site, prefer one of these: {listed}. "
        "If the user named a different site, use theirs instead."
    )


def _task_text(instruction: str, placeholders: set[str], source: TaskSource) -> str:
    extra = ["Only do what the user asked."]
    hint = _preferred_sites_hint()
    if hint:
        extra.append(hint)
    if "x_user" in placeholders and "x_pass" in placeholders:
        extra.append(
            "If there's a login form, sign in with username x_user and password x_pass."
        )
    extra.append(
        "If they asked you to send a message, type it, click send, and check it posted."
    )
    extra.append(
        "If you hit a login wall and can't continue, wait in this browser for someone "
        "to log in, then keep going."
    )
    if source == "phone":
        extra.append(
            "They're on a phone call. Final result should be a short spoken summary "
            "of what you did. No URLs, no markdown."
        )
    return instruction.strip() + "\n\n" + " ".join(extra)


async def _screenshot_jpeg_b64(browser: Browser) -> str | None:
    """Screenshot the page the agent is on."""
    page = await browser.get_current_page()
    if page is None:
        return None
    data = await page.screenshot(format="jpeg", quality=55)
    if not data:
        return None
    if isinstance(data, bytes):
        return base64.b64encode(data).decode("ascii")
    text = str(data)
    if text.startswith("data:"):
        return text.split(",", 1)[-1]
    return text


@dataclass
class TaskSession:
    instruction: str
    source: TaskSource = "web"
    task_id: str = field(default_factory=lambda: uuid4().hex)
    status: TaskStatus = "running"
    result: str | None = None
    logs: list[str] = field(default_factory=list)
    latest_frame: str | None = None
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    browser: Browser | None = None
    secrets: list[str] = field(default_factory=list)
    agent: Agent | None = None
    stop_requested: bool = False
    spoken: list[str] = field(default_factory=list)

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self.subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self.subscribers:
            self.subscribers.remove(queue)

    def push_spoken(self, text: str) -> str | None:
        """Record a narration line. Returns it, or None if dropped as a repeat."""
        line = _for_speech(text, extra=self.secrets)
        if not line:
            return None
        if self.spoken and _similar_speech(self.spoken[-1], line):
            return None
        self.spoken.append(line)
        return line

    async def emit(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "frame":
            self.latest_frame = message.get("data")
        elif kind == "log" and message.get("text"):
            message["text"] = _redact(str(message["text"]), extra=self.secrets)
            self.logs.append(str(message["text"]))
            # Also to the server log: a phone call has no WebSocket viewer, so
            # without this the agent's steps are invisible after the fact.
            logger.info("[%s] %s", self.task_id[:8], message["text"])
        elif kind == "done":
            logger.info("[%s] DONE %s", self.task_id[:8], message.get("result"))
        elif kind == "speak":
            logger.info("[%s] SPEAK %s", self.task_id[:8], message.get("text"))
        for queue in list(self.subscribers):
            await queue.put(message)


class AgentRunner:
    """One running task at a time."""

    def __init__(self) -> None:
        self.tasks: dict[str, TaskSession] = {}
        self.latest_task_id: str | None = None
        self._lock = asyncio.Lock()
        self._active: asyncio.Task | None = None
        self._display = None

    def current_running(self) -> TaskSession | None:
        if not self.latest_task_id:
            return None
        session = self.tasks.get(self.latest_task_id)
        if session and session.status == "running":
            return session
        return None

    async def start_task(
        self,
        instruction: str,
        source: TaskSource = "web",
    ) -> TaskSession:
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("instruction is required")

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set")

        sensitive = _build_sensitive_data()

        async with self._lock:
            running = self.current_running()
            if running:
                raise RuntimeError(f"A task is already running ({running.task_id})")
            session = TaskSession(instruction=instruction, source=source)
            # Redaction works off the real values, whatever domain they're scoped to.
            session.secrets = [v for v in _credentials().values() if v]
            self.tasks[session.task_id] = session
            self.latest_task_id = session.task_id
            self._active = asyncio.create_task(self._run(session, sensitive))
            return session

    async def stop_current(self) -> TaskSession:
        session = self.current_running()
        if session is None:
            raise RuntimeError("No running task")
        session.stop_requested = True
        if session.agent is not None:
            try:
                session.agent.stop()
            except Exception:
                logger.warning("agent.stop() failed", exc_info=True)
        await session.emit({"type": "log", "text": "Stop requested, shutting this run down."})
        return session

    async def reset(self) -> None:
        """Stop the run and clear demo state. Keeps the Chrome login profile."""
        session = self.current_running()
        if session is not None:
            session.stop_requested = True
            if session.agent is not None:
                try:
                    session.agent.stop()
                except Exception:
                    logger.warning("agent.stop() failed during reset", exc_info=True)
            if session.browser is not None:
                try:
                    await session.browser.kill()
                except Exception:
                    try:
                        await session.browser.stop()
                    except Exception:
                        logger.warning("browser close failed during reset", exc_info=True)
        active = self._active
        self.tasks.clear()
        self.latest_task_id = None
        self._active = None
        if active is not None and not active.done():
            active.cancel()
            try:
                await asyncio.wait_for(active, timeout=2)
            except Exception:
                pass

    async def _run(self, session: TaskSession, sensitive: dict[str, dict[str, str]]) -> None:
        if self._display is None:
            self._display = _ensure_virtual_display()

        stop = asyncio.Event()
        browser: Browser | None = None
        capture_task: asyncio.Task | None = None

        try:
            model = os.getenv("BROWSER_MODEL", "gpt-4.1")
            # Low by default: picking which element to click is not a place
            # for sampling variety. The voice agent runs hotter, separately.
            temperature = float(os.getenv("BROWSER_TEMPERATURE", "0.0"))
            llm = ChatOpenAI(model=model, temperature=temperature)
            browser = _make_browser()
            session.browser = browser
            await session.emit({"type": "log", "text": f"Starting agent with {model}"})
            chrome = _system_chrome_path()
            await session.emit(
                {
                    "type": "log",
                    "text": (
                        f"Browser: system Chrome ({chrome})"
                        if chrome
                        else "Browser: bundled Chromium (sites like Roblox often block this)"
                    ),
                }
            )
            await session.emit({"type": "log", "text": f"Instruction: {session.instruction}"})
            if sensitive:
                await session.emit(
                    {
                        "type": "log",
                        "text": (
                            "Login from .env (hidden), scoped to its domain. "
                            "Cookies stay in the local Chrome profile."
                        ),
                    }
                )

            async def on_step_end(agent: Agent) -> None:
                await session.emit({"type": "log", "text": _format_step(agent)})
                spoken = _speakable_step(agent)
                if spoken:
                    line = session.push_spoken(spoken)
                    if line:
                        # Pushed to the live call the moment the step lands.
                        await session.emit({"type": "speak", "text": line})

            placeholders = {k for creds in sensitive.values() for k in creds}
            agent_kwargs: dict[str, Any] = {
                "task": _task_text(session.instruction, placeholders, session.source),
                "llm": llm,
                "browser": browser,
                "use_vision": True,
                "extend_system_message": INTERACT_INSTRUCTIONS,
                # Batch related actions (a login is username + password + submit in one step).
                "max_actions_per_step": 4,
                "max_failures": 3,
                "step_timeout": STEP_TIMEOUT_S,
            }
            if sensitive:
                agent_kwargs["sensitive_data"] = sensitive

            agent = Agent(**agent_kwargs)
            session.agent = agent
            capture_task = asyncio.create_task(self._capture_loop(session, stop))
            # MAX_STEPS alone can't stop a run that stalls inside a step.
            history = await asyncio.wait_for(
                agent.run(on_step_end=on_step_end, max_steps=MAX_STEPS),
                timeout=MAX_RUN_SECONDS,
            )

            result = ""
            try:
                result = history.final_result() or ""
            except Exception:
                result = ""
            if not result:
                try:
                    extracted = history.extracted_content()
                    if extracted:
                        result = str(extracted[-1])
                except Exception:
                    result = str(history)

            if session.stop_requested:
                session.status = "done"
                session.result = "Stopped by user."
                await session.emit({"type": "log", "text": "Stopped by user."})
                await session.emit({"type": "done", "result": session.result})
                return

            session.status = "done"
            session.result = result
            await session.emit({"type": "log", "text": f"Done: {result}"})
            await session.emit({"type": "done", "result": result})
        except TimeoutError:
            logger.warning("Task %s hit the %ss budget", session.task_id, MAX_RUN_SECONDS)
            session.status = "error"
            session.result = f"Gave up after {MAX_RUN_SECONDS} seconds."
            await session.emit({"type": "log", "text": session.result})
            await session.emit({"type": "done", "result": session.result})
        except Exception as exc:
            if session.stop_requested:
                session.status = "done"
                session.result = "Stopped by user."
                await session.emit({"type": "log", "text": "Stopped by user."})
                await session.emit({"type": "done", "result": session.result})
                return
            logger.exception("Task %s failed", session.task_id)
            session.status = "error"
            session.result = str(exc)
            await session.emit({"type": "log", "text": f"Error: {exc}"})
            await session.emit({"type": "done", "result": str(exc)})
        finally:
            stop.set()
            if capture_task is not None:
                try:
                    await asyncio.wait_for(capture_task, timeout=2)
                except Exception:
                    capture_task.cancel()
            if browser is not None:
                try:
                    await browser.kill()
                except Exception:
                    try:
                        await browser.stop()
                    except Exception:
                        logger.warning("Failed to close browser for %s", session.task_id)
            session.browser = None
            session.agent = None

    async def _capture_loop(self, session: TaskSession, stop: asyncio.Event) -> None:
        while not stop.is_set():
            browser = session.browser
            if browser is not None:
                try:
                    frame = await _screenshot_jpeg_b64(browser)
                    if frame:
                        await session.emit({"type": "frame", "data": frame})
                except Exception:
                    logger.debug("Screenshot tick failed", exc_info=True)
            try:
                await asyncio.wait_for(stop.wait(), timeout=FRAME_INTERVAL_S)
            except TimeoutError:
                continue
