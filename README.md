# Dispatch

You call a number or type a task. A real Chrome window does the clicking, and you watch it happen.

## What's running it

- Python / FastAPI
- [browser-use](https://github.com/browser-use/browser-use) driving Chrome with GPT-4.1
- Real Chrome when it's installed on the machine
- Next.js page that streams screenshots of that Chrome window
- Twilio for the call, [Deepgram](https://developers.deepgram.com/docs/voice-agent) for the voice

## Setup

You need Python 3.11+, Node 20+, an [OpenAI key](https://platform.openai.com/api-keys), and Chrome (or `uvx browser-use install`).

Linux with no screen: install `Xvfb`.

### Backend

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvx browser-use install     # once per machine
cp .env.example .env        # then set OPENAI_API_KEY
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

Try:

```text
Go to news.ycombinator.com and tell me the top five stories with their points.
```

Hit **Run**. Chrome should show up in the live view.

## Where it goes

By default the agent goes wherever the task tells it to. Set `PREFERRED_SITES` in
`backend/.env` if you'd rather it favour particular sites when a task doesn't name one.

`ALLOWED_DOMAINS` restricts navigation to a list of domain patterns. It's empty by
default, which means unrestricted — worth setting before you leave this running.

## Signing in to a site

Optional. Set `LOGIN_DOMAIN`, `LOGIN_USERNAME`, and `LOGIN_PASSWORD` in `backend/.env`
(gitignored). Credentials are scoped to `LOGIN_DOMAIN`, so the agent can't type them
into any other site, and `ALLOWED_DOMAINS` has to cover that domain — if either is
missing the credentials are withheld and the run continues without them. Cookies
persist in `backend/.browser-profile`, so most sites only need this once.

## Phone

The call runs on Twilio Media Streams bridged to the Deepgram Voice Agent API:
Deepgram Flux handles transcription and turn detection, an OpenAI model handles the
conversation, and Deepgram speaks. You can interrupt it mid-sentence, and it narrates
each browser step as it lands rather than on a timer.

1. Get a Twilio number with Voice.
2. Expose the backend (Cloudflare Tunnel or ngrok).
3. In `backend/.env` set `PUBLIC_BASE_URL`, `TWILIO_AUTH_TOKEN`, `DEEPGRAM_API_KEY`,
   and `ALLOWED_CALLERS` (your number, E.164).
4. In Twilio, set **A call comes in** to `POST` `https://<your-tunnel>/voice`.
5. Restart the backend, call the number, say what you want.

`ALLOWED_CALLERS` is empty by default and empty means reject every call — anyone who
gets through drives a browser that may be signed in to your accounts. Inbound requests
are also checked against `X-Twilio-Signature`, so `TWILIO_AUTH_TOKEN` is required.

Voice tuning lives in `.env` too: `VOICE_EOT_THRESHOLD` decides how sure Deepgram has
to be that you've stopped talking (raise it if the agent cuts you off), and
`VOICE_EOT_TIMEOUT_MS` caps how long it waits regardless.

One thing at a time. A second run or call while something is going gets a 409.

## API

| Method | Path | What it does |
| --- | --- | --- |
| `POST` | `/tasks` | `{ "instruction": string }` → `{ "task_id": "..." }` |
| `POST` | `/tasks/stop` | Stop the current run |
| `POST` | `/tasks/reset` | Stop and clear the demo |
| `GET` | `/tasks` | Latest task (so the page can pick up a call) |
| `WS` | `/ws/{task_id}` | Screenshots, log lines, done |
| `POST` | `/voice` | Inbound call — verifies signature and caller, then connects the stream |
| `WS` | `/voice/stream` | Twilio Media Stream bridged to the Deepgram voice agent |
