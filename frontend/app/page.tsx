"use client";

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

const ACCEPTANCE =
  "Go to news.ycombinator.com and tell me the top five stories with their points.";

function backendBase() {
  const fromEnv = process.env.NEXT_PUBLIC_BACKEND_URL;
  if (typeof window === "undefined") {
    return fromEnv || "http://localhost:8000";
  }
  const { hostname, protocol } = window.location;
  if (hostname === "localhost" || hostname === "127.0.0.1") {
    return "http://localhost:8000";
  }
  if (hostname.endsWith("trycloudflare.com")) {
    return fromEnv || "http://localhost:8000";
  }
  return `${protocol}//${hostname}:8000`;
}

function wsUrl(taskId: string) {
  return `${backendBase().replace(/^http/, "ws")}/ws/${taskId}`;
}

type TaskStatus = "idle" | "running" | "done" | "error";

type TasksResponse = {
  latest_task_id: string | null;
  current: {
    task_id: string;
    instruction: string;
    status: string;
    source: string;
    result: string | null;
  } | null;
};

export default function Page() {
  const [instruction, setInstruction] = useState(ACCEPTANCE);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [status, setStatus] = useState<TaskStatus>("idle");
  const [source, setSource] = useState<string>("web");
  const [frame, setFrame] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const logEnd = useRef<HTMLLIElement | null>(null);
  const socketRef = useRef<WebSocket | null>(null);

  const attachSocket = useCallback((id: string) => {
    socketRef.current?.close();
    const socket = new WebSocket(wsUrl(id));
    socketRef.current = socket;
    setLogs([]);
    setFrame(null);
    setError(null);
    setStatus("running");

    socket.onmessage = (event) => {
      const msg = JSON.parse(event.data) as {
        type: string;
        data?: string;
        text?: string;
        result?: string;
      };
      if (msg.type === "frame" && msg.data) {
        setFrame(msg.data);
      } else if (msg.type === "log" && msg.text) {
        setLogs((prev) => [...prev, msg.text as string]);
      } else if (msg.type === "done") {
        if (msg.result) {
          setLogs((prev) =>
            prev.includes(`Done: ${msg.result}`) ? prev : [...prev, `Done: ${msg.result}`]
          );
        }
        setStatus("done");
        socket.close();
      }
    };

    socket.onerror = () => {
      setError("Can't reach the backend on port 8000.");
      setStatus("error");
    };
  }, []);

  const stopRun = async () => {
    setError(null);
    const res = await fetch(`${backendBase()}/tasks/stop`, { method: "POST" });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      setError(body.detail ?? "Nothing to stop");
    }
  };

  const resetDemo = async () => {
    setError(null);
    socketRef.current?.close();
    socketRef.current = null;
    try {
      await fetch(`${backendBase()}/tasks/reset`, { method: "POST" });
    } catch {
      // Clear the UI even if the backend is already idle.
    }
    setTaskId(null);
    setStatus("idle");
    setSource("web");
    setFrame(null);
    setLogs([]);
    setInstruction(ACCEPTANCE);
  };

  const startFromText = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    const res = await fetch(`${backendBase()}/tasks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        instruction,
      }),
    });
    const body = await res.json();
    if (!res.ok) {
      setError(body.detail ?? "Failed to start task");
      if (body.task_id) {
        setTaskId(body.task_id);
        attachSocket(body.task_id);
      }
      return;
    }
    setTaskId(body.task_id);
    setSource("web");
    attachSocket(body.task_id);
  };

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const res = await fetch(`${backendBase()}/tasks`);
        if (!res.ok) return;
        const data = (await res.json()) as TasksResponse;
        const latest = data.latest_task_id;
        if (!latest || cancelled) return;
        if (latest !== taskId) {
          setTaskId(latest);
          setSource(data.current?.source ?? "phone");
          if (data.current?.instruction) {
            setInstruction(data.current.instruction);
          }
          attachSocket(latest);
        }
      } catch {
        // Backend may not be up yet.
      }
    };
    const handle = window.setInterval(poll, 2000);
    void poll();
    return () => {
      cancelled = true;
      window.clearInterval(handle);
    };
  }, [attachSocket, taskId]);

  useEffect(() => {
    logEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  useEffect(() => {
    return () => socketRef.current?.close();
  }, []);

  const frameSrc = useMemo(
    () => (frame ? `data:image/jpeg;base64,${frame}` : null),
    [frame]
  );

  const busy = status === "running";

  return (
    <main className="shell">
      <header className="top">
        <div>
          <h1>Dispatch</h1>
          <p className="lede">
            Type a task or call in. Watch the browser do it.
          </p>
        </div>
        <div className="badge">
          <span className={`dot ${status}`} />
          {status}
          {source === "phone" ? " / phone" : ""}
          {taskId ? ` / ${taskId.slice(0, 8)}` : ""}
        </div>
      </header>

      <form className="composer" onSubmit={startFromText}>
        <div className="composer-fields">
          <textarea
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
            placeholder="Tell the agent which site to open and what to do there"
            disabled={busy}
          />
        </div>
        <div className="actions">
          <button type="submit" disabled={busy || !instruction.trim()}>
            Run
          </button>
          <button type="button" className="stop" onClick={stopRun} disabled={!busy}>
            Stop
          </button>
          <button type="button" className="reset" onClick={resetDemo}>
            Reset
          </button>
        </div>
      </form>

      {error ? <p className="error">{error}</p> : null}

      <section className="viewport-wrap">
        <div className="viewport-label">
          <span>Live browser 1280x720</span>
          <span>{frame ? "live" : "waiting"}</span>
        </div>
        <div className="viewport">
          {frameSrc ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img alt="Live agent browser" src={frameSrc} />
          ) : (
            <div className="placeholder">
              Type a task and hit Run.
            </div>
          )}
        </div>
      </section>

      <section className="log">
        <h2>Agent log</h2>
        <ul>
          {logs.length === 0 ? <li>No steps yet.</li> : null}
          {logs.map((line, i) => (
            <li key={`${i}-${line.slice(0, 24)}`}>{line}</li>
          ))}
          <li ref={logEnd} />
        </ul>
      </section>
    </main>
  );
}
