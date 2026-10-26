import { FormEvent, useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Session } from "@supabase/supabase-js";
import { AuthScreen } from "./AuthScreen";
import { getAccessToken, supabase, supabaseAuthConfigured } from "./supabase";

type RetrievedChunk = {
  id: string;
  window_start: string;
  window_end: string;
  sub_index: number;
  services: string[];
  trace_ids: string[];
  content: string;
  cosine_similarity?: number | null;
  max_level?: string | null;
  dense_rank?: number | null;
  bm25_rank?: number | null;
  rrf_score?: number | null;
  rerank_score?: number | null;
};

type AskResponse = {
  answer: string;
  retrieved_chunks: RetrievedChunk[];
  retrieval_note?: string | null;
  response_type?: ResponseType;
};

type ResponseType =
  | "supported"
  | "ambiguous"
  | "fallback"
  | "no_correlated_incident"
  | "out_of_scope"
  | "empty_retrieval"
  | "abstain";

type LogLevelName = "INFO" | "WARN" | "ERROR" | "CRITICAL";

type LogLevelInterval = {
  start: string;
  end: string;
  counts: Record<LogLevelName, number>;
  total: number;
};

type LogLevelStats = {
  status: "ready" | "empty";
  last_indexed: string | null;
  raw_logs: number;
  embedded_chunks: number;
  intervals: LogLevelInterval[];
};

type ProductionMetricPoint = {
  start: number;
  requests: number;
  errors: number;
  rate_limited: number;
  payload_rejected: number;
  average_latency_ms: number;
};

type ProductionMetrics = {
  status: "ready";
  window_minutes: number;
  series: ProductionMetricPoint[];
  totals: {
    requests: number;
    errors: number;
    rate_limited: number;
    payload_rejected: number;
    average_latency_ms: number;
    unique_visitors: number;
  };
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  question?: string;
  chunks?: RetrievedChunk[];
  retrievalNote?: string | null;
  responseType?: ResponseType;
  latencyMs?: number;
};

type LogEvent = {
  time: string;
  service: string;
  message: string;
  level?: string;
  sortKey: number;
};

const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8001").replace(
  /\/$/,
  "",
);

const EXAMPLE_QUESTIONS = [
  "Why did payment authorization time out?",
  "What caused the checkout failures?",
  "Were there any incidents involving inventory?",
];

const STORAGE_KEY = "sift-chat-history";
const ADMIN_EMAILS = new Set(
  String(import.meta.env.VITE_ADMIN_EMAILS || "")
    .split(",")
    .map((value) => value.trim().toLowerCase())
    .filter(Boolean),
);
const ADMIN_USER_IDS = new Set(
  String(import.meta.env.VITE_ADMIN_USER_IDS || "")
    .split(",")
    .map((value) => value.trim().toLowerCase())
    .filter(Boolean),
);

class ApiRequestError extends Error {
  status: number;
  retryAfterSeconds: number | null;

  constructor(message: string, status: number, retryAfter: string | null) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
    this.retryAfterSeconds = retryAfter ? Number.parseInt(retryAfter, 10) : null;
  }
}

type RequestErrorState = {
  message: string;
  status?: number;
};

function readHistory(): ChatMessage[] {
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (!saved) return [];
    const messages = JSON.parse(saved) as ChatMessage[];
    let previousQuestion = "";
    return messages.map((message) => {
      if (message.role === "user") {
        previousQuestion = message.text;
        return message;
      }
      return {
        ...message,
        question: message.question || previousQuestion,
      };
    });
  } catch {
    return [];
  }
}

function friendlyRequestError(error: unknown): string {
  const status = error instanceof ApiRequestError ? error.status : undefined;
  const retryAfter = error instanceof ApiRequestError ? error.retryAfterSeconds : null;

  if (status === 429) {
    if (retryAfter && Number.isFinite(retryAfter)) {
      const unit = retryAfter === 1 ? "second" : "seconds";
      return `Sift is receiving many requests right now. Please try again in about ${retryAfter} ${unit}.`;
    }
    return "Sift is receiving many requests right now. Please wait a moment and try again.";
  }
  if (status === 413) {
    return "That request is too large. Please shorten your question and try again.";
  }
  if (status === 401) {
    return "Your session has expired. Please sign in again to continue.";
  }
  if (status === 403) {
    return "You do not have permission to access Sift. Please check your account.";
  }
  if (status === 503) {
    return "Sift is temporarily unavailable. Please try again in a moment.";
  }
  if (status === 502) {
    return "Sift could not complete that request. Please try again in a moment.";
  }
  if (error instanceof Error && error.message === "Failed to fetch") {
    return "We couldn't connect to the log API. Check your connection and try again.";
  }
  return "Something went wrong while processing your question. Please try again.";
}

function requestErrorTitle(status?: number): string {
  if (status === 429) return "Sift is busy";
  if (status === 413) return "Request too large";
  if (status === 401 || status === 403) return "Sign-in required";
  if (status === 502 || status === 503) return "Sift is temporarily unavailable";
  return "We couldn't complete that request";
}

function formatWindow(start: string, end: string): string {
  const startDate = new Date(start);
  const endDate = new Date(end);
  return `${startDate.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  })} — ${endDate.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  })}`;
}

function traceLabel(traceId: string): string {
  const labels: Record<string, string> = {
    "trace-payment-timeout-001": "Payment worker OOM",
    "trace-postgres-lock-001": "PostgreSQL lock contention",
    "trace-checkout-502-001": "Checkout 502 / crash loop",
    "trace-notification-rate-limit-001": "Notification rate limit",
  };
  return labels[traceId] || traceId;
}

function groundingLabel(
  chunks: RetrievedChunk[],
  note?: string | null,
  responseType: ResponseType = "supported",
): string {
  const normalized = (note || "").toLowerCase();
  if (responseType === "out_of_scope") return "Question is outside the log corpus";
  if (responseType === "abstain" || responseType === "empty_retrieval") return "No evidence found";
  if (responseType === "no_correlated_incident") return "No correlated incident found";
  if (responseType === "fallback") return "Answer uses fallback context";
  if (responseType === "ambiguous") return "Multiple incidents may explain this";
  if (!chunks.length) {
    if (normalized.includes("outside") || normalized.includes("scope")) {
      return "Question is outside the log corpus";
    }
    return "No evidence found";
  }
  if (normalized.includes("no correlated incident")) {
    return "No correlated incident found";
  }
  if (normalized.includes("fallback") || normalized.includes("requested time range")) {
    return "Answer uses fallback context";
  }
  const traces = new Set(chunks.flatMap((chunk) => chunk.trace_ids));
  if (traces.size > 1 || normalized.includes("multiple incident")) {
    return "Multiple incidents may explain this";
  }
  return "Supported by retrieved logs";
}

function parseEvents(chunks: RetrievedChunk[]): LogEvent[] {
  const events: LogEvent[] = [];
  chunks.forEach((chunk, chunkIndex) => {
    const windowStart = new Date(chunk.window_start);
    chunk.content.split("\n").forEach((line, lineIndex) => {
      const match = line.match(
        /^\[([^\]]+)\]\s+(INFO|WARN|ERROR|CRITICAL)\s+([^:]+):\s*(.*)$/i,
      );
      if (!match) return;
      const [, time, level, service, message] = match;
      const timeParts = time.split(":").map(Number);
      const eventDate = new Date(windowStart);
      if (timeParts.length === 3 && timeParts.every(Number.isFinite)) {
        eventDate.setUTCHours(timeParts[0], timeParts[1], timeParts[2], 0);
        if (eventDate.getTime() < windowStart.getTime()) {
          eventDate.setUTCDate(eventDate.getUTCDate() + 1);
        }
      }
      events.push({
        time,
        service: service.trim(),
        message,
        level: level.toUpperCase(),
        sortKey: eventDate.getTime() + chunkIndex / 1000 + lineIndex / 1000000,
      });
    });
  });
  return events.sort((a, b) => a.sortKey - b.sortKey);
}

function shouldHideEvidence(
  chunks: RetrievedChunk[],
  responseType: ResponseType = "supported",
): boolean {
  if (!chunks.length) return true;
  return ["out_of_scope", "no_correlated_incident", "empty_retrieval", "abstain"].includes(responseType);
}

function traceGroups(chunks: RetrievedChunk[]) {
  const groups = new Map<string, RetrievedChunk[]>();
  chunks.forEach((chunk) => {
    chunk.trace_ids.forEach((traceId) => {
      const existing = groups.get(traceId) || [];
      existing.push(chunk);
      groups.set(traceId, existing);
    });
  });
  return [...groups.entries()];
}

function firstNameForSession(session: Session): string {
  const metadata = session.user.user_metadata as Record<string, unknown>;
  const candidate = [metadata.first_name, metadata.given_name, metadata.full_name, metadata.name]
    .find((value) => typeof value === "string" && value.trim());
  if (typeof candidate === "string") return candidate.trim().split(/\s+/)[0];
  return session.user.email?.split("@")[0] || "there";
}

function isAdminSession(session: Session): boolean {
  return (
    ADMIN_USER_IDS.has(session.user.id.toLowerCase()) ||
    Boolean(session.user.email && ADMIN_EMAILS.has(session.user.email.toLowerCase()))
  );
}

function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [showAuthScreen, setShowAuthScreen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>(readHistory);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<RequestErrorState | null>(null);
  const [lastQuestion, setLastQuestion] = useState("");
  const [logStats, setLogStats] = useState<LogLevelStats | null>(null);
  const [logStatsError, setLogStatsError] = useState<string | null>(null);
  const [view, setView] = useState<"chat" | "architecture" | "metrics">("chat");
  const isAdmin = session ? isAdminSession(session) : false;
  const [darkMode, setDarkMode] = useState(() => {
    return window.localStorage.getItem("sift-theme") === "dark";
  });

  useEffect(() => {
    if (!supabase) return;
    let mounted = true;
    void supabase.auth.getSession().then(({ data }) => {
      if (mounted) {
        setSession(data.session);
        if (data.session) setShowAuthScreen(false);
      }
    });
    const { data: listener } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      if (mounted) {
        setSession(nextSession);
        if (nextSession) setShowAuthScreen(false);
      }
    });
    return () => {
      mounted = false;
      listener.subscription.unsubscribe();
    };
  }, []);

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
  }, [messages]);

  useEffect(() => {
    document.documentElement.dataset.theme = darkMode ? "dark" : "light";
    window.localStorage.setItem("sift-theme", darkMode ? "dark" : "light");
  }, [darkMode]);

  useEffect(() => {
    if (!session) return;
    let cancelled = false;

    async function loadLogStats() {
      try {
        const accessToken = await getAccessToken();
        const response = await fetch(`${API_URL}/stats/log-levels`, {
          headers: { Authorization: `Bearer ${accessToken}` },
        });
        const body = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new ApiRequestError(
            body.detail || `Stats request failed with status ${response.status}`,
            response.status,
            response.headers.get("Retry-After"),
          );
        }
        if (!cancelled) {
          setLogStats(body as LogLevelStats);
          setLogStatsError(null);
        }
      } catch (statsRequestError) {
        if (!cancelled) {
          setLogStatsError(friendlyRequestError(statsRequestError));
        }
      }
    }

    void loadLogStats();
    const refreshId = window.setInterval(() => void loadLogStats(), 15000);
    return () => {
      cancelled = true;
      window.clearInterval(refreshId);
    };
  }, [session]);

  useEffect(() => {
    if (view === "metrics" && !isAdmin) setView("chat");
  }, [isAdmin, view]);

  async function ask(rawQuestion: string) {
    const trimmed = rawQuestion.trim();
    if (!trimmed || loading) return;
    if (!session) {
      setShowAuthScreen(true);
      return;
    }

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      text: trimmed,
    };
    setMessages((current) => [...current, userMessage]);
    setQuestion("");
    setLastQuestion(trimmed);
    setError(null);
    setLoading(true);
    const started = performance.now();

    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${await getAccessToken()}`,
        },
        body: JSON.stringify({ question: trimmed }),
      });
      const body = (await response.json().catch(() => ({}))) as Partial<AskResponse> & {
        detail?: string;
      };
      if (!response.ok) {
        throw new ApiRequestError(
          body.detail || `Request failed with status ${response.status}`,
          response.status,
          response.headers.get("Retry-After"),
        );
      }

      const assistantMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        text: body.answer || "The API returned an empty answer.",
        question: trimmed,
        chunks: body.retrieved_chunks || [],
        retrievalNote: body.retrieval_note,
        responseType: body.response_type || "supported",
        latencyMs: performance.now() - started,
      };
      setMessages((current) => [...current, assistantMessage]);
    } catch (requestError) {
      console.error("Sift request failed", requestError);
      setError({
        message: friendlyRequestError(requestError),
        status: requestError instanceof ApiRequestError ? requestError.status : undefined,
      });
    } finally {
      setLoading(false);
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    void ask(question);
  }

  function clearChat() {
    setMessages([]);
    setError(null);
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark">S</div>
          <div>
            <div className="brand-name">Sift</div>
            <div className="brand-caption">Incident intelligence for application logs</div>
          </div>
        </div>
        <div className="topbar-actions">
          <span className="api-status">
            <span className="status-dot" /> {session ? "API connected" : "Login required for log access"}
          </span>
          <button className="icon-button" onClick={() => setDarkMode((value) => !value)} aria-label="Toggle dark mode">
            {darkMode ? "☼" : "☾"}
          </button>
          {view === "chat" && (
            <button className="text-button" onClick={clearChat} disabled={!messages.length}>
              Clear chat
            </button>
          )}
          {view === "chat" && (
            <>
              <button className="text-button" onClick={() => setView("architecture")}>
                Architecture
              </button>
              {isAdmin && (
                <button className="text-button admin-metrics-button" onClick={() => setView("metrics")}>
                  Production metrics
                </button>
              )}
            </>
          )}
          {view === "architecture" && (
            <>
              <button className="text-button" onClick={() => setView("chat")}>
                Back to chat
              </button>
              {isAdmin && (
                <button className="text-button admin-metrics-button" onClick={() => setView("metrics")}>
                  Production metrics
                </button>
              )}
            </>
          )}
          {view === "metrics" && (
            <>
              <button className="text-button" onClick={() => setView("architecture")}>
                Architecture
              </button>
              <button className="text-button" onClick={() => setView("chat")}>
                Back to chat
              </button>
            </>
          )}
          {session ? (
            <>
              <span className="user-email" title={session.user.email || undefined}>
                {session.user.email || "Signed in"}
              </span>
              <button
                className="icon-button sign-out-button"
                onClick={() => void supabase?.auth.signOut()}
                aria-label="Sign out"
                title="Sign out"
              >
                <svg className="auth-icon" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M5 3h5v2H5v14h5v2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2Z" />
                  <path d="m13 7 1.4 1.4-2.6 2.6H21v2h-9.2l2.6 2.6L13 17l-5-5 5-5Z" />
                </svg>
              </button>
            </>
          ) : (
            <button
              className="icon-button sign-in-button"
              onClick={() => { setView("chat"); setShowAuthScreen(true); }}
              aria-label="Sign in to ask Sift"
              title="Sign in to ask Sift"
            >
              <svg className="auth-icon" viewBox="0 0 24 24" aria-hidden="true">
                <path d="M14 3h5a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-5v-2h5V5h-5V3Z" />
                <path d="m11 7-1.4 1.4 2.6 2.6H3v2h9.2l-2.6 2.6L11 17l5-5-5-5Z" />
              </svg>
            </button>
          )}
        </div>
      </header>

      <main className="workspace">
        {view === "architecture" ? <ArchitecturePage /> : view === "metrics" ? <AdminMetricsPage session={session} /> : <>
        <section className="chat-column">
          {(!session || messages.length === 0) && (
            <section className="welcome-card">
              <div className="eyebrow">LOG QA / TRACE-AWARE RETRIEVAL</div>
              {session && <div className="welcome-greeting">Hi, {firstNameForSession(session)}.</div>}
              <h1>Find the incident behind the noise.</h1>
              <p>
                Ask Sift why an operation failed. Answers are grounded in retrieved log evidence,
                chronological traces, and the services involved.
              </p>
              <div className="example-grid">
                {EXAMPLE_QUESTIONS.map((example) => (
                  <button key={example} className="example-card" onClick={() => void ask(example)}>
                    <span className="example-arrow">↗</span>
                    <span>{example}</span>
                  </button>
                ))}
              </div>
            </section>
          )}

          {session && (
            <div className="message-list">
              {messages.map((message) => (
                  <MessageCard key={message.id} message={message} onRetry={ask} />
              ))}
              {loading && (
                <div className="assistant-row">
                  <div className="avatar assistant-avatar">S</div>
                  <div className="message-bubble assistant-bubble loading-bubble">
                    <div className="loading-label">Sifting through the logs<span className="loading-dots">...</span></div>
                    <div className="loading-bar" />
                    <div className="loading-bar short" />
                  </div>
                </div>
              )}
            </div>
          )}

          {!session && showAuthScreen && <AuthScreen configured={supabaseAuthConfigured} />}

          {session && error && (
            <div className="error-card" role="alert">
              <div>
                <strong>{requestErrorTitle(error.status)}</strong>
                <span>{error.message}</span>
              </div>
              <button className="secondary-button" onClick={() => void ask(lastQuestion)} disabled={!lastQuestion || loading}>
                Try again
              </button>
            </div>
          )}

          {session ? (
            <form className="composer" onSubmit={submit}>
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Ask about an incident, service, or failure..."
                rows={2}
                disabled={loading}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    submit(event);
                  }
                }}
              />
              <div className="composer-footer">
                <span>Enter to ask · Shift + Enter for a new line</span>
                <button className="primary-button" type="submit" disabled={loading || !question.trim()}>
                  {loading ? "Investigating..." : "Ask Sift"}
                  <span>↗</span>
                </button>
              </div>
            </form>
          ) : (
            <div className="composer auth-composer">
              <div className="auth-composer-icon">⌁</div>
              <div className="auth-composer-copy">
                <strong>Sign in required to ask Sift</strong>
                <span>Authenticate to search the indexed logs and view live corpus data.</span>
              </div>
              <button className="primary-button" type="button" onClick={() => setShowAuthScreen(true)}>
                Sign in <span>↗</span>
              </button>
            </div>
          )}
        </section>

        <aside className="side-column">
          <div className="side-card system-card">
            <div className="side-card-header"><span>PIPELINE STATUS</span><span className="live-pill">LIVE</span></div>
            <div className="pipeline-line"><span className="pipeline-node done">✓</span><span>Supabase corpus</span><span className="pipeline-value">ready</span></div>
            <div className="pipeline-line"><span className="pipeline-node done">✓</span><span>Hybrid retrieval</span><span className="pipeline-value">active</span></div>
            <div className="pipeline-line"><span className="pipeline-node done">✓</span><span>Evidence grounding</span><span className="pipeline-value">active</span></div>
          </div>
          {session && <>
            <LiveIndexStatus stats={logStats} />
            <LogVolumeGraph stats={logStats} error={logStatsError} />
            <LogLevelGraph stats={logStats} error={logStatsError} />
          </>}
          <div className="side-card about-card">
            <div className="side-card-header"><span>ABOUT THIS VIEW</span></div>
            <p>Every response includes the retrieved evidence, incident traces, and time windows used to answer.</p>
            <div className="legend-row"><span className="legend-swatch evidence" /> Supported by retrieved logs</div>
            <div className="legend-row"><span className="legend-swatch ambiguity" /> Multiple traces detected</div>
            <div className="legend-row"><span className="legend-swatch quiet" /> No correlated incident</div>
          </div>
        </aside>
        </>}
      </main>
    </div>
  );
}

function ArchitecturePage() {
  const flow = [
    { title: "Generate", detail: "Seeded templates + scenarios", logo: ["Python", "python", "3776AB"] },
    { title: "Collect", detail: "FastAPI validates and queues", logo: ["FastAPI", "fastapi", "009688"] },
    { title: "Store", detail: "Redis stream → raw_logs", logo: ["Redis", "redis", "DC382D"] },
    { title: "Embed", detail: "60-second chunks + pgvector", logo: ["Hugging Face", "huggingface", "FFD21E"] },
    { title: "Retrieve", detail: "Hybrid search + trace expansion", logo: ["PostgreSQL", "postgresql", "4169E1"] },
    { title: "Answer", detail: "Grounded QA in the React app", logo: ["React", "react", "61DAFB"] },
  ];
  const queryTypes = [
    ["Topic", "Why did payment authorization time out?"],
    ["Service", "What was postgres doing?"],
    ["Temporal", "What happened around midnight?"],
    ["Combined", "What happened in postgres last night?"],
    ["Out of scope", "Which customer segment had the most failures?"],
  ];
  const services = [
    "api-gateway",
    "nginx",
    "checkout-service",
    "payment-service",
    "inventory-service",
    "notification-service",
    "postgres",
  ];
  const technologies = [
    { category: "Frontend", stack: "React · TypeScript · Vite · React Markdown", icons: [["React", "react", "61DAFB"], ["TypeScript", "typescript", "3178C6"], ["Vite", "vite", "646CFF"]] },
    { category: "API", stack: "Python · FastAPI · Pydantic · asyncpg · LangChain Core", icons: [["Python", "python", "3776AB"], ["FastAPI", "fastapi", "009688"], ["LangChain", "langchain", "1C3C3C"]] },
    { category: "Storage", stack: "PostgreSQL · Supabase · pgvector · Redis Streams", icons: [["PostgreSQL", "postgresql", "4169E1"], ["Supabase", "supabase", "3FCF8E"], ["Redis", "redis", "DC382D"]] },
    { category: "Retrieval", stack: "BGE-small embeddings · BM25/tsvector · RRF · Cross-Encoder", icons: [["Hugging Face", "huggingface", "FFD21E"], ["PostgreSQL", "postgresql", "4169E1"]] },
    { category: "LLM", stack: "LangChain adapters · Groq / Cerebras-compatible models", icons: [["Groq", "groq", "F55036"], ["Cerebras", "cerebras", "F54E42"]] },
    { category: "Operations", stack: "Docker Compose · Google Cloud Run · Cloud Build", icons: [["Docker", "docker", "2496ED"], ["Google Cloud", "googlecloud", "4285F4"]] },
    { category: "Deployment", stack: "GitHub Actions CI/CD · Cloud Run API and worker jobs · Vercel frontend", icons: [["GitHub Actions", "githubactions", "2088FF"], ["Google Cloud", "googlecloud", "4285F4"], ["Vercel", "vercel", "000000"]] },
    { category: "Evaluation", stack: "pytest · RAGAS · custom behavior checks", icons: [["Python", "python", "3776AB"], ["GitHub Actions", "githubactions", "2088FF"]] },
  ];

  return (
    <section className="architecture-page">
      <div className="architecture-hero">
        <div className="eyebrow">SIFT / SYSTEM OVERVIEW</div>
        <h1>From noisy logs to grounded incident answers.</h1>
        <p>
          Sift is a trace-aware log investigation tool. It combines a live ingestion pipeline,
          hybrid retrieval, and evidence checks so answers stay tied to the available logs.
        </p>
      </div>

      <section className="architecture-section">
        <div className="section-label">ARCHITECTURE FLOW</div>
        <p className="architecture-section-intro">Each stage has a focused responsibility, from generating realistic traffic to presenting trace-backed answers.</p>
        <div className="architecture-flow">
          {flow.map((step, index) => (
            <div className="architecture-flow-step" key={step.title}>
              <div className="architecture-flow-head">
                <div className="architecture-step-number">0{index + 1}</div>
                <img
                  className="architecture-flow-logo"
                  src={`https://cdn.simpleicons.org/${step.logo[1]}/${step.logo[2]}`}
                  alt={step.logo[0]}
                  title={step.logo[0]}
                  onError={(event) => { event.currentTarget.style.display = "none"; }}
                />
              </div>
              <strong>{step.title}</strong>
              <span>{step.detail}</span>
            </div>
          ))}
        </div>
      </section>

      <div className="architecture-columns">
        <section className="architecture-card">
          <div className="section-label">QUERY TYPES</div>
          <p className="architecture-card-intro">The query parser extracts intent before retrieval.</p>
          <div className="query-type-list">
            {queryTypes.map(([kind, sample]) => (
              <div className="query-type-row" key={kind}>
                <span>{kind}</span>
                <code>{sample}</code>
              </div>
            ))}
          </div>
        </section>

        <section className="architecture-card">
          <div className="section-label">SAMPLE LOG SERVICES</div>
          <p className="architecture-card-intro">The generator combines application, edge, and database signals.</p>
          <div className="service-chip-list">
            {services.map((service) => <span className="service-chip" key={service}>{service}</span>)}
          </div>
          <div className="scenario-note">
            Sample incidents include payment-service OOM and timeout, PostgreSQL lock contention,
            checkout crash-loop / 502 responses, and notification provider rate limiting.
          </div>
        </section>
      </div>

      <section className="architecture-card retrieval-card">
        <div className="section-label">RETRIEVAL AND GROUNDING</div>
        <div className="retrieval-pillars">
          <div><strong>1. Understand</strong><span>Detect service and time filters, plus unsupported questions.</span></div>
          <div><strong>2. Retrieve</strong><span>Combine pgvector similarity and PostgreSQL full-text search with RRF.</span></div>
          <div><strong>3. Expand</strong><span>Use incident trace IDs to recover the chronological cross-service context.</span></div>
          <div><strong>4. Answer safely</strong><span>Show evidence, flag ambiguity, abstain, or report no correlated incident.</span></div>
        </div>
      </section>

      <section className="architecture-card technologies-card">
        <div className="section-label">TECHNOLOGIES USED</div>
        <p className="architecture-card-intro">The stack combines a lightweight React interface, Python services, and Postgres-native retrieval.</p>
        <div className="technology-list">
          {technologies.map((technology) => (
            <div className="technology-row" key={technology.category}>
              <div className="technology-row-header">
                <strong>{technology.category}</strong>
                <div className="technology-icon-strip" aria-label={`${technology.category} technologies`}>
                  {technology.icons.map(([name, slug, color]) => (
                    <img
                      key={slug}
                      className="technology-icon"
                      src={`https://cdn.simpleicons.org/${slug}/${color}`}
                      alt={name}
                      title={name}
                      onError={(event) => { event.currentTarget.style.display = "none"; }}
                    />
                  ))}
                </div>
              </div>
              <span>{technology.stack}</span>
            </div>
          ))}
        </div>
      </section>
    </section>
  );
}

function LiveIndexStatus({ stats }: { stats: LogLevelStats | null }) {
  const lastIndexed = stats?.last_indexed
    ? new Date(stats.last_indexed).toLocaleString([], {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "Loading...";

  return (
    <div className="side-card live-index-card">
      <div className="side-card-header">
        <span>INDEX STATUS</span>
        <span className="live-pill">{stats?.status === "ready" ? "LIVE" : "LOADING"}</span>
      </div>
      <div className="index-status-line">
        <span className={`pipeline-node ${stats?.status === "ready" ? "done" : "muted"}`}>{stats?.status === "ready" ? "✓" : "•"}</span>
        <span>Indexer status</span>
        <strong>{stats?.status || "loading"}</strong>
      </div>
      <div className="index-stat-grid">
        <div><span>Last indexed</span><strong>{lastIndexed}</strong></div>
        <div><span>Raw logs</span><strong>{stats?.raw_logs.toLocaleString() || "—"}</strong></div>
        <div><span>Embedded chunks</span><strong>{stats?.embedded_chunks.toLocaleString() || "—"}</strong></div>
      </div>
      <p className="placeholder-note">Corpus-wide counts refresh automatically.</p>
    </div>
  );
}

function LogVolumeGraph({ stats, error }: { stats: LogLevelStats | null; error: string | null }) {
  const intervals = stats?.intervals || [];
  const [selectedIndex, setSelectedIndex] = useState(0);

  useEffect(() => {
    setSelectedIndex(intervals.length ? intervals.length - 1 : 0);
  }, [intervals.length]);

  const selectedInterval = intervals[selectedIndex];
  const maximumTotal = Math.max(...intervals.map((interval) => interval.total), 1);

  return (
    <div className="side-card log-volume-card">
      <div className="side-card-header">
        <span>LOG VOLUME</span>
        <span className="live-pill">LIVE</span>
      </div>
      <div className="log-level-subtitle">
        {intervals.length
          ? "Total records per time window"
          : error
            ? "Unable to load indexed logs"
            : "Loading indexed logs"}
      </div>
      {selectedInterval ? (
        <>
          <div className="interval-controls">
            <button className="interval-button" onClick={() => setSelectedIndex((current) => current - 1)} disabled={selectedIndex === 0} aria-label="Previous volume interval">←</button>
            <span>{formatWindow(selectedInterval.start, selectedInterval.end)}</span>
            <button className="interval-button" onClick={() => setSelectedIndex((current) => current + 1)} disabled={selectedIndex === intervals.length - 1} aria-label="Next volume interval">→</button>
          </div>
          <div className="volume-summary">
            <strong>{selectedInterval.total.toLocaleString()}</strong>
            <span>logs in selected window</span>
          </div>
          <div className="volume-strip" aria-label="Total logs per time interval">
            {intervals.map((interval, index) => {
              const height = `${Math.max(10, (interval.total / maximumTotal) * 72)}px`;
              return (
                <button
                  key={`${interval.start}-${interval.end}`}
                  className={`volume-bar ${index === selectedIndex ? "selected" : ""}`}
                  onClick={() => setSelectedIndex(index)}
                  style={{ height }}
                  aria-label={`${interval.total} logs for ${formatWindow(interval.start, interval.end)}`}
                  title={`${interval.total} logs · ${formatWindow(interval.start, interval.end)}`}
                />
              );
            })}
          </div>
          <div className="volume-caption">
            <span>Older</span>
            <span>Newest</span>
          </div>
        </>
      ) : (
        <div className="empty-evidence">{error || "No indexed logs are available yet."}</div>
      )}
    </div>
  );
}

function LogLevelGraph({ stats, error }: { stats: LogLevelStats | null; error: string | null }) {
  const intervals = stats?.intervals || [];
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [expanded, setExpanded] = useState(false);
  useEffect(() => {
    setSelectedIndex(intervals.length ? intervals.length - 1 : 0);
  }, [intervals.length]);
  useEffect(() => {
    if (!expanded) return;
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") setExpanded(false);
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [expanded]);

  const levels = [
    { name: "INFO", colorClass: "info" },
    { name: "WARN", colorClass: "warn" },
    { name: "ERROR", colorClass: "error" },
    { name: "CRITICAL", colorClass: "critical" },
  ] as const;
  const selectedInterval = intervals[selectedIndex];
  const selectedCounts = selectedInterval?.counts || { INFO: 0, WARN: 0, ERROR: 0, CRITICAL: 0 };
  const selectedTotal = Object.values(selectedCounts).reduce((sum, count) => sum + count, 0);

  return (
    <>
      {expanded && <button className="chart-backdrop" type="button" onClick={() => setExpanded(false)} aria-label="Close enlarged log-level chart" />}
      <div className={`side-card log-level-card ${expanded ? "expanded" : ""}`} role={expanded ? "dialog" : undefined} aria-modal={expanded || undefined}>
      <div className="side-card-header">
        <span>LOG LEVEL DISTRIBUTION</span>
        <div className="log-level-actions">
          <span className="live-pill">LIVE</span>
          <button
            className="expand-button"
            type="button"
            onClick={() => setExpanded((current) => !current)}
            aria-label={expanded ? "Close enlarged log-level chart" : "Enlarge log-level chart"}
            title={expanded ? "Close" : "Enlarge"}
          >
            {expanded ? "×" : "⤢"}
          </button>
        </div>
      </div>
      <div className="log-level-subtitle">
        {intervals.length
          ? `${intervals.length} time windows · ${stats?.raw_logs.toLocaleString()} indexed logs`
          : error
            ? "Unable to load indexed logs"
            : "Loading indexed logs"}
      </div>
      {selectedInterval ? (
        <>
          <div className="interval-controls">
            <button className="interval-button" onClick={() => setSelectedIndex((current) => current - 1)} disabled={selectedIndex === 0} aria-label="Previous time interval">←</button>
            <span>{formatWindow(selectedInterval.start, selectedInterval.end)}</span>
            <button className="interval-button" onClick={() => setSelectedIndex((current) => current + 1)} disabled={selectedIndex === intervals.length - 1} aria-label="Next time interval">→</button>
          </div>
          <div className="level-chart" aria-label="Retrieved logs by level for selected time interval">
            <div className="level-stack-bar">
              {levels.map((level) => {
                const count = selectedCounts[level.name];
                return <span key={level.name} className={`level-fill ${level.colorClass}`} style={{ width: selectedTotal ? `${(count / selectedTotal) * 100}%` : "0%" }} title={`${level.name}: ${count}`} />;
              })}
            </div>
            <div className="level-legend">
              {levels.map((level) => <span key={level.name}><i className={`legend-dot ${level.colorClass}`} />{level.name} {selectedCounts[level.name]}</span>)}
            </div>
          </div>
          <div className="interval-strip" aria-label="Available log time intervals">
            {intervals.map((interval, index) => {
              const total = Object.values(interval.counts).reduce((sum, count) => sum + count, 0);
              const height = `${Math.max(8, Math.min(42, total * 4))}px`;
              return <button key={`${interval.start}-${interval.end}`} className={`interval-preview ${index === selectedIndex ? "selected" : ""}`} onClick={() => setSelectedIndex(index)} style={{ height }} aria-label={`Show logs for ${formatWindow(interval.start, interval.end)}`} title={formatWindow(interval.start, interval.end)} />;
            })}
          </div>
        </>
      ) : (
        <div className="empty-evidence">{error || "No indexed logs are available yet."}</div>
      )}
      <p className="placeholder-note">All incidents and background noise · refreshes every 15 seconds.</p>
      </div>
    </>
  );
}

function AdminMetricsPage({ session }: { session: Session | null }) {
  const [metrics, setMetrics] = useState<ProductionMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!session) return;
    let cancelled = false;

    async function loadMetrics() {
      try {
        const accessToken = await getAccessToken();
        const response = await fetch(`${API_URL}/admin/metrics`, {
          headers: { Authorization: `Bearer ${accessToken}` },
        });
        const body = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new ApiRequestError(
            body.detail || `Metrics request failed with status ${response.status}`,
            response.status,
            response.headers.get("Retry-After"),
          );
        }
        if (!cancelled) {
          setMetrics(body as ProductionMetrics);
          setError(null);
        }
      } catch (requestError) {
        if (!cancelled) setError(friendlyRequestError(requestError));
      }
    }

    void loadMetrics();
    const refreshId = window.setInterval(() => void loadMetrics(), 30000);
    return () => {
      cancelled = true;
      window.clearInterval(refreshId);
    };
  }, [session]);

  const points = (metrics?.series || []).filter(
    (point) => point.requests || point.errors || point.rate_limited || point.payload_rejected,
  );
  const maximumRequests = Math.max(...points.map((point) => point.requests), 1);
  const timeLabelStep = Math.max(1, Math.ceil(points.length / 7));
  const totals = metrics?.totals;

  return (
    <section className="admin-metrics-page">
      <div className="admin-metrics-hero">
        <div>
          <div className="eyebrow">SIFT / PRIVATE OPERATIONS</div>
          <h1>Production metrics.</h1>
          <p>Request traffic and API protection signals from the Redis-backed metrics store.</p>
        </div>
        <span className="admin-badge">ADMIN ONLY</span>
      </div>

      {error && (
        <div className="error-card admin-metrics-error" role="alert">
          <div>
            <strong>Metrics are temporarily unavailable</strong>
            <span>{error}</span>
          </div>
        </div>
      )}

      {metrics && (
        <>
          <div className="metrics-summary-grid">
            <div className="metrics-summary-card"><span>Total requests</span><strong>{totals?.requests.toLocaleString()}</strong><small>Since metrics collection began</small></div>
            <div className="metrics-summary-card"><span>Server errors</span><strong>{totals?.errors.toLocaleString()}</strong><small>HTTP 5xx responses</small></div>
            <div className="metrics-summary-card"><span>Rate limited</span><strong>{totals?.rate_limited.toLocaleString()}</strong><small>HTTP 429 responses</small></div>
            <div className="metrics-summary-card"><span>Average latency</span><strong>{totals?.average_latency_ms.toFixed(0)} ms</strong><small>All recorded requests</small></div>
            <div className="metrics-summary-card"><span>Unique visitors</span><strong>{totals?.unique_visitors.toLocaleString()}</strong><small>Authenticated users in this window</small></div>
          </div>

          <section className="metrics-panel">
            <div className="metrics-panel-header">
              <div>
                <div className="section-label">REQUEST TRAFFIC</div>
                <p>Non-empty minute buckets in the last {Math.round((metrics.window_minutes || 0) / 60)} hours.</p>
              </div>
              <span className="live-pill">REDIS · LIVE</span>
            </div>
            {points.length ? (
              <div className="metrics-bar-chart" aria-label="Request traffic over time">
                {points.map((point, index) => {
                  const showTimeLabel =
                    index === 0 || index === points.length - 1 || index % timeLabelStep === 0;
                  const timestamp = new Date(point.start * 1000);
                  return (
                  <div className="metrics-bar-column" key={point.start} title={`${timestamp.toLocaleString()} · ${point.requests} requests`}>
                    <div className="metrics-bar-stack">
                      <span className="metrics-bar requests" style={{ height: `${Math.max(8, (point.requests / maximumRequests) * 150)}px` }} />
                      {point.errors > 0 && <span className="metrics-bar errors" style={{ height: `${Math.max(5, (point.errors / maximumRequests) * 150)}px` }} />}
                    </div>
                    <small className={showTimeLabel ? "metrics-time-label" : "metrics-time-label metrics-time-label-hidden"}>
                      {showTimeLabel ? timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false }) : "·"}
                    </small>
                  </div>
                  );
                })}
              </div>
            ) : (
              <div className="empty-evidence">No production requests have been recorded yet.</div>
            )}
            <div className="metrics-legend"><span><i className="metrics-legend-dot requests" /> Requests</span><span><i className="metrics-legend-dot errors" /> Server errors</span></div>
          </section>

          <section className="metrics-panel metrics-breakdown">
            <div className="section-label">PROTECTION EVENTS</div>
            <div className="metrics-breakdown-grid">
              <div><strong>{totals?.rate_limited.toLocaleString()}</strong><span>Rate-limited requests</span></div>
              <div><strong>{totals?.payload_rejected.toLocaleString()}</strong><span>Oversized payloads</span></div>
              <div><strong>{points.length}</strong><span>Active time buckets</span></div>
            </div>
          </section>
        </>
      )}
    </section>
  );
}

function MessageCard({
  message,
  onRetry,
}: {
  message: ChatMessage;
  onRetry: (question: string) => Promise<void>;
}) {
  if (message.role === "user") {
    return (
      <div className="user-message-group">
        <div className="user-row">
          <div className="message-bubble user-bubble">{message.text}</div>
        </div>
        <div className="user-message-actions">
          <button
            className="retry-icon-button"
            type="button"
            onClick={() => void onRetry(message.text)}
            aria-label="Retry this question"
            title="Retry this question"
          >
            ↻
          </button>
        </div>
      </div>
    );
  }

  const chunks = message.chunks || [];
  return (
    <div className="assistant-row">
      <div className="avatar assistant-avatar">S</div>
      <div className="assistant-content">
        <div className="message-bubble assistant-bubble">
          <div className="answer-toolbar">
            <span className="grounding-badge"><span className="badge-dot" />{groundingLabel(chunks, message.retrievalNote, message.responseType)}</span>
            {message.latencyMs !== undefined && <span className="latency">{(message.latencyMs / 1000).toFixed(1)}s</span>}
          </div>
          <div className="markdown-answer"><ReactMarkdown remarkPlugins={[remarkGfm]}>{message.text}</ReactMarkdown></div>
          <div className="answer-actions">
            <CopyButton text={message.text} />
            <ExportButton message={message} format="md" />
            <ExportButton message={message} format="json" />
          </div>
        </div>
        {!shouldHideEvidence(chunks, message.responseType) && (
          <EvidencePanel chunks={chunks} note={message.retrievalNote} />
        )}
      </div>
    </div>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }
  return <button className="copy-button" onClick={() => void copy()}>{copied ? "Copied" : "Copy answer"}</button>;
}

function ExportButton({ message, format }: { message: ChatMessage; format: "md" | "json" }) {
  function exportAnswer() {
    const payload = format === "json"
      ? JSON.stringify(
          {
            answer: message.text,
            response_type: message.responseType || "supported",
            retrieval_note: message.retrievalNote || null,
            retrieved_chunks: message.chunks || [],
          },
          null,
          2,
        )
      : `# Sift answer\n\n${message.text}\n\n## Retrieval note\n\n${message.retrievalNote || "None"}\n`;
    const blob = new Blob([payload], { type: format === "json" ? "application/json" : "text/markdown" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `sift-answer.${format}`;
    link.click();
    URL.revokeObjectURL(url);
  }

  return <button className="copy-button" onClick={exportAnswer}>Export {format.toUpperCase()}</button>;
}

function EvidencePanel({ chunks, note }: { chunks: RetrievedChunk[]; note?: string | null }) {
  const [search, setSearch] = useState("");
  const normalizedSearch = search.trim().toLowerCase();
  const filteredChunks = useMemo(() => {
    if (!normalizedSearch) return chunks;
    return chunks.filter((chunk) =>
      [chunk.content, ...chunk.services, ...chunk.trace_ids].join(" ").toLowerCase().includes(normalizedSearch),
    );
  }, [chunks, normalizedSearch]);
  const traces = traceGroups(chunks);
  const events = parseEvents(chunks);

  return (
    <section className="evidence-panel">
      <div className="evidence-header">
        <div>
          <div className="evidence-title">Evidence & timeline</div>
          <div className="evidence-subtitle">{chunks.length} retrieved chunk{chunks.length === 1 ? "" : "s"} · {traces.length} trace{traces.length === 1 ? "" : "s"}</div>
        </div>
        <span className="evidence-icon">⌁</span>
      </div>
      {note && <div className="retrieval-note"><span>i</span>{note}</div>}
      {chunks.length === 0 ? (
        <div className="empty-evidence">No log chunks were retrieved for this question.</div>
      ) : (
        <>
          {traces.length > 1 && (
            <div className="trace-comparison">
              <div className="section-label">TRACE COMPARISON</div>
              <div className="trace-grid">
                {traces.map(([traceId, traceChunks]) => {
                  const services = [...new Set(traceChunks.flatMap((chunk) => chunk.services))];
                  const traceEvents = parseEvents(traceChunks);
                  const lastTraceEvent = traceEvents[traceEvents.length - 1];
                  return (
                    <div className="trace-card" key={traceId}>
                      <strong>{traceLabel(traceId)}</strong>
                      <span>{traceId}</span>
                      <small>{services.join(" · ")}</small>
                      {traceEvents[0] && <small>First logged event: {traceEvents[0].time} · {traceEvents[0].service}</small>}
                      {lastTraceEvent && <small>Last logged symptom: {lastTraceEvent.time} · {lastTraceEvent.service}</small>}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
          <div className="section-label timeline-label">CHRONOLOGICAL TIMELINE</div>
          <div className="timeline">
            {events.map((event, index) => (
              <div className="timeline-event" key={`${event.time}-${event.service}-${index}`}>
                <span className="timeline-time">{event.time}</span>
                <span className={`timeline-dot ${event.level?.toLowerCase() || "info"}`} />
                <span className="timeline-service">{event.service}</span>
                <span className="timeline-message">{event.message}</span>
              </div>
            ))}
          </div>
          <div className="chunk-list-header">
            <div className="section-label">RETRIEVED CHUNKS</div>
            <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Filter evidence..." aria-label="Filter retrieved evidence" />
          </div>
          <div className="chunk-list">
            {filteredChunks.map((chunk, index) => <ChunkCard key={chunk.id} chunk={chunk} number={index + 1} />)}
            {!filteredChunks.length && <div className="empty-evidence">No chunks match “{search}”.</div>}
          </div>
        </>
      )}
    </section>
  );
}

function ChunkCard({ chunk, number }: { chunk: RetrievedChunk; number: number }) {
  return (
    <details className="chunk-card">
      <summary>
        <span className="chunk-number">{String(number).padStart(2, "0")}</span>
        <span className="chunk-summary"><strong>{chunk.services.join(" · ") || "Background noise"}</strong><span>{formatWindow(chunk.window_start, chunk.window_end)}</span></span>
        <span className="chunk-chevron">⌄</span>
      </summary>
      <div className="chunk-details">
        <div className="chip-row">
          {(chunk.trace_ids.length ? chunk.trace_ids : ["noise"]).map((traceId) => <span className="chip" key={traceId}>{traceId}</span>)}
          {chunk.max_level && <span className="chip level-chip">{chunk.max_level}</span>}
        </div>
        <pre>{chunk.content}</pre>
        <div className="chunk-meta"><span>ID {chunk.id}</span>{chunk.cosine_similarity != null && <span>Similarity {chunk.cosine_similarity.toFixed(3)}</span>}</div>
      </div>
    </details>
  );
}

export default App;
