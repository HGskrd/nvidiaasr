import {
  collectBrowserDiagnostics,
  collectHighEntropyHints,
  DiagnosticRecord,
  serializeDiagnosticValue
} from "./diagnostics";

// Best-effort forwarding of diagnostic records to the /admin collector so the
// same logs visible in the Runtime Log are aggregated server-side across every
// machine under test. On a static host (no collector) it self-disables after a
// few failed posts and the app is otherwise unaffected.

const INGEST_URL = "/admin/ingest";
const FLUSH_INTERVAL_MS = 4000;
const MAX_QUEUE = 1000;
const MAX_FAILURES = 3;
const LABEL_KEY = "nemotron-machine-label";

function makeSessionId(): string {
  const uuid = globalThis.crypto?.randomUUID?.();
  return uuid ?? `s-${Date.now().toString(36)}-${Math.random().toString(16).slice(2, 10)}`;
}

function readMachineLabel(): string | undefined {
  try {
    const fromQuery = new URL(window.location.href).searchParams.get("machine");
    if (fromQuery) {
      window.localStorage?.setItem(LABEL_KEY, fromQuery);
      return fromQuery;
    }
    return window.localStorage?.getItem(LABEL_KEY) ?? undefined;
  } catch {
    return undefined;
  }
}

interface IngestPayload {
  sessionId: string;
  label?: string;
  userAgent: string;
  meta?: unknown;
  records: DiagnosticRecord[];
}

class Telemetry {
  readonly sessionId = makeSessionId();
  private label = readMachineLabel();
  private queue: DiagnosticRecord[] = [];
  private timer: ReturnType<typeof setTimeout> | undefined;
  private failures = 0;
  private enabled = true;
  private metaSent = false;
  private meta?: unknown;

  /** Capture machine/session metadata and begin forwarding records. */
  start(extra?: Record<string, unknown>): void {
    this.meta = serializeDiagnosticValue({
      ...(collectBrowserDiagnostics() as Record<string, unknown>),
      sessionId: this.sessionId,
      ...(this.label ? { machineLabel: this.label } : {}),
      appMode: import.meta.env.MODE,
      pageUrl: window.location.href,
      startedAt: new Date().toISOString(),
      ...extra
    });

    window.addEventListener("pagehide", () => this.flushBeacon());
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "hidden") void this.flush();
    });

    this.scheduleFlush(0);

    // Real CPU architecture resolves asynchronously; fold it in and resend meta.
    void collectHighEntropyHints().then((hints) => {
      if (!hints || typeof this.meta !== "object" || this.meta === null) return;
      this.meta = { ...(this.meta as Record<string, unknown>), clientHints: hints };
      this.metaSent = false;
      this.scheduleFlush(0);
    });
  }

  record(record: DiagnosticRecord): void {
    if (!this.enabled) return;
    this.queue.push(record);
    if (this.queue.length > MAX_QUEUE) this.queue.splice(0, this.queue.length - MAX_QUEUE);
    this.scheduleFlush(record.level === "info" ? FLUSH_INTERVAL_MS : 0);
  }

  private scheduleFlush(delay: number): void {
    if (!this.enabled) return;
    if (delay === 0) {
      if (this.timer !== undefined) {
        clearTimeout(this.timer);
        this.timer = undefined;
      }
      void this.flush();
      return;
    }
    if (this.timer !== undefined) return;
    this.timer = setTimeout(() => {
      this.timer = undefined;
      void this.flush();
    }, delay);
  }

  private takePayload(): IngestPayload {
    const records = this.queue;
    this.queue = [];
    const payload: IngestPayload = {
      sessionId: this.sessionId,
      label: this.label,
      userAgent: navigator.userAgent,
      records
    };
    if (!this.metaSent) payload.meta = this.meta;
    return payload;
  }

  private async flush(): Promise<void> {
    if (!this.enabled) return;
    if (this.queue.length === 0 && this.metaSent) return;
    const payload = this.takePayload();
    try {
      const response = await fetch(INGEST_URL, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
        keepalive: true
      });
      if (!response.ok) throw new Error(`ingest ${response.status}`);
      this.metaSent = true;
      this.failures = 0;
    } catch {
      // Requeue (bounded) so a transient hiccup doesn't drop records.
      this.queue = payload.records.concat(this.queue).slice(-MAX_QUEUE);
      this.failures++;
      if (this.failures >= MAX_FAILURES) this.enabled = false;
    }
  }

  private flushBeacon(): void {
    if (!this.enabled || (this.queue.length === 0 && this.metaSent)) return;
    try {
      const payload = this.takePayload();
      const blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
      if (!navigator.sendBeacon?.(INGEST_URL, blob)) this.queue = payload.records.concat(this.queue);
    } catch {
      /* ignore */
    }
  }
}

export const telemetry = new Telemetry();
