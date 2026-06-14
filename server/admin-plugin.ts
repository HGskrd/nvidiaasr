import { createHash, timingSafeEqual } from "node:crypto";
import type { Connect, Plugin, ViteDevServer } from "vite";

// Aggregates diagnostic logs POSTed by every browser/machine under test and
// serves them on a password-protected /admin dashboard, so failures can be
// inspected centrally without copy-pasting between machines.

const ADMIN_USER = process.env.ADMIN_USER ?? "admin";
const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD ?? "nemotron-admin";
const ADMIN_REALM = "Nemotron ASR Admin";
const MAX_SESSIONS = 200;
const MAX_RECORDS_PER_SESSION = 4000;
const MAX_BODY_BYTES = 4_000_000;

type Level = "info" | "warn" | "error";

interface StoredRecord {
  seq: number;
  at: string;
  serverAt: string;
  level: Level;
  message: string;
  detail?: unknown;
}

interface Session {
  sessionId: string;
  label?: string;
  meta?: unknown;
  ip?: string;
  userAgent?: string;
  firstSeen: string;
  lastSeen: string;
  counts: { info: number; warn: number; error: number };
  records: StoredRecord[];
}

const sessions = new Map<string, Session>();
let globalSeq = 0;

function evictIfNeeded(): void {
  if (sessions.size <= MAX_SESSIONS) return;
  const ordered = [...sessions.values()].sort((a, b) => a.lastSeen.localeCompare(b.lastSeen));
  for (const session of ordered) {
    if (sessions.size <= MAX_SESSIONS) break;
    sessions.delete(session.sessionId);
  }
}

function normalizeLevel(value: unknown): Level {
  return value === "warn" || value === "error" ? value : "info";
}

function ingest(body: unknown, ip: string | undefined): void {
  if (typeof body !== "object" || body === null) return;
  const payload = body as {
    sessionId?: unknown;
    label?: unknown;
    meta?: unknown;
    userAgent?: unknown;
    records?: unknown;
  };
  const sessionId = typeof payload.sessionId === "string" ? payload.sessionId : undefined;
  if (!sessionId) return;
  const now = new Date().toISOString();

  let session = sessions.get(sessionId);
  if (!session) {
    session = {
      sessionId,
      firstSeen: now,
      lastSeen: now,
      counts: { info: 0, warn: 0, error: 0 },
      records: []
    };
    sessions.set(sessionId, session);
  }

  session.lastSeen = now;
  session.ip = ip;
  if (typeof payload.label === "string") session.label = payload.label;
  if (payload.meta !== undefined) session.meta = payload.meta;
  if (typeof payload.userAgent === "string") session.userAgent = payload.userAgent;

  if (Array.isArray(payload.records)) {
    for (const raw of payload.records) {
      if (typeof raw !== "object" || raw === null) continue;
      const record = raw as { at?: unknown; level?: unknown; message?: unknown; detail?: unknown };
      const level = normalizeLevel(record.level);
      session.records.push({
        seq: ++globalSeq,
        at: typeof record.at === "string" ? record.at : now,
        serverAt: now,
        level,
        message: typeof record.message === "string" ? record.message : String(record.message ?? ""),
        detail: record.detail
      });
      session.counts[level]++;
    }
    if (session.records.length > MAX_RECORDS_PER_SESSION) {
      session.records.splice(0, session.records.length - MAX_RECORDS_PER_SESSION);
    }
  }

  evictIfNeeded();
}

function snapshot(levelFilter?: Level): { serverTime: string; sessions: unknown[] } {
  const rank: Record<Level, number> = { info: 0, warn: 1, error: 2 };
  const minRank = levelFilter ? rank[levelFilter] : 0;
  const out = [...sessions.values()]
    .sort((a, b) => b.lastSeen.localeCompare(a.lastSeen))
    .map((session) => ({
      sessionId: session.sessionId,
      label: session.label,
      ip: session.ip,
      userAgent: session.userAgent,
      meta: session.meta,
      firstSeen: session.firstSeen,
      lastSeen: session.lastSeen,
      counts: session.counts,
      records: session.records.filter((record) => rank[record.level] >= minRank).slice(-1500)
    }));
  return { serverTime: new Date().toISOString(), sessions: out };
}

function percentile(sortedAsc: number[], p: number): number {
  if (sortedAsc.length === 0) return 0;
  const index = Math.min(sortedAsc.length - 1, Math.ceil((p / 100) * sortedAsc.length) - 1);
  return Math.round(sortedAsc[Math.max(0, index)]);
}

function parseBrowser(ua: string | undefined): string | undefined {
  if (!ua) return undefined;
  const m =
    /(Edg|OPR|Chrome|Firefox|Safari)\/(\d+)/.exec(ua) ?? /(Edge)\/(\d+)/.exec(ua);
  if (!m) return undefined;
  const name = { Edg: "Edge", OPR: "Opera" }[m[1]] ?? m[1];
  return `${name} ${m[2]}`;
}

function parseOs(ua: string | undefined): string | undefined {
  if (!ua) return undefined;
  if (/Mac OS X/.test(ua)) return "macOS";
  if (/Windows NT 10/.test(ua)) return "Windows";
  if (/Android/.test(ua)) return "Android";
  if (/(iPhone|iPad)/.test(ua)) return "iOS";
  if (/Linux/.test(ua)) return "Linux";
  return undefined;
}

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {};
}

// Collapse a session's record stream into a compact summary: machine/GPU facts,
// load timings, a chunk-latency rollup, and the full warn/error list. This is
// what /admin/api/data returns by default; ?full=1 returns the raw records.
function digestSession(session: Session): unknown {
  const meta = asRecord(session.meta);
  const hints = asRecord(meta.clientHints);
  const ua = (meta.userAgent as string | undefined) ?? session.userAgent;

  const chunkMs: number[] = [];
  let emittedTokens = 0;
  let lastTokenCount = 0;
  let slowChunks = 0;
  const load: Record<string, number> = {};
  let mic: Record<string, unknown> | undefined;
  let audio: Record<string, unknown> | undefined;
  let webgpuFeatures: number | undefined;
  let shaderF16: boolean | undefined;
  let adapter: unknown;
  const issues: Array<{ at: string; level: Level; message: string; detail?: unknown }> = [];
  const budgetMs = 560;

  for (const record of session.records) {
    if (record.level !== "info") {
      issues.push({ at: record.at, level: record.level, message: record.message, detail: record.detail });
    }
    const detail = asRecord(record.detail);
    const message = record.message;

    if (message.startsWith("Chunk ") && message.endsWith(" complete")) {
      if (typeof detail.elapsedMs === "number") {
        chunkMs.push(detail.elapsedMs);
        if (detail.elapsedMs > budgetMs) slowChunks++;
      }
      if (typeof detail.emittedTokens === "number") emittedTokens += detail.emittedTokens;
      if (typeof detail.tokenCount === "number") lastTokenCount = detail.tokenCount;
    } else if (message.startsWith("[session] Created ONNX WebGPU session for") && typeof detail.elapsedMs === "number") {
      load[message.replace("[session] Created ONNX WebGPU session for ", "")] = detail.elapsedMs;
    } else if (message.includes("Loaded vocab.txt") && typeof detail.elapsedMs === "number") {
      load["vocab.txt"] = detail.elapsedMs;
    } else if (message === "Microphone started") {
      mic = {
        scriptBuffer: detail.scriptProcessorBufferSize,
        chunkSamples: detail.chunkSamples,
        targetSampleRate: detail.targetSampleRate
      };
    } else if (message === "AudioContext ready") {
      audio = { sampleRate: detail.sampleRate, baseLatency: detail.baseLatency, state: detail.state };
    } else if (message === "WebGPU adapter available") {
      const features = Array.isArray(detail.features) ? detail.features : [];
      webgpuFeatures = features.length;
      shaderF16 = features.includes("shader-f16");
      const info = asRecord(detail.info);
      if (Object.keys(info).length > 0) adapter = info;
    }
  }

  chunkMs.sort((a, b) => a - b);
  const chunks = chunkMs.length
    ? {
        count: chunkMs.length,
        p50Ms: percentile(chunkMs, 50),
        p95Ms: percentile(chunkMs, 95),
        maxMs: chunkMs[chunkMs.length - 1],
        budgetMs,
        slowChunks,
        realtime: percentile(chunkMs, 95) <= budgetMs,
        tokens: lastTokenCount,
        emittedTokens
      }
    : undefined;

  return {
    id: session.sessionId.slice(0, 8),
    ...(session.label ? { label: session.label } : {}),
    ip: session.ip,
    lastSeen: session.lastSeen,
    durationMs: Date.parse(session.lastSeen) - Date.parse(session.firstSeen),
    machine: {
      browser: parseBrowser(ua),
      os: parseOs(ua),
      platform: meta.platform,
      arch: hints.architecture
        ? `${hints.architecture}${hints.bitness ? `/${hints.bitness}` : ""}`
        : "unknown (UA-CH unavailable)",
      cores: meta.hardwareConcurrency,
      memGB: meta.deviceMemoryGB
    },
    webgpu:
      meta.webgpu === false
        ? { available: false }
        : { available: true, features: webgpuFeatures, shaderF16, adapter },
    env: {
      secureContext: meta.secureContext,
      crossOriginIsolated: meta.crossOriginIsolated,
      mediaDevices: meta.mediaDevices,
      cacheStorage: meta.cacheStorage,
      audioContext: meta.audioContext
    },
    counts: session.counts,
    load: Object.keys(load).length > 0 ? load : undefined,
    mic,
    audio,
    chunks,
    issues: issues.slice(-25)
  };
}

function digest(): { serverTime: string; note: string; sessions: unknown[] } {
  return {
    serverTime: new Date().toISOString(),
    note: "Compact digest. Append ?full=1 for raw records.",
    sessions: [...sessions.values()]
      .sort((a, b) => b.lastSeen.localeCompare(a.lastSeen))
      .map(digestSession)
  };
}

function readBody(req: Connect.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = "";
    let size = 0;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(new Error("payload too large"));
        req.destroy();
        return;
      }
      data += chunk.toString("utf8");
    });
    req.on("end", () => resolve(data));
    req.on("error", reject);
  });
}

function constantTimeEquals(a: string, b: string): boolean {
  const ha = createHash("sha256").update(a).digest();
  const hb = createHash("sha256").update(b).digest();
  return timingSafeEqual(ha, hb);
}

function isAuthorized(req: Connect.IncomingMessage): boolean {
  const header = req.headers.authorization;
  if (typeof header !== "string" || !header.startsWith("Basic ")) return false;
  let decoded: string;
  try {
    decoded = Buffer.from(header.slice(6), "base64").toString("utf8");
  } catch {
    return false;
  }
  const separator = decoded.indexOf(":");
  if (separator === -1) return false;
  const user = decoded.slice(0, separator);
  const password = decoded.slice(separator + 1);
  return constantTimeEquals(user, ADMIN_USER) && constantTimeEquals(password, ADMIN_PASSWORD);
}

function requireAuth(req: Connect.IncomingMessage, res: Connect.ServerResponse): boolean {
  if (isAuthorized(req)) return true;
  res.statusCode = 401;
  res.setHeader("WWW-Authenticate", `Basic realm="${ADMIN_REALM}", charset="UTF-8"`);
  res.setHeader("Content-Type", "text/plain; charset=utf-8");
  res.end("Authentication required");
  return false;
}

function clientIp(req: Connect.IncomingMessage): string | undefined {
  const forwarded = req.headers["x-forwarded-for"];
  if (typeof forwarded === "string" && forwarded.length > 0) return forwarded.split(",")[0].trim();
  return req.socket?.remoteAddress ?? undefined;
}

function sendJson(res: Connect.ServerResponse, status: number, value: unknown): void {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Cache-Control", "no-store");
  res.end(JSON.stringify(value));
}

function adminMiddleware(logger: ViteDevServer["config"]["logger"]): Connect.HandleFunction {
  return async (req, res, next) => {
    const url = new URL(req.url ?? "/", "http://localhost");
    const path = url.pathname;
    if (path !== "/admin" && !path.startsWith("/admin/")) {
      next();
      return;
    }

    const method = (req.method ?? "GET").toUpperCase();

    // Open ingest endpoint: test machines POST here without a human present.
    if (path === "/admin/ingest") {
      if (method !== "POST") {
        res.statusCode = 405;
        res.end("Method not allowed");
        return;
      }
      try {
        ingest(JSON.parse(await readBody(req)), clientIp(req));
        sendJson(res, 200, { ok: true });
      } catch (error) {
        logger.warn(`[admin] bad ingest: ${error instanceof Error ? error.message : String(error)}`);
        sendJson(res, 400, { ok: false });
      }
      return;
    }

    // Everything below is the operator-facing surface and needs the password.
    if (!requireAuth(req, res)) return;

    if (path === "/admin/api/data") {
      if (url.searchParams.get("full") === "1") {
        const levelParam = url.searchParams.get("level");
        const level = levelParam === "warn" || levelParam === "error" ? levelParam : undefined;
        sendJson(res, 200, snapshot(level));
      } else {
        sendJson(res, 200, digest());
      }
      return;
    }

    if (path === "/admin/api/clear") {
      if (method !== "POST") {
        res.statusCode = 405;
        res.end("Method not allowed");
        return;
      }
      sessions.clear();
      globalSeq = 0;
      sendJson(res, 200, { ok: true });
      return;
    }

    if (path === "/admin" || path === "/admin/") {
      res.statusCode = 200;
      res.setHeader("Content-Type", "text/html; charset=utf-8");
      res.setHeader("Cache-Control", "no-store");
      res.end(DASHBOARD_HTML);
      return;
    }

    res.statusCode = 404;
    res.end("Not found");
  };
}

export function adminPlugin(): Plugin {
  let warned = false;
  const warnDefaultPassword = (logger: ViteDevServer["config"]["logger"]) => {
    if (warned) return;
    warned = true;
    logger.info(`[admin] diagnostics dashboard at /admin (user "${ADMIN_USER}")`);
    if (!process.env.ADMIN_PASSWORD) {
      logger.warn('[admin] using default password "nemotron-admin"; set ADMIN_PASSWORD to override');
    }
  };
  return {
    name: "nemotron-admin",
    configureServer(server) {
      warnDefaultPassword(server.config.logger);
      server.middlewares.use(adminMiddleware(server.config.logger));
    },
    configurePreviewServer(server) {
      warnDefaultPassword(server.config.logger);
      server.middlewares.use(adminMiddleware(server.config.logger));
    }
  };
}

const DASHBOARD_HTML = /* html */ `<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Nemotron ASR — Diagnostics</title>
<style>
  :root { color-scheme: light dark; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
  body { margin: 0; background: #0f1419; color: #e6edf3; }
  header { position: sticky; top: 0; z-index: 2; display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
    padding: 12px 16px; background: #161b22; border-bottom: 1px solid #30363d; }
  header h1 { font-size: 16px; margin: 0 12px 0 0; }
  header .spacer { flex: 1; }
  label.f { font-size: 13px; display: inline-flex; align-items: center; gap: 6px; color: #9aa7b4; }
  select, input[type=search], button { font: inherit; font-size: 13px; background: #0d1117; color: #e6edf3;
    border: 1px solid #30363d; border-radius: 6px; padding: 6px 8px; }
  button { cursor: pointer; }
  button:hover { border-color: #58a6ff; }
  main { padding: 16px; display: grid; gap: 14px; }
  .session { border: 1px solid #30363d; border-radius: 10px; overflow: hidden; background: #11161d; }
  .session > summary { cursor: pointer; padding: 10px 14px; list-style: none; display: flex; flex-wrap: wrap;
    gap: 10px; align-items: center; background: #161b22; }
  .session > summary::-webkit-details-marker { display: none; }
  .tag { font-weight: 700; }
  .muted { color: #8b949e; font-size: 12px; }
  .badge { font-size: 12px; padding: 1px 8px; border-radius: 999px; border: 1px solid #30363d; }
  .badge.error { color: #ffa198; border-color: #f85149; }
  .badge.warn { color: #f0c674; border-color: #d29922; }
  .badge.info { color: #79c0ff; border-color: #1f6feb; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  td { padding: 6px 10px; border-top: 1px solid #21262d; vertical-align: top; }
  tr.error td { background: rgba(248,81,73,0.08); }
  tr.warn td { background: rgba(210,153,34,0.08); }
  td.lvl { white-space: nowrap; font-weight: 700; }
  td.lvl.error { color: #ff7b72; }
  td.lvl.warn { color: #e3b341; }
  td.lvl.info { color: #6e7681; }
  td.time { white-space: nowrap; color: #8b949e; font-variant-numeric: tabular-nums; }
  pre { margin: 6px 0 0; white-space: pre-wrap; word-break: break-word; color: #adbac7;
    background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 8px; max-height: 320px; overflow: auto; }
  .meta { padding: 8px 14px; }
  .empty { color: #8b949e; padding: 40px; text-align: center; }
</style>
</head>
<body>
<header>
  <h1>Nemotron ASR Diagnostics</h1>
  <label class="f">Level <select id="level">
    <option value="all">all</option><option value="warn">warn+error</option><option value="error">error</option>
  </select></label>
  <input id="search" type="search" placeholder="filter text…" />
  <label class="f"><input type="checkbox" id="auto" checked /> auto-refresh</label>
  <span class="spacer"></span>
  <span class="muted" id="status">loading…</span>
  <button id="refresh">Refresh</button>
  <button id="download">Download</button>
  <button id="clear">Clear all</button>
</header>
<main id="main"><div class="empty">No sessions yet. Open the app on a test machine with biasing/mic and failures will appear here.</div></main>
<script>
const $ = (id) => document.getElementById(id);
let latest = { sessions: [] };
const open = new Set();

function fmtTime(iso) { try { return new Date(iso).toLocaleTimeString(); } catch { return iso; } }
function esc(s) { return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }

function machineName(s) {
  if (s.label) return s.label;
  const m = s.meta || {};
  const ua = s.userAgent || m.userAgent || "";
  const plat = m.platform ? " · " + m.platform : "";
  const gpu = m.webgpu === false ? " · no-webgpu" : "";
  return (ua.slice(0, 48) || s.sessionId.slice(0, 8)) + plat + gpu;
}

function render() {
  const level = $("level").value;
  const q = $("search").value.trim().toLowerCase();
  const main = $("main");
  const visible = [];
  for (const s of latest.sessions) {
    let records = s.records || [];
    if (level === "warn") records = records.filter((r) => r.level !== "info");
    else if (level === "error") records = records.filter((r) => r.level === "error");
    if (q) records = records.filter((r) => (r.message + " " + JSON.stringify(r.detail || "")).toLowerCase().includes(q));
    visible.push({ s, records });
  }
  if (visible.every((v) => v.records.length === 0) && !q && level === "all" && latest.sessions.length === 0) {
    main.innerHTML = '<div class="empty">No sessions yet.</div>';
    return;
  }
  main.innerHTML = visible.map(({ s, records }) => {
    const id = s.sessionId;
    const rows = records.slice().reverse().map((r) =>
      '<tr class="' + r.level + '"><td class="time">' + fmtTime(r.at) + '</td><td class="lvl ' + r.level + '">' +
      r.level + '</td><td>' + esc(r.message) +
      (r.detail !== undefined && r.detail !== null ? '<pre>' + esc(JSON.stringify(r.detail, null, 2)) + '</pre>' : '') +
      '</td></tr>').join("");
    const c = s.counts || { info: 0, warn: 0, error: 0 };
    const metaJson = s.meta ? '<div class="meta"><pre>' + esc(JSON.stringify(s.meta, null, 2)) + '</pre></div>' : '';
    return '<details class="session" data-id="' + id + '"' + (open.has(id) ? ' open' : '') + '>' +
      '<summary><span class="tag">' + esc(machineName(s)) + '</span>' +
      '<span class="badge error">' + c.error + ' err</span>' +
      '<span class="badge warn">' + c.warn + ' warn</span>' +
      '<span class="badge info">' + c.info + ' info</span>' +
      '<span class="muted">' + esc(s.ip || "") + ' · last ' + fmtTime(s.lastSeen) + '</span></summary>' +
      metaJson + '<table>' + rows + '</table></details>';
  }).join("");

  main.querySelectorAll("details.session").forEach((d) => {
    d.addEventListener("toggle", () => {
      const id = d.getAttribute("data-id");
      if (d.open) open.add(id); else open.delete(id);
    });
  });
}

async function refresh() {
  try {
    const res = await fetch("/admin/api/data?full=1", { headers: { accept: "application/json" } });
    if (!res.ok) throw new Error("HTTP " + res.status);
    latest = await res.json();
    $("status").textContent = latest.sessions.length + " session(s) · " + fmtTime(latest.serverTime);
    render();
  } catch (e) {
    $("status").textContent = "error: " + e.message;
  }
}

$("level").addEventListener("change", render);
$("search").addEventListener("input", render);
$("refresh").addEventListener("click", refresh);
$("download").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(latest, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "nemotron-diagnostics-" + new Date().toISOString().replace(/[:.]/g, "-") + ".json";
  a.click();
});
$("clear").addEventListener("click", async () => {
  if (!confirm("Clear all collected diagnostics?")) return;
  await fetch("/admin/api/clear", { method: "POST" });
  open.clear();
  refresh();
});

setInterval(() => { if ($("auto").checked) refresh(); }, 4000);
refresh();
</script>
</body>
</html>`;
