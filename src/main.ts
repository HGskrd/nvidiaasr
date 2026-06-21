import { Copy, Cpu, createIcons, Gauge, Loader2, Mic, RotateCcw, Square, Trash2, Upload } from "lucide";
import { decodeAudioFile, ModelChunker, StreamingLinearResampler } from "./audio";
import {
  collectBrowserDiagnostics,
  detectWasmSimd,
  DiagnosticLevel,
  DiagnosticRecord,
  errorMessage,
  probeWebGpu,
  serializeDiagnosticValue,
  serializeError
} from "./diagnostics";
import {
  formatBytes,
  LANG_ID_PRESETS,
  MODEL_FILES,
  MODEL_ID,
  MODEL_REVISION,
  NEMOTRON_CONFIG
} from "./model-config";
import { AsrClient } from "./asr-client";
import type { LoadOptions, LoadProgress, StreamProgress } from "./nemotron-asr";
import { telemetry } from "./telemetry";
import "./styles.css";

type StatusKind = "idle" | "busy" | "ok" | "error";

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("Missing #app");

const modelBytes =
  MODEL_FILES.encoder.bytes + MODEL_FILES.decoder.bytes + MODEL_FILES.joint.bytes + MODEL_FILES.tokenizer.bytes;

app.innerHTML = `
  <main class="shell">
    <header class="topbar">
      <div>
        <h1>Nemotron 3.5 ONNX WebGPU</h1>
        <p>${MODEL_ID} @ ${MODEL_REVISION.slice(0, 7)}</p>
      </div>
      <div class="status" id="status"><span></span><strong>starting</strong></div>
    </header>

    <section class="control-band">
      <div class="primary-actions">
        <button id="load-model" class="primary-button" data-mode="load" type="button">
          <i class="icon-load" data-lucide="cpu"></i>
          <i class="icon-loading" data-lucide="loader-2"></i>
          <span>Load ONNX Model</span>
        </button>
        <button id="microphone" class="secondary-button" data-mode="start" type="button">
          <i class="icon-mic" data-lucide="mic"></i>
          <i class="icon-stop" data-lucide="square"></i>
          <span>Start Mic</span>
        </button>
        <button id="reset" class="icon-button" title="Reset transcript" aria-label="Reset transcript" type="button">
          <i data-lucide="rotate-ccw"></i>
        </button>
      </div>

      <div class="input-actions">
        <label class="select-control" for="language">
          <span>Language ID</span>
          <select id="language">
            ${LANG_ID_PRESETS.map((preset) => `<option value="${preset.id}">${preset.label}</option>`).join("")}
          </select>
        </label>
        <label class="file-control" for="audio-file">
          <i data-lucide="upload"></i>
          <span>Audio File</span>
          <input id="audio-file" type="file" accept="audio/*" />
        </label>
        <label class="file-control" for="benchmark-file" title="Run the file through every backend and chunk size">
          <i data-lucide="gauge"></i>
          <span>Benchmark</span>
          <input id="benchmark-file" type="file" accept="audio/*" />
        </label>
      </div>
    </section>

    <section class="biasing">
      <label class="toggle-control" for="bias-enabled">
        <input id="bias-enabled" type="checkbox" />
        <span>Contextual biasing</span>
      </label>
      <label class="bias-field" for="bias-terms">
        <span>Bias terms (one per line)</span>
        <textarea id="bias-terms" rows="3" spellcheck="false" placeholder="Kubernetes&#10;NVIDIA TensorRT&#10;Nemotron"></textarea>
      </label>
      <label class="select-control" for="bias-beam">
        <span>Beam size</span>
        <input id="bias-beam" type="number" min="1" max="8" step="1" value="4" />
      </label>
    </section>

    <section class="runtime-panel">
      <p id="detail">Ready</p>
      <div class="meters" aria-label="Runtime details">
        <div><span>Provider</span><strong>WebGPU</strong></div>
        <div><span>Weights</span><strong>${formatBytes(modelBytes)}</strong></div>
        <div><span>Sample Rate</span><strong>${NEMOTRON_CONFIG.sampleRate / 1000} kHz</strong></div>
        <div><span>Chunks</span><strong id="chunk-count">0</strong></div>
        <div><span>Queue</span><strong id="queue-depth">0</strong></div>
        <div><span>Tokens</span><strong id="token-count">0</strong></div>
        <div><span>Top Token</span><strong id="top-token">-</strong></div>
        <div><span>Blank Score</span><strong id="blank-score">-</strong></div>
      </div>
    </section>

    <section class="workspace">
      <section class="transcript-panel">
        <div class="panel-head">
          <h2>Transcript</h2>
          <span id="word-count">0 words</span>
        </div>
        <output id="transcript" data-empty="true"></output>
      </section>

      <aside class="log-panel">
        <div class="panel-head">
          <h2>Runtime Log</h2>
          <div class="log-actions">
            <span id="state">Idle</span>
            <button id="copy-log" class="icon-button small" title="Copy diagnostic log" aria-label="Copy diagnostic log" type="button">
              <i data-lucide="copy"></i>
            </button>
            <button id="clear-log" class="icon-button small" title="Reset runtime log" aria-label="Reset runtime log" type="button">
              <i data-lucide="trash-2"></i>
            </button>
          </div>
        </div>
        <div id="log" class="log"></div>
      </aside>
    </section>
  </main>
`;

createIcons({ icons: { Copy, Cpu, Gauge, Loader2, Mic, RotateCcw, Square, Trash2, Upload } });

const statusEl = document.querySelector<HTMLDivElement>("#status")!;
const detailEl = document.querySelector<HTMLParagraphElement>("#detail")!;
const loadButton = document.querySelector<HTMLButtonElement>("#load-model")!;
const loadLabel = loadButton.querySelector("span")!;
const micButton = document.querySelector<HTMLButtonElement>("#microphone")!;
const micLabel = micButton.querySelector("span")!;
const resetButton = document.querySelector<HTMLButtonElement>("#reset")!;
const languageSelect = document.querySelector<HTMLSelectElement>("#language")!;
const fileInput = document.querySelector<HTMLInputElement>("#audio-file")!;
const benchmarkFileInput = document.querySelector<HTMLInputElement>("#benchmark-file")!;
const biasEnabledInput = document.querySelector<HTMLInputElement>("#bias-enabled")!;
const biasTermsInput = document.querySelector<HTMLTextAreaElement>("#bias-terms")!;
const biasBeamInput = document.querySelector<HTMLInputElement>("#bias-beam")!;
const transcriptEl = document.querySelector<HTMLOutputElement>("#transcript")!;
const logEl = document.querySelector<HTMLDivElement>("#log")!;
const stateEl = document.querySelector<HTMLSpanElement>("#state")!;
const copyLogButton = document.querySelector<HTMLButtonElement>("#copy-log")!;
const clearLogButton = document.querySelector<HTMLButtonElement>("#clear-log")!;
const wordCountEl = document.querySelector<HTMLSpanElement>("#word-count")!;
const chunkCountEl = document.querySelector<HTMLElement>("#chunk-count")!;
const queueDepthEl = document.querySelector<HTMLElement>("#queue-depth")!;
const tokenCountEl = document.querySelector<HTMLElement>("#token-count")!;
const topTokenEl = document.querySelector<HTMLElement>("#top-token")!;
const blankScoreEl = document.querySelector<HTMLElement>("#blank-score")!;

const asr = new AsrClient();
const audioQueue: Float32Array[] = [];
const logRecords: DiagnosticRecord[] = [];

let isLoading = false;
let isLoaded = false;
let isListening = false;
let isStopping = false;
let isProcessingFile = false;
let benchmarkRunning = false;
// Samples per chunk for the active run. Normally the model default; the
// benchmark overrides it per pass to sweep larger chunks.
let activeChunkSamples: number = NEMOTRON_CONFIG.chunkSamples;
let chunkIndex = 0;
const perf = {
  chunks: 0,
  elapsedMs: 0,
  chunkElapsedMs: [] as number[],
  featureMs: 0,
  encoderMs: 0,
  decodeMs: 0,
  jointMs: 0,
  decoderMs: 0,
  jointCalls: 0,
  decoderCalls: 0,
  emittedTokens: 0,
  blankFrames: 0,
  finalTokenCount: 0,
  maxQueueDepth: 0
};
let drainPromise: Promise<void> | undefined;
let transcript = "";
let mediaStream: MediaStream | undefined;
let audioContext: AudioContext | undefined;
let sourceNode: MediaStreamAudioSourceNode | undefined;
let processorNode: ScriptProcessorNode | undefined;
let resampler: StreamingLinearResampler | undefined;
let chunker = new ModelChunker();

function setStatus(kind: StatusKind, text: string): void {
  statusEl.dataset.kind = kind;
  statusEl.querySelector("strong")!.textContent = text;
}

function setDetail(text: string): void {
  detailEl.textContent = text;
}

function appendLog(text: string, level: DiagnosticLevel = "info", detail?: unknown, options?: { telemetry?: boolean }): void {
  const record: DiagnosticRecord = {
    id: logRecords.length + 1,
    at: new Date().toISOString(),
    level,
    message: text,
    detail: detail === undefined ? undefined : serializeDiagnosticValue(detail)
  };
  logRecords.push(record);
  // Skip forwarding records already carried by session meta or that are
  // high-frequency duplicates; they stay in the on-device Runtime Log.
  if (options?.telemetry !== false) telemetry.record(record);
  while (logRecords.length > 300) logRecords.shift();

  const row = document.createElement("div");
  row.className = "log-row";
  row.dataset.level = level;

  const summary = document.createElement("div");
  summary.className = "log-summary";

  const time = document.createElement("span");
  time.className = "log-time";
  time.textContent = new Date(record.at).toLocaleTimeString();

  const badge = document.createElement("span");
  badge.className = "log-level";
  badge.textContent = level;

  const message = document.createElement("span");
  message.className = "log-message";
  message.textContent = text;

  summary.append(time, badge, message);
  row.append(summary);

  if (record.detail !== undefined) {
    const pre = document.createElement("pre");
    pre.className = "log-detail";
    pre.textContent = JSON.stringify(record.detail, null, 2);
    row.append(pre);
  }

  logEl.prepend(row);
  while (logEl.childElementCount > 300) logEl.lastElementChild?.remove();
  copyLogButton.disabled = logRecords.length === 0;
  clearLogButton.disabled = logRecords.length === 0;
}

function appendErrorLog(text: string, error: unknown): void {
  appendLog(text, "error", serializeError(error));
}

function diagnosticLogText(): string {
  return logRecords
    .map((record) =>
      [
        `[${record.at}] ${record.level.toUpperCase()} ${record.message}`,
        record.detail === undefined ? "" : JSON.stringify(record.detail, null, 2)
      ]
        .filter(Boolean)
        .join("\n")
    )
    .join("\n\n");
}

function resetRuntimeLog(): void {
  logRecords.length = 0;
  logEl.replaceChildren();
  copyLogButton.disabled = true;
  clearLogButton.disabled = true;
}

function formatScore(value: number): string {
  if (!Number.isFinite(value)) return "-";
  return value.toFixed(2);
}

function selectedLangId(): number {
  return Number(languageSelect.value);
}

// Posts the bias config to the worker synchronously (so it stays ordered ahead
// of a following startStream), then logs the phrase counts once the worker
// replies. Awaiting is optional for callers — ordering holds regardless.
async function applyBiasing(): Promise<void> {
  const terms = biasTermsInput.value.split("\n").map((line) => line.trim()).filter(Boolean);
  const beamSize = Math.min(8, Math.max(1, Math.round(Number(biasBeamInput.value) || 4)));
  biasBeamInput.value = String(beamSize);
  const result = await asr.setBiasing({ enabled: biasEnabledInput.checked, terms, beamSize });
  appendLog("Contextual biasing updated", "info", {
    enabled: biasEnabledInput.checked,
    beamSize,
    phrases: result.phraseCount,
    tokenizedPhrases: result.tokenized
  });
}

function countWords(text: string): number {
  return text ? text.split(/\s+/).filter(Boolean).length : 0;
}

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[index];
}

function hashText(text: string): string {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

// Single roll-up line for a run so the server log can be pasted wholesale and
// the timing breakdown read at a glance. realtimeFactor > 1 means we process
// slower than audio arrives (i.e. we cannot keep up live).
function logPerfSummary(): void {
  if (perf.chunks === 0) return;
  const chunkAudioMs = (activeChunkSamples / NEMOTRON_CONFIG.sampleRate) * 1000;
  const avgElapsedMs = perf.elapsedMs / perf.chunks;
  appendLog("Performance summary", "info", {
    chunks: perf.chunks,
    chunkAudioMs: Math.round(chunkAudioMs),
    avgElapsedMs: Math.round(avgElapsedMs),
    p50ElapsedMs: percentile(perf.chunkElapsedMs, 50),
    p95ElapsedMs: percentile(perf.chunkElapsedMs, 95),
    maxElapsedMs: Math.max(...perf.chunkElapsedMs),
    avgFeatureMs: Math.round(perf.featureMs / perf.chunks),
    avgEncoderMs: Math.round(perf.encoderMs / perf.chunks),
    avgDecodeMs: Math.round(perf.decodeMs / perf.chunks),
    avgJointMs: Math.round(perf.jointMs / perf.chunks),
    avgDecoderMs: Math.round(perf.decoderMs / perf.chunks),
    avgJointCalls: Math.round(perf.jointCalls / perf.chunks),
    avgDecoderCalls: Math.round(perf.decoderCalls / perf.chunks),
    msPerJointCall: perf.jointCalls ? Number((perf.jointMs / perf.jointCalls).toFixed(2)) : 0,
    msPerDecoderCall: perf.decoderCalls ? Number((perf.decoderMs / perf.decoderCalls).toFixed(2)) : 0,
    realtimeFactor: Number((avgElapsedMs / chunkAudioMs).toFixed(2)),
    encoderShare: Number((perf.encoderMs / perf.elapsedMs).toFixed(2)),
    emittedTokens: perf.emittedTokens,
    blankFrames: perf.blankFrames,
    finalTokenCount: perf.finalTokenCount,
    transcriptWords: countWords(transcript),
    transcriptHash: hashText(transcript),
    maxQueueDepth: perf.maxQueueDepth,
    provider: benchmarkOptions().provider ?? "webgpu"
  });
}

function renderTranscript(text: string): void {
  transcript = text.trim();
  transcriptEl.textContent = transcript;
  transcriptEl.dataset.empty = transcript ? "false" : "true";
  const words = countWords(transcript);
  wordCountEl.textContent = `${words} word${words === 1 ? "" : "s"}`;
}

function renderProgress(progress: StreamProgress): void {
  renderTranscript(progress.text);
  chunkCountEl.textContent = String(progress.chunkIndex + 1);
  tokenCountEl.textContent = String(progress.tokenCount);
  topTokenEl.textContent = String(progress.topToken);
  blankScoreEl.textContent = formatScore(progress.blankScore);
}

function renderQueue(): void {
  queueDepthEl.textContent = String(audioQueue.length);
}

function setState(): void {
  if (benchmarkRunning) {
    stateEl.textContent = "Benchmarking";
  } else if (isLoading) {
    stateEl.textContent = "Loading";
  } else if (isStopping) {
    stateEl.textContent = "Stopping";
  } else if (isListening) {
    stateEl.textContent = "Listening";
  } else if (isProcessingFile || drainPromise) {
    stateEl.textContent = "Processing";
  } else if (isLoaded) {
    stateEl.textContent = "Ready";
  } else {
    stateEl.textContent = "Idle";
  }
}

function refreshControls(): void {
  const canRun = "gpu" in navigator && detectWasmSimd();
  const busy = isLoading || isStopping || isProcessingFile || benchmarkRunning || Boolean(drainPromise);
  loadButton.disabled = !canRun || isLoaded || busy || isListening;
  micButton.disabled = !isLoaded || isLoading || isProcessingFile || isStopping || benchmarkRunning;
  resetButton.disabled = busy || isListening;
  copyLogButton.disabled = logRecords.length === 0;
  clearLogButton.disabled = logRecords.length === 0;
  languageSelect.disabled = isLoading || isListening || isProcessingFile || benchmarkRunning || Boolean(drainPromise);
  fileInput.disabled = !isLoaded || isLoading || isListening || isStopping || benchmarkRunning || Boolean(drainPromise);
  // The benchmark loads its own sessions, so it doesn't require a pre-loaded
  // model; it only needs a capable browser and nothing else running.
  benchmarkFileInput.disabled = !canRun || isListening || busy;

  const biasLocked = isLoading || isListening || isStopping || isProcessingFile || benchmarkRunning || Boolean(drainPromise);
  biasEnabledInput.disabled = biasLocked;
  biasTermsInput.disabled = biasLocked;
  biasBeamInput.disabled = biasLocked;

  loadButton.dataset.mode = isLoading ? "loading" : "load";
  loadLabel.textContent = isLoading ? "Loading" : isLoaded ? "Model Loaded" : "Load ONNX Model";
  micButton.dataset.mode = isListening || isStopping ? "stop" : "start";
  micLabel.textContent = isListening || isStopping ? "Stop Mic" : "Start Mic";
  setState();
}

function resetRun(): void {
  asr.reset();
  chunker.reset();
  audioQueue.length = 0;
  chunkIndex = 0;
  perf.chunks = 0;
  perf.elapsedMs = 0;
  perf.chunkElapsedMs = [];
  perf.featureMs = 0;
  perf.encoderMs = 0;
  perf.decodeMs = 0;
  perf.jointMs = 0;
  perf.decoderMs = 0;
  perf.jointCalls = 0;
  perf.decoderCalls = 0;
  perf.emittedTokens = 0;
  perf.blankFrames = 0;
  perf.finalTokenCount = 0;
  perf.maxQueueDepth = 0;
  renderTranscript("");
  renderQueue();
  chunkCountEl.textContent = "0";
  tokenCountEl.textContent = "0";
  topTokenEl.textContent = "-";
  blankScoreEl.textContent = "-";
}

function beginStream(): void {
  asr.startStream();
  resetRun();
}

function runtimeContext(): Record<string, unknown> {
  return {
    sessionId: telemetry.sessionId,
    webgpu: "gpu" in navigator,
    secureContext: window.isSecureContext,
    state: { isLoading, isLoaded, isListening, isStopping, isProcessingFile, draining: Boolean(drainPromise) },
    langId: selectedLangId(),
    chunkIndex,
    queueDepth: audioQueue.length,
    transcriptWords: countWords(transcript),
    biasing: {
      enabled: biasEnabledInput.checked,
      beamSize: Number(biasBeamInput.value),
      termCount: biasTermsInput.value.split("\n").map((line) => line.trim()).filter(Boolean).length
    },
    audio: audioContext
      ? { sampleRate: audioContext.sampleRate, state: audioContext.state, baseLatency: audioContext.baseLatency }
      : null
  };
}

function handleRuntimeError(error: unknown, label: string): void {
  const message = errorMessage(error);
  setStatus("error", label);
  setDetail(message);
  // Pair the stack with a snapshot of app state so failures are reproducible
  // across machines from the /admin dashboard alone.
  appendLog(`${label}: ${message}`, "error", { error: serializeError(error), context: runtimeContext() });
  audioQueue.length = 0;
  renderQueue();
  closeAudioInput();
  isListening = false;
  isStopping = false;
  isProcessingFile = false;
  refreshControls();
}

async function drainQueue(): Promise<void> {
  if (drainPromise) return drainPromise;

  drainPromise = (async () => {
    refreshControls();
    while (audioQueue.length > 0) {
      const chunk = audioQueue.shift();
      if (!chunk) continue;
      renderQueue();
      setStatus("busy", benchmarkRunning ? "benchmarking" : "processing");
      setDetail("Running Nemotron ONNX inference");
      const started = performance.now();
      // During a benchmark this loop runs hundreds of chunks across 9 passes;
      // skip the per-chunk log spam (and telemetry) and keep only the summaries.
      if (!benchmarkRunning) {
        appendLog(
          `Processing chunk ${chunkIndex}`,
          "info",
          { langId: selectedLangId(), samples: chunk.length, queueDepth: audioQueue.length },
          { telemetry: false }
        );
      }
      const progress = await asr.acceptAudioChunk(chunk, selectedLangId(), chunkIndex, activeChunkSamples);
      chunkIndex += 1;
      renderProgress(progress);
      const elapsedMs = Math.round(performance.now() - started);
      perf.chunks += 1;
      perf.elapsedMs += elapsedMs;
      perf.chunkElapsedMs.push(elapsedMs);
      perf.featureMs += progress.featureMs;
      perf.encoderMs += progress.encoderMs;
      perf.decodeMs += progress.decodeMs;
      perf.jointMs += progress.jointMs;
      perf.decoderMs += progress.decoderMs;
      perf.jointCalls += progress.jointCalls;
      perf.decoderCalls += progress.decoderCalls;
      perf.emittedTokens += progress.emittedTokens;
      perf.blankFrames += progress.blankFrames;
      perf.finalTokenCount = progress.tokenCount;
      perf.maxQueueDepth = Math.max(perf.maxQueueDepth, audioQueue.length);
      if (!benchmarkRunning) {
        appendLog(`Chunk ${progress.chunkIndex} complete`, "info", {
          elapsedMs,
          featureMs: Math.round(progress.featureMs),
          encoderMs: Math.round(progress.encoderMs),
          decodeMs: Math.round(progress.decodeMs),
          jointMs: Math.round(progress.jointMs),
          decoderMs: Math.round(progress.decoderMs),
          jointCalls: progress.jointCalls,
          decoderCalls: progress.decoderCalls,
          audioPeak: Number(progress.audioPeak.toFixed(4)),
          audioRms: Number(progress.audioRms.toFixed(4)),
          tokenCount: progress.tokenCount,
          emittedTokens: progress.emittedTokens,
          blankFrames: progress.blankFrames,
          topToken: progress.topToken,
          topScore: progress.topScore,
          blankScore: progress.blankScore
        });
      }
    }

    renderQueue();
  })();

  try {
    await drainPromise;
  } catch (error) {
    handleRuntimeError(error, "inference failed");
    throw error;
  } finally {
    drainPromise = undefined;
    if (!isListening && !isStopping && !isProcessingFile && isLoaded) {
      setStatus("ok", "ready");
      setDetail(transcript ? "Ready" : "Model loaded");
    }
    refreshControls();
  }
}

function enqueueChunks(chunks: Float32Array[]): void {
  if (chunks.length === 0) return;
  audioQueue.push(...chunks);
  renderQueue();
  void drainQueue().catch(() => undefined);
}

function closeAudioInput(): void {
  processorNode?.disconnect();
  sourceNode?.disconnect();
  for (const track of mediaStream?.getTracks() ?? []) track.stop();
  void audioContext?.close();

  processorNode = undefined;
  sourceNode = undefined;
  mediaStream = undefined;
  audioContext = undefined;
  resampler = undefined;
}

// Benchmark knobs read from the URL, e.g. ?provider=wasm&profile=1. Absent in
// normal use, so production keeps the WebGPU defaults.
function benchmarkOptions(): LoadOptions {
  const params = new URLSearchParams(window.location.search);
  const provider = params.get("provider");
  return {
    provider: provider === "wasm" || provider === "webgpu" ? provider : undefined,
    profile: params.get("profile") === "1",
    // Hybrid (decoder/joint on WASM) is off by default — it lost to pure WebGPU
    // on the AMD APU while WASM was single-threaded. ?hybrid=1 opts back in.
    hybrid: params.get("hybrid") === "1" ? true : undefined,
    // ?encoder=fp16 loads the locally-converted fp16 encoder instead of int4.
    encoderVariant: params.get("encoder") === "fp16" ? "fp16" : undefined,
    // ?batchedJoint=0 restores the original one-joint-run-per-frame greedy loop.
    batchedJoint: params.get("batchedJoint") === "0" ? false : undefined,
    // ?graphCapture=1 captures the fixed-shape one-frame joint session. It
    // disables batchedJoint internally because capture needs static buffers.
    graphCapture: params.get("graphCapture") === "1" ? true : undefined
  };
}

async function loadModel(): Promise<void> {
  if (isLoaded || isLoading) return;

  const options = benchmarkOptions();
  isLoading = true;
  setStatus("busy", "loading");
  setDetail("Loading Nemotron ONNX weights");
  appendLog("Loading model", "info", {
    modelId: MODEL_ID,
    modelRevision: MODEL_REVISION,
    totalWeightBytes: modelBytes,
    provider: options.provider ?? "webgpu",
    profile: options.profile
  });
  refreshControls();

  try {
    // Re-probe locally before session creation; the boot probe already shipped.
    await probeWebGpu((level, message, detail) => appendLog(message, level, detail, { telemetry: false }));
    await asr.load((progress: LoadProgress) => {
      const detail = progress.detail ?? progress.stage;
      setDetail(detail);
      appendLog(`[${progress.stage}] ${detail}`, progress.level ?? "info", progress.data);
    }, options);

    isLoaded = true;
    setStatus("ok", "ready");
    setDetail("Model loaded");
    appendLog("Ready");
    // Tokenize any bias terms entered before the model (and tokenizer) loaded.
    await applyBiasing();
  } catch (error) {
    handleRuntimeError(error, "load failed");
  } finally {
    isLoading = false;
    refreshControls();
  }
}

async function startMic(): Promise<void> {
  if (!isLoaded || isListening || isStopping) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    handleRuntimeError(new Error("Microphone capture is not available in this browser"), "mic unavailable");
    return;
  }

  try {
    await applyBiasing();
    beginStream();
    setStatus("busy", "opening mic");
    setDetail("Opening microphone");
    refreshControls();

    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false
      }
    });
    const [track] = mediaStream.getAudioTracks();
    appendLog("Microphone stream acquired", "info", {
      trackSettings: track?.getSettings(),
      trackCapabilities: track?.getCapabilities?.()
    });
    audioContext = new AudioContext();
    await audioContext.resume();
    appendLog("AudioContext ready", "info", {
      sampleRate: audioContext.sampleRate,
      outputLatency: "outputLatency" in audioContext ? audioContext.outputLatency : null,
      baseLatency: audioContext.baseLatency,
      state: audioContext.state
    });
    resampler = new StreamingLinearResampler(audioContext.sampleRate, NEMOTRON_CONFIG.sampleRate);
    sourceNode = audioContext.createMediaStreamSource(mediaStream);
    processorNode = audioContext.createScriptProcessor(4096, 1, 1);

    processorNode.onaudioprocess = (event) => {
      for (let channel = 0; channel < event.outputBuffer.numberOfChannels; channel++) {
        event.outputBuffer.getChannelData(channel).fill(0);
      }
      if (!isListening || !resampler) return;
      const input = event.inputBuffer.getChannelData(0);
      const samples = resampler.push(input);
      enqueueChunks(chunker.push(samples));
    };

    sourceNode.connect(processorNode);
    processorNode.connect(audioContext.destination);

    isListening = true;
    setStatus("busy", "listening");
    setDetail("Streaming microphone into Nemotron ONNX");
    appendLog("Microphone started", "info", {
      scriptProcessorBufferSize: 4096,
      targetSampleRate: NEMOTRON_CONFIG.sampleRate,
      chunkSamples: NEMOTRON_CONFIG.chunkSamples
    });
  } catch (error) {
    handleRuntimeError(error, "mic failed");
  } finally {
    refreshControls();
  }
}

async function stopMic(): Promise<void> {
  if (!isListening && !isStopping) return;

  isListening = false;
  isStopping = true;
  setStatus("busy", "stopping");
  setDetail("Finalizing microphone chunks");
  refreshControls();
  closeAudioInput();

  const finalChunk = chunker.flushPadded();
  if (finalChunk) enqueueChunks([finalChunk]);
  try {
    await drainQueue();
  } catch {
    return;
  }

  isStopping = false;
  setStatus("ok", "stopped");
  setDetail("Stopped");
  appendLog("Microphone stopped", "info", {
    chunksProcessed: chunkIndex,
    transcriptWords: countWords(transcript)
  });
  logPerfSummary();
  refreshControls();
}

async function processAudioFile(file: File): Promise<void> {
  if (!isLoaded || isProcessingFile || isListening) return;

  try {
    await applyBiasing();
    beginStream();
    isProcessingFile = true;
    setStatus("busy", "decoding");
    setDetail(`Decoding ${file.name}`);
    appendLog(`Decoding ${file.name}`, "info", {
      name: file.name,
      type: file.type || "unknown",
      bytes: file.size,
      lastModified: new Date(file.lastModified).toISOString()
    });
    refreshControls();

    const samples = await decodeAudioFile(file);
    appendLog("Audio file decoded", "info", {
      samples: samples.length,
      durationSeconds: samples.length / NEMOTRON_CONFIG.sampleRate,
      targetSampleRate: NEMOTRON_CONFIG.sampleRate
    });
    const fileChunker = new ModelChunker();
    const chunks = fileChunker.push(samples);
    appendLog("Audio file chunked", "info", {
      chunkCount: chunks.length,
      chunkSamples: NEMOTRON_CONFIG.chunkSamples
    });
    enqueueChunks(chunks);
    const finalChunk = fileChunker.flushPadded();
    if (finalChunk) enqueueChunks([finalChunk]);
    await drainQueue();

    setStatus("ok", "ready");
    setDetail("File processed");
    appendLog("File processed", "info", {
      chunksProcessed: chunkIndex,
      transcriptWords: countWords(transcript)
    });
    logPerfSummary();
  } catch (error) {
    handleRuntimeError(error, "file failed");
  } finally {
    isProcessingFile = false;
    refreshControls();
  }
}

// Benchmark cases matching the manual URLs we compare on weak GPUs. Each case
// rebuilds ONNX sessions, so they are the outer loop; chunk sizes re-run the
// same sessions and are the inner loop.
const BENCHMARK_CASES: { key: string; label: string; options: LoadOptions }[] = [
  {
    key: "default",
    label: "Default WebGPU (batched joint)",
    options: { provider: "webgpu", hybrid: false, batchedJoint: true, graphCapture: false }
  },
  {
    key: "stepwise",
    label: "WebGPU stepwise joint (?batchedJoint=0)",
    options: { provider: "webgpu", hybrid: false, batchedJoint: false, graphCapture: false }
  },
  {
    key: "graphcapture",
    label: "WebGPU joint graph capture (?graphCapture=1)",
    options: { provider: "webgpu", hybrid: false, batchedJoint: false, graphCapture: true }
  },
  {
    key: "hybrid",
    label: "Hybrid WebGPU encoder / WASM decode (?hybrid=1)",
    options: { provider: "webgpu", hybrid: true, batchedJoint: true, graphCapture: false }
  }
];

// Chunk sizes (ms of audio) tried for every backend. The encoder was exported
// with a static 65-frame input (= 560 ms: 9 pre-encode cache + 56 mel frames),
// so anything else fails OrtRun with "Got invalid dimensions for audio_signal".
// Only re-add larger sizes here if the model is re-exported with a dynamic time
// axis (which is the real lever for amortizing per-run encoder overhead).
const BENCHMARK_CHUNK_MS = [560];

function chunkSamplesForMs(ms: number): number {
  return Math.round((ms / 1000) * NEMOTRON_CONFIG.sampleRate);
}

// URL overrides so the matrix can be trimmed without a rebuild, e.g.
// ?benchCases=default,graphcapture or ?benchChunks=560,280 (only meaningful
// once the encoder is re-exported).
function benchmarkCases(): typeof BENCHMARK_CASES {
  const params = new URLSearchParams(window.location.search);
  const param = params.get("benchCases") ?? params.get("benchBackends");
  if (!param) return BENCHMARK_CASES;
  const byKey: Record<string, (typeof BENCHMARK_CASES)[number]> = {
    default: BENCHMARK_CASES[0],
    webgpu: BENCHMARK_CASES[0],
    batched: BENCHMARK_CASES[0],
    stepwise: BENCHMARK_CASES[1],
    unbatched: BENCHMARK_CASES[1],
    graph: BENCHMARK_CASES[2],
    graphcapture: BENCHMARK_CASES[2],
    capture: BENCHMARK_CASES[2],
    hybrid: BENCHMARK_CASES[3]
  };
  const chosen = param
    .split(",")
    .map((key) => byKey[key.trim().toLowerCase()])
    .filter((entry): entry is (typeof BENCHMARK_CASES)[number] => Boolean(entry));
  return chosen.length ? chosen : BENCHMARK_CASES;
}

function benchmarkChunkMs(): number[] {
  const param = new URLSearchParams(window.location.search).get("benchChunks");
  if (!param) return BENCHMARK_CHUNK_MS;
  const parsed = param
    .split(",")
    .map((value) => Number(value.trim()))
    .filter((value) => Number.isFinite(value) && value > 0);
  return parsed.length ? parsed : BENCHMARK_CHUNK_MS;
}

// Run the decoded audio through the currently loaded sessions at one chunk size
// and return the timing summary for that pass.
async function runBenchmarkPass(samples: Float32Array, chunkSamples: number): Promise<Record<string, number | string>> {
  activeChunkSamples = chunkSamples;
  beginStream();
  const passChunker = new ModelChunker(chunkSamples);
  enqueueChunks(passChunker.push(samples));
  const finalChunk = passChunker.flushPadded();
  if (finalChunk) enqueueChunks([finalChunk]);
  await drainQueue();

  const chunkAudioMs = (chunkSamples / NEMOTRON_CONFIG.sampleRate) * 1000;
  const divisor = Math.max(1, perf.chunks);
  const avgElapsedMs = perf.elapsedMs / divisor;
  return {
    chunkMs: Math.round(chunkAudioMs),
    chunks: perf.chunks,
    totalElapsedMs: Math.round(perf.elapsedMs),
    avgElapsedMs: Math.round(avgElapsedMs),
    p50ElapsedMs: percentile(perf.chunkElapsedMs, 50),
    p95ElapsedMs: percentile(perf.chunkElapsedMs, 95),
    maxElapsedMs: Math.max(0, ...perf.chunkElapsedMs),
    avgFeatureMs: Math.round(perf.featureMs / divisor),
    avgEncoderMs: Math.round(perf.encoderMs / divisor),
    avgDecodeMs: Math.round(perf.decodeMs / divisor),
    avgJointMs: Math.round(perf.jointMs / divisor),
    avgDecoderMs: Math.round(perf.decoderMs / divisor),
    avgJointCalls: Math.round(perf.jointCalls / divisor),
    avgDecoderCalls: Math.round(perf.decoderCalls / divisor),
    totalJointCalls: perf.jointCalls,
    totalDecoderCalls: perf.decoderCalls,
    msPerJointCall: perf.jointCalls ? Number((perf.jointMs / perf.jointCalls).toFixed(2)) : 0,
    msPerDecoderCall: perf.decoderCalls ? Number((perf.decoderMs / perf.decoderCalls).toFixed(2)) : 0,
    emittedTokens: perf.emittedTokens,
    blankFrames: perf.blankFrames,
    finalTokenCount: perf.finalTokenCount,
    transcriptChars: transcript.length,
    transcriptWords: countWords(transcript),
    transcriptHash: hashText(transcript),
    realtimeFactor: Number((avgElapsedMs / chunkAudioMs).toFixed(2)),
    maxQueueDepth: perf.maxQueueDepth
  };
}

async function runBenchmark(file: File): Promise<void> {
  if (benchmarkRunning || isListening || isProcessingFile || isLoading || Boolean(drainPromise)) return;
  if (!("gpu" in navigator) || !detectWasmSimd()) {
    handleRuntimeError(new Error("WebGPU + WASM SIMD are required to benchmark"), "benchmark unavailable");
    return;
  }

  benchmarkRunning = true;
  // The previous manual load (if any) is about to be replaced backend-by-backend.
  isLoaded = false;
  const results: Record<string, unknown>[] = [];
  const startedAt = performance.now();
  try {
    await applyBiasing();
    const cases = benchmarkCases();
    const chunkMsList = benchmarkChunkMs();
    setStatus("busy", "benchmarking");
    setDetail(`Benchmarking ${file.name}`);
    refreshControls();
    appendLog(`Benchmark started for ${file.name}`, "info", {
      cases: cases.map((b) => b.label),
      chunkMs: chunkMsList,
      bytes: file.size
    });

    const samples = await decodeAudioFile(file);
    appendLog("Benchmark audio decoded", "info", {
      samples: samples.length,
      durationSeconds: Number((samples.length / NEMOTRON_CONFIG.sampleRate).toFixed(2))
    });

    for (const benchCase of cases) {
      setDetail(`Loading ${benchCase.label}`);
      setState();
      const loadStart = performance.now();
      const { encoderVariant, profile } = benchmarkOptions();
      const loadOptions: LoadOptions = { ...benchCase.options, encoderVariant, profile };
      try {
        await asr.load((progress: LoadProgress) => {
          setDetail(`${benchCase.label}: ${progress.detail ?? progress.stage}`);
        }, loadOptions);
      } catch (error) {
        appendErrorLog(`Benchmark: failed to load ${benchCase.label}`, error);
        results.push({ case: benchCase.key, backend: benchCase.label, status: "load failed", error: errorMessage(error) });
        continue;
      }
      const loadMs = Math.round(performance.now() - loadStart);
      appendLog(`Benchmark: loaded ${benchCase.label}`, "info", { case: benchCase.key, ...loadOptions, loadMs });

      for (const chunkMs of chunkMsList) {
        setDetail(`${benchCase.label} @ ${chunkMs} ms chunks`);
        try {
          const summary = await runBenchmarkPass(samples, chunkSamplesForMs(chunkMs));
          const row = { case: benchCase.key, backend: benchCase.label, status: "ok", loadMs, ...summary };
          results.push(row);
          appendLog(`Benchmark: ${benchCase.label} @ ${chunkMs} ms`, "info", row);
        } catch (error) {
          appendErrorLog(`Benchmark: run failed for ${benchCase.label} @ ${chunkMs} ms`, error);
          results.push({ case: benchCase.key, backend: benchCase.label, chunkMs, status: "run failed", error: errorMessage(error) });
        }
      }
    }

    // Single roll-up record: copy the diagnostic log and the whole matrix is in
    // one place, sorted best-RTF-first.
    const ranked = [...results].sort((a, b) => {
      const ra = typeof a.realtimeFactor === "number" ? a.realtimeFactor : Number.POSITIVE_INFINITY;
      const rb = typeof b.realtimeFactor === "number" ? b.realtimeFactor : Number.POSITIVE_INFINITY;
      return ra - rb;
    });
    appendLog("Benchmark complete", "info", {
      file: file.name,
      totalMs: Math.round(performance.now() - startedAt),
      best: ranked[0],
      results: ranked
    });
  } catch (error) {
    handleRuntimeError(error, "benchmark failed");
  } finally {
    activeChunkSamples = NEMOTRON_CONFIG.chunkSamples;
    benchmarkRunning = false;
    isLoaded = asr.isLoaded;
    if (isLoaded) {
      setStatus("ok", "ready");
      setDetail("Benchmark finished");
    }
    refreshControls();
  }
}

loadButton.addEventListener("click", () => {
  void loadModel();
});

micButton.addEventListener("click", () => {
  if (isListening || isStopping) {
    void stopMic();
    return;
  }
  void startMic();
});

resetButton.addEventListener("click", () => {
  resetRun();
  setStatus(isLoaded ? "ok" : "idle", isLoaded ? "ready" : "idle");
  setDetail(isLoaded ? "Model loaded" : "Ready");
  refreshControls();
});

copyLogButton.addEventListener("click", () => {
  const text = diagnosticLogText();
  if (!navigator.clipboard?.writeText) {
    appendLog("Clipboard API unavailable", "warn", collectBrowserDiagnostics());
    return;
  }

  navigator.clipboard
    .writeText(text)
    .then(() => appendLog("Diagnostic log copied"))
    .catch((error: unknown) => appendErrorLog("Failed to copy diagnostic log", error));
});

clearLogButton.addEventListener("click", resetRuntimeLog);

biasEnabledInput.addEventListener("change", applyBiasing);
biasBeamInput.addEventListener("change", applyBiasing);
biasTermsInput.addEventListener("change", applyBiasing);

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  fileInput.value = "";
  if (file) void processAudioFile(file);
});

benchmarkFileInput.addEventListener("change", () => {
  const file = benchmarkFileInput.files?.[0];
  benchmarkFileInput.value = "";
  if (file) void runBenchmark(file);
});

window.addEventListener("error", (event) => {
  appendLog("Window error", "error", {
    message: event.message,
    filename: event.filename,
    lineno: event.lineno,
    colno: event.colno,
    error: serializeError(event.error),
    context: runtimeContext()
  });
});

window.addEventListener("unhandledrejection", (event) => {
  appendLog("Unhandled promise rejection", "error", {
    reason: serializeError(event.reason),
    context: runtimeContext()
  });
});

telemetry.start({ modelId: MODEL_ID, modelRevision: MODEL_REVISION });
appendLog("Session started", "info", {
  sessionId: telemetry.sessionId,
  appMode: import.meta.env.MODE,
  modelId: MODEL_ID,
  modelRevision: MODEL_REVISION
});

if (!("gpu" in navigator)) {
  setStatus("error", "no webgpu");
  setDetail("WebGPU is not available in this browser");
} else if (!detectWasmSimd()) {
  setStatus("error", "no wasm simd");
  setDetail("WebAssembly SIMD is unavailable. Use a recent desktop Chrome, Edge, or Firefox.");
} else {
  setStatus("idle", "ready");
  setDetail("Load the ONNX model");
}
appendLog("Browser diagnostics", "info", collectBrowserDiagnostics(), { telemetry: false });
void probeWebGpu((level, message, detail) => appendLog(message, level, detail));
renderQueue();
refreshControls();
