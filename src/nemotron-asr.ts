import * as ort from "onnxruntime-web/webgpu";
import { StreamingFeatureBuilder } from "./audio";
import { detectWasmSimd } from "./diagnostics";
import { ContextGraph, ContextNode, logSoftmax, topTokens } from "./context-biasing";
import { modelAssetUrls, MODEL_FILES, MODEL_REVISION, NEMOTRON_CONFIG, ORT_RUNTIME_PATH } from "./model-config";
import { NemotronTokenizer } from "./tokenizer";

export interface LoadProgress {
  stage: string;
  detail?: string;
  level?: "info" | "warn" | "error";
  data?: unknown;
}

// Benchmark knobs, normally absent (production uses the WebGPU defaults).
export interface LoadOptions {
  provider?: "webgpu" | "wasm";
  profile?: boolean;
}

export interface StreamProgress {
  chunkIndex: number;
  text: string;
  tokenCount: number;
  emittedTokens: number;
  blankFrames: number;
  topToken: number;
  topScore: number;
  blankScore: number;
  featureMs: number;
  encoderMs: number;
  decodeMs: number;
  audioPeak: number;
  audioRms: number;
}

interface SessionBundle {
  encoder: ort.InferenceSession;
  decoder: ort.InferenceSession;
  joint: ort.InferenceSession;
}

interface EncoderCache {
  channel: ort.Tensor;
  time: ort.Tensor;
  channelLen: ort.Tensor;
}

interface DecoderState {
  hidden: ort.Tensor;
  cell: ort.Tensor;
  lastToken: number;
}

interface BeamHypothesis {
  tokens: number[];
  score: number;
  hidden: ort.Tensor;
  cell: ort.Tensor;
  lastToken: number;
  context: ContextNode;
  decoderOutput?: { output: ort.Tensor; hidden: ort.Tensor; cell: ort.Tensor };
}

export interface BiasingConfig {
  enabled: boolean;
  terms: string[];
  beamSize?: number;
  boost?: number;
}

const MODEL_CACHE_NAME = `nemotron-asr-${MODEL_REVISION}`;

// Stand-in root used by hypotheses when no bias graph is active; never extended.
const EMPTY_CONTEXT_ROOT: ContextNode = { children: new Map(), score: 0, isEnd: false };

function zeros(size: number): Float32Array {
  return new Float32Array(size);
}

function int64(value: number): BigInt64Array {
  return new BigInt64Array([BigInt(value)]);
}

async function createSession(
  modelFile: string,
  externalDataFile: string,
  executionProviders: ort.InferenceSession.ExecutionProviderConfig[],
  onProgress?: (progress: LoadProgress) => void
): Promise<ort.InferenceSession> {
  const modelData = await fetchModelFile(modelFile, onProgress);
  const externalData = await fetchModelFile(externalDataFile, onProgress);
  const started = performance.now();
  onProgress?.({
    stage: "session",
    detail: `Creating ONNX WebGPU session for ${modelFile}`,
    data: { modelBytes: modelData.byteLength, externalDataBytes: externalData.byteLength }
  });

  try {
    const session = await ort.InferenceSession.create(modelData, {
      executionProviders,
      graphOptimizationLevel: "all",
      externalData: [{ path: externalDataFile, data: externalData }]
    });
    onProgress?.({
      stage: "session",
      detail: `Created ONNX WebGPU session for ${modelFile}`,
      data: { elapsedMs: Math.round(performance.now() - started) }
    });
    return session;
  } catch (error) {
    onProgress?.({
      stage: "session",
      level: "error",
      detail: `Failed to create ONNX WebGPU session for ${modelFile}`,
      data: error
    });
    throw error;
  }
}

async function fetchModelFile(
  filename: string,
  onProgress?: (progress: LoadProgress) => void
): Promise<Uint8Array> {
  const cacheKey = modelAssetUrls(filename)[0];
  let cache: Cache | undefined;

  if ("caches" in window) {
    try {
      cache = await caches.open(MODEL_CACHE_NAME);
      const cached = await cache.match(cacheKey);
      if (cached) {
        const bytes = new Uint8Array(await cached.arrayBuffer());
        onProgress?.({
          stage: "cache",
          detail: `Using cached ${filename}`,
          data: { bytes: bytes.byteLength, responseType: cached.type }
        });
        return bytes;
      }
    } catch (error) {
      onProgress?.({
        stage: "cache",
        level: "warn",
        detail: `Browser cache unavailable for ${filename}; downloading`,
        data: error
      });
      console.warn(error);
    }
  }

  const started = performance.now();
  const urls = modelAssetUrls(filename);
  let response: Response | undefined;
  let lastError: unknown;

  for (const url of urls) {
    onProgress?.({ stage: "download", detail: `Downloading ${filename}`, data: { url } });
    try {
      response = await fetch(url);
    } catch (error) {
      lastError = error;
      onProgress?.({ stage: "download", level: "warn", detail: `Network fetch failed for ${filename}`, data: error });
      continue;
    }

    const responseData = {
      elapsedMs: Math.round(performance.now() - started),
      responseType: response.type,
      contentLength: response.headers.get("content-length"),
      contentType: response.headers.get("content-type"),
      url
    };
    onProgress?.({
      stage: "download",
      detail: `${filename} response ${response.status} ${response.statusText}`,
      data: responseData
    });

    const contentType = response.headers.get("content-type") ?? "";
    if (response.ok && !contentType.includes("text/html")) break;

    lastError = new Error(`Failed to fetch ${filename}: ${response.status} ${response.statusText}`);
    response = undefined;
  }

  if (!response) {
    throw lastError instanceof Error ? lastError : new Error(`Failed to fetch ${filename}`);
  }

  // Persist into Cache Storage by streaming the response body straight to the
  // cache and then reading the bytes back, instead of buffering the whole file
  // into a Uint8Array and caching a second copy. For the 690 MB encoder this is
  // the difference between loading once and re-downloading every reload:
  //   - peak memory stays low (the body streams to disk, never a 690 MB JS
  //     allocation), which is what made the download fail outright on phones; and
  //   - cache.put is atomic, so a dropped connection writes nothing rather than
  //     leaving a half-file that silently re-downloads next time.
  if (cache) {
    try {
      await cache.put(cacheKey, response);
      const cached = await cache.match(cacheKey);
      if (!cached) throw new Error("cache.match returned nothing after put");
      const bytes = new Uint8Array(await cached.arrayBuffer());
      onProgress?.({
        stage: "download",
        detail: `Downloaded ${filename}`,
        data: { bytes: bytes.byteLength, elapsedMs: Math.round(performance.now() - started) }
      });
      onProgress?.({ stage: "cache", detail: `Cached ${filename}`, data: { bytes: bytes.byteLength } });
      return bytes;
    } catch (error) {
      onProgress?.({
        stage: "cache",
        level: "warn",
        detail: `Could not cache ${filename}; re-fetching uncached for this load`,
        data: error
      });
      console.warn(error);
      // cache.put consumed the body, so the original response is spent; re-fetch.
      response = await fetch(response.url);
    }
  }

  const bytes = new Uint8Array(await response.arrayBuffer());
  onProgress?.({
    stage: "download",
    detail: `Downloaded ${filename}`,
    data: { bytes: bytes.byteLength, elapsedMs: Math.round(performance.now() - started) }
  });
  return bytes;
}

function createEncoderCache(): EncoderCache {
  const cfg = NEMOTRON_CONFIG;
  return {
    channel: new ort.Tensor(
      "float32",
      zeros(1 * cfg.encoderLayers * cfg.leftContext * cfg.encoderHiddenSize),
      [1, cfg.encoderLayers, cfg.leftContext, cfg.encoderHiddenSize]
    ),
    time: new ort.Tensor(
      "float32",
      zeros(1 * cfg.encoderLayers * cfg.encoderHiddenSize * cfg.convContext),
      [1, cfg.encoderLayers, cfg.encoderHiddenSize, cfg.convContext]
    ),
    channelLen: new ort.Tensor("int64", int64(0), [1])
  };
}

function createDecoderState(): DecoderState {
  const cfg = NEMOTRON_CONFIG;
  return {
    hidden: new ort.Tensor("float32", zeros(cfg.decoderLayers * cfg.decoderHiddenSize), [
      cfg.decoderLayers,
      1,
      cfg.decoderHiddenSize
    ]),
    cell: new ort.Tensor("float32", zeros(cfg.decoderLayers * cfg.decoderHiddenSize), [
      cfg.decoderLayers,
      1,
      cfg.decoderHiddenSize
    ]),
    lastToken: cfg.blankId
  };
}

function readScalarInt(tensor: ort.Tensor): number {
  if (tensor.type === "int64") return Number((tensor.data as BigInt64Array)[0]);
  if (tensor.type === "int32") return (tensor.data as Int32Array)[0];
  return Number((tensor.data as Float32Array)[0]);
}

function argmaxWithBlankPenalty(logits: Float32Array): number {
  const cfg = NEMOTRON_CONFIG;
  let best = 0;
  let bestScore = logits[0];
  for (let i = 1; i < logits.length; i++) {
    const score = i === cfg.blankId ? logits[i] - cfg.blankPenalty : logits[i];
    if (score > bestScore) {
      bestScore = score;
      best = i;
    }
  }
  return best;
}

// Sort hypotheses by score, drop duplicates that decode to the same token
// sequence (keeping the highest-scoring), and keep the top `beamSize`.
function pruneHypotheses(hypotheses: BeamHypothesis[], beamSize: number): BeamHypothesis[] {
  hypotheses.sort((a, b) => b.score - a.score);
  const seen = new Set<string>();
  const kept: BeamHypothesis[] = [];
  for (const hyp of hypotheses) {
    const key = hyp.tokens.join(",");
    if (seen.has(key)) continue;
    seen.add(key);
    kept.push(hyp);
    if (kept.length >= beamSize) break;
  }
  return kept;
}

export class NemotronBrowserASR {
  private sessions?: SessionBundle;
  private tokenizer?: NemotronTokenizer;
  private encoderCache = createEncoderCache();
  private decoderState = createDecoderState();
  private readonly featureBuilder = new StreamingFeatureBuilder();
  private tokens: number[] = [];
  private decoderOutputCache?: { output: ort.Tensor; hidden: ort.Tensor; cell: ort.Tensor };

  private biasingEnabled = false;
  private beamSize = 4;
  private boost = 2;
  private biasTerms: string[] = [];
  private contextGraph?: ContextGraph;
  private beam: BeamHypothesis[] = [];

  get isLoaded(): boolean {
    return Boolean(this.sessions && this.tokenizer);
  }

  /**
   * Configure contextual biasing. Changes take effect on the next startStream()
   * (the UI disables these controls while a stream is running). With biasing off
   * the decoder uses the original greedy path unchanged.
   */
  setBiasing(config: BiasingConfig): { phraseCount: number; tokenized: number } {
    this.biasingEnabled = config.enabled;
    if (config.beamSize) this.beamSize = Math.min(8, Math.max(1, Math.round(config.beamSize)));
    if (config.boost !== undefined) this.boost = config.boost;
    this.biasTerms = config.terms.map((term) => term.trim()).filter(Boolean);
    return this.rebuildContextGraph();
  }

  private rebuildContextGraph(): { phraseCount: number; tokenized: number } {
    if (!this.tokenizer || this.biasTerms.length === 0) {
      this.contextGraph = undefined;
      return { phraseCount: 0, tokenized: 0 };
    }
    const sequences = this.biasTerms.map((term) => this.tokenizer!.encode(term)).filter((seq) => seq.length > 0);
    this.contextGraph = sequences.length > 0 ? ContextGraph.build(sequences, this.boost) : undefined;
    return { phraseCount: this.biasTerms.length, tokenized: sequences.length };
  }

  private rootHypothesis(): BeamHypothesis {
    const state = createDecoderState();
    return {
      tokens: [],
      score: 0,
      hidden: state.hidden,
      cell: state.cell,
      lastToken: state.lastToken,
      context: this.contextGraph?.root ?? EMPTY_CONTEXT_ROOT
    };
  }

  async load(onProgress?: (progress: LoadProgress) => void, options: LoadOptions = {}): Promise<void> {
    const provider = options.provider ?? "webgpu";

    if (provider === "webgpu" && !("gpu" in navigator)) {
      throw new Error("WebGPU is not available in this browser");
    }

    // onnxruntime-web's WebGPU backend (JSEP) only ships a SIMD wasm artifact and
    // hard-throws if WASM SIMD is missing — but only after the model is downloaded.
    // Probe up front so we fail fast before fetching ~790 MB of weights.
    if (!detectWasmSimd()) {
      throw new Error(
        "WebAssembly SIMD is required but unavailable in this browser. Use a recent desktop Chrome, Edge, or Firefox (not an in-app/embedded browser)."
      );
    }

    ort.env.logLevel = "warning";
    // The WASM artifacts are served at ORT_RUNTIME_PATH by the (preview) server,
    // so point ORT there in dev and whenever the WASM provider is benchmarked.
    if (import.meta.env.DEV || provider === "wasm") {
      ort.env.wasm.wasmPaths = ORT_RUNTIME_PATH;
    }

    // Benchmark knob: WebGPU per-kernel timings are printed to the console so we
    // can see which encoder ops dominate instead of treating it as a black box.
    if (options.profile && provider === "webgpu") {
      ort.env.webgpu.profiling = { mode: "default" };
    }

    const executionProviders: ort.InferenceSession.ExecutionProviderConfig[] =
      provider === "wasm" ? ["wasm"] : ["webgpu"];

    onProgress?.({
      stage: "runtime",
      detail: `ONNX Runtime configured for ${provider}`,
      data: {
        provider,
        profile: Boolean(options.profile),
        crossOriginIsolated: globalThis.crossOriginIsolated,
        wasmThreads: ort.env.wasm.numThreads,
        wasmPaths: import.meta.env.DEV ? ORT_RUNTIME_PATH : "bundled"
      }
    });

    this.tokenizer = await NemotronTokenizer.fromHuggingFace(onProgress);

    onProgress?.({ stage: "decoder", detail: "Loading decoder.onnx and decoder.onnx.data" });
    const decoder = await createSession(MODEL_FILES.decoder.onnx, MODEL_FILES.decoder.data, executionProviders, onProgress);

    onProgress?.({ stage: "joint", detail: "Loading joint.onnx and joint.onnx.data" });
    const joint = await createSession(MODEL_FILES.joint.onnx, MODEL_FILES.joint.data, executionProviders, onProgress);

    onProgress?.({ stage: "encoder", detail: "Loading encoder.onnx and encoder.onnx.data" });
    const encoder = await createSession(MODEL_FILES.encoder.onnx, MODEL_FILES.encoder.data, executionProviders, onProgress);

    this.sessions = { encoder, decoder, joint };
    onProgress?.({ stage: "ready", detail: "All ONNX sessions loaded" });
  }

  reset(): void {
    this.encoderCache = createEncoderCache();
    this.decoderState = createDecoderState();
    this.featureBuilder.reset();
    this.tokens = [];
    this.decoderOutputCache = undefined;
    this.beam = [this.rootHypothesis()];
  }

  startStream(): void {
    if (!this.isLoaded) throw new Error("Model is not loaded");
    this.reset();
  }

  async acceptAudioChunk(chunk: Float32Array, langId: number, chunkIndex: number): Promise<StreamProgress> {
    if (!this.sessions) throw new Error("Model is not loaded");
    if (!this.tokenizer) throw new Error("Tokenizer is not loaded");
    const normalizedChunk = new Float32Array(NEMOTRON_CONFIG.chunkSamples);
    normalizedChunk.set(chunk.subarray(0, NEMOTRON_CONFIG.chunkSamples));
    const stats = await this.processChunk(normalizedChunk, langId);
    return {
      chunkIndex,
      text: this.tokenizer.decode(this.tokens),
      tokenCount: this.tokens.length,
      ...stats
    };
  }

  private async processChunk(
    chunk: Float32Array,
    langId: number
  ): Promise<{
    emittedTokens: number;
    blankFrames: number;
    topToken: number;
    topScore: number;
    blankScore: number;
    featureMs: number;
    encoderMs: number;
    decodeMs: number;
    audioPeak: number;
    audioRms: number;
  }> {
    if (!this.sessions) throw new Error("Model is not loaded");
    const cfg = NEMOTRON_CONFIG;

    // Energy of the raw audio reaching the model. If these are ~0 while the user
    // is speaking, the problem is upstream (wrong/silent mic, capture/resample),
    // not the decoder — every frame will correctly predict blank on silence.
    let audioPeak = 0;
    let sumSquares = 0;
    for (let i = 0; i < chunk.length; i++) {
      const sample = chunk[i];
      const abs = sample < 0 ? -sample : sample;
      if (abs > audioPeak) audioPeak = abs;
      sumSquares += sample * sample;
    }
    const audioRms = Math.sqrt(sumSquares / chunk.length);
    const featureStart = performance.now();
    const features = this.featureBuilder.build(chunk);
    const featureMs = performance.now() - featureStart;

    const encoderFeeds: Record<string, ort.Tensor> = {
      [cfg.inputs.encoderAudio]: new ort.Tensor("float32", features.data, features.dims),
      [cfg.inputs.encoderLength]: new ort.Tensor("int64", int64(features.frames), [1]),
      [cfg.inputs.cacheLastChannel]: this.encoderCache.channel,
      [cfg.inputs.cacheLastTime]: this.encoderCache.time,
      [cfg.inputs.cacheLastChannelLen]: this.encoderCache.channelLen,
      [cfg.inputs.langId]: new ort.Tensor("int64", int64(langId), [1])
    };

    const encoderStart = performance.now();
    const encoderOutputs = await this.sessions.encoder.run(encoderFeeds);
    const encoderMs = performance.now() - encoderStart;
    const encoded = encoderOutputs[cfg.outputs.encoder];
    const encodedLen = readScalarInt(encoderOutputs[cfg.outputs.encoderLength]);
    this.encoderCache = {
      channel: encoderOutputs[cfg.outputs.cacheLastChannel],
      time: encoderOutputs[cfg.outputs.cacheLastTime],
      channelLen: encoderOutputs[cfg.outputs.cacheLastChannelLen]
    };

    const decodeStart = performance.now();
    const decodeStats = await (this.biasingEnabled
      ? this.decodeEncodedFramesBeam(encoded, encodedLen)
      : this.decodeEncodedFrames(encoded, encodedLen));
    const decodeMs = performance.now() - decodeStart;

    return { ...decodeStats, featureMs, encoderMs, decodeMs, audioPeak, audioRms };
  }

  private async decodeEncodedFrames(
    encoded: ort.Tensor,
    encodedLen: number
  ): Promise<{ emittedTokens: number; blankFrames: number; topToken: number; topScore: number; blankScore: number }> {
    const cfg = NEMOTRON_CONFIG;
    const shape = encoded.dims;
    if (shape.length !== 3) throw new Error(`Expected encoder output rank 3, got ${shape.join("x")}`);
    const timeSteps = Math.min(shape[1], encodedLen);
    const hidden = shape[2];
    const encodedData = encoded.data as Float32Array;
    let timeStep = 0;
    let symbolStep = 0;
    let emittedTokens = 0;
    let blankFrames = 0;
    let topToken: number = cfg.blankId;
    let topScore = Number.NEGATIVE_INFINITY;
    let blankScore = Number.NEGATIVE_INFINITY;

    while (timeStep < timeSteps) {
      const start = timeStep * hidden;
      const frame = new ort.Tensor("float32", encodedData.slice(start, start + hidden), [1, 1, hidden]);
      const decoderOutput = await this.currentDecoderOutput();
      const logits = await this.runJoint(frame, this.asDecoderFrame(decoderOutput));
      const logitsData = logits.data as Float32Array;
      const best = argmaxWithBlankPenalty(logitsData);
      topToken = best;
      topScore = logitsData[best] ?? Number.NEGATIVE_INFINITY;
      blankScore = logitsData[cfg.blankId] ?? Number.NEGATIVE_INFINITY;

      if (best === cfg.blankId) {
        blankFrames++;
        timeStep++;
        symbolStep = 0;
        continue;
      }

      this.decoderState.lastToken = best;
      this.decoderState.hidden = decoderOutput.hidden;
      this.decoderState.cell = decoderOutput.cell;
      this.decoderOutputCache = undefined;
      this.tokens.push(best);
      emittedTokens++;

      symbolStep++;
      if (symbolStep >= cfg.maxSymbolsPerStep) {
        timeStep++;
        symbolStep = 0;
      }
    }

    return { emittedTokens, blankFrames, topToken, topScore, blankScore };
  }

  // Time-synchronous RNN-T beam search with context-graph biasing. Hypotheses
  // carry their own decoder state and trie node and persist across chunks. With
  // beamSize 1 this is essentially greedy decoding plus biasing.
  private async decodeEncodedFramesBeam(
    encoded: ort.Tensor,
    encodedLen: number
  ): Promise<{ emittedTokens: number; blankFrames: number; topToken: number; topScore: number; blankScore: number }> {
    const cfg = NEMOTRON_CONFIG;
    const shape = encoded.dims;
    if (shape.length !== 3) throw new Error(`Expected encoder output rank 3, got ${shape.join("x")}`);
    const timeSteps = Math.min(shape[1], encodedLen);
    const hidden = shape[2];
    const encodedData = encoded.data as Float32Array;
    const graph = this.contextGraph;

    if (this.beam.length === 0) this.beam = [this.rootHypothesis()];
    const tokensBefore = this.beam[0].tokens.length;
    let blankFrames = 0;
    let topToken: number = cfg.blankId;
    let topScore = Number.NEGATIVE_INFINITY;
    let blankScore = Number.NEGATIVE_INFINITY;

    for (let timeStep = 0; timeStep < timeSteps; timeStep++) {
      const start = timeStep * hidden;
      const encoderFrame = new ort.Tensor("float32", encodedData.slice(start, start + hidden), [1, 1, hidden]);
      const beamTokensAtFrameStart = this.beam[0].tokens.length;

      let active = this.beam;
      const finished: BeamHypothesis[] = [];
      let bestFinishedScore = Number.NEGATIVE_INFINITY;

      for (let symbol = 0; symbol < cfg.maxSymbolsPerStep && active.length > 0; symbol++) {
        const expanded: BeamHypothesis[] = [];

        for (const hyp of active) {
          const decoderOutput = await this.decoderOutputFor(hyp);
          const logits = (await this.runJoint(encoderFrame, this.asDecoderFrame(decoderOutput))).data as Float32Array;
          const logProbs = logSoftmax(logits);

          if (hyp === active[0] && symbol === 0) {
            const best = argmaxWithBlankPenalty(logits);
            topToken = best;
            topScore = logits[best] ?? Number.NEGATIVE_INFINITY;
            blankScore = logits[cfg.blankId] ?? Number.NEGATIVE_INFINITY;
          }

          // Blank: the hypothesis stays put and advances to the next frame.
          const blankScoreNext = hyp.score + logProbs[cfg.blankId];
          finished.push({ ...hyp, score: blankScoreNext });
          if (blankScoreNext > bestFinishedScore) bestFinishedScore = blankScoreNext;

          // Non-blank expansions, ranked with the context match bonus.
          const matchScores = graph ? graph.matchScores(hyp.context) : undefined;
          for (const candidate of topTokens(logProbs, cfg.blankId, this.beamSize, matchScores)) {
            const advance = graph
              ? graph.step(hyp.context, candidate.token)
              : { node: hyp.context, bias: 0 };
            expanded.push({
              tokens: [...hyp.tokens, candidate.token],
              score: hyp.score + candidate.logProb + advance.bias,
              hidden: decoderOutput.hidden,
              cell: decoderOutput.cell,
              lastToken: candidate.token,
              context: advance.node
            });
          }
        }

        if (expanded.length === 0) break;
        active = pruneHypotheses(expanded, this.beamSize);
        // Emitting more symbols only appends further (mostly negative) log-probs,
        // so once the best surviving continuation can no longer beat the best
        // blank-advanced hypothesis, stop expanding this frame. This keeps cost
        // near greedy (≈1-2 symbol steps) instead of always running the cap.
        if (active[0].score <= bestFinishedScore) break;
      }

      // Hypotheses still emitting at the symbol cap also carry to the next frame.
      for (const hyp of active) finished.push(hyp);
      this.beam = pruneHypotheses(finished, this.beamSize);
      if (this.beam[0].tokens.length === beamTokensAtFrameStart) blankFrames++;
    }

    const top = this.beam[0];
    this.tokens = top.tokens;
    this.decoderState = { hidden: top.hidden, cell: top.cell, lastToken: top.lastToken };
    this.decoderOutputCache = undefined;

    return {
      emittedTokens: Math.max(0, this.tokens.length - tokensBefore),
      blankFrames,
      topToken,
      topScore,
      blankScore
    };
  }

  private async decoderOutputFor(
    hyp: BeamHypothesis
  ): Promise<{ output: ort.Tensor; hidden: ort.Tensor; cell: ort.Tensor }> {
    if (!hyp.decoderOutput) {
      hyp.decoderOutput = await this.runDecoderState(hyp.lastToken, hyp.hidden, hyp.cell);
    }
    return hyp.decoderOutput;
  }

  // The prediction network output depends only on (lastToken, hidden, cell),
  // none of which change while the joint network keeps emitting blanks. Cache
  // it so consecutive blank frames (and chunk boundaries that land on a blank)
  // reuse one decoder pass instead of re-running the LSTM every frame.
  private async currentDecoderOutput(): Promise<{ output: ort.Tensor; hidden: ort.Tensor; cell: ort.Tensor }> {
    if (!this.decoderOutputCache) {
      this.decoderOutputCache = await this.runDecoder();
    }
    return this.decoderOutputCache;
  }

  private runDecoder(): Promise<{ output: ort.Tensor; hidden: ort.Tensor; cell: ort.Tensor }> {
    return this.runDecoderState(this.decoderState.lastToken, this.decoderState.hidden, this.decoderState.cell);
  }

  private async runDecoderState(
    lastToken: number,
    hidden: ort.Tensor,
    cell: ort.Tensor
  ): Promise<{ output: ort.Tensor; hidden: ort.Tensor; cell: ort.Tensor }> {
    if (!this.sessions) throw new Error("Model is not loaded");
    const cfg = NEMOTRON_CONFIG;
    const outputs = await this.sessions.decoder.run({
      [cfg.inputs.decoderTargets]: new ort.Tensor("int64", int64(lastToken), [1, 1]),
      [cfg.inputs.decoderHidden]: hidden,
      [cfg.inputs.decoderCell]: cell
    });
    return {
      output: outputs[cfg.outputs.decoder],
      hidden: outputs[cfg.outputs.decoderHidden],
      cell: outputs[cfg.outputs.decoderCell]
    };
  }

  private asDecoderFrame(decoderOutput: { output: ort.Tensor }): ort.Tensor {
    const data = decoderOutput.output.data as Float32Array;
    return new ort.Tensor("float32", data, [1, 1, data.length]);
  }

  private async runJoint(encoderFrame: ort.Tensor, decoderFrame: ort.Tensor): Promise<ort.Tensor> {
    if (!this.sessions) throw new Error("Model is not loaded");
    const cfg = NEMOTRON_CONFIG;
    const outputs = await this.sessions.joint.run({
      [cfg.inputs.joinerEncoder]: encoderFrame,
      [cfg.inputs.joinerDecoder]: decoderFrame
    });
    return outputs[cfg.outputs.joint];
  }
}
