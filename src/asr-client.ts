// Main-thread proxy for the ASR pipeline running in asr-worker.ts. Mirrors the
// subset of NemotronBrowserASR that main.ts uses, turning each call into a
// worker message. Requests are answered by id; load streams progress events
// before its final result. The worker processes messages serially, so the
// fire-and-forget methods (startStream/reset) stay correctly ordered with the
// awaited ones as long as each posts synchronously.
import type { BiasingConfig, LoadOptions, LoadProgress, StreamProgress } from "./nemotron-asr";

type WorkerMessage =
  | { type: "progress"; id: number; progress: LoadProgress }
  | { type: "result"; id: number; ok: true; result?: unknown }
  | { type: "result"; id: number; ok: false; error: { name?: string; message?: string; stack?: string } };

interface Pending {
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
  onProgress?: (progress: LoadProgress) => void;
}

function reviveError(serialized: { name?: string; message?: string; stack?: string }): Error {
  const error = new Error(serialized.message ?? "ASR worker error");
  if (serialized.name) error.name = serialized.name;
  if (serialized.stack) error.stack = serialized.stack;
  return error;
}

export class AsrClient {
  private readonly worker = new Worker(new URL("./asr-worker.ts", import.meta.url), { type: "module" });
  private readonly pending = new Map<number, Pending>();
  private nextId = 1;
  private loaded = false;

  constructor() {
    this.worker.onmessage = (event: MessageEvent<WorkerMessage>) => this.handleMessage(event.data);
    this.worker.onerror = (event) => {
      const error = new Error(`ASR worker crashed: ${event.message}`);
      for (const [, pending] of this.pending) pending.reject(error);
      this.pending.clear();
      this.loaded = false;
    };
  }

  get isLoaded(): boolean {
    return this.loaded;
  }

  private handleMessage(data: WorkerMessage): void {
    if (data.type === "progress") {
      this.pending.get(data.id)?.onProgress?.(data.progress);
      return;
    }
    const pending = this.pending.get(data.id);
    if (!pending) return;
    this.pending.delete(data.id);
    if (data.ok) pending.resolve(data.result);
    else pending.reject(reviveError(data.error));
  }

  private request<T>(
    message: Record<string, unknown>,
    transfer: Transferable[] = [],
    onProgress?: (progress: LoadProgress) => void
  ): Promise<T> {
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (value: unknown) => void, reject, onProgress });
      this.worker.postMessage({ ...message, id }, transfer);
    });
  }

  async load(onProgress?: (progress: LoadProgress) => void, options: LoadOptions = {}): Promise<void> {
    this.loaded = false;
    await this.request<void>({ type: "load", options }, [], onProgress);
    this.loaded = true;
  }

  startStream(): void {
    void this.request({ type: "startStream" });
  }

  reset(): void {
    void this.request({ type: "reset" });
  }

  async dispose(): Promise<void> {
    await this.request<void>({ type: "dispose" });
    this.loaded = false;
  }

  setBiasing(config: BiasingConfig): Promise<{ phraseCount: number; tokenized: number }> {
    return this.request<{ phraseCount: number; tokenized: number }>({ type: "setBiasing", config });
  }

  acceptAudioChunk(chunk: Float32Array, langId: number, chunkIndex: number, chunkSamples: number): Promise<StreamProgress> {
    // Transfer the audio buffer instead of copying it; main.ts has already read
    // chunk.length for logging and never reuses the buffer after enqueueing.
    return this.request<StreamProgress>(
      { type: "acceptAudioChunk", chunk, langId, chunkIndex, chunkSamples },
      [chunk.buffer]
    );
  }
}
