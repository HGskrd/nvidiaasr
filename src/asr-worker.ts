/// <reference lib="webworker" />
// Runs the entire ASR pipeline (feature extraction + all three ONNX sessions)
// off the main thread. Session creation takes ~12-16 s for the encoder and the
// WASM execution provider runs synchronously; doing either on the main thread
// blocks it long enough to trigger the browser's "Page Unresponsive" dialog.
// Here it only blocks the worker, so the UI stays live.
import { serializeError } from "./diagnostics";
import { BiasingConfig, LoadOptions, LoadProgress, NemotronBrowserASR } from "./nemotron-asr";

type Request =
  | { id: number; type: "load"; options: LoadOptions }
  | { id: number; type: "startStream" }
  | { id: number; type: "reset" }
  | { id: number; type: "dispose" }
  | { id: number; type: "setBiasing"; config: BiasingConfig }
  | { id: number; type: "acceptAudioChunk"; chunk: Float32Array; langId: number; chunkIndex: number; chunkSamples: number };

const ctx = self as unknown as DedicatedWorkerGlobalScope;
const asr = new NemotronBrowserASR();

// Messages are processed strictly one at a time. The handlers are async (model
// load, inference), so without serialization a second message could interleave
// at an await point and corrupt the streaming caches/decoder state.
const queue: Request[] = [];
let processing = false;

ctx.onmessage = (event: MessageEvent<Request>) => {
  queue.push(event.data);
  if (!processing) void drain();
};

async function drain(): Promise<void> {
  processing = true;
  while (queue.length > 0) {
    const msg = queue.shift()!;
    await handle(msg);
  }
  processing = false;
}

async function handle(msg: Request): Promise<void> {
  try {
    let result: unknown;
    switch (msg.type) {
      case "load":
        await asr.load((progress: LoadProgress) => ctx.postMessage({ type: "progress", id: msg.id, progress }), msg.options);
        break;
      case "startStream":
        asr.startStream();
        break;
      case "reset":
        asr.reset();
        break;
      case "dispose":
        await asr.dispose();
        break;
      case "setBiasing":
        result = asr.setBiasing(msg.config);
        break;
      case "acceptAudioChunk":
        result = await asr.acceptAudioChunk(msg.chunk, msg.langId, msg.chunkIndex, msg.chunkSamples);
        break;
    }
    ctx.postMessage({ type: "result", id: msg.id, ok: true, result });
  } catch (error) {
    ctx.postMessage({ type: "result", id: msg.id, ok: false, error: serializeError(error) });
  }
}
