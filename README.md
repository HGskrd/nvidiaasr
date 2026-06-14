# Nemotron ONNX WebGPU ASR

Browser-only runner for `onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4`.

The app loads the Nemotron encoder, decoder, joiner, and tokenizer ONNX assets in the browser and runs inference through `onnxruntime-web/webgpu`. Microphone capture and audio-file decoding use browser APIs, then the local log-mel feature pipeline feeds the ONNX model.

It does not use the Web Speech API, MLX, Python, a backend inference path, or a WASM execution fallback.

## Start Locally

```bash
cd /Users/ishanbaichoo/Documents/nvidiaasr
npm install
npm run dev
```

Open `http://localhost:5173/` in a WebGPU-enabled browser.

Build the static app:

```bash
npm run build
```

## Model Cache

The Vite dev and preview servers expose a same-origin cache at `/model-cache/<revision>/...`.
On the first request for each pinned Hugging Face asset, the server downloads it into `.cache/huggingface/...`.
Later loads, including loads from another browser, are served from local disk instead of Hugging Face.

ONNX Runtime WebGPU runtime assets are also served from `/ort-runtime/` so the WASM loader does not fall back to the app HTML.

## Browser Requirements

- WebGPU-enabled Chromium or Edge.
- Microphone permission for live capture.
- Network access to the pinned Hugging Face model revision on the server's first cache miss.
- The browser also caches downloaded model files with Cache Storage.

## Diagnostics

The Runtime Log records browser capability checks, WebGPU adapter/device probes, model fetch/cache details, ONNX session creation timings, microphone settings, chunk inference stats, and full error stacks. Errors also capture an app-state snapshot (load/stream flags, language, biasing config, audio context) so they are reproducible from the log alone. Use the copy button in the Runtime Log header to grab a single browser's log.

### /admin dashboard (cross-machine)

Every browser also forwards its log records to the host running the dev/preview server, which aggregates them per session and serves a dashboard at `/admin`. This lets you watch failures from several test machines in one place instead of copy-pasting between them.

- Open `http://<host>:5173/admin` (Basic auth). Default user `admin`, password `nemotron-admin`.
- Set a strong password — and a username if you like — via env vars when starting the server:

  ```bash
  ADMIN_USER=ops ADMIN_PASSWORD='choose-something' npm run dev   # or: npm run build && ADMIN_PASSWORD=... npx vite preview
  ```

- Tag each machine with `?machine=<name>` once (e.g. `http://<host>:5173/?machine=dell-xps`); the label is remembered in `localStorage` and shown on the dashboard.
- The dashboard filters by level, full-text searches messages/details, auto-refreshes, and can download everything as JSON or clear the store.
- `GET /admin/api/data` returns a compact per-session **digest** (machine + real CPU architecture via UA Client Hints, WebGPU summary, load timings, a chunk-latency rollup with `realtime` flag, and only warn/error records). Add `?full=1` for the raw record stream (what the dashboard renders). Note: `navigator.platform`/UA report `MacIntel` on all Macs including Apple Silicon — the digest's `machine.arch` is the real value.
- Log ingestion (`POST /admin/ingest`) is unauthenticated so client machines can post without a prompt — keep the server on a trusted network. Only the dashboard and its APIs require the password. Logs are kept in memory and reset when the server restarts.

A purely static deploy (no dev/preview server) has no collector; clients detect this and stop posting after a few attempts, with no effect on the app.
