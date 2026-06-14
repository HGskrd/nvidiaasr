import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, rename, stat, unlink } from "node:fs/promises";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";
import { defineConfig, type Connect, type Plugin, type ViteDevServer } from "vite";
import { adminPlugin } from "./server/admin-plugin";
import {
  hfUrl,
  MODEL_ASSET_PATH,
  MODEL_FILES,
  MODEL_ID,
  MODEL_REVISION,
  ORT_RUNTIME_PATH
} from "./src/model-config";

const projectRoot = path.dirname(fileURLToPath(import.meta.url));
const modelCacheDir = path.join(
  projectRoot,
  ".cache",
  "huggingface",
  MODEL_ID.replaceAll("/", "--"),
  MODEL_REVISION
);
const ortDistDir = path.join(projectRoot, "node_modules", "onnxruntime-web", "dist");

const modelFileNames = new Set(
  Object.values(MODEL_FILES)
    .flatMap((group) => Object.values(group))
    .filter((value): value is string => typeof value === "string")
);
const ortRuntimeFiles = new Set([
  "ort-wasm-simd-threaded.jsep.mjs",
  "ort-wasm-simd-threaded.jsep.wasm",
  "ort-wasm-simd-threaded.asyncify.mjs",
  "ort-wasm-simd-threaded.asyncify.wasm",
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm"
]);

const downloads = new Map<string, Promise<void>>();

function contentType(filename: string): string {
  if (filename.endsWith(".mjs") || filename.endsWith(".js")) return "text/javascript; charset=utf-8";
  if (filename.endsWith(".wasm")) return "application/wasm";
  if (filename.endsWith(".txt")) return "text/plain; charset=utf-8";
  return "application/octet-stream";
}

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await stat(filePath);
    return true;
  } catch {
    return false;
  }
}

async function downloadModelFile(filename: string, targetPath: string, logger: ViteDevServer["config"]["logger"]) {
  await mkdir(path.dirname(targetPath), { recursive: true });
  const tmpPath = `${targetPath}.${process.pid}.${Date.now()}.tmp`;
  const response = await fetch(hfUrl(filename));
  if (!response.ok || !response.body) {
    throw new Error(`Failed to download ${filename}: ${response.status} ${response.statusText}`);
  }

  logger.info(`[model-cache] downloading ${filename}`);
  try {
    await pipeline(Readable.fromWeb(response.body), createWriteStream(tmpPath));
    await rename(tmpPath, targetPath);
    logger.info(`[model-cache] cached ${filename}`);
  } catch (error) {
    await unlink(tmpPath).catch(() => undefined);
    throw error;
  }
}

async function ensureModelFileCached(filename: string, logger: ViteDevServer["config"]["logger"]): Promise<"hit" | "miss"> {
  const targetPath = path.join(modelCacheDir, filename);
  if (await fileExists(targetPath)) return "hit";

  let pending = downloads.get(filename);
  if (!pending) {
    pending = downloadModelFile(filename, targetPath, logger).finally(() => downloads.delete(filename));
    downloads.set(filename, pending);
  }
  await pending;
  return "miss";
}

function safeFilenameFromPathname(pathname: string, prefix: string): string | undefined {
  if (!pathname.startsWith(prefix)) return undefined;
  const filename = decodeURIComponent(pathname.slice(prefix.length));
  if (!filename || filename.includes("/") || filename.includes("\\") || filename === "." || filename === "..") {
    return undefined;
  }
  return filename;
}

function serveFile(res: Connect.ServerResponse, filePath: string, filename: string, cacheState?: string): void {
  stat(filePath)
    .then((stats) => {
      res.statusCode = 200;
      res.setHeader("Content-Type", contentType(filename));
      res.setHeader("Content-Length", stats.size);
      res.setHeader("Cache-Control", "public, max-age=31536000, immutable");
      if (cacheState) res.setHeader("X-Model-Cache", cacheState);
      createReadStream(filePath)
        .on("error", () => {
          if (!res.headersSent) res.statusCode = 500;
          res.end();
        })
        .pipe(res);
    })
    .catch(() => {
      res.statusCode = 404;
      res.end("Not found");
    });
}

function cacheMiddleware(logger: ViteDevServer["config"]["logger"]): Connect.HandleFunction {
  return async (req, res, next) => {
    const requestUrl = new URL(req.url ?? "/", "http://localhost");

    const modelFilename = safeFilenameFromPathname(requestUrl.pathname, `${MODEL_ASSET_PATH}/`);
    if (modelFilename) {
      if (!modelFileNames.has(modelFilename)) {
        res.statusCode = 404;
        res.end("Unknown model asset");
        return;
      }

      try {
        const cacheState = await ensureModelFileCached(modelFilename, logger);
        serveFile(res, path.join(modelCacheDir, modelFilename), modelFilename, cacheState);
      } catch (error) {
        logger.error(`[model-cache] ${error instanceof Error ? error.message : String(error)}`);
        res.statusCode = 502;
        res.end(`Failed to cache ${modelFilename}`);
      }
      return;
    }

    const ortFilename = safeFilenameFromPathname(requestUrl.pathname, ORT_RUNTIME_PATH);
    if (ortFilename) {
      if (!ortRuntimeFiles.has(ortFilename)) {
        res.statusCode = 404;
        res.end("Unknown ONNX Runtime asset");
        return;
      }
      serveFile(res, path.join(ortDistDir, ortFilename), ortFilename);
      return;
    }

    next();
  };
}

function modelCachePlugin(): Plugin {
  return {
    name: "nemotron-model-cache",
    configureServer(server) {
      server.middlewares.use(cacheMiddleware(server.config.logger));
    },
    configurePreviewServer(server) {
      server.middlewares.use(cacheMiddleware(server.config.logger));
    }
  };
}

export default defineConfig({
  plugins: [modelCachePlugin(), adminPlugin()]
});
