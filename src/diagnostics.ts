export type DiagnosticLevel = "info" | "warn" | "error";

type Jsonish = null | boolean | number | string | Jsonish[] | { [key: string]: Jsonish };

type MinimalGpu = {
  requestAdapter(options?: { powerPreference?: "low-power" | "high-performance" }): Promise<MinimalGpuAdapter | null>;
};

type MinimalGpuAdapter = {
  features?: Iterable<string>;
  limits?: Record<string, number>;
  info?: Record<string, unknown>;
  requestAdapterInfo?: () => Promise<Record<string, unknown>>;
  requestDevice?: () => Promise<{ destroy?: () => void }>;
};

export interface DiagnosticRecord {
  id: number;
  at: string;
  level: DiagnosticLevel;
  message: string;
  detail?: Jsonish;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function primitive(value: unknown): Jsonish {
  if (value === null || typeof value === "boolean" || typeof value === "number" || typeof value === "string") {
    return value;
  }
  if (typeof value === "bigint") return value.toString();
  if (typeof value === "undefined") return "undefined";
  if (typeof value === "symbol") return value.toString();
  if (typeof value === "function") return `[Function ${value.name || "anonymous"}]`;
  return String(value);
}

export function serializeError(error: unknown, depth = 0): Jsonish {
  if (!isObject(error) || depth > 4) return primitive(error);

  const details: Record<string, Jsonish> = {
    type: error.constructor?.name ?? "Object"
  };

  if ("name" in error) details.name = primitive(error.name);
  if ("message" in error) details.message = primitive(error.message);
  if ("code" in error) details.code = primitive(error.code);
  if ("stack" in error) details.stack = primitive(error.stack);
  if ("cause" in error && error.cause !== undefined) details.cause = serializeError(error.cause, depth + 1);

  for (const [key, value] of Object.entries(error)) {
    if (!(key in details)) details[key] = serializeDiagnosticValue(value, depth + 1);
  }

  return details;
}

export function serializeDiagnosticValue(value: unknown, depth = 0): Jsonish {
  if (!isObject(value)) return primitive(value);
  if (depth > 4) return "[Max depth]";
  if (value instanceof Error) return serializeError(value, depth);
  if (Array.isArray(value)) return value.slice(0, 80).map((item) => serializeDiagnosticValue(item, depth + 1));

  const output: Record<string, Jsonish> = {};
  for (const [key, item] of Object.entries(value).slice(0, 80)) {
    output[key] = serializeDiagnosticValue(item, depth + 1);
  }
  return output;
}

export function errorMessage(error: unknown): string {
  if (error instanceof Error) return `${error.name}: ${error.message}`;
  return String(error);
}

interface UserAgentData {
  brands?: Array<{ brand: string; version: string }>;
  mobile?: boolean;
  platform?: string;
  getHighEntropyValues?: (hints: string[]) => Promise<Record<string, unknown>>;
}

export function collectBrowserDiagnostics(): Jsonish {
  const nav = navigator as Navigator & { deviceMemory?: number; gpu?: MinimalGpu; userAgentData?: UserAgentData };
  const audioCtor = window.AudioContext ?? (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  const uaData = nav.userAgentData;

  return {
    userAgent: navigator.userAgent,
    // navigator.platform is frozen to "MacIntel"/"Win32" on modern browsers and
    // does not reflect the real CPU; the architecture hint below is the truth.
    platform: navigator.platform,
    uaPlatform: uaData?.platform ?? null,
    uaBrands: uaData?.brands ? uaData.brands.map((b) => `${b.brand} ${b.version}`) : null,
    uaMobile: uaData?.mobile ?? null,
    language: navigator.language,
    languages: Array.from(navigator.languages ?? []),
    hardwareConcurrency: navigator.hardwareConcurrency,
    deviceMemoryGB: nav.deviceMemory ?? null,
    secureContext: window.isSecureContext,
    crossOriginIsolated: window.crossOriginIsolated,
    location: {
      protocol: window.location.protocol,
      host: window.location.host
    },
    webgpu: Boolean(nav.gpu),
    mediaDevices: Boolean(navigator.mediaDevices?.getUserMedia),
    cacheStorage: "caches" in window,
    audioContext: Boolean(audioCtor),
    indexedDB: "indexedDB" in window
  };
}

// Real CPU architecture etc. live behind async UA Client Hints (Chromium only);
// navigator.platform cannot distinguish Apple Silicon from Intel.
export async function collectHighEntropyHints(): Promise<Jsonish> {
  const uaData = (navigator as Navigator & { userAgentData?: UserAgentData }).userAgentData;
  if (!uaData?.getHighEntropyValues) return null;
  try {
    const values = await uaData.getHighEntropyValues(["architecture", "bitness", "platformVersion", "model"]);
    return serializeDiagnosticValue(values);
  } catch {
    return null;
  }
}

function selectedLimits(adapter: MinimalGpuAdapter): Jsonish {
  const keys = [
    "maxTextureDimension2D",
    "maxBufferSize",
    "maxStorageBufferBindingSize",
    "maxComputeWorkgroupStorageSize",
    "maxComputeInvocationsPerWorkgroup",
    "maxComputeWorkgroupSizeX",
    "maxComputeWorkgroupsPerDimension"
  ];
  const limits = adapter.limits ?? {};
  const output: Record<string, Jsonish> = {};
  for (const key of keys) {
    const value = limits[key];
    if (typeof value === "number") output[key] = value;
  }
  return output;
}

export async function probeWebGpu(
  log: (level: DiagnosticLevel, message: string, detail?: unknown) => void
): Promise<void> {
  const gpu = (navigator as Navigator & { gpu?: MinimalGpu }).gpu;
  if (!gpu) {
    log("error", "WebGPU API missing", {
      reason: "navigator.gpu is undefined",
      secureContext: window.isSecureContext,
      protocol: window.location.protocol
    });
    return;
  }

  const started = performance.now();
  let adapter: MinimalGpuAdapter | null = null;
  try {
    adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  } catch (error) {
    log("error", "WebGPU requestAdapter failed", serializeError(error));
    return;
  }

  if (!adapter) {
    log("error", "WebGPU requestAdapter returned null", {
      elapsedMs: Math.round(performance.now() - started),
      reason: "Browser exposes WebGPU but no compatible adapter is available"
    });
    return;
  }

  let adapterInfo: Record<string, unknown> | undefined = adapter.info;
  if (!adapterInfo && adapter.requestAdapterInfo) {
    try {
      adapterInfo = await adapter.requestAdapterInfo();
    } catch (error) {
      log("warn", "WebGPU adapter info unavailable", serializeError(error));
    }
  }

  log("info", "WebGPU adapter available", {
    elapsedMs: Math.round(performance.now() - started),
    info: serializeDiagnosticValue(adapterInfo ?? {}),
    features: Array.from(adapter.features ?? []),
    limits: selectedLimits(adapter)
  });

  if (!adapter.requestDevice) {
    log("warn", "WebGPU adapter does not expose requestDevice");
    return;
  }

  const deviceStarted = performance.now();
  try {
    const device = await adapter.requestDevice();
    log("info", "WebGPU device request succeeded", {
      elapsedMs: Math.round(performance.now() - deviceStarted)
    });
    device.destroy?.();
  } catch (error) {
    log("error", "WebGPU requestDevice failed", serializeError(error));
  }
}
