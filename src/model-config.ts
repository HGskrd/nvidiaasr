export const MODEL_ID = "onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4";
export const MODEL_REVISION = "da2bdea2670c93e95fa23cb4b7708d6e145b0764";
export const MODEL_ASSET_PATH = `/model-cache/${MODEL_REVISION}`;
export const ORT_RUNTIME_PATH = "/ort-runtime/";

export const MODEL_FILES = {
  encoder: {
    onnx: "encoder.onnx",
    data: "encoder.onnx.data",
    bytes: 692_767_532
  },
  decoder: {
    onnx: "decoder.onnx",
    data: "decoder.onnx.data",
    bytes: 59_789_912
  },
  joint: {
    onnx: "joint.onnx",
    data: "joint.onnx.data",
    bytes: 37_832_792
  },
  tokenizer: {
    vocab: "vocab.txt",
    bytes: 64_024
  }
} as const;

export const NEMOTRON_CONFIG = {
  vocabSize: 13_088,
  blankId: 13_087,
  blankPenalty: 0,
  sampleRate: 16_000,
  chunkSamples: 8_960,
  numMels: 128,
  fftSize: 512,
  hopLength: 160,
  winLength: 400,
  preemph: 0.97,
  logEps: 5.960_464_48e-8,
  subsamplingFactor: 8,
  leftContext: 56,
  convContext: 8,
  preEncodeCacheSize: 9,
  encoderLayers: 24,
  encoderHiddenSize: 1024,
  decoderHiddenSize: 640,
  decoderLayers: 2,
  maxSymbolsPerStep: 10,
  inputs: {
    encoderAudio: "audio_signal",
    encoderLength: "length",
    cacheLastChannel: "cache_last_channel",
    cacheLastTime: "cache_last_time",
    cacheLastChannelLen: "cache_last_channel_len",
    langId: "lang_id",
    decoderTargets: "targets",
    decoderHidden: "h_in",
    decoderCell: "c_in",
    joinerEncoder: "encoder_output",
    joinerDecoder: "decoder_output"
  },
  outputs: {
    encoder: "outputs",
    encoderLength: "encoded_lengths",
    cacheLastChannel: "cache_last_channel_next",
    cacheLastTime: "cache_last_time_next",
    cacheLastChannelLen: "cache_last_channel_len_next",
    decoder: "decoder_output",
    decoderHidden: "h_out",
    decoderCell: "c_out",
    joint: "joint_output"
  }
} as const;

export const LANG_ID_PRESETS = [
  { id: 101, label: "101 auto" },
  { id: 0, label: "0 en-US" },
  { id: 1, label: "1 en-GB" },
  { id: 8, label: "8 fr-FR" },
  { id: 100, label: "100 fr-CA" },
  { id: 2, label: "2 es-ES" },
  { id: 3, label: "3 es-US" },
  { id: 9, label: "9 de-DE" },
  { id: 15, label: "15 it-IT" },
  { id: 12, label: "12 pt-BR" },
  { id: 13, label: "13 pt-PT" },
  { id: 16, label: "16 nl-NL" },
  { id: 11, label: "11 ru-RU" },
  { id: 14, label: "14 ko-KR" },
  { id: 10, label: "10 ja-JP" },
  { id: 4, label: "4 zh-CN" },
  { id: 7, label: "7 ar-AR" },
  { id: 6, label: "6 hi-IN" },
  { id: 33, label: "33 vi-VN" },
  { id: 19, label: "19 uk-UA" },
  { id: 18, label: "18 tr-TR" },
  { id: 17, label: "17 pl-PL" },
  { id: 24, label: "24 sv-SE" },
  { id: 22, label: "22 cs-CZ" },
  { id: 103, label: "103 nb-NO" },
  { id: 25, label: "25 da-DK" },
  { id: 30, label: "30 bg-BG" },
  { id: 26, label: "26 fi-FI" },
  { id: 29, label: "29 hr-HR" },
  { id: 28, label: "28 sk-SK" },
  { id: 23, label: "23 hu-HU" },
  { id: 20, label: "20 ro-RO" },
  { id: 60, label: "60 et-EE" },
  { id: 21, label: "21 el-GR" },
  { id: 31, label: "31 lt-LT" },
  { id: 61, label: "61 lv-LV" },
  { id: 102, label: "102 mt-MT" },
  { id: 62, label: "62 sl-SI" },
  { id: 64, label: "64 he-IL" },
  { id: 32, label: "32 th-TH" },
  { id: 104, label: "104 nn-NO" }
];

export function hfUrl(filename: string): string {
  return `https://huggingface.co/${MODEL_ID}/resolve/${MODEL_REVISION}/${filename}`;
}

export function modelAssetUrl(filename: string): string {
  return `${MODEL_ASSET_PATH}/${encodeURIComponent(filename)}`;
}

export function modelAssetUrls(filename: string): string[] {
  return [hfUrl(filename), modelAssetUrl(filename)];
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}
