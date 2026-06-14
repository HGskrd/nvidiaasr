import FFT from "fft.js";
import { NEMOTRON_CONFIG } from "./model-config";

export interface StreamingFeatureChunk {
  data: Float32Array;
  frames: number;
  dims: [number, number, number];
}

export class StreamingLinearResampler {
  private readonly ratio: number;
  private sourceBuffer = new Float32Array(0);
  private position = 0;

  constructor(sourceRate: number, targetRate: number) {
    this.ratio = sourceRate / targetRate;
  }

  push(input: Float32Array): Float32Array {
    if (input.length === 0) return new Float32Array(0);
    if (this.ratio === 1 && this.sourceBuffer.length === 0) return input.slice();

    const source = new Float32Array(this.sourceBuffer.length + input.length);
    source.set(this.sourceBuffer);
    source.set(input, this.sourceBuffer.length);

    const output: number[] = [];
    while (this.position + 1 < source.length) {
      const left = Math.floor(this.position);
      const frac = this.position - left;
      output.push(source[left] * (1 - frac) + source[left + 1] * frac);
      this.position += this.ratio;
    }

    const keepFrom = Math.max(0, Math.floor(this.position));
    this.sourceBuffer = source.slice(keepFrom);
    this.position -= keepFrom;
    return Float32Array.from(output);
  }

  reset(): void {
    this.sourceBuffer = new Float32Array(0);
    this.position = 0;
  }
}

export class ModelChunker {
  private pending = new Float32Array(0);

  push(samples: Float32Array): Float32Array[] {
    if (samples.length === 0) return [];
    const cfg = NEMOTRON_CONFIG;
    const source = new Float32Array(this.pending.length + samples.length);
    source.set(this.pending);
    source.set(samples, this.pending.length);

    const chunks: Float32Array[] = [];
    let offset = 0;
    while (offset + cfg.chunkSamples <= source.length) {
      chunks.push(source.slice(offset, offset + cfg.chunkSamples));
      offset += cfg.chunkSamples;
    }

    this.pending = source.slice(offset);
    return chunks;
  }

  flushPadded(): Float32Array | undefined {
    if (this.pending.length === 0) return undefined;
    const chunk = new Float32Array(NEMOTRON_CONFIG.chunkSamples);
    chunk.set(this.pending);
    this.pending = new Float32Array(0);
    return chunk;
  }

  reset(): void {
    this.pending = new Float32Array(0);
  }
}

function hannWindow(length: number): Float32Array {
  const win = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (length - 1));
  }
  return win;
}

function hzToMelSlaney(hz: number): number {
  const fSp = 200 / 3;
  const minLogHz = 1000;
  const minLogMel = minLogHz / fSp;
  const logStep = Math.log(6.4) / 27;
  return hz < minLogHz ? hz / fSp : minLogMel + Math.log(hz / minLogHz) / logStep;
}

function melToHzSlaney(mel: number): number {
  const fSp = 200 / 3;
  const minLogHz = 1000;
  const minLogMel = minLogHz / fSp;
  const logStep = Math.log(6.4) / 27;
  return mel < minLogMel ? mel * fSp : minLogHz * Math.exp(logStep * (mel - minLogMel));
}

// Mirrors librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2,
// htk=False, norm="slaney") — the filterbank NeMo's preprocessor uses. The
// triangles are evaluated on the continuous FFT frequency grid with fractional
// edges; flooring mel points to integer bins (as the old code did) collapses
// the densely packed low-frequency bands to zero at 128 mels / 257 FFT bins.
function buildMelFilterBank(): Float32Array[] {
  const cfg = NEMOTRON_CONFIG;
  const fftBins = Math.floor(cfg.fftSize / 2) + 1;

  // FFT bin centre frequencies: linspace(0, sr/2, fftBins).
  const fftFreqs = new Float64Array(fftBins);
  for (let k = 0; k < fftBins; k++) fftFreqs[k] = (k * cfg.sampleRate) / cfg.fftSize;

  // numMels + 2 mel points spanning [0, sr/2], converted back to Hz.
  const minMel = hzToMelSlaney(0);
  const maxMel = hzToMelSlaney(cfg.sampleRate / 2);
  const hzPoints = new Float64Array(cfg.numMels + 2);
  for (let i = 0; i < hzPoints.length; i++) {
    hzPoints[i] = melToHzSlaney(minMel + ((maxMel - minMel) * i) / (cfg.numMels + 1));
  }

  const filters: Float32Array[] = [];
  for (let m = 0; m < cfg.numMels; m++) {
    const filter = new Float32Array(fftBins);
    const lowerHz = hzPoints[m];
    const centerHz = hzPoints[m + 1];
    const upperHz = hzPoints[m + 2];
    const leftWidth = centerHz - lowerHz;
    const rightWidth = upperHz - centerHz;
    // Slaney normalisation: area-preserving across the non-uniform mel widths.
    const enorm = 2 / (upperHz - lowerHz);

    for (let k = 0; k < fftBins; k++) {
      const freq = fftFreqs[k];
      const lower = leftWidth > 0 ? (freq - lowerHz) / leftWidth : 0;
      const upper = rightWidth > 0 ? (upperHz - freq) / rightWidth : 0;
      const weight = Math.max(0, Math.min(lower, upper));
      filter[k] = weight * enorm;
    }
    filters.push(filter);
  }
  return filters;
}

export class NemoLogMelExtractor {
  private readonly fft = new FFT(NEMOTRON_CONFIG.fftSize);
  private readonly window = hannWindow(NEMOTRON_CONFIG.winLength);
  private readonly filters = buildMelFilterBank();
  private readonly fftInput = new Float64Array(NEMOTRON_CONFIG.fftSize);
  private readonly spectrum = this.fft.createComplexArray();
  private readonly power = new Float32Array(Math.floor(NEMOTRON_CONFIG.fftSize / 2) + 1);
  private previousSample = 0;
  private previousEmphasizedTail = new Float32Array(0);

  process(samples: Float32Array): { data: Float32Array; frames: number } {
    const cfg = NEMOTRON_CONFIG;
    const emphasized = new Float32Array(samples.length);
    if (samples.length > 0) {
      emphasized[0] = samples[0] - cfg.preemph * this.previousSample;
      for (let i = 1; i < samples.length; i++) {
        emphasized[i] = samples[i] - cfg.preemph * samples[i - 1];
      }
      this.previousSample = samples[samples.length - 1];
    }

    const centerPad = Math.floor(cfg.fftSize / 2);
    // When winLength < fftSize, librosa/torch.stft center the (padded) window
    // inside the FFT frame. Placing it at offset 0 instead miscentres every
    // frame by (fftSize - winLength) / 2 samples relative to its hop position.
    const winOffset = Math.floor((cfg.fftSize - cfg.winLength) / 2);
    const frames = Math.floor(samples.length / cfg.hopLength);
    const mel = new Float32Array(cfg.numMels * frames);

    for (let frame = 0; frame < frames; frame++) {
      this.fftInput.fill(0);
      const start = frame * cfg.hopLength - centerPad;
      for (let i = 0; i < cfg.winLength; i++) {
        const src = start + winOffset + i;
        let value = 0;
        if (src >= 0 && src < emphasized.length) {
          value = emphasized[src];
        } else if (src < 0 && this.previousEmphasizedTail.length > 0) {
          const tailIndex = this.previousEmphasizedTail.length + src;
          value = tailIndex >= 0 ? this.previousEmphasizedTail[tailIndex] : emphasized[Math.min(emphasized.length - 1, -src)];
        } else if (src < 0 && emphasized.length > 0) {
          value = emphasized[Math.min(emphasized.length - 1, -src)];
        }
        this.fftInput[winOffset + i] = value * this.window[i];
      }

      this.fft.realTransform(this.spectrum, this.fftInput);
      this.fft.completeSpectrum(this.spectrum);

      for (let k = 0; k < this.power.length; k++) {
        const re = this.spectrum[2 * k];
        const im = this.spectrum[2 * k + 1];
        this.power[k] = re * re + im * im;
      }

      for (let m = 0; m < cfg.numMels; m++) {
        const filter = this.filters[m];
        let energy = 0;
        for (let k = 0; k < filter.length; k++) energy += filter[k] * this.power[k];
        mel[m * frames + frame] = Math.log(Math.max(energy, 0) + cfg.logEps);
      }
    }

    const tailSize = Math.min(centerPad, emphasized.length);
    this.previousEmphasizedTail = emphasized.slice(emphasized.length - tailSize);

    return { data: mel, frames };
  }

  reset(): void {
    this.previousSample = 0;
    this.previousEmphasizedTail = new Float32Array(0);
  }
}

export class StreamingFeatureBuilder {
  private readonly extractor = new NemoLogMelExtractor();
  private readonly cache = new Float32Array(NEMOTRON_CONFIG.preEncodeCacheSize * NEMOTRON_CONFIG.numMels);
  private cachePos = 0;

  build(samples: Float32Array): StreamingFeatureChunk {
    const cfg = NEMOTRON_CONFIG;
    const mel = this.extractor.process(samples);
    const totalFrames = cfg.preEncodeCacheSize + mel.frames;
    const output = new Float32Array(totalFrames * cfg.numMels);

    const firstRun = Math.min(cfg.preEncodeCacheSize - this.cachePos, cfg.preEncodeCacheSize);
    output.set(this.cache.subarray(this.cachePos * cfg.numMels, (this.cachePos + firstRun) * cfg.numMels), 0);
    if (firstRun < cfg.preEncodeCacheSize) {
      output.set(this.cache.subarray(0, (cfg.preEncodeCacheSize - firstRun) * cfg.numMels), firstRun * cfg.numMels);
    }

    for (let t = 0; t < mel.frames; t++) {
      const dst = (cfg.preEncodeCacheSize + t) * cfg.numMels;
      for (let m = 0; m < cfg.numMels; m++) {
        output[dst + m] = mel.data[m * mel.frames + t];
      }
    }

    const framesToCache = Math.min(mel.frames, cfg.preEncodeCacheSize);
    const cacheStartFrame = cfg.preEncodeCacheSize + mel.frames - framesToCache;
    for (let i = 0; i < framesToCache; i++) {
      const src = (cacheStartFrame + i) * cfg.numMels;
      const dstFrame = (this.cachePos + i) % cfg.preEncodeCacheSize;
      this.cache.set(output.subarray(src, src + cfg.numMels), dstFrame * cfg.numMels);
    }
    this.cachePos = (this.cachePos + framesToCache) % cfg.preEncodeCacheSize;

    return { data: output, frames: totalFrames, dims: [1, totalFrames, cfg.numMels] };
  }

  reset(): void {
    this.extractor.reset();
    this.cache.fill(0);
    this.cachePos = 0;
  }
}

export async function decodeAudioFile(file: File): Promise<Float32Array> {
  const arrayBuffer = await file.arrayBuffer();
  const audioContext = new AudioContext();
  try {
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    return downmixAndResample(audioBuffer, NEMOTRON_CONFIG.sampleRate);
  } finally {
    await audioContext.close();
  }
}

export function downmixAndResample(audioBuffer: AudioBuffer, targetRate: number): Float32Array {
  const channelCount = audioBuffer.numberOfChannels;
  const source = new Float32Array(audioBuffer.length);
  for (let c = 0; c < channelCount; c++) {
    const data = audioBuffer.getChannelData(c);
    for (let i = 0; i < data.length; i++) source[i] += data[i] / channelCount;
  }

  return resamplePcm(source, audioBuffer.sampleRate, targetRate);
}

export function resamplePcm(source: Float32Array, sourceRate: number, targetRate: number): Float32Array {
  if (sourceRate === targetRate) return source;

  const ratio = sourceRate / targetRate;
  const outLength = Math.max(1, Math.floor(source.length / ratio));
  const output = new Float32Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const srcIndex = i * ratio;
    const left = Math.floor(srcIndex);
    const right = Math.min(source.length - 1, left + 1);
    const frac = srcIndex - left;
    output[i] = source[left] * (1 - frac) + source[right] * frac;
  }
  return output;
}
