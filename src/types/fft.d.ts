declare module "fft.js" {
  export default class FFT {
    constructor(size: number);
    createComplexArray(): Float64Array;
    realTransform(out: Float64Array, input: Float64Array | Float32Array): void;
    completeSpectrum(out: Float64Array): void;
  }
}
