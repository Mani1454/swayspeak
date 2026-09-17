// static/pcmWorkletProcessor.js
class PCMWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16000;
    this.phase = 0.0;
    this.lastSample = 0.0;
  }

  process(inputs) {
    const in32 = inputs[0][0];
    if (!in32 || in32.length === 0) return true;

    const sourceRate = (typeof sampleRate !== 'undefined' && sampleRate > 0) ? sampleRate : 48000;

    // Direct 1:1 if already 16kHz
    if (sourceRate === this.targetSampleRate) {
      const int16 = new Int16Array(in32.length);
      for (let i = 0; i < in32.length; i++) {
        let s = in32[i];
        s = s < -1 ? -1 : s > 1 ? 1 : s;
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      this.port.postMessage(int16.buffer, [int16.buffer]);
      return true;
    }

    // High-fidelity continuous-phase downsampling to exactly 16,000 Hz
    const ratio = sourceRate / this.targetSampleRate;
    const outSamples = [];

    while (this.phase < in32.length) {
      const idx = Math.floor(this.phase);
      const frac = this.phase - idx;
      const s0 = in32[idx];
      const s1 = (idx + 1 < in32.length) ? in32[idx + 1] : s0;

      let s = (1.0 - frac) * s0 + frac * s1;
      s = s < -1.0 ? -1.0 : s > 1.0 ? 1.0 : s;
      outSamples.push(s < 0 ? s * 0x8000 : s * 0x7FFF);

      this.phase += ratio;
    }

    this.phase -= in32.length;
    this.lastSample = in32[in32.length - 1];

    if (outSamples.length > 0) {
      const int16 = new Int16Array(outSamples.length);
      for (let i = 0; i < outSamples.length; i++) {
        int16[i] = outSamples[i];
      }
      this.port.postMessage(int16.buffer, [int16.buffer]);
    }

    return true;
  }
}

registerProcessor('pcm-worklet-processor', PCMWorkletProcessor);

