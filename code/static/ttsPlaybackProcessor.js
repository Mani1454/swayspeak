class TTSPlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufferQueue = [];
    this.currentChunk = null;
    this.currentChunkIndex = 0;
    this.fractionalPos = 0.0;
    this.sourceSampleRate = 24000;
    this.isPlaying = false;
    this.isBuffering = true;
    this.hasSamplePair = false;
    this.lastSample = 0.0;
    this.nextSample = 0.0;
    this.silenceFrames = 0;

    // Minimum samples to accumulate before beginning playback (~170ms at 24kHz)
    this.prebufferSamples = 4096;

    // Listen for incoming messages
    this.port.onmessage = (event) => {
      if (event.data && typeof event.data === "object") {
        if (event.data.type === "clear") {
          this.bufferQueue = [];
          this.currentChunk = null;
          this.currentChunkIndex = 0;
          this.fractionalPos = 0.0;
          this.hasSamplePair = false;
          this.lastSample = 0.0;
          this.nextSample = 0.0;
          this.isBuffering = true;
          this.silenceFrames = 0;
          if (this.isPlaying) {
            this.isPlaying = false;
            this.port.postMessage({ type: 'ttsPlaybackStopped' });
          }
          return;
        }
        if (event.data.type === "setSourceSampleRate") {
          this.sourceSampleRate = Number(event.data.sampleRate) || 24000;
          return;
        }
      }

      // Incoming audio data (Int16Array or ArrayBuffer)
      if (event.data instanceof Int16Array) {
        this.bufferQueue.push(event.data);
      } else if (event.data instanceof ArrayBuffer) {
        this.bufferQueue.push(new Int16Array(event.data));
      }
    };
  }

  _getBufferedSampleCount() {
    let count = 0;
    if (this.currentChunk && this.currentChunkIndex < this.currentChunk.length) {
      count += (this.currentChunk.length - this.currentChunkIndex);
    }
    for (let i = 0; i < this.bufferQueue.length; i++) {
      count += this.bufferQueue[i].length;
    }
    return count;
  }

  _getNextSourceSample() {
    while (!this.currentChunk || this.currentChunkIndex >= this.currentChunk.length) {
      if (this.bufferQueue.length === 0) {
        this.currentChunk = null;
        return null;
      }
      this.currentChunk = this.bufferQueue.shift();
      this.currentChunkIndex = 0;
    }
    return this.currentChunk[this.currentChunkIndex++] / 32768.0;
  }

  process(inputs, outputs) {
    const outputChannel = outputs[0][0];
    if (!outputChannel) return true;

    // Exact hardware/context sample rate (e.g., 48000, 44100, 16000)
    const hardwareRate = (typeof sampleRate !== 'undefined' && sampleRate > 0) ? sampleRate : 48000;
    const ratio = this.sourceSampleRate / hardwareRate;
    const bufferCount = this._getBufferedSampleCount();

    // Hangover window: ~300ms of empty buffer before declaring true completion
    const hangoverThreshold = Math.max(10, Math.floor((0.300 * hardwareRate) / outputChannel.length));

    // 1. Initial Jitter Pre-buffering
    if (this.isBuffering) {
      if (bufferCount >= this.prebufferSamples) {
        this.isBuffering = false;
        this.silenceFrames = 0;
      } else {
        outputChannel.fill(0);
        return true;
      }
    }

    // 2. Buffer Underrun / Hangover
    if (bufferCount === 0 && !this.hasSamplePair) {
      outputChannel.fill(0);
      this.silenceFrames++;
      if (this.silenceFrames >= hangoverThreshold) {
        if (this.isPlaying) {
          this.isPlaying = false;
          this.isBuffering = true;
          this.port.postMessage({ type: 'ttsPlaybackStopped' });
        }
      }
      return true;
    }

    // Reset silence frames as soon as audio is available
    this.silenceFrames = 0;

    // 3. Signal playback start
    if (!this.isPlaying) {
      this.isPlaying = true;
      this.port.postMessage({ type: 'ttsPlaybackStarted' });
    }

    // 4. Initialize interpolation pair if starting or resuming
    if (!this.hasSamplePair) {
      const first = this._getNextSourceSample();
      if (first === null) {
        outputChannel.fill(0);
        return true;
      }
      const second = this._getNextSourceSample();
      this.lastSample = first;
      this.nextSample = second !== null ? second : first;
      this.hasSamplePair = true;
      this.fractionalPos = 0.0;
    }

    // 5. Linear interpolation resampled to exact hardware DAC clock
    for (let i = 0; i < outputChannel.length; i++) {
      while (this.fractionalPos >= 1.0) {
        const next = this._getNextSourceSample();
        if (next === null) {
          this.hasSamplePair = false;
          break;
        }
        this.lastSample = this.nextSample;
        this.nextSample = next;
        this.fractionalPos -= 1.0;
      }

      if (this.hasSamplePair) {
        outputChannel[i] = (1.0 - this.fractionalPos) * this.lastSample + this.fractionalPos * this.nextSample;
        this.fractionalPos += ratio;
      } else {
        // Smoothly zero out remainder of quantum during temporary starvation
        for (let j = i; j < outputChannel.length; j++) {
          outputChannel[j] = 0;
        }
        break;
      }
    }

    return true;
  }
}

registerProcessor('tts-playback-processor', TTSPlaybackProcessor);
