class TTSPlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufferQueue = [];
    this.currentChunk = null;
    this.currentChunkIndex = 0;
    this.fractionalPos = 0.0;
    this.sourceSampleRate = 24000;
    this.isPlaying = false;
    this.hasStartedSample = false;
    this.hasPendingSample = false;
    this.lastSample = 0.0;
    this.nextSample = 0.0;

    // Listen for incoming messages
    this.port.onmessage = (event) => {
      if (event.data && typeof event.data === "object") {
        if (event.data.type === "clear") {
          // Clear the TTS buffer and reset playback state
          this.bufferQueue = [];
          this.currentChunk = null;
          this.currentChunkIndex = 0;
          this.fractionalPos = 0.0;
          this.hasStartedSample = false;
          this.hasPendingSample = false;
          this.lastSample = 0.0;
          this.nextSample = 0.0;
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

    // Hardware/context sample rate (e.g., 48000, 96000, 44100)
    const hardwareRate = (typeof sampleRate !== 'undefined' && sampleRate > 0) ? sampleRate : 48000;
    const ratio = this.sourceSampleRate / hardwareRate;

    const hasData = (this.currentChunk && this.currentChunkIndex < this.currentChunk.length) || this.bufferQueue.length > 0;

    if (!hasData && !this.hasPendingSample) {
      outputChannel.fill(0);
      if (this.isPlaying) {
        this.isPlaying = false;
        this.port.postMessage({ type: 'ttsPlaybackStopped' });
      }
      return true;
    }

    if (!this.isPlaying && (hasData || this.hasPendingSample)) {
      this.isPlaying = true;
      this.port.postMessage({ type: 'ttsPlaybackStarted' });
    }

    // Initialize first two samples for interpolation
    if (!this.hasStartedSample) {
      const first = this._getNextSourceSample();
      if (first === null) {
        outputChannel.fill(0);
        return true;
      }
      this.lastSample = first;
      const second = this._getNextSourceSample();
      this.nextSample = second !== null ? second : first;
      this.hasStartedSample = true;
      this.hasPendingSample = true;
      this.fractionalPos = 0.0;
    }

    for (let i = 0; i < outputChannel.length; i++) {
      while (this.fractionalPos >= 1.0) {
        const next = this._getNextSourceSample();
        if (next === null) {
          this.hasPendingSample = false;
          break;
        }
        this.lastSample = this.nextSample;
        this.nextSample = next;
        this.fractionalPos -= 1.0;
      }

      if (this.hasPendingSample) {
        // High-fidelity linear interpolation resampled to exact hardware clock
        outputChannel[i] = (1.0 - this.fractionalPos) * this.lastSample + this.fractionalPos * this.nextSample;
        this.fractionalPos += ratio;
      } else {
        outputChannel[i] = 0;
        this.hasStartedSample = false;
        for (let j = i + 1; j < outputChannel.length; j++) {
          outputChannel[j] = 0;
        }
        break;
      }
    }

    return true;
  }
}

registerProcessor('tts-playback-processor', TTSPlaybackProcessor);
