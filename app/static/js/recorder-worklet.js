// AudioWorkletProcessor: kumpulkan sampel mono lalu kirim per ~4096 sampel ke main thread.
class RecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(4096);
    this.filled = 0;
  }
  process(inputs) {
    const input = inputs[0];
    if (input && input.length > 0) {
      const ch0 = input[0];
      const n = ch0.length;
      for (let i = 0; i < n; i++) {
        let v = ch0[i];
        if (input.length > 1) { let s = 0; for (let c = 0; c < input.length; c++) s += input[c][i]; v = s / input.length; }
        this.buffer[this.filled++] = v;
        if (this.filled === this.buffer.length) {
          this.port.postMessage(this.buffer.slice(0));
          this.filled = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor("recorder-processor", RecorderProcessor);
