// Perekam WAV 16 kHz mono PCM16 di browser (cross-browser: resampling di JS secara streaming,
// sehingga memori client kecil: 15 menit ≈ 29 MB). Butuh secure context (HTTPS / localhost / flag).
const TARGET_RATE = 16000;

class StreamResampler {
  constructor(inRate, outRate) { this.ratio = inRate / outRate; this.pending = new Float32Array(0); this.pos = 0; }
  process(input) {
    const data = new Float32Array(this.pending.length + input.length);
    data.set(this.pending); data.set(input, this.pending.length);
    const r = this.ratio;
    const out = [];
    let pos = this.pos;
    while (pos + r <= data.length) {
      const s = Math.floor(pos);
      const e = Math.max(s + 1, Math.floor(pos + r)); // box filter (anti-alias sederhana)
      let sum = 0;
      for (let i = s; i < e; i++) sum += data[i];
      out.push(sum / (e - s));
      pos += r;
    }
    const consumed = Math.floor(pos);
    this.pending = data.slice(consumed);
    this.pos = pos - consumed;
    return Float32Array.from(out);
  }
}

export class WavRecorder {
  constructor() { this.chunks = []; this.samples = 0; this.running = false; this.onLevel = null; }

  static supported() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.isSecureContext);
  }

  async start() {
    if (!WavRecorder.supported()) {
      throw new Error("Mikrofon tidak tersedia: buka lewat HTTPS (mkcert) atau aktifkan flag " +
        "chrome://flags/#unsafely-treat-insecure-origin-as-secure untuk alamat server ini.");
    }
    // Matikan pemrosesan browser agar fitur akustik konsisten antara enrollment & sesi.
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    this.ctx = new AudioContext();
    await this.ctx.audioWorklet.addModule("/static/js/recorder-worklet.js");
    this.source = this.ctx.createMediaStreamSource(this.stream);
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 2048;
    this.node = new AudioWorkletNode(this.ctx, "recorder-processor");
    this.resampler = new StreamResampler(this.ctx.sampleRate, TARGET_RATE);
    this.chunks = []; this.samples = 0;
    this.node.port.onmessage = (ev) => {
      const out = this.resampler.process(ev.data);
      const pcm = new Int16Array(out.length);
      for (let i = 0; i < out.length; i++) {
        const v = Math.max(-1, Math.min(1, out[i]));
        pcm[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
      }
      this.chunks.push(pcm); this.samples += pcm.length;
    };
    this.source.connect(this.analyser);
    this.source.connect(this.node);
    // Sambungkan ke destination lewat gain 0: menjamin graph di-render di semua browser tanpa feedback suara.
    this.mute = this.ctx.createGain();
    this.mute.gain.value = 0;
    this.node.connect(this.mute).connect(this.ctx.destination);
    this.running = true;
  }

  get seconds() { return this.samples / TARGET_RATE; }

  level() {
    if (!this.analyser) return 0;
    const buf = new Float32Array(this.analyser.fftSize);
    this.analyser.getFloatTimeDomainData(buf);
    let sum = 0;
    for (const v of buf) sum += v * v;
    return Math.min(1, Math.sqrt(sum / buf.length) * 4);
  }

  drawWaveform(canvas) {
    const draw = () => {
      if (!this.running) return;
      const g = canvas.getContext("2d");
      const w = (canvas.width = canvas.clientWidth * devicePixelRatio);
      const h = (canvas.height = canvas.clientHeight * devicePixelRatio);
      const buf = new Float32Array(this.analyser.fftSize);
      this.analyser.getFloatTimeDomainData(buf);
      g.fillStyle = "#0d1b24"; g.fillRect(0, 0, w, h);
      g.strokeStyle = "#5fd3c6"; g.lineWidth = 2 * devicePixelRatio; g.beginPath();
      for (let i = 0; i < buf.length; i++) {
        const x = (i / (buf.length - 1)) * w;
        const y = h / 2 + buf[i] * h * 0.45;
        if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
      }
      g.stroke();
      requestAnimationFrame(draw);
    };
    requestAnimationFrame(draw);
  }

  async stop() {
    this.running = false;
    if (this.source) this.source.disconnect();
    if (this.node) { this.node.port.onmessage = null; this.node.disconnect(); }
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    if (this.ctx) await this.ctx.close();
    return encodeWav(this.chunks, this.samples, TARGET_RATE);
  }
}

export function encodeWav(chunks, totalSamples, rate) {
  const header = new ArrayBuffer(44);
  const v = new DataView(header);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
  const dataBytes = totalSamples * 2;
  writeStr(0, "RIFF"); v.setUint32(4, 36 + dataBytes, true); writeStr(8, "WAVE");
  writeStr(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  writeStr(36, "data"); v.setUint32(40, dataBytes, true);
  return new Blob([header, ...chunks], { type: "audio/wav" });
}
