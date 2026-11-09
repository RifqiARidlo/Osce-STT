# Benchmark — laptop server

## Lingkungan
| Komponen | Versi |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop, 6141 MiB, driver 596.36 (CUDA 13.2) |
| Python | 3.11.9 (`.venv`) |
| faster-whisper / ctranslate2 | 1.2.1 / 4.8.5 |
| cuBLAS / cuDNN (pip, di `.venv`) | `nvidia-cublas-cu12` 12.9.2.10 / `nvidia-cudnn-cu12` 9.28.1.4 |
| Whisper | `small`, GPU `int8_float16`, CPU `int8`, beam 5, VAD filter aktif |

## Fase 2 — Whisper + VAD di GPU (2026-11-09)

**Audio uji:** suara TTS Windows "Microsoft Andika" (id-ID) yang membacakan naskah anamnesis fiktif,
dikonversi ke 16 kHz mono. Audio TTS bersih dan hanya satu pembicara, jadi angka WER di bawah adalah
**batas atas optimistis**. Rekaman mikrofon asli masih perlu diuji.

### VRAM (`nvidia-smi`, sampel 0.5 s)
| Kondisi | VRAM terpakai | Selisih dari idle |
|---|---|---|
| Idle (sebelum model dimuat) | 423 MiB | — |
| Whisper small dimuat | 783 MiB | +360 MiB |
| **Puncak saat job berjalan** | **993 MiB** | **+570 MiB** |
| Setelah `WhisperASR.close()` + `gc.collect()` | 513 MiB | +90 MiB (CUDA context) |

VRAM Whisper **dilepas** setelah `close()`. Sisa ~90 MiB adalah CUDA context ctranslate2 yang tetap
ada selama proses hidup. Swap Whisper → LLM yang sebenarnya belum diukur karena Ollama belum
terpasang (dilanjutkan di Fase 3).

### Latensi per tahap (GPU, server lewat HTTP, model sudah dimuat)
Dua run per sesi. Muat model pertama kali: 5.0 s.

| Sesi | Durasi audio | ASR run 1 / run 2 | RTF ASR | Total wall (run 2) | Denoise | VAD | Fitur audio |
|---|---|---|---|---|---|---|---|
| A | 17.9 s | 2.41 / 2.55 s | 0.14 | 4.0 s | 0.29 s | 0.11 s | 0.15 s |
| B | 40.1 s | 5.56 / 4.46 s | 0.11–0.14 | 6.3 s | 0.35 s | 0.20 s | 0.30 s |
| C | 115.1 s | 13.39 / 11.55 s | 0.10–0.12 | 14.8 s | 1.07 s | 0.64 s | 1.04 s |

**CPU vs GPU (sesi B, 40.1 s):** CPU int8 ASR 26.17 s (RTF 0.65) vs GPU 4.5–5.6 s → GPU ~5× lebih cepat.

### WER (sesi TTS)
"Mentah" = teks referensi dengan angka ditulis sebagai kata. "Format-digit" = referensi pakai digit,
dan `x`/`mg`/`permenit` dari Whisper dinormalisasi, sehingga yang dihitung hanya salah dengar.

| Sesi | WER mentah | WER format-digit |
|---|---|---|
| A (GPU) | 5.9% | 2.9% |
| B (GPU) | 47.1% | 14.9% |
| C (GPU) | 29.3% | 13.4% |
| B (CPU int8) | — | 17.6% |

**Kesalahan substantif (sesi B, GPU):** "amok sisi lin" (amoksisilin), "para setamol" (parasetamol),
"ke GD" (IGD), "diparuh" (di paru), "kejauhan" (kehijauan), "yabu" (ya Bu), "ronkhi" (ronki).
Nama obat terpecah walau `amoksisilin` sudah ada di glossary prompt. Kandidat perbaikan: koreksi pasca-ASR
lewat lexicon (penggabungan token), bukan menambah prompt.

**Format angka:** Whisper menulis tanda vital sebagai digit ("130x80", "92xpermenit", "38,5"). Untuk ekstraksi
LLM ini justru menguntungkan, tetapi perlu diperhitungkan saat evaluasi WER.

### Belum diukur
- WER pada rekaman mikrofon asli (2–3 sesi, pembicara manusia).
- VRAM saat swap Whisper → LLM (Fase 3).
- Latensi end-to-end dengan LLM untuk rekaman 3/5/10 menit (Fase 3).
