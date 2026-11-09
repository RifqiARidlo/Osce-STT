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

## Fase 3 — LLM asli (Ollama) (2026-11-09)
| Komponen | Versi |
|---|---|
| Ollama | 0.40.15 (`format` = JSON Schema didukung) |
| Model | `qwen2.5:3b` (id 357c53fb659c, 1.9 GB), `num_ctx` 8192, `temperature` 0, `keep_alive` 0 |

**Audio uji:** dialog TTS fiktif per template (60–88 s). Dokter memakai suara "Andika" asli, pasien memakai
"Andika" yang di-pitch-shift (resample 1.2×). Dokter TTS di-enroll dari 27 s suara Andika asli.
Rekaman panjang adalah gabungan keempat dialog yang diulang, lalu dipotong ke 3/5/10 menit.

### Bug yang ditemukan & diperbaiki
JSON Schema yang dikirim ke Ollama tidak punya `required` (semua field Pydantic punya default), sehingga
constrained decoding mengizinkan object minimal. qwen2.5:3b hanya mengeluarkan
`{"rubrik": {"memberikan_edukasi": true}}` (26 token), dan seluruh field lain terisi default
"Tidak disebutkan". Perbaikan: `llm_json_schema()` mewajibkan semua properti, dan schema ikut
dimasukkan ke prompt (`build_user_prompt(..., schema)`). Regresi diuji oleh
`test_llm_schema_requires_all_fields_and_is_in_prompt`.

### Hasil 4 template (setelah perbaikan)
| Template | Field terisi (sebelum → sesudah) | `summary_source` | Warning "tanpa sumber" | LLM | Total wall |
|---|---|---|---|---|---|
| osce_respirasi | 0/14 → 9/14 | llm | 0 | 9.0 s | 15.6 s |
| osce_kardiovaskular | 2/16 → 16/16 | llm | 0 | 10.9 s | 15.4 s |
| calgary_cambridge | 2/12 (salah tempat) → 9/12 | llm | 0 | 8.3 s | 13.5 s |
| soap_umum | 2/8 → 6/8 | llm | 0 | 6.4 s | 10.4 s |

Muat model LLM pertama kali (cold start dari disk): ~48 s. Setelah itu model dimuat ulang per job
(`keep_alive=0`) dan sudah termasuk di kolom LLM.

**Kesalahan klinis yang masih ada (batas kemampuan model 3B + error ASR):**
- Respirasi: demam & sesak napas "Tidak disebutkan", padahal disebut pasien; riwayat merokok tidak terisi.
- Kardiovaskular: skala nyeri 10 (ucapan: 8); penjalaran "Tidak menjalar" (ucapan: ke lengan kiri & rahang);
  palpitasi "Tidak" (ucapan: berdebar sedikit); faktor risiko "Diabetes melitus" **halusinasi**, merokok terlewat.
- Calgary-Cambridge: onset "Kronis (>12 minggu)" (ucapan: 2 minggu); pertanyaan dokter masuk ke riwayat
  penyakit dahulu; kekhawatiran ("takut tumor"), riwayat keluarga, dan sosial kosong.
- SOAP: keluhan utama "mual" (seharusnya nyeri ulu hati); "tidak muntah" menjadi "vomitus (muntah)"
  (negasi hilang).

Rangkuman valid secara schema dan sumber, tetapi **wajib direview manusia** (Stage 5). Kandidat perbaikan:
`qwen2.5:7b` (~4.7 GB Q4; muat karena Whisper sudah di-swap keluar), penanganan negasi di lexicon mapper.

### Latensi rekaman panjang (template `soap_umum`, 1 run per durasi)
| Durasi | Utterance | Denoise | VAD | ASR | Fitur audio | LLM | Total wall | Prompt LLM |
|---|---|---|---|---|---|---|---|---|
| 3 menit | 43 | 2.4 s | 1.3 s | 21.9 s | 1.8 s | 9.8 s | 42.1 s | — |
| 5 menit | 75 | 3.0 s | 1.8 s | 35.2 s | 3.1 s | 13.0 s | 61.0 s | — |
| 10 menit | 149 | 6.4 s | 3.9 s | 34.6 s | 1.3 s | 16.6 s | 68.2 s | 3708 token, `truncated = 0` |

ASR 10 menit ≈ 5 menit dalam run ini; seluruh 10:00 terbukti diproses (log faster-whisper). Variasi
antar-run belum diukur (perlu ≥ 3 run per durasi).

### VRAM saat swap Whisper → LLM
- Log server: `Model 'asr' di-unload dari VRAM` sebelum LLM dipanggil, jadi `GpuModelSlot` swap berjalan.
- Puncak per job dengan LLM: 2504–2544 MiB (job pertama setelah start: 2944 MiB). Tidak pernah mendekati 6 GB.
- Observasi: VRAM terpakai setelah job naik ~8 MiB per job dalam satu proses server (91 → 139 MiB setelah
  7 job). Kemungkinan sisa alokasi CUDA dari muat-ulang Whisper. Perlu dipantau pada sesi panjang
  (≥ 50 job); belum dapat dipastikan apakah ini kebocoran.
