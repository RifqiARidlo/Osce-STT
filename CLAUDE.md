# CLAUDE.md — konteks proyek untuk Claude Code

Dibaca otomatis oleh Claude Code setiap sesi. Agent di sesi ini TIDAK punya akses ke percakapan
claude.ai tempat kode ini dibuat. File ini, `README.md`, dan `docs/NEXT_STEPS.md` adalah satu-satunya
sumber konteks.

## Tujuan
Sistem "Offline Medical STT + Custom Speaker & Clinical Intent Classification + OSCE Case Bank"
berbasis web di LAN. Hanya laptop server yang punya GPU (4–6 GB VRAM). Laptop dokter/penguji cukup
browser. Proyek kelompok (Kelompok 1) MK Machine Learning for Medical Data, Teknologi Kedokteran ITS.
- Syarat akademik: 2 modul ML yang dilatih sendiri, dengan metrik (confusion matrix, P/R/F1).
- Jadwal: presentasi final 15 Des 2026, pameran 18 Des 2026.

## Arsitektur (ringkas — detail di README §1–2)
Browser (AudioWorklet → WAV 16 kHz) → FastAPI → JobQueue (1 consumer, FIFO, `asyncio.to_thread`) →
S1 spectral gating + Silero VAD (ONNX dari faster-whisper) → S2 Faster-Whisper word timestamps →
segmentasi utterance → [ML-1] Speaker Role (anchor-relative) → PII scrub → S3 [ML-2] Intent
S/O/A/P/CHITCHAT + lexicon mapper → S4 LLM (Ollama/llama.cpp, JSON Schema terkunci) → S5 review manusia
→ SQLite `osce_cases` + `runtime/training_data/*.jsonl` (active learning).
`GpuModelSlot` menjamin Whisper & LLM tidak pernah berada di VRAM bersamaan (`APP_GPU_SWAP_POLICY=swap`).

Peta modul: `app/pipeline/orchestrator.py` (alur), `app/ml/` (2 modul ML), `app/core/` (antrean, model
slot, audio IO), `app/api/` (routes), `app/static/` (UI tanpa build step), `scripts/` (data & training).

## Keputusan desain yang JANGAN dibalik tanpa diskusi dengan user
1. **Fitur speaker relatif terhadap anchor dokter + relatif-sesi**, bukan MFCC/F0 absolut. Fitur
   absolut membuat model menghafal identitas suara, bukan peran, sehingga gagal pada dokter baru.
   Diuji oleh `tests/test_ml_audio.py::test_trained_model_generalizes_to_unseen_doctor`.
2. **CMVN tidak dipakai di jalur anchor**, karena menghapus mean cepstral yang dipakai anchor.
3. **Fitur speaker diambil dari audio MENTAH**; denoise hanya untuk VAD/ASR. Berlaku di
   `orchestrator.py`, `routes_doctors._enroll`, dan `scripts/train_speaker_role.py`. Ketiganya harus
   konsisten.
4. **PII di-scrub langsung setelah ASR**, sebelum apa pun disimpan. Audio mentah dihapus setelah
   diproses (`APP_KEEP_RAW_AUDIO=false`).
5. **Uvicorn wajib 1 worker.** Multi-worker berarti model terduplikasi di VRAM → OOM.
6. **Split train/test berbasis grup** (template/job/dokter). Split acak pada data template = kebocoran.
7. **Lexicon memetakan ke gejala, bukan diagnosis**, dan teks asli dipertahankan dalam kurung.
8. **Setiap butir rangkuman LLM wajib punya `sumber`** (ID utterance). ID yang tidak valid dibuang.

## Perintah
```bash
pip install -r requirements.txt            # + requirements-optional.txt untuk pytest/matplotlib
pytest -q                                   # 26 test CPU, harus tetap hijau
python scripts/generate_synthetic_text.py --n-per-template 25
python scripts/train_text_intent.py --data data/synthetic/text_intent.jsonl
python scripts/train_speaker_role.py --data runtime/training_data/speaker_role.jsonl --model svm
uvicorn app.main:app --host 0.0.0.0 --port 8000 --ssl-keyfile key.pem --ssl-certfile cert.pem
```

## Status verifikasi (per pembuatan awal)
- SUDAH diuji (CPU, model palsu untuk ASR/LLM): fitur audio, speaker heuristik & terlatih, intent,
  lexicon, PII, schema, retry/fallback LLM, memory swap, antrean, end-to-end + verifikasi + training.
- SUDAH diuji di Fase 1 (CPU): lapisan FastAPI via `tests/test_api.py` (routes, upload, auth 401) dan
  manual di browser localhost (enroll, rekam, review, verify, `/cases`); Faster-Whisper small CPU int8
  dan Silero VAD asli (faster-whisper 1.2.1).
- SUDAH diuji di Fase 2: Faster-Whisper small GPU `int8_float16` (cuBLAS/cuDNN dari pip di `.venv`),
  hanya dengan audio TTS. Angka di `docs/benchmark.md`.
- BELUM PERNAH dijalankan: Ollama asli (structured outputs), llama-cpp backend, ECAPA/speechbrain,
  akses LAN/HTTPS dari laptop client, WER pada rekaman mikrofon asli.
- Dikembangkan di Python 3.13 + scikit-learn 1.9. Laptop server: Python 3.11.9 di `.venv`
  (Fase 0, 23 test lolos tanpa perubahan kode).
- Pelepasan VRAM CTranslate2: `close()` + gc melepas VRAM model (sisa ~90 MiB CUDA context). Swap ke
  LLM asli belum diukur.

## Aturan kerja untuk agent
- Sebelum mengubah kode: cek struktur folder, `git status`/`git diff`, dan baca file terkait.
- Ubah seminimal mungkin dan pertahankan arsitektur. Jalankan `pytest -q` setelah perubahan Python.
- Tanyakan user dulu sebelum: memasang dependency berat (torch, speechbrain, llama-cpp-python CUDA),
  mengunduh model besar, mengubah skema DB, atau mengubah keputusan desain di atas.
- Bila perintah lama/menggantung atau error tidak jelas: berhenti dan laporkan ke user, jangan
  troubleshoot berputar-putar.
- Jangan menaruh data pasien nyata di test, fixture, log, atau commit. `runtime/` dan `*.pem` ada di
  `.gitignore`.
- Git: Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`), atomic, tanpa debug code.
- Jangan mengarang API/library. Bila tidak yakin dengan signature library (mis. faster-whisper VAD,
  Ollama `format`), cek source/dokumentasi yang terpasang (`python -c "import inspect, ..."`) dan
  laporkan "belum dapat dipastikan" bila tetap ragu.

## Gaya komunikasi dengan user
Bahasa Indonesia, istilah teknis tetap English. Langsung ke jawaban, tanpa basa-basi. Koreksi asumsi
yang salah secara langsung beserta alasan teknisnya. Untuk rekomendasi arsitektur/algoritma, sertakan
trade-off (latensi, memori, kompleksitas, failure mode). Kode lengkap dan runnable; untuk edit file
yang ada, berikan blok/fungsi yang berubah. Saat debugging, jelaskan root cause dulu.
