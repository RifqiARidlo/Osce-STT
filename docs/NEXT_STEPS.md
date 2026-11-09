# NEXT_STEPS — rencana eksekusi di laptop server

Kerjakan berurutan. Tiap fase punya kriteria selesai. Centang `[x]` setelah lolos dan commit hasilnya.
Agent: kerjakan SATU fase per sesi/permintaan, laporkan hasilnya, lalu tunggu konfirmasi user.

## Fase 0 — Setup repo & environment
- [x] `git init`, lalu commit awal: `chore: initial import of osce-stt scaffold`
- [x] Buat venv Python 3.10/3.11, `pip install -r requirements.txt -r requirements-optional.txt`
- [x] `pytest -q` → 23 passed. Bila gagal karena beda versi library: perbaiki kompatibilitasnya, jangan
      melonggarkan assertion test.
- **Selesai bila:** test hijau di Python target.

## Fase 1 — Smoke test HTTP tanpa GPU (lapisan yang belum pernah dijalankan)
- [x] `.env`: `APP_WHISPER_DEVICE=cpu`, `APP_WHISPER_COMPUTE_TYPE=int8`, `APP_LLM_BACKEND=none`
- [x] `uvicorn app.main:app --port 8000` (localhost, HTTP). Cek `/api/health`, `/docs`, semua halaman
      (`/`, `/enroll`, `/review`, `/metrics`, `/cases`) tanpa error di console browser.
      _Catatan: semua route + 6 aset JS/CSS = 200. Satu-satunya 404: `/favicon.ico` dan
      `/.well-known/appspecific/com.chrome.devtools.json` (request otomatis browser/DevTools)._
- [x] Enrollment 15 detik → profil tersimpan. Rekam sesi 1–2 menit (role-play) → job `done`.
      _Catatan: enrollment 13.4 s ucapan → anchor tersimpan, rekaman dihapus. Sesi uji hanya 21 s,
      1 pembicara (tidak ada partner role-play). Silero VAD asli + Whisper small CPU int8 berjalan
      (ASR 4.7 s); transkrip berkualitas rendah — dievaluasi ulang di Fase 2._
- [x] Review: tukar peran, ganti kelas, simpan draft, verifikasi → kasus muncul di `/cases`, baris baru
      di `runtime/training_data/*.jsonl`.
      _Catatan: tukar peran, draft, verifikasi, `/cases`, 7+7 baris jsonl lolos manual di browser.
      Ganti kelas hanya diverifikasi lewat `tests/test_api.py` (belum diuji manual di UI)._
- [x] Tambahkan test API dengan `fastapi.testclient.TestClient` (ASR/LLM di-mock seperti
      `tests/test_pipeline.py`) untuk: submit job, get job (`light=true`), review, verify, auth token 401.
- **Selesai bila:** alur penuh berjalan di browser lokal dan test API ikut hijau.
- **Sebelum Fase 5:** data smoke test di `runtime/` (dokter uji, 1 kasus, 7 baris jsonl berlabel
  CHITCHAT dari transkrip rusak) jangan ikut dilatih — backup/hapus dulu.

## Fase 2 — GPU: Whisper + VAD asli
- [ ] Verifikasi `from faster_whisper.vad import VadOptions, get_speech_timestamps` sesuai versi
      terpasang. Bila signature berbeda, sesuaikan `app/pipeline/stage1_preprocess.py::silero_vad`.
- [ ] `APP_WHISPER_DEVICE=cuda`. Bila error cuBLAS/cuDNN: ikuti README faster-whisper. Laporkan ke user
      sebelum memasang paket NVIDIA.
- [ ] Jalankan 1 job sambil memantau `nvidia-smi -l 1`. Catat VRAM puncak dan apakah VRAM turun setelah
      swap ke LLM.
- **Selesai bila:** WER kualitatif wajar pada 2–3 rekaman, angka VRAM tercatat di `docs/benchmark.md`.

## Fase 3 — LLM asli (Ollama)
- [ ] `ollama pull qwen2.5:3b`, `APP_LLM_BACKEND=ollama`. Pastikan versi Ollama mendukung `format`
      berupa JSON Schema.
- [ ] Uji keempat template. Cek `summary_source == "llm"` dan jumlah warning "tanpa sumber".
- [ ] Catat `timings` per tahap untuk rekaman 3, 5, dan 10 menit di `docs/benchmark.md`.
- **Selesai bila:** rangkuman valid untuk keempat template dan ada tabel latensi nyata.

## Fase 4 — LAN multi-laptop
- [ ] mkcert untuk IP server, pasang rootCA di laptop client, buka port 8000 di firewall.
- [ ] Dua laptop client submit bersamaan → antrean berurutan, tidak ada OOM.
- [ ] Set `APP_ACCESS_TOKEN`.
- **Selesai bila:** rekam dari laptop client lewat HTTPS berhasil tanpa flag Chrome.

## Fase 5 — Data & training (inti penilaian ML)
- [ ] `scripts/generate_synthetic_llm.py --n-dialogs 60` (Ollama). Cek manual 50 baris acak, perbaiki
      label yang salah.
- [ ] Role-play terstruktur: ≥ 3 "dokter" berbeda × ≥ 4 sesi, pasien bervariasi, verifikasi via UI.
- [ ] `train_text_intent.py --test-source human_verified` (latih sintetis, uji data nyata).
- [ ] `train_speaker_role.py` (split per dokter otomatis bila ≥ 3 dokter). Simpan tabel ablation
      relatif vs absolut.
- [ ] Klik "Muat ulang model" di `/metrics`.
- **Selesai bila:** `models/*/metrics.json` + `confusion_matrix.png` tersedia dari data nyata.

## Fase 6 — Materi presentasi (deadline 15 Des)
- [ ] Ekspor angka dari `/metrics`: perbandingan baseline, CV mean±std, ablation.
- [ ] Demo end-to-end 1 skenario OSCE respirasi.
- [ ] Jelaskan keterbatasan: overlap speech, code-switching di luar lexicon, bias data sintetis.

## Risiko yang diketahui
| Risiko | Gejala | Mitigasi |
|---|---|---|
| VRAM tidak terlepas setelah unload Whisper | OOM saat tahap LLM | Whisper di CPU, atau `qwen2.5:3b` dengan `num_ctx` lebih kecil (`APP_LLM_CTX=4096`) |
| Akurasi peran rendah pada suara mirip (sesama gender/usia) | Banyak koreksi peran | ECAPA (`APP_USE_ECAPA=true`), SOP jeda giliran bicara |
| Model sklearn tidak kompatibel antar-versi | Error saat `joblib.load` | Selalu latih ulang di mesin server, jangan menyalin `.joblib` dari mesin lain |
| Browser memblokir mikrofon | Error "Mikrofon tidak tersedia" | HTTPS mkcert + rootCA di client |
