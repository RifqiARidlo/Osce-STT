# Offline Medical STT · Speaker & Clinical Intent Classification · OSCE Case Bank

Sistem client–server di LAN. **Hanya laptop server yang butuh GPU.** Laptop dokter/penguji cukup
membuka browser (Chrome/Edge/Firefox terbaru): perekaman, encoding WAV 16 kHz, dan UI review berjalan
di browser. Semua inferensi (Whisper, 2 model Custom ML, LLM) berjalan di server, tanpa internet.

**Mulai dari sini:** instalasi komputer lab + URL publik Cloudflare → [`docs/INSTALASI_LAB.md`](docs/INSTALASI_LAB.md) ·
pengujian data nyata → [`docs/TUTORIAL_TESTING.md`](docs/TUTORIAL_TESTING.md) ·
cara kerja & alasan tiap tahap → [`docs/BELAJAR_SISTEM.md`](docs/BELAJAR_SISTEM.md).
Mode Cloudflare tunnel membuat sistem **tidak lagi offline** (audio melewati Cloudflare): hanya untuk data fiktif.

```
 Laptop dokter (browser, spesifikasi rendah)            Laptop server (GPU 4–6 GB)
 ┌──────────────────────────────┐   HTTPS/LAN   ┌────────────────────────────────────────────┐
 │ AudioWorklet → WAV 16k mono  │ ────────────▶ │ FastAPI ─ JobQueue (1 worker, FIFO)        │
 │ Waveform, Queue Monitor      │ ◀──────────── │  S1 Spectral gating + Silero VAD (ONNX)    │
 │ Review: chat ⟷ form schema   │   polling     │  S2 Faster-Whisper → segmentasi utterance  │
 └──────────────────────────────┘               │     + [ML-1] Speaker Role (anchor-relative)│
                                                │  PII scrub (sebelum apa pun disimpan)      │
                                                │  S3 [ML-2] Intent S/O/A/P/CHITCHAT          │
                                                │     + Medical Lexicon Mapper               │
                                                │  S4 LLM (Ollama/llama.cpp) + JSON Schema   │
                                                │  S5 HITL verifikasi → SQLite + active learn│
                                                └────────────────────────────────────────────┘
          GpuModelSlot: Whisper dan LLM tidak pernah berada di VRAM bersamaan (policy=swap)
```

---

## 1. Koreksi terhadap spesifikasi awal (dan alasannya)

| # | Spesifikasi awal | Implementasi | Alasan teknis |
|---|---|---|---|
| 1 | Classifier peran dilatih di atas MFCC/F0 absolut + cosine anchor | Fitur **relatif terhadap anchor dokter** (\|z\| per dimensi) + **relatif-sesi** (dikurangi median sesi, rank, cluster KMeans 2) | Fitur absolut mengkodekan *identitas* speaker & mikrofon, bukan *peran*. Model akan menghafal suara dokter di data latih dan gagal pada dokter baru. Fitur relatif mengubah tugas menjadi *speaker verification* terhadap anchor → bisa digeneralisasi. Ini diuji di `test_trained_model_generalizes_to_unseen_doctor` dan dibandingkan lewat ablation di script training. |
| 2 | CMVN saat ekstraksi MFCC | **Tidak** dipakai di jalur anchor (opsi tetap ada di `extract_segment_features(apply_cmvn=True)`) | CMVN per-rekaman menghapus mean cepstral = informasi yang justru dipakai anchor. Kompensasi domain shift dilakukan oleh fitur relatif-sesi (dokter & pasien berbagi kanal yang sama) + augmentasi. |
| 3 | Fitur speaker dari audio hasil denoise | Fitur speaker dari **audio mentah**; denoise hanya untuk VAD/ASR | Spectral gating mengubah MFCC/centroid secara tidak konsisten antar segmen (tergantung SNR lokal). Pada uji sintetis, separasi jarak-ke-anchor antara dokter dan pasien hilang setelah denoise (anchor dari audio mentah). |
| 4 | `asyncio.Lock()` | **Single-consumer FIFO queue** + `asyncio.to_thread` | Jaminan eksklusif GPU sama, plus posisi antrean untuk Queue Monitor; event loop tetap melayani upload/polling laptop lain. |
| 5 | PII scrubber di Stage 5 | Scrub **langsung setelah ASR**, diulang saat verifikasi; audio mentah **dihapus** setelah diproses (`APP_KEEP_RAW_AUDIO=false`) | Prinsip *privacy by design*: transkrip ber-PII tidak pernah masuk DB. Suara = data biometrik. |
| 6 | Pydantic + Outlines | JSON Schema langsung ke **Ollama `format`** / **llama-cpp `response_format`** | Keduanya sudah melakukan grammar-constrained decoding; Outlines menambah dependency tanpa manfaat tambahan di sini. |
| 7 | WhisperX untuk alignment | **Word timestamps faster-whisper** + segmentasi (batas VAD, jeda > 0.6 s, tanda baca) | WhisperX menarik torch + model wav2vec2 (VRAM tambahan). Untuk 1 mikrofon dengan SOP giliran bicara, word timestamps cukup. Overlap speech tetap tidak tertangani (Gap 1). |
| 8 | RNNoise | Spectral gating stationary (numpy/scipy) | RNNoise butuh binding C; sering bermasalah di Windows. |
| 9 | librosa | Ekstraktor numpy/scipy sendiri (MFCC HTK-mel + DCT-II, delta, F0 autokorelasi, centroid, ZCR, RMS) | Menghilangkan dependency berat (numba). Nilai tidak identik bit-per-bit dengan librosa; train & inference memakai ekstraktor yang sama sehingga konsisten. |
| 10 | Mapping "nggliyeng" → "Vertigo atau Cephalgia" | Mapping ke **gejala** ("pusing berputar (nggliyeng)"), teks asli dipertahankan | Mengubah istilah awam langsung menjadi diagnosis adalah keputusan klinis, bukan normalisasi bahasa → sumber halusinasi diagnostik. |
| 11 | Estimasi proses 15–30 detik | **Belum dapat dipastikan** | Tergantung durasi rekaman, model, dan GPU. Setiap job mencatat `timings` per tahap (panel review → "Timing per tahap") untuk diukur. |
| 12 | Qwen-2.5-7B Q4_K_M untuk 4 GB | Default **qwen2.5:3b** | File 7B Q4_K_M ≈ 4.7 GB (perkiraan) → tidak muat penuh di 4 GB, sebagian layer jatuh ke CPU (lambat). 7B realistis mulai 6–8 GB. |

Tambahan anti-halusinasi: setiap butir naratif rangkuman wajib membawa `sumber` (ID utterance). ID yang
tidak ada dibuang dan butir tanpa sumber ditandai "wajib dicek manual". Klik chip `#id` di form →
bubble transkrip terkait disorot.

---

## 2. Struktur proyek

```
app/
  main.py                FastAPI app, auth token, routing halaman
  config.py              Settings dari env/.env
  api/                   routes_jobs, routes_doctors, routes_misc (template, kasus, metrik, health)
  core/                  audio_io, job_queue, model_slot (memory swap), services (komposisi)
  pipeline/              stage1_preprocess, stage2_asr, stage4_summarize, stage5_finalize, orchestrator
  ml/                    audio_features, speaker_role [ML-1], text_intent [ML-2], augment, evaluation,
                         speaker_embedding (ECAPA opsional)
  nlp/                   lexicon (mapper), pii (scrubber)
  llm/                   backends (Ollama, llama-cpp-python)
  schemas/               pipeline (Utterance, Result), templates (SOAP, Calgary-Cambridge, OSCE)
  static/                UI web (HTML + ES modules, tanpa framework/build step)
data/lexicon, data/glossary, data/synthetic
scripts/                 generate_synthetic_text, generate_synthetic_llm, train_text_intent, train_speaker_role
tests/                   23 unit/integration test (CPU, tanpa GPU/model)
```

---

## 3. Instalasi server (laptop dengan GPU)

Cara otomatis di Windows: `scripts\windows\setup_lab.ps1` (lihat `docs/INSTALASI_LAB.md`). Langkah manual:

**Prasyarat:** Python 3.10/3.11, driver NVIDIA terbaru, ffmpeg di PATH (hanya untuk unggah non-WAV).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-optional.txt    # matplotlib + pytest
```

- **faster-whisper GPU** (CTranslate2 ≥ 4.5) butuh cuBLAS untuk CUDA 12 dan cuDNN 9. Cara termudah:
  `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*` lalu pastikan DLL/so-nya ada di PATH/LD_LIBRARY_PATH
  (lihat README faster-whisper bagian *GPU*). Bila bermasalah: `APP_WHISPER_DEVICE=cpu` (model `small` int8
  masih layak di CPU, lebih lambat).
- **LLM (default Ollama):** pasang Ollama, lalu `ollama pull qwen2.5:3b`. Server Ollama berjalan di
  `127.0.0.1:11434`. Pastikan versi Ollama mendukung *structured outputs* (`format` = JSON Schema).
- Model Whisper diunduh otomatis saat pertama dipakai ke `models/whisper/` → **lakukan sekali saat
  online**, setelah itu offline.

**Panduan pemilihan model per VRAM (perkiraan, verifikasi dengan `nvidia-smi`):**

| VRAM | Whisper | LLM | `APP_GPU_SWAP_POLICY` |
|---|---|---|---|
| 4 GB | `small` int8_float16 (atau CPU) | `qwen2.5:3b` | `swap` |
| 6 GB | `small`/`medium` int8_float16 | `qwen2.5:3b` (7B dengan offload parsial) | `swap` |
| ≥ 12 GB | `medium`/`large-v3` | `qwen2.5:7b` | `coexist` |

Catatan: pelepasan VRAM CTranslate2 setelah model dihapus **belum dapat dipastikan** 100% di semua versi;
cek `nvidia-smi` setelah satu job selesai. Bila VRAM tidak turun, jalankan Whisper di CPU atau naikkan VRAM.

### Melatih model ML awal (cold start)

```bash
python scripts/generate_synthetic_text.py --n-per-template 25
python scripts/train_text_intent.py --data data/synthetic/text_intent.jsonl
# opsional (variasi bahasa jauh lebih kaya; butuh Ollama):
python scripts/generate_synthetic_llm.py --n-dialogs 60
python scripts/train_text_intent.py --data data/synthetic/text_intent.jsonl data/synthetic/text_intent_llm.jsonl
```

Speaker Role Classifier **tidak bisa** dilatih dari data sintetis teks; sebelum ada data, server memakai
heuristik jarak-anchor (ditandai `heuristic` di UI). Lihat §6.

---

## 4. Menjalankan untuk akses lintas laptop (LAN)

1. Server & client di Wi-Fi/hotspot yang sama. Cek IP server (`ipconfig` / `ip a`), misal `192.168.1.15`.
   Sebaiknya reservasi IP statis di router (sertifikat terikat ke IP).
2. Buka port 8000 di firewall server (Windows Defender Firewall → Inbound Rule → TCP 8000).
3. **HTTPS lokal (disarankan) — mkcert:**
   ```bash
   mkcert -install
   mkcert -key-file key.pem -cert-file cert.pem 192.168.1.15 localhost 127.0.0.1
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --ssl-keyfile key.pem --ssl-certfile cert.pem
   ```
   Di **setiap laptop client**, pasang `rootCA.pem` dari folder `mkcert -CAROOT` (server) ke
   *Trusted Root Certification Authorities* (Windows) / keychain (macOS). Tanpa ini browser menampilkan
   peringatan sertifikat.
4. **Alternatif uji cepat (HTTP):** `uvicorn app.main:app --host 0.0.0.0 --port 8000`, lalu di Chrome client
   buka `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, isi `http://192.168.1.15:8000`,
   Enabled, relaunch. Data audio berjalan tanpa enkripsi → hanya untuk uji coba.
5. **Jangan** pakai `--workers > 1`: tiap worker = proses terpisah = model terpisah di VRAM → OOM.
6. Set `APP_ACCESS_TOKEN` di `.env`; browser akan meminta token sekali (disimpan di localStorage).

Halaman: `/` rekam sesi · `/enroll` enrollment · `/review?job=<id>` · `/cases` bank kasus · `/metrics` evaluasi ML ·
`/docs` dokumentasi API (OpenAPI).

---

## 5. Alur pemakaian

1. **Enrollment** (`/enroll`): bacakan teks standar 15 detik **dengan mikrofon & posisi yang sama** seperti
   saat konsultasi. Server menyimpan profil fitur (`runtime/anchors/<id>.npz`), bukan audionya.
2. **Konfigurasi sesi** (`/`): pilih dokter, template (SOAP Umum / Calgary-Cambridge / OSCE Respirasi /
   OSCE Kardiovaskular), opsional kunci parameter (onset, skala nyeri, sistem organ, dst.), isi nama yang
   wajib dianonimkan (meningkatkan recall PII).
3. **Rekam** → "Selesai & Proses". Status antrean & tahap terlihat langsung.
4. **Review** (`/review?job=…`): kiri chat bubble (klik peran untuk menukar, ganti kelas via dropdown,
   edit teks); kanan form terstandar dari JSON Schema. Panel ML menampilkan confidence & probabilitas per
   kelas; confidence < 65% diberi warna merah.
5. **Verifikasi & Ekspor**: data terkunci ke `osce_cases` (dual view: transkrip + kunci jawaban), dan
   koreksi label masuk ke `runtime/training_data/*.jsonl` (active learning).

---

## 6. Protokol training & evaluasi (untuk laporan mata kuliah)

Tutorial pengujian data nyata: [`docs/TUTORIAL_TESTING.md`](docs/TUTORIAL_TESTING.md). Protokol evaluasi:
[`docs/EVALUASI.md`](docs/EVALUASI.md). Hasil terbaru: [`docs/LAPORAN_EVALUASI.md`](docs/LAPORAN_EVALUASI.md).

**Modul 2 – Intent (teks).** `train_text_intent.py` melakukan split **berbasis grup** (template/dialog/job),
GroupKFold-5, perbandingan dengan baseline *majority* & *keyword rule*, perbandingan LogReg vs LinearSVC,
dan ablation fitur (word / char / role). Contoh hasil pada data template saja (template di test tidak
pernah dilihat saat training): macro-F1 ≈ 0.63 (CV ≈ 0.72), baseline keyword ≈ 0.58, majority ≈ 0.11.
Angka ini sengaja jujur: data template punya variasi terbatas. Protokol yang paling valid untuk laporan:

```bash
python scripts/train_text_intent.py \
  --data data/synthetic/text_intent.jsonl data/synthetic/text_intent_llm.jsonl runtime/training_data/text_intent.jsonl \
  --test-source human_verified
```
(latih di sintetis, uji di percakapan nyata hasil verifikasi manusia).

**Modul 1 – Speaker role (audio).** Kumpulkan data role-play: idealnya **≥ 3 dokter berbeda** (agar split
berbasis dokter bisa menguji generalisasi ke dokter baru), masing-masing beberapa sesi dengan pasien
berbeda. Dua jalur data:
- Otomatis: setiap sesi yang diverifikasi di UI → `runtime/training_data/speaker_role.jsonl`.
- Manual: rekaman + label segmen dalam `manifest.jsonl` (format di docstring script), mendukung `--augment N`
  (noise, reverb, speed/pitch perturbation, gain).

```bash
python scripts/train_speaker_role.py --data runtime/training_data/speaker_role.jsonl --model svm
python scripts/train_speaker_role.py --manifest data/labeled/manifest.jsonl --augment 2 --model rf
```
Output: perbandingan SVM/RF/MLP/LogReg + baseline heuristik + ablation **fitur relatif vs absolut**
(bukti empiris koreksi #1). Setelah training: klik "Muat ulang model" di `/metrics` (tanpa restart).

`metrics.json` dan `confusion_matrix.png` (bila matplotlib terpasang) tersimpan di `models/<modul>/`.

**Pedoman anotasi singkat:** pertanyaan anamnesis dokter = `S_KELUHAN`; instruksi saat pemeriksaan
("tarik napas") = `O_PEMERIKSAAN`; pertanyaan pasien soal obat = `P_EDUKASI_TERAPI`; "iya/baik dok" tanpa
isi = `CHITCHAT`, "iya dok, ada demam" = `S_KELUHAN`.

---

## 7. Trade-off utama

| Keputusan | Pro | Kontra / bottleneck |
|---|---|---|
| Polling HTTP (2–3 s) vs WebSocket | Sederhana, tahan reconnect Wi-Fi | Latensi status ±3 s, request ringan berulang (`?light=true`) |
| Swap model per job | Muat di 4–6 GB | Overhead load model tiap job (lihat `models.load_*` di timing); `coexist` menghilangkannya bila VRAM cukup |
| Batch (rekam dulu, proses kemudian) vs streaming real-time | Akurasi ASR lebih baik (konteks penuh), satu GPU melayani banyak client | Hasil baru tersedia setelah rekaman selesai |
| ECAPA opsional (CPU) | Embedding speaker lebih diskriminatif | Menarik torch (~2 GB disk); model yang dilatih dengan ECAPA wajib punya ECAPA saat inference |
| Rule-based PII | Deterministik, tanpa model | Recall nama terbatas → verifikasi manusia diwajibkan sebelum ekspor |
| SQLite | Nol konfigurasi, satu file | Satu penulis per waktu (cukup untuk skala lab); pindah PostgreSQL bila multi-server |

**Gap yang belum diselesaikan:** overlapping speech (butuh separation model, VRAM tambahan), code-switching di
luar kamus lexicon (perluas `data/lexicon/medical_lexicon.json`), dan bias data sintetis.

---

## 8. Konfigurasi (env / `.env`)

Lihat `.env.example` dan `app/config.py`. Yang paling sering diubah: `APP_WHISPER_MODEL`, `APP_WHISPER_DEVICE`,
`APP_LLM_BACKEND` (`ollama`/`llamacpp`/`none`), `APP_OLLAMA_MODEL`, `APP_GPU_SWAP_POLICY`, `APP_ACCESS_TOKEN`,
`APP_USE_ECAPA`, `APP_KEEP_RAW_AUDIO`. Dengan `APP_LLM_BACKEND=none` rangkuman diisi rule-based dari hasil
klasifikasi Stage 3 (berguna untuk demo modul ML tanpa LLM).

## 9. Testing

```bash
pytest -q        # 23 test, CPU saja: fitur audio, generalisasi speaker ke dokter baru, intent, lexicon,
                 # PII, schema, retry/fallback LLM, memory swap, antrean, end-to-end + verifikasi + training
```
Lapisan HTTP (FastAPI) dan Whisper/Ollama asli tidak tercakup unit test — uji manual di server.
