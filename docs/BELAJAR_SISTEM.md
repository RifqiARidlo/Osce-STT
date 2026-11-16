# BELAJAR SISTEM — cara kerja, alasan, dan parameter OSCE-STT

Dokumen belajar untuk anggota kelompok. Tujuannya agar bisa menjelaskan **apa** yang dilakukan setiap
bagian sistem, **jenis metode ML-nya apa**, **kenapa dipilih**, dan **parameter apa yang dipakai beserta
efeknya**. Semua angka parameter diambil langsung dari kode (lokasi file disebut di tiap bagian).

Urutan baca yang disarankan:
1. §1–2 (gambaran dan konsep dasar);
2. §3 (per tahap, sambil membuka file kodenya);
3. §4–5 (model kecil vs besar, data);
4. §8 (latihan tanya-jawab sebelum presentasi).

---

## 1. Gambaran besar

**Masalah.** Saat ujian OSCE atau konsultasi, dokter harus mendengar, bertanya, memeriksa, lalu
menulis dokumentasi (SOAP). Sistem ini merekam percakapan dan menghasilkan:
1. transkrip yang sudah dipisah per pembicara;
2. label tiap kalimat (keluhan / pemeriksaan / diagnosis / terapi / basa-basi);
3. rangkuman terstruktur sesuai template, yang diverifikasi manusia lalu disimpan ke Bank Kasus OSCE.

**Alur (pipeline).** Satu rekaman diproses berurutan; tiap tahap memakai metode yang paling cocok.
```
Rekaman WAV 16 kHz
  │
  ├─ S1a Spectral gating ............ DSP klasik (tanpa training) ─────────── hapus derau stasioner
  ├─ S1b Silero VAD ................. neural net kecil PRETRAINED ─────────── di mana ada suara?
  ├─ S2a Whisper small .............. transformer besar PRETRAINED ────────── suara → teks + waktu tiap kata
  ├─ S2b Segmentasi ucapan .......... aturan (rule-based) ─────────────────── kata → kalimat/ucapan
  ├─ S2c [ML-1] Speaker Role ........ SUPERVISED (+ fitur UNSUPERVISED) ──── tiap ucapan: DOKTER/PASIEN   ← dilatih sendiri
  ├─ PII scrub ...................... aturan regex ─────────────────────────── sensor nama, NIK, telepon
  ├─ S3 [ML-2] Clinical Intent ...... SUPERVISED ───────────────────────────── tiap ucapan: S/O/A/P/CHITCHAT ← dilatih sendiri
  │    └ Lexicon mapper ............. kamus ────────────────────────────────── "nggliyeng" → "pusing berputar (nggliyeng)"
  ├─ S4 LLM qwen2.5:3b .............. LLM GENERATIF PRETRAINED + JSON Schema ─ rangkuman terstruktur
  └─ S5 Review manusia .............. human-in-the-loop ───────────────────── koreksi → Bank Kasus + data training baru
```
File utama alur: `app/pipeline/orchestrator.py`.

**Yang "dilatih sendiri" (syarat mata kuliah) hanya dua:** ML-1 dan ML-2. Model lain dipakai apa adanya
(*pretrained*), karena melatih ASR atau LLM sendiri butuh data dan komputasi ribuan kali lebih besar.

---

## 2. Konsep dasar ML yang dipakai

| Konsep | Arti singkat | Di mana dipakai |
|---|---|---|
| **Supervised learning** | Belajar dari contoh berlabel (input → jawaban benar) | ML-1 (label DOKTER/PASIEN), ML-2 (label S/O/A/P/CHITCHAT) |
| **Unsupervised learning** | Mencari struktur tanpa label | KMeans k=2 di ML-1: mengelompokkan ucapan satu sesi menjadi 2 kelompok suara, lalu hasilnya dipakai sebagai fitur |
| **Pretrained / transfer** | Model sudah dilatih pihak lain di data besar, kita pakai langsung | Silero VAD, Whisper, qwen2.5 |
| **Generative model** | Menghasilkan teks/keluaran baru, bukan memilih label | LLM (rangkuman); juga dipakai membuat data latih sintetis |
| **Feature engineering** | Mengubah data mentah menjadi angka bermakna sebelum masuk model | ML-1: 89 fitur akustik → 95 fitur relatif; ML-2: TF-IDF |
| **Train/test split** | Data uji tidak boleh terlihat saat training | Semua evaluasi |
| **Data leakage** | Informasi test bocor ke train sehingga skor terlalu bagus | Dicegah dengan **split berbasis grup**: satu template/dialog/sesi/dokter hanya ada di train **atau** test |
| **Cross-validation (GroupKFold-5)** | Train set dibagi 5, bergiliran jadi validasi → mean ± std | Mengukur kestabilan model, dan memilih model tanpa mengintip test |
| **Baseline** | Pembanding paling sederhana | Majority class, keyword rule, heuristik anchor |
| **Ablation** | Mencabut satu komponen untuk melihat kontribusinya | ML-2: word/char/role; ML-1: fitur relatif vs absolut |
| **Class imbalance** | Jumlah contoh per kelas tidak seimbang | `class_weight="balanced"`: kesalahan pada kelas kecil diberi bobot lebih besar |
| **Kalibrasi probabilitas** | Agar skor keyakinan model bermakna sebagai peluang | SVM + Platt scaling (`CalibratedClassifierCV`) untuk confidence di UI |
| **Human-in-the-loop / active learning** | Manusia mengoreksi output model; koreksi menjadi data latih berikutnya | Halaman review → `runtime/training_data/*.jsonl` |

### Metrik
- **Accuracy** = proporsi prediksi benar. Menyesatkan bila kelas tidak seimbang. Contoh: menebak
  "S_KELUHAN" untuk semua kalimat sudah memberi accuracy 0.36, padahal modelnya tidak belajar apa-apa.
- **Precision** kelas X = dari semua yang diprediksi X, berapa yang benar X ("seberapa bisa dipercaya").
- **Recall** kelas X = dari semua yang sebenarnya X, berapa yang tertangkap ("seberapa lengkap").
- **F1** = rata-rata harmonik precision dan recall.
- **Macro-F1** = rata-rata F1 semua kelas dengan bobot sama. Ini **metrik utama** di proyek ini, karena
  kelas kecil (mis. A_DIAGNOSIS) sama pentingnya dengan kelas besar.
- **Weighted-F1** = rata-rata F1 dibobot jumlah contoh.
- **Confusion matrix** = tabel label sebenarnya (baris) × prediksi (kolom). Menunjukkan kelas mana yang
  sering tertukar.

---

## 3. Per tahap: apa, kenapa, parameter

### S0. Format rekaman: WAV 16 kHz mono
- Browser merekam lewat AudioWorklet, lalu mengirim WAV PCM 16-bit 16 kHz (`app/static/js/`).
- **Kenapa 16 kHz:** energi penting suara manusia ada di bawah 8 kHz (teorema Nyquist: sampling 16 kHz
  menangkap frekuensi sampai 8 kHz). Whisper dan Silero juga dilatih pada 16 kHz. Sampling lebih tinggi
  hanya memperbesar file. Ukurannya ≈ 1.9 MB per menit.

### S1a. Spectral gating (pengurangan derau) — DSP klasik
File: `app/pipeline/stage1_preprocess.py::spectral_gate`
1. Audio diubah ke domain frekuensi dengan **STFT**: jendela `nperseg=512` sampel (32 ms),
   `noverlap=384` (pergeseran 8 ms).
2. **15% frame paling sunyi** (`noise_quantile=0.15`) dianggap "derau saja". Dari frame itu dihitung
   rata-rata dan standar deviasi energi per frekuensi = **profil derau**.
3. Ambang per frekuensi = `mean + 1.5 × std` (`n_std=1.5`). Komponen di atas ambang = suara.
4. Mask dihaluskan (`uniform_filter 3×5`) untuk mengurangi artefak "musical noise".
5. Komponen di bawah ambang diredam **85%** (`prop_decrease=0.85`), tidak dihapus total, supaya suara
   tidak terdengar "robotik".

| Kenapa metode ini | Kelemahan |
|---|---|
| Tanpa training, cepat (±1 s untuk 2 menit audio), tanpa dependensi berat. Efektif untuk derau **stasioner** (AC, kipas, dengung). RNNoise sempat dipertimbangkan, tapi butuh binding C yang sering bermasalah di Windows. | Tidak efektif untuk derau tidak stasioner (orang lain bicara, pintu). |

**Aturan penting (keputusan desain #3):** hasil denoise **hanya** dipakai untuk VAD dan Whisper. Fitur
speaker (ML-1) diambil dari audio **mentah**, karena spectral gating mengubah MFCC secara tidak konsisten
antar-segmen (tergantung SNR lokal) dan merusak perbandingan dengan profil dokter.

### S1b. Silero VAD (Voice Activity Detection) — neural net kecil pretrained
File: `stage1_preprocess.py::silero_vad`; parameter di `app/config.py`.
- Model ONNX `silero_vad_v6.onnx` (**1.2 MB**) bawaan faster-whisper, dijalankan **tanpa PyTorch**.
- Output: daftar wilayah waktu yang berisi suara manusia.

| Parameter | Nilai | Efek bila dinaikkan |
|---|---|---|
| `APP_VAD_THRESHOLD` | 0.5 | Lebih ketat: suara pelan bisa terlewat |
| `APP_VAD_MIN_SILENCE_MS` | 300 | Jeda harus lebih lama untuk dianggap pemisah → ucapan lebih panjang |
| `vad_min_speech_ms` | 250 | Bunyi pendek (batuk, "eh") lebih banyak diabaikan |
| `vad_speech_pad_ms` | 100 | Tepi wilayah suara diperlebar → awal/akhir kata tidak terpotong |

Fallback: bila Silero gagal dimuat, dipakai VAD berbasis energi (`energy_vad`).

### S2a. Whisper `small` (ASR, speech-to-text) — transformer pretrained
File: `app/pipeline/stage2_asr.py`; implementasi **faster-whisper** (CTranslate2).
- Arsitektur encoder–decoder transformer dari OpenAI, dilatih pada ±680 ribu jam audio multibahasa.
  Ukuran `small` = **244 juta parameter**.

| Ukuran Whisper | Parameter | Catatan |
|---|---|---|
| tiny / base | 39 M / 74 M | Cepat, tapi akurasi bahasa Indonesia rendah |
| **small (dipakai)** | **244 M** | Kompromi akurasi vs VRAM. Puncak VRAM +570 MiB, RTF ±0.1 di RTX 4050 |
| medium / large | 769 M / 1.55 B | Lebih akurat, tapi VRAM dan waktu berlipat; sulit berbagi GPU 6 GB dengan LLM |

| Parameter | Nilai | Alasan |
|---|---|---|
| `compute_type` | `int8_float16` (GPU), `int8` (CPU) | **Kuantisasi**: bobot disimpan 8-bit, hitungan 16-bit → VRAM ±½ dan lebih cepat, penurunan akurasi kecil |
| `language` | `id` | Mencegah salah deteksi bahasa |
| `beam_size` | 5 | Decoder menyimpan 5 kandidat kalimat terbaik, bukan 1 (greedy). Lebih akurat, sedikit lebih lambat |
| `temperature` | (0.0, 0.2, 0.4) | Mulai deterministik; naik hanya bila hasil gagal (fallback) |
| `vad_filter` | True | Bagian sunyi dilewati → mencegah *halusinasi* teks saat hening |
| `condition_on_previous_text` | False | Tidak memakai teks segmen sebelumnya sebagai konteks → mencegah loop mengulang kalimat |
| `word_timestamps` | True | Waktu tiap kata, dibutuhkan untuk memotong per ucapan dan mengambil audio per ucapan |
| `initial_prompt` | Glossary medis (`data/glossary/medical_glossary.txt`, maks. 700 karakter ≈ 224 token) | Memberi "petunjuk ejaan" istilah medis (amoksisilin, ronki, ...) |

Hasil di audio TTS: WER 3–15% (`docs/benchmark.md`). Nama obat masih sering terpecah ("amok sisi lin").

### S2b. Segmentasi kata → ucapan — rule-based
File: `stage2_asr.py::segment_words`. Kata digabung jadi satu ucapan, lalu dipotong bila:
- pindah wilayah VAD;
- jeda antar-kata **> 0.6 s** (`APP_UTTERANCE_GAP_S`);
- kalimat berakhir tanda baca dan jeda > 0.2 s;
- durasi > 15 s (`utterance_max_s`).

**Kenapa:** sistem memakai **satu mikrofon**, jadi pergantian pembicara hanya bisa ditebak dari jeda.
Karena itu SOP rekaman meminta jeda ±0.5 s saat bergantian. **Keterbatasan:** ucapan yang tumpang
tindih (*overlap speech*) tergabung jadi satu ucapan dengan satu peran.

### S2c. [ML-1] Speaker Role Classifier (DOKTER vs PASIEN) — dilatih sendiri
File: `app/ml/audio_features.py`, `app/ml/speaker_role.py`, training `scripts/train_speaker_role.py`.

**Ide utama (keputusan desain #1).** Model tidak belajar "suara dokter terdengar seperti apa", karena
itu sama dengan menghafal suara orang di data latih dan akan gagal pada dokter baru. Model belajar
**"seberapa mirip ucapan ini dengan profil suara dokter yang sedang bertugas"**. Profil itu (*anchor*)
direkam saat enrollment. Tugasnya berubah menjadi *speaker verification* terhadap anchor, yang bisa
dipakai pada dokter yang belum pernah dilihat.

**Langkah 1: 89 fitur akustik per ucapan** (`FeatureConfig`):

| Fitur | Jumlah | Makna | Parameter |
|---|---|---|---|
| MFCC mean & std | 20 + 20 | "Warna suara" (bentuk saluran vokal) | 40 filter mel, 20 koefisien, frame 25 ms / hop 10 ms, FFT 512, 20–7600 Hz, pre-emphasis 0.97 |
| Delta MFCC mean & std | 20 + 20 | Perubahan warna suara dari waktu ke waktu (dinamika artikulasi) | regresi width 2 |
| F0 mean, std, voiced ratio | 3 | Tinggi nada suara (pitch) dan proporsi bagian bersuara | autokorelasi, 65–400 Hz, ambang voicing 0.45 |
| Spectral centroid mean & std | 2 | "Terang/gelap" suara | |
| ZCR mean & std | 2 | Seberapa sering sinyal melintasi nol (desis/frikatif) | |
| log-RMS mean & std | 2 | Kekerasan suara (juga terkait jarak ke mikrofon) | |

Diimplementasikan dengan numpy/scipy, tanpa librosa, untuk mengurangi dependensi. Training dan inference
**wajib** memakai ekstraktor yang sama.

**Langkah 2: anchor dokter.** Rekaman enrollment 15 detik dipotong per 2 s (geser 1 s). Dari 89 fitur
tiap potongan dihitung `mean` dan `scale` (std dengan batas bawah, agar tidak membagi angka ≈ 0).

**Langkah 3: 95 fitur relatif per ucapan** (`build_role_features`):
- 89 × **|z|** = |(fitur − mean_anchor) / scale_anchor|, dipotong maksimal 10. Artinya "berapa standar
  deviasi ucapan ini menjauh dari profil dokter" per dimensi.
- `dist_mean_abs_z`, `dist_rms_z`: ringkasan jarak ke anchor.
- `dist_rel_session`: jarak dikurangi **median jarak satu sesi**. Mengkompensasi mikrofon/ruangan,
  karena dokter dan pasien berbagi kanal yang sama.
- `dist_rank_pct`: peringkat jarak dalam sesi (0 = paling mirip dokter).
- `in_anchor_cluster`, `cluster_margin`: hasil **KMeans k=2 (unsupervised)** atas semua ucapan satu
  sesi. Cluster yang pusatnya paling dekat ke anchor dianggap "cluster dokter".

Fitur ke-3 sampai ke-5 bersifat **relatif-sesi**: membandingkan ucapan dengan ucapan lain di sesi yang
sama, bukan dengan angka absolut.

**Langkah 4: classifier** (`make_estimator`). Empat model dibandingkan; yang terbaik dipilih dari CV:

| Model | Parameter | Kenapa ada |
|---|---|---|
| **SVM RBF** (default) | `C=1.0`, `gamma="scale"`, `class_weight="balanced"`, + `StandardScaler`, + kalibrasi Platt (`cv=3`) | Kuat pada data kecil dan fitur kontinu; batas keputusan non-linear |
| Random Forest | 300 pohon, `min_samples_leaf=2`, balanced | Tahan outlier, tidak butuh scaling, ada *feature importance* |
| MLP | layer (64, 32), `alpha=1e-3`, early stopping | Neural net kecil, pembanding non-linear |
| Logistic Regression | balanced, + scaler | Baseline linear, mudah ditafsirkan |

- `StandardScaler` dipakai karena SVM, MLP, dan LogReg sensitif skala fitur; Random Forest tidak.
- `C` adalah kebalikan kekuatan regularisasi. C besar = model lebih "menghafal" data latih.

**Baseline tanpa training:** heuristik jarak anchor + cluster (`_heuristic`). Ini yang dipakai server
**sebelum** model ML-1 dilatih. Di `/api/health` tampil sebagai `heuristic-anchor-distance`.

**Augmentasi** (`app/ml/augment.py`, opsi `--augment N` pada jalur manifest):
- speed/pitch ±4–8% (peluang 50%);
- reverb RT60 0.15–0.5 s (30%);
- derau SNR 5–25 dB (70%);
- gain −8 s.d. +4 dB.

Tujuannya agar model tahan terhadap variasi ruangan dan mikrofon. Salinan augmentasi selalu berada di
grup yang sama dengan aslinya, supaya tidak bocor ke test.

**Evaluasi:**
- split **per dokter** bila ≥ 3 dokter;
- ablation **fitur relatif vs absolut** (MFCC/F0 mentah + SVM). Ini bukti empiris ide utama di atas.

**Keputusan terkait:**
- CMVN (normalisasi mean cepstral) tidak dipakai, karena menghapus informasi yang justru dibandingkan
  dengan anchor.
- Enrollment dan sesi harus memakai mikrofon dan posisi yang sama.

### PII scrub (anonimisasi) — rule-based
File: `app/nlp/pii.py`. Dijalankan **langsung setelah ASR, sebelum apa pun disimpan** (keputusan #4).
- Regex: email, NIK (16 digit), BPJS (13 digit), nomor telepon, nomor rekam medis, tanggal lahir,
  RT/RW, alamat.
- Nama setelah pemicu ("nama saya ...", "Pak/Bu <Kapital>").
- Nama yang diisi manual di UI.

**Kenapa rule-based:** deterministik dan bisa diaudit. **Keterbatasan:** recall nama terbatas, jadi
verifikasi manusia diwajibkan sebelum ekspor. Audio mentah dihapus setelah diproses
(`APP_KEEP_RAW_AUDIO=false`), karena suara termasuk data biometrik.

### S3. [ML-2] Clinical Intent Classifier — dilatih sendiri
File: `app/ml/text_intent.py`, training `scripts/train_text_intent.py`.

**Label (5 kelas):**
- `S_KELUHAN`: subjektif, anamnesis;
- `O_PEMERIKSAAN`: objektif, pemeriksaan dan tanda vital;
- `A_DIAGNOSIS`: assessment;
- `P_EDUKASI_TERAPI`: plan;
- `CHITCHAT`: basa-basi.

**Fitur:**
| Fitur | Parameter | Fungsi |
|---|---|---|
| TF-IDF kata | n-gram 1–2, `sublinear_tf`, maks. 50.000 fitur | Kata dan pasangan kata kunci ("tekanan darah", "kali sehari") |
| TF-IDF karakter (`char_wb`) | n-gram 3–5, `min_df=2`, maks. 80.000 | Tahan salah ketik/salah dengar ASR, singkatan, imbuhan, bahasa campur ("nyetuhin", "maketengah") |
| Peran (one-hot) | DOKTER/PASIEN dari ML-1 | Kalimat yang sama bisa beda makna tergantung siapa yang bicara |

TF-IDF = frekuensi istilah di kalimat × kebalikan frekuensi istilah di seluruh data. Kata umum ("saya",
"ya") mendapat bobot kecil; kata khas ("ronki", "antasida") mendapat bobot besar.

**Classifier:**
- **Logistic Regression** (default): `C=4.0`, `class_weight="balanced"`, `max_iter=3000`.
- Pembanding: **LinearSVC** `C=0.5` + kalibrasi.

**Kenapa TF-IDF + model linear, bukan BERT:**
- data latih kecil (±2.400 kalimat);
- training hitungan detik di CPU;
- inference milidetik;
- bobot bisa ditafsirkan;
- tidak memakai VRAM, yang dibutuhkan Whisper dan LLM.

IndoBERT tersedia sebagai opsi (`--embedding`), tapi menarik PyTorch (±2 GB).

**Fallback** tanpa model: aturan kata kunci (`rule_based_intent`), juga dipakai sebagai baseline.

**Lexicon mapper** (`app/nlp/lexicon.py`, `data/lexicon/medical_lexicon.json`) dijalankan setelah
klasifikasi. Istilah awam/daerah dipetakan ke **gejala**, bukan diagnosis, dan teks asli dipertahankan
dalam kurung: "nggliyeng" → "pusing berputar (nggliyeng)", **bukan** "vertigo" (keputusan #7). Menebak
diagnosis dari istilah awam adalah keputusan klinis, bukan normalisasi bahasa.

**Kenapa recall kelas klinis penting:** ucapan berlabel `CHITCHAT` **tidak dikirim ke LLM**
(`stage4_summarize.build_user_prompt`). Bila keluhan salah diberi label CHITCHAT, keluhan itu hilang dari
rangkuman. Kesalahan sebaliknya (basa-basi masuk sebagai keluhan) hanya menambah noise.

### S4. LLM qwen2.5:3b (rangkuman terstruktur) — generatif pretrained
File: `app/pipeline/stage4_summarize.py`, `app/llm/backends.py`, `app/schemas/templates.py`.
- Model: **Qwen2.5 3B Instruct** (±3 miliar parameter), kuantisasi 4-bit (file 1.9 GB), dijalankan oleh
  Ollama.

| Parameter | Nilai | Alasan |
|---|---|---|
| `temperature` | 0 | Deterministik: input sama → rangkuman sama. Mengurangi "kreativitas" yang berbahaya di dokumen medis |
| `num_ctx` (`APP_LLM_CTX`) | 8192 token | Cukup untuk transkrip ±10 menit + schema |
| `format` | JSON Schema template | **Constrained decoding**: LLM hanya bisa menghasilkan token yang membentuk JSON valid sesuai schema |
| Semua field `required` | `llm_json_schema()` | Tanpa ini, model kecil mengeluarkan objek minimal dan field lain terisi default (bug yang ditemukan di Fase 3) |
| `keep_alive` | 0 (mode swap) | Model langsung dikeluarkan dari VRAM setelah menjawab, memberi tempat untuk Whisper |
| Retry | 1× dengan pesan error | Bila JSON tidak lolos validasi Pydantic |
| Fallback | rangkuman rule-based | Sistem tidak pernah gagal total |

**Anti-halusinasi:**
1. System prompt melarang menambah informasi yang tidak diucapkan.
2. Setiap butir rangkuman wajib punya `sumber` (ID ucapan). ID yang tidak ada dibuang; butir tanpa
   sumber ditandai "wajib dicek manual" (keputusan #8).
3. Manusia memverifikasi sebelum ekspor.

### S5. Review manusia (human-in-the-loop)
Manusia memperbaiki peran, intent, teks, dan rangkuman, lalu memverifikasi. Hasilnya menjadi Bank Kasus
**dan** data latih baru (active learning). Kolom `model_label` mencatat tebakan model sebelum dikoreksi,
sehingga tingkat kesalahan model di lapangan bisa diukur.

### Infrastruktur yang membuat semua ini muat di satu GPU 4–6 GB
- **GpuModelSlot** (`app/core/model_slot.py`, `APP_GPU_SWAP_POLICY=swap`): Whisper dan LLM **tidak
  pernah** berada di VRAM bersamaan; yang satu dilepas sebelum yang lain dimuat. Puncak VRAM terukur
  ±2.5 GB. Trade-off: ada waktu muat model setiap job.
- **JobQueue** (`app/core/job_queue.py`): antrean FIFO dengan **satu** pekerja. Banyak laptop boleh
  mengunggah bersamaan, tapi GPU memproses satu per satu.
- **Uvicorn 1 worker** (keputusan #5): tiap worker = proses terpisah dengan salinan model sendiri →
  VRAM habis.

---

## 4. Kenapa ada model kecil dan model besar?

| Model | Ukuran | Jenis | Waktu per job | Dilatih sendiri | Tugas |
|---|---|---|---|---|---|
| ML-2 LogReg TF-IDF | bobot per fitur TF-IDF per kelas (file ±1.3 MB) | supervised klasik | milidetik (CPU) | Ya | klasifikasi teks |
| ML-1 SVM | 95 fitur, ratusan support vector | supervised klasik + KMeans | milidetik (CPU) | Ya | klasifikasi peran |
| Silero VAD | 1.2 MB | neural net kecil | < 1 s | Tidak | deteksi suara |
| Whisper small | 244 M parameter | transformer | ±0.1 × durasi audio (GPU) | Tidak | suara → teks |
| qwen2.5:3b | ±3 B parameter | LLM generatif | belasan–puluhan detik (GPU) | Tidak | rangkuman |
| qwen2.5:7b | ±7 B parameter | LLM generatif | ±28 s/dialog | Tidak | **hanya** membuat data latih sintetis (offline) |

**Prinsip: pakai model terkecil yang sanggup mengerjakan tugasnya.**
- **Klasifikasi** (memilih 1 dari 2 atau 5 label) cukup dengan model kecil. Model kecil cepat, murah,
  bisa dilatih dengan ratusan–ribuan contoh, hasilnya bisa diukur dengan metrik jelas (P/R/F1), dan bisa
  ditafsirkan. Ini yang diminta mata kuliah: melatih dan mengevaluasi model sendiri.
- **Persepsi suara** (ASR) butuh model besar pretrained, karena belajar mengenali bahasa dari nol butuh
  ratusan ribu jam audio.
- **Generasi teks** (rangkuman) butuh LLM, karena harus memahami konteks dan menyusun kalimat. LLM
  mahal (VRAM, waktu), bisa berhalusinasi, dan sulit diukur, jadi dipakai **paling akhir** dan **hanya
  untuk tugas yang tidak bisa dikerjakan model kecil**. Outputnya dikunci schema dan diverifikasi
  manusia.
- **3B vs 7B:** server memakai 3B karena harus bergantian dengan Whisper di GPU 6 GB dan cepat
  menjawab. Data sintetis dibuat dengan 7B karena di situ kualitas lebih penting daripada kecepatan.
  Terbukti: 3B dengan temperature 0 mengulang satu kalimat terus-menerus, 7B menghasilkan dialog valid
  60/60.
- **Model kecil membantu model besar:** ML-2 menyaring CHITCHAT, sehingga prompt LLM lebih pendek dan
  fokus; lexicon menerjemahkan istilah awam lebih dulu. Model besar juga membantu model kecil: LLM 7B
  membuat data latih untuk ML-2.

---

## 5. Data: dari mana dan kenapa begitu

| Data | Sumber | Kegunaan | Risiko |
|---|---|---|---|
| Template sintetis (1650 kalimat, 66 template) | `generate_synthetic_text.py` | Cold start ML-2 | Variasi sempit; skor terlalu optimistis bila split acak |
| Dialog LLM (789 kalimat, 60 dialog) | `generate_synthetic_llm.py` + qwen2.5:7b | Variasi bahasa lebih kaya | Label noise ±18% (audit), bias gaya LLM |
| Role-play terverifikasi | UI review | **Test set utama (E4)** dan data ML-1 | Butuh waktu manusia |

**Split berbasis grup:** 25 kalimat dari template yang sama hampir identik. Bila sebagian masuk train
dan sebagian test, model "sudah melihat jawabannya" → skor palsu tinggi. Karena itu satu
template/dialog/sesi/dokter selalu utuh di train **atau** test (keputusan #6).

Hasil evaluasi saat ini dan interpretasinya: [`LAPORAN_EVALUASI.md`](LAPORAN_EVALUASI.md).

---

## 6. Parameter yang paling sering perlu disetel

| Gejala | Parameter | Arah |
|---|---|---|
| Ucapan dua orang tergabung jadi satu | `APP_UTTERANCE_GAP_S` (0.6), `APP_VAD_MIN_SILENCE_MS` (300) | Turunkan sedikit (mis. 0.45 / 250) |
| Satu kalimat terpotong jadi banyak | sama | Naikkan |
| Suara pelan tidak tertranskrip | `APP_VAD_THRESHOLD` (0.5) | Turunkan (0.35–0.4) |
| Transkrip kurang akurat, GPU masih longgar | `APP_WHISPER_MODEL` | `small` → `medium` (cek VRAM & waktu) |
| Istilah medis salah eja terus | `data/glossary/medical_glossary.txt` | Tambahkan istilah (maks. ±700 karakter total) |
| Istilah awam tidak dinormalisasi | `data/lexicon/medical_lexicon.json` | Tambah entri (ke gejala, bukan diagnosis) |
| Rangkuman sering fallback | `APP_LLM_CTX`, model Ollama | Cek log; jangan naikkan ke 7B di GPU 6 GB tanpa uji VRAM |
| GPU ≥ 12 GB, ingin lebih cepat | `APP_GPU_SWAP_POLICY` | `coexist` (Whisper & LLM tetap di VRAM) |
| ML-1 overfit / underfit | `--model svm/rf/mlp/logreg` | Pilih dari **CV**, bukan test |

Setiap perubahan parameter harus diikuti uji ulang (pytest + 1–2 sesi nyata). Catat di laporan bila
memengaruhi angka.

---

## 7. Peta file (untuk membaca kode)

| File | Isi |
|---|---|
| `app/pipeline/orchestrator.py` | Urutan semua tahap dan pencatatan waktu |
| `app/pipeline/stage1_preprocess.py` | Spectral gating, VAD |
| `app/pipeline/stage2_asr.py` | Whisper, glossary, segmentasi |
| `app/ml/audio_features.py` | 89 fitur akustik |
| `app/ml/speaker_role.py` | Anchor, fitur relatif, classifier ML-1, heuristik |
| `app/ml/text_intent.py` | TF-IDF, classifier ML-2, fallback |
| `app/ml/evaluation.py` | Metrik, confusion matrix |
| `app/nlp/pii.py`, `app/nlp/lexicon.py` | Anonimisasi, normalisasi istilah |
| `app/pipeline/stage4_summarize.py` | Prompt, validasi, retry, fallback LLM |
| `app/core/model_slot.py`, `job_queue.py` | Swap VRAM, antrean |
| `scripts/train_*.py` | Training + evaluasi lengkap |
| `app/config.py` | Semua parameter `APP_*` dan default-nya |

---

## 8. Latihan tanya-jawab (persiapan presentasi)

**Kenapa tidak memakai speaker diarization siap pakai (pyannote, dll.)?**
Diarization hanya menjawab "pembicara A vs B", bukan "mana dokter". Model seperti itu juga menarik
PyTorch dan VRAM tambahan. Dengan anchor enrollment, sistem langsung tahu mana dokter, dan tugas ini
menjadi modul ML yang kami latih dan ukur sendiri.

**Kenapa fitur relatif, bukan langsung MFCC?**
MFCC absolut mengkodekan identitas suara dan mikrofon. Model akan menghafal dokter di data latih.
Fitur relatif terhadap anchor + sesi membuat model belajar "mirip dokter atau tidak", yang tetap berlaku
untuk dokter baru. Buktinya ada di ablation relatif vs absolut dan di test split-dokter.

**Di mana unsupervised learning-nya?**
KMeans k=2 per sesi di ML-1: tanpa label, ucapan dikelompokkan menjadi dua kelompok suara. Posisi tiap
ucapan terhadap cluster yang paling dekat ke anchor menjadi fitur untuk classifier supervised.

**Kenapa macro-F1, bukan accuracy?**
Kelas tidak seimbang. Model yang selalu menebak kelas terbesar bisa mendapat accuracy 0.36 dengan
macro-F1 hanya 0.10. Macro-F1 memberi bobot sama ke kelas kecil seperti A_DIAGNOSIS.

**Kenapa skor sintetis tidak boleh diklaim sebagai kinerja nyata?**
Data sintetis lebih bersih dan seragam dibanding percakapan asli dengan salah dengar ASR. Contohnya,
E1–E3 memakai teks bersih. Karena itu protokol utama adalah E4: latih sintetis, uji pada data nyata yang
diverifikasi manusia.

**Kenapa LLM tidak dipakai untuk semuanya (mis. klasifikasi intent)?**
- Lebih lambat (detik vs milidetik per kalimat);
- butuh VRAM yang sedang dipakai Whisper;
- tidak bisa dilatih ulang dengan murah dari koreksi pengguna;
- sulit dievaluasi dan dijelaskan;
- tidak memenuhi syarat "model yang dilatih sendiri".

Classifier kecil mengerjakan tugas itu lebih murah dan terukur.

**Bagaimana mencegah LLM mengarang?**
- Temperature 0;
- prompt melarang informasi tambahan;
- JSON Schema membatasi bentuk output;
- setiap butir wajib menyebut ID ucapan sumber (yang tidak valid dibuang);
- manusia memverifikasi sebelum ekspor.

**Apa keterbatasan utama?**
- Overlap speech (satu mikrofon);
- code-switching di luar lexicon;
- data latih sebagian besar sintetis;
- jumlah dokter role-play kecil;
- mode tunnel membuat sistem tidak offline.

---

## 9. Glosarium singkat

| Istilah | Arti |
|---|---|
| ASR | Automatic Speech Recognition, suara → teks |
| VAD | Voice Activity Detection, mendeteksi bagian yang berisi suara |
| STFT | Short-Time Fourier Transform, sinyal → spektrum per potongan waktu |
| MFCC | Mel-Frequency Cepstral Coefficients, ringkasan "warna suara" |
| F0 | Frekuensi dasar = tinggi nada suara |
| SNR | Signal-to-Noise Ratio, perbandingan kekuatan suara dan derau (dB) |
| RTF | Real-Time Factor, waktu proses / durasi audio |
| WER | Word Error Rate, (substitusi + hapus + sisip) / jumlah kata referensi |
| TF-IDF | Pembobotan kata: sering di kalimat ini, jarang di semua kalimat |
| Kuantisasi | Menyimpan bobot model dengan bit lebih sedikit (8/4 bit) agar hemat memori |
| Constrained decoding | Membatasi token yang boleh dihasilkan LLM agar output sesuai grammar/schema |
| Anchor | Profil statistik suara dokter dari enrollment |
| HITL | Human-in-the-loop, manusia memeriksa dan mengoreksi output model |
| PII | Personally Identifiable Information, data identitas pribadi |
