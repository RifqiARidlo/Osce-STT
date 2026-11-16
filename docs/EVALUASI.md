# EVALUASI — panduan step by step (Fase 5)

Panduan ini menjelaskan cara **mereproduksi** seluruh evaluasi dua modul ML yang dilatih sendiri.
Hasil angka dan interpretasinya ada di [`LAPORAN_EVALUASI.md`](LAPORAN_EVALUASI.md).

Semua perintah dijalankan dari root repo di laptop server, dengan venv aktif
(`.venv\Scripts\activate` di Windows). Contoh memakai `python`; di Git Bash pakai `.venv/Scripts/python.exe`.

| Modul | Tugas | Input | Script |
|---|---|---|---|
| ML-1 Speaker Role | DOKTER vs PASIEN per utterance | fitur akustik relatif-anchor dokter | `scripts/train_speaker_role.py` |
| ML-2 Clinical Intent | S_KELUHAN / O_PEMERIKSAAN / A_DIAGNOSIS / P_EDUKASI_TERAPI / CHITCHAT | teks + peran | `scripts/train_text_intent.py` |

---

## 0. Prasyarat

```bash
pip install -r requirements.txt -r requirements-optional.txt   # matplotlib untuk confusion_matrix.png
pytest -q                                                       # harus hijau sebelum mulai
ollama list                                                     # perlu qwen2.5:7b untuk langkah 1b
```

Pastikan `runtime/training_data/` **tidak** berisi data smoke test (Fase 1). Data itu sudah dipindah ke
`runtime/_backup_smoke/` dan jangan ikut dilatih (label CHITCHAT dari transkrip rusak).

---

## 1. Data ML-2 (Intent)

Tiga sumber data, dibedakan oleh field `source` dan dikelompokkan oleh `template_id`/`job_id`:

| Sumber (`source`) | File | Grup split | Peran dalam evaluasi |
|---|---|---|---|
| `synthetic_template` | `data/synthetic/text_intent.jsonl` | `template_id` (66 template) | train |
| `synthetic_llm` | `data/synthetic/text_intent_llm.jsonl` | `template_id` = `llm_<seed>_<i>` (1 dialog = 1 grup) | train / uji domain-shift |
| `human_verified` | `runtime/training_data/text_intent.jsonl` | `job_id` (1 sesi = 1 grup) | **test set utama** |

### 1a. Data template (sudah ada)
```bash
python scripts/generate_synthetic_text.py --n-per-template 25
```

### 1b. Data dialog LLM
```bash
python scripts/generate_synthetic_llm.py --n-dialogs 60 --model qwen2.5:7b --seed 0
```
- Satu dialog = satu kombinasi unik (kasus × gaya pasien), diambil tanpa pengembalian (maks. 120).
- `temperature` 0.7 + `repeat_penalty` 1.1 + `num_predict` 2500. **Jangan** pakai temperature 0:
  qwen2.5:3b terbukti looping mengulang satu kalimat sampai timeout (root cause file LLM yang dulu kosong).
- Script menambah (`append`) ke file. Hapus file lama dulu bila ingin generate ulang dari nol.
- Waktu: ±28 s/dialog di RTX 4050 (7B) → 60 dialog ±30 menit. Jangan sleep laptop selama proses.

### 1c. Audit label LLM (wajib sebelum dipakai)
Label dari LLM belum tentu benar. Ambil 50 baris acak (seed tetap), periksa terhadap pedoman anotasi
(README §6), catat hasilnya, lalu terapkan koreksi:

```bash
python scripts/audit_llm_labels.py sample --n 50 --seed 7          # -> data/synthetic/audit_llm_labels.jsonl
# isi kolom "label_audit" (dan "catatan") untuk tiap baris pada file audit
python scripts/audit_llm_labels.py apply                            # terapkan koreksi + cetak akurasi label LLM
python scripts/audit_llm_labels.py fix-ack                          # "Ya dok"/"Oke" tanpa isi -> CHITCHAT (seluruh file)
```
Ketiga perintah idempoten. Label asli LLM disimpan di field `label_llm`.
Laporkan **akurasi label LLM pada sampel audit** (estimasi tingkat noise label) di laporan.

Pedoman anotasi ringkas:
- Pertanyaan anamnesis dokter → `S_KELUHAN`.
- Instruksi saat pemeriksaan ("tarik napas"), hasil tanda vital → `O_PEMERIKSAAN`.
- Simpulan/dugaan diagnosis → `A_DIAGNOSIS`.
- Obat, edukasi, rencana, rujukan, kontrol, pertanyaan pasien soal obat → `P_EDUKASI_TERAPI`.
- "iya/baik dok" tanpa isi, salam, terima kasih → `CHITCHAT`; "iya dok, ada demam" → `S_KELUHAN`.

### 1d. Data verifikasi manusia (role-play)
Terisi otomatis setiap sesi diverifikasi di halaman `/review` (lihat §3, langkah yang sama dengan ML-1).
Label = intent setelah dikoreksi manusia; kolom `corrected` mencatat apakah model salah.

---

## 2. Evaluasi ML-2 (Intent)

Setiap eksperimen menghasilkan `metrics.json` + `confusion_matrix.png` di folder `--out`.
Isi `metrics.json`: `test` (accuracy, macro/weighted F1, P/R/F1 per kelas, confusion matrix),
`cv` (GroupKFold-5 macro-F1 mean ± std pada train), `comparison` (baseline majority, keyword rule,
LogReg vs LinearSVC), `ablation` (word / char / role).

| ID | Pertanyaan | Perintah |
|---|---|---|
| E1 | Generalisasi ke template yang tidak dilihat | `python scripts/train_text_intent.py --data data/synthetic/text_intent.jsonl --out reports/text_intent/E1_template_only` |
| E2 | Template → dialog LLM (domain shift, proksi data nyata) | `python scripts/train_text_intent.py --data data/synthetic/text_intent.jsonl data/synthetic/text_intent_llm.jsonl --test-source synthetic_llm --no-refit --out reports/text_intent/E2_template_to_llm` |
| E3 | Gabungan sintetis, split grup | `python scripts/train_text_intent.py --data data/synthetic/text_intent.jsonl data/synthetic/text_intent_llm.jsonl --out models/text_intent` lalu salin `metrics.json` + `confusion_matrix.png` ke `reports/text_intent/E3_combined/` |
| E4 | **Protokol utama:** latih sintetis, uji data nyata | `python scripts/train_text_intent.py --data data/synthetic/text_intent.jsonl data/synthetic/text_intent_llm.jsonl runtime/training_data/text_intent.jsonl --test-source human_verified --out models/text_intent` |

Stabilitas split (E3 dengan beberapa seed; model tidak disimpan ke `models/`):
```bash
for s in 0 1 2 3 4; do
  python scripts/train_text_intent.py --data data/synthetic/text_intent.jsonl data/synthetic/text_intent_llm.jsonl \
    --seed $s --no-refit --out reports/_seed_tmp/s$s | grep TEST
done
```
Laporkan mean ± std macro-F1 test (bersama seed 42 dari E3), bukan satu angka.

Catatan:
- `models/` ada di `.gitignore`. Bukti untuk laporan disimpan di `reports/` (hanya `metrics.json` + PNG;
  hapus `model.joblib` dari `reports/`, model sklearn tidak portabel antar-versi).
- E3 menghasilkan model produksi sementara selama data nyata belum ada. Setelah role-play terkumpul,
  E4 menggantikan E3 di `models/text_intent/` (model tersimpan di-refit pada seluruh data).
- E4 butuh ≥ 20 baris `human_verified`; disarankan ≥ 300 baris dari ≥ 12 sesi agar semua kelas terwakili.
- `--test-source` menjadikan SELURUH baris sumber itu test set; train = sumber lain → tidak ada kebocoran.
- Split acak per baris **dilarang** (kalimat dari template/dialog yang sama bocor ke test).

---

## 3. Data ML-1 (Speaker Role) — SOP role-play

Data speaker **harus** dari suara manusia asli. Audio TTS/pitch-shift tidak dipakai untuk penilaian
karena terlalu mudah dipisahkan.

**Target minimum:** ≥ 3 "dokter" berbeda × ≥ 4 sesi tiap dokter (≥ 12 sesi), pasien bervariasi
(ganti orang yang berperan pasien antar-sesi; campur gender bila memungkinkan, termasuk pasangan
dokter–pasien bergender sama karena itu kasus tersulit).

Langkah per dokter:
1. Buka `https://<ip-server>:8000/enroll` → isi nama samaran (mis. "Dokter A", **jangan nama asli
   pasien**) → rekam 15–20 detik suara dokter sendirian di ruangan dan mikrofon yang sama dengan sesi.
2. Untuk tiap sesi (2–5 menit): halaman utama → pilih dokter → pilih template OSCE → rekam role-play.
   Skenario fiktif; pasien boleh memakai identitas karangan.
3. SOP rekaman: satu mikrofon/laptop di tengah, jarak 50–100 cm, beri jeda ±0.5 s saat bergantian bicara,
   hindari bicara bersamaan (overlap speech belum ditangani).
4. Buka `/review` untuk job tersebut → koreksi **peran** dan **intent** setiap utterance → klik
   **Verifikasi**. Baris otomatis ditambahkan ke:
   - `runtime/training_data/speaker_role.jsonl` (vektor fitur + label peran),
   - `runtime/training_data/text_intent.jsonl` (`source=human_verified`).
5. Catat di lembar kerja: ID sesi, dokter, pemeran pasien, durasi, jumlah koreksi peran/intent.

Jalur alternatif (manifest manual) untuk rekaman di luar aplikasi: lihat docstring
`scripts/train_speaker_role.py` (format `manifest.jsonl`, mendukung `--augment N`). Rekaman harus mentah
(tanpa denoise), konsisten dengan pipeline inference.

Cek kecukupan data sebelum training:
```bash
python -c "import json,collections as c; r=[json.loads(l) for l in open('runtime/training_data/speaker_role.jsonl',encoding='utf-8')]; print(len({x['session_id'] for x in r}),'sesi', len({x['doctor_id'] for x in r}),'dokter', c.Counter(x['label'] for x in r))"
```

---

## 4. Evaluasi ML-1 (Speaker Role)

```bash
python scripts/train_speaker_role.py --data runtime/training_data/speaker_role.jsonl --model svm
```
Script otomatis:
- split **berbasis dokter** bila ≥ 3 dokter (dokter di test tidak pernah dilihat saat training),
  selain itu berbasis sesi (dan menulis bahwa generalisasi ke dokter baru belum teruji);
- GroupKFold CV pada train;
- perbandingan SVM / RF / MLP / LogReg + baseline heuristik anchor (tanpa training);
- **ablation fitur relatif-anchor vs absolut (MFCC/F0 mentah)** → bukti empiris keputusan desain #1.

Ulangi dengan `--model rf` / `--model mlp` bila ingin memilih model lain; pilih berdasarkan macro-F1 CV,
bukan skor test (hindari memilih model dari test set).

---

## 5. Aktifkan model baru

Buka `/metrics` → klik **Muat ulang model** (atau `POST /api/ml/reload` dengan token). Tidak perlu restart.
Halaman `/metrics` menampilkan isi `models/*/metrics.json`.

## 6. Checklist laporan
- [ ] Tabel ringkasan data (jumlah baris/sesi/dokter per sumber, distribusi kelas).
- [ ] Akurasi audit label LLM (n = 50).
- [ ] ML-2: E1–E3 (+ E4 setelah data nyata): accuracy, macro-F1, CV mean ± std, per kelas, confusion matrix.
- [ ] ML-2: baseline (majority, keyword) dan ablation fitur.
- [ ] ML-1: test (split dokter), CV, perbandingan model, baseline heuristik, ablation relatif vs absolut.
- [ ] Keterbatasan: data sintetis, overlap speech, code-switching, jumlah dokter kecil.
