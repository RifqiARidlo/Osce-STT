# TUTORIAL TESTING — dari menyalakan server sampai angka laporan

Panduan praktis untuk hari pengujian dengan data nyata (role-play). Setiap langkah ditulis dengan
**Lakukan** (perintah/klik) dan **Fungsi** (kenapa langkah itu perlu), supaya alurnya bisa dipahami,
bukan hanya diikuti.

Dokumen terkait:
- [`EVALUASI.md`](EVALUASI.md): protokol evaluasi dan arti tiap eksperimen (E1–E4).
- [`LAPORAN_EVALUASI.md`](LAPORAN_EVALUASI.md): hasil yang sudah ada dan bagian yang masih kosong.
- README §4–5: detail akses LAN dan alur aplikasi.

Perintah ditulis untuk **PowerShell** di laptop server, dari folder repo, dengan venv aktif:
```powershell
cd C:\Users\Ridlo\Downloads\osce-stt\osce-stt
.\.venv\Scripts\Activate.ps1
```

---

## Peta besar

```
[A] Persiapan (sekali)            [B] Per sesi role-play (berulang)        [C] Setelah data cukup
 1. Backup & reset runtime         5. Rekam sesi                             9.  Cek kesiapan data
 2. Nyalakan server                6. Review & verifikasi                    10. Training ML-1 + E4 ML-2
 3. Siapkan akses (server/client)  7. Isi lembar kerja                       11. Aktifkan model baru
 4. Enroll tiap dokter             8. (sebagian sesi) simpan audio untuk WER 12. Hitung WER
                                                                             13. Rapikan data LLM (opsional)
                                                                             14. Isi laporan
```

Target minimum data: **≥ 3 dokter berbeda × ≥ 4 sesi per dokter** (≥ 12 sesi, masing-masing 2–5 menit),
plus **3–5 sesi untuk WER**. WER boleh memakai sesi yang sama.

| Alat | Fungsi singkat |
|---|---|
| `uvicorn app.main:app ...` | Menjalankan server web (UI + API + pipeline). |
| `scripts/roleplay_status.py` | Menghitung data yang sudah terkumpul per dokter dan memberi tahu apakah sudah cukup untuk training. |
| `scripts/train_speaker_role.py` | Melatih + mengevaluasi ML-1 (DOKTER vs PASIEN). |
| `scripts/train_text_intent.py` | Melatih + mengevaluasi ML-2 (S/O/A/P/CHITCHAT). |
| `scripts/eval_wer.py` | Mengukur akurasi transkripsi Whisper pada rekaman asli (WER). |
| `scripts/audit_llm_labels.py` | Meninjau/memperbaiki label data sintetis LLM lewat CSV (Excel/Sheets). |
| `docs/templates/lembar_kerja_roleplay.csv` | Catatan manual tiap sesi (siapa, skenario, kendala). |

---

## [A] Persiapan

### 1. Backup & reset `runtime/` (sekali, sebelum sesi pertama)

**Lakukan** (server dalam keadaan mati):
```powershell
$bk = "runtime\_backup_pre_roleplay_$(Get-Date -Format yyyyMMdd)"
New-Item -ItemType Directory -Force $bk | Out-Null
Move-Item runtime\osce.sqlite3*, runtime\anchors, runtime\features $bk
Get-ChildItem runtime
```
Folder `runtime\training_data` sudah kosong. Data smoke test lama ada di `runtime\_backup_smoke`; biarkan.

**Fungsi:** database saat ini masih berisi 19 job uji (dokter "dr. Ringlo" dan "dr. TTS Andika (dummy)")
yang belum diverifikasi. Kalau salah satunya terverifikasi tanpa sengaja, audio TTS/uji ikut masuk data
training dan mencemari evaluasi. Dengan reset, daftar dokter dan antrean dimulai bersih. File dipindah,
bukan dihapus, jadi bisa dikembalikan. Server membuat database baru otomatis saat start.

### 2. Nyalakan server

**Lakukan:**
```powershell
ollama list                      # harus ada qwen2.5:3b (dipakai server untuk rangkuman)
nvidia-smi                       # cek GPU terdeteksi
uvicorn app.main:app --host 0.0.0.0 --port 8000 --ssl-keyfile key.pem --ssl-certfile cert.pem
```
Tunggu sampai muncul `Uvicorn running on https://0.0.0.0:8000`. Biarkan jendela ini terbuka.
Cek di browser server: `https://localhost:8000/api/health` (tanpa token). Pastikan `"status": "ok"`,
`"llm_backend": "ollama"`, dan `intent_model` berisi `logreg@<tanggal>` (bukan `rule-based-fallback`).
`speaker_model` masih `heuristic-anchor-distance` sampai ML-1 dilatih di §10. Itu normal: sesi awal
memakai heuristik anchor, lalu dikoreksi manual di review.

**Fungsi:**
- `--host 0.0.0.0` membuat server bisa diakses laptop lain di jaringan yang sama.
- `--ssl-*` mengaktifkan HTTPS. Browser **hanya** mengizinkan mikrofon di HTTPS (atau localhost).
- Jangan menambah `--workers`: tiap worker memuat model sendiri ke VRAM → kehabisan memori GPU.
- Ollama berjalan sebagai aplikasi terpisah. Server memanggilnya untuk membuat rangkuman. Bila Ollama
  mati, rangkuman jatuh ke mode fallback (tanpa LLM), tetapi transkrip dan klasifikasi tetap jalan.
- Token akses ada di `.env` (`APP_ACCESS_TOKEN`). Browser meminta token sekali; bagikan hanya ke anggota.

### 3. Siapkan akses

**Opsi A — semua rekaman di laptop server (paling cepat).** Buka `https://localhost:8000` di Chrome
laptop server. Tidak perlu langkah tambahan.

**Opsi B — rekam dari laptop anggota lain.**
1. Cek IP server: `ipconfig` → "IPv4 Address" di adapter Wi-Fi. Sertifikat saat ini dibuat untuk
   `10.127.19.183`. Bila IP berubah, buat ulang:
   `mkcert -key-file key.pem -cert-file cert.pem <IP_BARU> LEPI-RIDLO localhost 127.0.0.1`,
   lalu restart server.
2. Firewall (PowerShell **Administrator**, sekali):
   ```powershell
   New-NetFirewallRule -DisplayName "OSCE-STT 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -RemoteAddress LocalSubnet -Action Allow
   ```
3. Di **setiap laptop client**: salin `%LOCALAPPDATA%\mkcert\rootCA.pem` dari server (lewat flashdisk).
   **Jangan** salin `rootCA-key.pem`. Lalu di client:
   ```powershell
   certutil -addstore -user Root rootCA.pem
   ```
   Tutup semua jendela Chrome, buka lagi `https://<IP_SERVER>:8000`.

**Fungsi:** `rootCA.pem` membuat laptop client percaya pada sertifikat server, sehingga tidak ada
peringatan dan mikrofon diizinkan. `rootCA-key.pem` adalah kunci rahasia: siapa pun yang memegangnya bisa
membuat sertifikat palsu yang dipercaya laptop itu. Rule firewall dibatasi ke port 8000 dan subnet lokal
supaya server tidak terbuka ke jaringan lain.

### 4. Enroll tiap dokter (sekali per orang)

**Lakukan:** buka `/enroll` → isi **Nama dokter** dengan kode samaran (mis. `D1`, `D2`, `D3`; jangan nama
lengkap) → **Rekam 15 detik**. Dokter membaca teks yang tampil, sendirian, dengan mikrofon dan posisi
duduk yang **sama** seperti saat sesi → **Simpan Profil**.

**Fungsi:** enrollment membuat *voice anchor*, yaitu profil statistik suara dokter
(`runtime/anchors/<id>.npz`). ML-1 tidak menebak "suara siapa"; ML-1 mengukur **seberapa mirip tiap
potongan suara dengan anchor dokter**. Karena itu model bisa dipakai pada dokter baru. Mikrofon dan posisi
harus sama, karena perbedaan mikrofon mengubah profil suara dan menurunkan akurasi. Rekaman enrollment
tidak disimpan; hanya profil fiturnya.

---

## [B] Per sesi role-play

### 5. Rekam sesi

**Lakukan:** halaman utama `/`:
1. **Profil dokter**: pilih dokter yang sedang berperan.
2. **Standar template penulisan**: pilih template (mis. OSCE Respirasi). Variasikan antar-sesi.
3. **Nama yang wajib dianonimkan**: isi nama samaran pasien yang akan diucapkan (pisahkan koma).
4. **● Mulai Rekam** → jalankan role-play 2–5 menit → **■ Selesai & Proses**.
5. Pantau status di bagian **Antrean server & job saya** sampai `done`.

SOP rekaman:
- Satu laptop/mikrofon di tengah meja, jarak 50–100 cm dari keduanya.
- Beri jeda ±0.5 detik saat bergantian bicara; hindari bicara bersamaan.
- Skenario fiktif. Jangan memakai identitas atau data pasien nyata.
- Ganti pemeran pasien antar-sesi. Usahakan ada pasangan dokter–pasien bergender sama (kasus tersulit).
- Lengkapi alur: salam → anamnesis → pemeriksaan/tanda vital → diagnosis → terapi/edukasi → penutup.

**Fungsi:** audio dikirim ke antrean server (diproses satu per satu agar GPU tidak kehabisan memori).
Pipeline: hapus derau → deteksi suara (VAD) → Whisper (teks + waktu per kata) → potong per ucapan →
ML-1 (peran) → hapus PII → ML-2 (intent) → LLM (rangkuman terstruktur). Jeda giliran penting karena
sistem memotong ucapan berdasarkan jeda. Ucapan yang tumpang tindih akan tergabung dan diberi satu peran.
Pasien bervariasi membuat model belajar "peran", bukan menghafal satu suara pasien.

### 6. Review & verifikasi

**Lakukan:** klik job → halaman `/review?job=<id>`:
- **Panel kiri — Transkrip (Dokter ⟷ Pasien):**
  - klik label peran untuk menukar DOKTER/PASIEN bila salah;
  - ganti intent lewat dropdown bila salah (lihat pedoman di bawah);
  - perbaiki teks bila Whisper salah dengar.
- **Panel kanan — Rangkuman terstandar:** periksa dan perbaiki isi form hasil LLM. Tiap butir punya
  `sumber` (ID ucapan); cocokkan dengan transkrip.
- Confidence < 65% ditandai merah. Periksa baris itu lebih teliti, tapi tetap cek semua baris.
- **Simpan Draft** bila belum selesai. **Verifikasi & Ekspor ke Bank Kasus OSCE** bila semua sudah benar.

Pedoman intent:
| Ucapan | Label |
|---|---|
| Pertanyaan anamnesis dokter, keluhan pasien, "iya dok, ada demam" | `S_KELUHAN` |
| Instruksi pemeriksaan ("tarik napas"), hasil tanda vital/pemeriksaan | `O_PEMERIKSAAN` |
| Dugaan/simpulan diagnosis | `A_DIAGNOSIS` |
| Obat, edukasi, rencana, rujukan, kontrol, pertanyaan pasien soal obat/kontrol | `P_EDUKASI_TERAPI` |
| Salam, terima kasih, "iya/baik dok" tanpa isi, basa-basi | `CHITCHAT` |

**Fungsi:** verifikasi adalah satu-satunya jalan data nyata masuk training. Saat diklik, sistem menulis:
- `runtime/training_data/speaker_role.jsonl`: fitur suara tiap ucapan + **peran yang sudah dikoreksi**
  (data ML-1);
- `runtime/training_data/text_intent.jsonl`: teks + **intent yang sudah dikoreksi**, `source=human_verified`
  (data ML-2 dan test set E4);
- kasus ke Bank Kasus (`/cases`).

Label yang salah saat verifikasi menjadi "kunci jawaban" yang salah. Kolom `model_label` juga mencatat
tebakan model sebelum dikoreksi, sehingga jumlah koreksi = ukuran kesalahan model versi lama.
Job yang belum diverifikasi **tidak** menghasilkan data training.

### 7. Isi lembar kerja

**Lakukan:** salin `docs/templates/lembar_kerja_roleplay.csv` ke `runtime/lembar_kerja_roleplay.csv`, lalu
buka di Excel/Sheets dan isi satu baris per sesi. Hapus baris CONTOH.
Kolom `job_id` diambil dari URL review.

**Fungsi:** script hanya tahu angka. Lembar kerja mencatat konteks yang tidak terekam: siapa pemeran
pasien, gender, ruangan, ada overlap atau tidak, kendala. Konteks ini dipakai saat menganalisis error di
laporan (mis. "akurasi rendah di sesi dengan dokter dan pasien sama-sama perempuan"). Disimpan di
`runtime/` karena berisi kode anggota; folder itu tidak di-commit.

### 8. (3–5 sesi) Simpan audio untuk WER

Audio mentah otomatis dihapus setelah diproses (privasi). Untuk mengukur WER, audio dibutuhkan, jadi
simpan **sementara** untuk beberapa sesi saja.

**Lakukan:**
1. Matikan server (Ctrl+C). Di `.env` ubah `APP_KEEP_RAW_AUDIO=false` menjadi `APP_KEEP_RAW_AUDIO=true`.
   Nyalakan server lagi.
2. Rekam 3–5 sesi seperti biasa (langkah 5–6).
3. Matikan server, kembalikan `APP_KEEP_RAW_AUDIO=false`, nyalakan lagi.
4. Pindahkan audionya:
   ```powershell
   New-Item -ItemType Directory -Force runtime\wer | Out-Null
   Move-Item runtime\uploads\<job_id>.wav runtime\wer\sesi01.wav     # ulangi per sesi
   ```
5. Buat transkrip referensi `runtime\wer\sesi01.txt` (nama sama dengan audionya) dengan mendengarkan
   audio. Lihat aturan di §12.

**Fungsi:** WER membandingkan hasil Whisper dengan transkrip buatan manusia. Mode simpan-audio hanya
dinyalakan sebentar dan hanya untuk role-play fiktif, sehingga prinsip "audio mentah tidak disimpan"
tetap berlaku untuk pemakaian normal. File dipindah ke `runtime\wer\` agar tidak tercampur dengan upload
lain. Folder `runtime\` tidak masuk git.

---

## [C] Setelah data cukup

### 9. Cek kesiapan data

**Lakukan:**
```powershell
python scripts/roleplay_status.py
python scripts/roleplay_status.py --csv runtime/status_roleplay.csv     # opsional: tabel per sesi
```
Contoh bagian akhir output:
```
=== Kesiapan training
  [x] >= 3 dokter dengan >= 4 sesi  (3 dokter memenuhi)
  [x] setiap sesi punya segmen DOKTER dan PASIEN
  [ ] >= 300 baris intent manusia  (240 baris)
  [x] kelima kelas intent ada
  -> python scripts/train_speaker_role.py --data runtime/training_data/speaker_role.jsonl --model svm
```

**Fungsi:** script membaca data training dan database (read-only, tidak mengubah apa pun), lalu
menampilkan:
- per dokter: jumlah sesi, segmen DOKTER/PASIEN, dan persentase koreksi peran (= error model saat itu);
- data intent: jumlah baris, distribusi kelas, persentase koreksi;
- **job selesai yang belum diverifikasi**: datanya belum masuk training. Buka link-nya dan verifikasi;
- checklist kesiapan dan perintah berikutnya.

Syarat ≥ 3 dokter ada karena script training memisahkan train/test **per dokter**: dokter di test set
tidak pernah dilihat saat training. Hanya cara ini yang membuktikan model bekerja pada dokter baru.
Dengan < 3 dokter, split jatuh ke per sesi dan klaim generalisasi belum teruji.

### 10. Training & evaluasi

**Lakukan (ML-1 Speaker Role):**
```powershell
python scripts/train_speaker_role.py --data runtime/training_data/speaker_role.jsonl --model svm
```
**Lakukan (ML-2 Intent, protokol utama E4):**
```powershell
python scripts/train_text_intent.py --data data/synthetic/text_intent.jsonl data/synthetic/text_intent_llm.jsonl runtime/training_data/text_intent.jsonl --test-source human_verified
```
Lalu simpan bukti untuk laporan (folder `models/` tidak di-commit):
```powershell
New-Item -ItemType Directory -Force reports\speaker_role, reports\text_intent\E4_human_verified | Out-Null
Copy-Item models\speaker_role\metrics.json, models\speaker_role\confusion_matrix.png reports\speaker_role\
Copy-Item models\text_intent\metrics.json, models\text_intent\confusion_matrix.png reports\text_intent\E4_human_verified\
```

**Fungsi:**
- ML-1: script membangun fitur relatif-anchor per sesi, memisahkan dokter train/test, lalu menjalankan
  CV, membandingkan 4 model (SVM/RF/MLP/LogReg) dan baseline heuristik tanpa training, serta
  **ablation fitur relatif vs absolut**. Ablation ini bukti angka untuk keputusan desain bahwa fitur
  absolut (MFCC/F0 mentah) menghafal suara, bukan peran.
- ML-2 (E4): model dilatih **hanya** dari data sintetis, lalu diuji pada **seluruh** ucapan nyata yang
  diverifikasi manusia. Ini angka paling jujur untuk laporan: "seberapa baik model sintetis pada
  percakapan nyata". Model yang tersimpan kemudian di-refit dengan semua data, termasuk data nyata,
  untuk dipakai server.
- Output tiap script: `metrics.json` (accuracy, precision/recall/F1 per kelas, confusion matrix, CV
  mean ± std, perbandingan, ablation) dan `confusion_matrix.png` untuk slide.
- Opsional: bandingkan `--model rf` / `--model mlp` untuk ML-1. Pilih model dari skor **CV**, bukan skor
  test; memilih dari test set sama dengan "mengintip kunci jawaban".

### 11. Aktifkan model baru

**Lakukan:** buka `/metrics` → **Muat ulang model**.

**Fungsi:** server memuat ulang `models/*/model.joblib` tanpa restart. Halaman `/metrics` (Academic ML
Evaluation Panel) menampilkan angka dari `metrics.json` dan bisa dipakai langsung saat presentasi.

### 12. Hitung WER

**Aturan transkrip referensi** (`runtime\wer\<nama>.txt`):
- Tulis kata yang **benar-benar diucapkan**, termasuk salah ucap pembicara. Jangan dirapikan.
- Hilangkan filler ("eh", "emm", "hmm") dan pengulangan tak sengaja ("saya, saya"). Whisper juga
  membuangnya.
- Istilah medis pakai ejaan baku ("parasetamol", "amoksisilin").
- Angka bebas digit atau kata: script menyamakan "tiga"/"3", "120/80"/"120 per 80", "38,5"/"38.5".
- Tanpa label pembicara dan tanpa timestamp; cukup teks berurutan.
- Butuh ±10–15 menit kerja per menit audio. Putar audio dengan kecepatan 0.75× bila perlu.

**Lakukan** (server sebaiknya idle atau mati, karena script memuat Whisper sendiri ke GPU):
```powershell
python scripts/eval_wer.py
python scripts/eval_wer.py --no-denoise --out reports/wer/wer_no_denoise.json   # opsional: efek denoise
```
Contoh output:
```
sesi01.wav                    182.4 s  WER 14.2%  (S 51 D 12 I 9 / 507 kata)  RTF 0.118
TOTAL WER 14.2% (51 S + 12 D + 9 I / 507 kata)
Substitusi tersering: parasetamol -> para (3), ronki -> ronkhi (2), ...
```

**Fungsi:**
- WER = (S + D + I) / jumlah kata referensi. S = kata salah, D = kata hilang, I = kata tambahan.
  Makin kecil makin baik.
- Script memakai jalur yang **sama persis** dengan server (denoise → Whisper dengan glossary dari
  `.env`). Jadi angkanya mewakili kualitas transkrip yang dilihat dokter di halaman review.
- RTF (real-time factor) = waktu proses / durasi audio. 0.12 berarti 1 menit audio diproses ±7 detik.
- Hasil: `runtime\wer\<nama>.hyp.txt` (teks Whisper, untuk dibandingkan manual) dan
  `reports/wer/wer_results.json` (angka + daftar substitusi tersering).
- Daftar substitusi menunjukkan istilah yang perlu ditambahkan ke `data/glossary/medical_glossary.txt`
  atau `data/lexicon/medical_lexicon.json`.
- Setelah selesai, hapus audio: `Remove-Item runtime\wer\*.wav`.

### 13. (Opsional, disarankan) Rapikan data LLM

Laporan §3.5 menemukan banyak ucapan pasien berlabel `S_KELUHAN` di data LLM yang seharusnya `CHITCHAT`
atau `P_EDUKASI_TERAPI`.

**Lakukan:**
```powershell
python scripts/audit_llm_labels.py sample --role PASIEN --label S_KELUHAN --n 0 --prefill --audit data/synthetic/review_pasien_s.csv
```
Buka `data/synthetic/review_pasien_s.csv` di Excel/Sheets. Ubah kolom `label_audit` **hanya** pada baris
yang salah (pakai tabel pedoman di §6), boleh isi `catatan`, lalu simpan tetap sebagai CSV. Setelah itu:
```powershell
python scripts/audit_llm_labels.py apply --audit data/synthetic/review_pasien_s.csv
```
Lalu ulangi E2/E3 (perintah di `EVALUASI.md` §2) dan E4 (§10).

**Fungsi:**
- `sample --role PASIEN --label S_KELUHAN --n 0` mengambil **semua** baris subset itu (212 baris).
- `--prefill` mengisi `label_audit` dengan label sekarang, jadi peninjau cukup mengubah yang salah.
- File memakai pemisah `;` dan baris `sep=;` agar Excel berbahasa Indonesia langsung membuka per kolom.
- `apply` menulis koreksi ke `data/synthetic/text_intent_llm.jsonl`; label asli LLM tetap disimpan di
  `label_llm`. Script menolak bila teks di file data sudah berubah, dan aman dijalankan ulang.
- Angka "akurasi" dari tinjauan ini **bukan** estimasi noise seluruh data, karena subsetnya dipilih,
  bukan acak. Estimasi noise tetap memakai audit acak 50 baris (82%).

### 14. Isi laporan

Bagian yang perlu diisi/diperbarui di `docs/LAPORAN_EVALUASI.md`:
| Bagian | Sumber angka |
|---|---|
| §2 tabel data (baris `human_verified`) | output `roleplay_status.py` |
| §3 tambah baris E4 + confusion matrix | `reports/text_intent/E4_human_verified/metrics.json` |
| §4 ML-1: test split-dokter, CV, perbandingan, baseline, ablation | `reports/speaker_role/metrics.json` |
| WER mikrofon asli (juga `docs/benchmark.md`) | `reports/wer/wer_results.json` |
| Analisis error per kondisi | lembar kerja + `status_roleplay.csv` |

Agent (Claude Code) bisa diminta mengisi bagian ini dari file-file di atas. Setelah itu commit:
`docs/`, `reports/`, `data/synthetic/` (bila dirapikan). Jangan commit isi `runtime/`.

---

## Troubleshooting

| Gejala | Penyebab | Solusi |
|---|---|---|
| "Mikrofon tidak tersedia" | Halaman dibuka lewat HTTP atau sertifikat tidak dipercaya | Pakai `https://`; client: pasang `rootCA.pem` (§3) |
| Peringatan "Your connection is not private" | `rootCA.pem` belum dipasang, atau IP server berubah | §3 langkah 1 dan 3 |
| Laptop client tidak bisa membuka halaman | Beda jaringan, firewall, atau IP salah | Satu Wi-Fi/hotspot; rule firewall §3; cek `ipconfig` |
| Muncul prompt token berulang / error 401 | Token kosong/salah | UI meminta token ulang otomatis saat 401; isi persis nilai `APP_ACCESS_TOKEN` dari `.env` |
| Job lama di `queued` | Job lain sedang diproses (antrean FIFO, satu per satu) | Normal; tunggu. Cek jendela uvicorn untuk error |
| Job `error`: "Tidak ada suara terdeteksi (VAD kosong)" | Mic salah/mati atau volume terlalu kecil | Cek input mic di Windows, dekatkan laptop |
| Rangkuman kosong/fallback, `summary_source` bukan `llm` | Ollama tidak berjalan | Jalankan aplikasi Ollama, `ollama list` |
| CUDA out of memory | Proses lain memakai GPU (mis. `eval_wer.py` / generator LLM saat server sibuk) | Jalankan script berat saat server idle; jangan pakai `--workers` |
| Banyak peran tertukar | Enrollment beda posisi/mic, suara mirip, atau overlap | Enroll ulang di posisi sesi; jeda giliran; tetap koreksi di review |
| `roleplay_status.py` menyebut job belum diverifikasi | Review belum diklik "Verifikasi & Ekspor" | Buka link yang tercetak dan verifikasi |
| `train_speaker_role.py`: "Anchor dokter ... tidak ditemukan" | File `runtime/anchors/<id>.npz` terhapus/dipindah | Jangan pindahkan `anchors` setelah sesi direkam |
| `eval_wer.py`: "Tidak ada pasangan" | Nama `.wav` dan `.txt` tidak sama | `sesi01.wav` + `sesi01.txt` di `runtime\wer\` |
