# INSTALASI DI KOMPUTER LAB (server) + AKSES PUBLIK CLOUDFLARE

Panduan memasang OSCE-STT dari nol di komputer lab yang akan menyala terus dan menjadi **server**.
Anggota lain cukup memakai browser. Mereka tidak perlu memasang apa pun.

Setelah instalasi selesai, lanjut ke [`TUTORIAL_TESTING.md`](TUTORIAL_TESTING.md) untuk pengujian, dan
[`BELAJAR_SISTEM.md`](BELAJAR_SISTEM.md) untuk memahami cara kerja sistem.

---

## 0. Gambaran: siapa menjalankan apa

```
 HP/laptop anggota (browser saja)                     Komputer lab (server, menyala terus)
 ───────────────────────────────                      ───────────────────────────────────────────
 https://xxxx.trycloudflare.com  ──HTTPS──►  Cloudflare  ◄──koneksi keluar──  cloudflared (jendela 2)
                                                                                    │ http://127.0.0.1:8000
                                                                                    ▼
                                                                              uvicorn + FastAPI (jendela 1)
                                                                              ├─ Whisper (GPU)
                                                                              ├─ ML-1, ML-2 (CPU)
                                                                              └─ Ollama qwen2.5:3b (GPU)
```

- **Satu komputer = server.** Semua model berjalan di sini. Anggota hanya membuka URL di browser,
  merekam, lalu me-review.
- **Cloudflare Quick Tunnel** (gratis, tanpa akun, tanpa domain) membuat URL publik HTTPS.
  - Karena HTTPS-nya sah, mikrofon langsung diizinkan browser. Tidak perlu mkcert atau `rootCA.pem`.
  - `cloudflared` hanya membuat koneksi **keluar**, sehingga tetap jalan walau firewall kampus
    memblokir koneksi masuk.
- Server sendiri hanya mendengarkan di `127.0.0.1` (lokal). Komputer lain di jaringan lab tidak bisa
  mengaksesnya langsung; semua akses lewat tunnel.

### Konsekuensi yang wajib dipahami (dan ditulis di laporan)
1. **Sistem tidak lagi offline.** Audio dan transkrip melewati server Cloudflare dalam keadaan
   terenkripsi, tetapi enkripsi itu dibuka di Cloudflare sebelum diteruskan ke komputer lab. Ini
   bertentangan dengan prinsip "offline, data tidak keluar dari jaringan lokal" di desain awal.
   Untuk role-play fiktif dan demo, ini wajar. **Untuk data pasien nyata: pakai mode LAN (§7), jangan tunnel.**
2. **URL bersifat publik.** Siapa pun yang tahu URL bisa membuka halamannya. Semua data dilindungi
   token akses (§5).
3. **URL berubah** setiap kali tunnel dinyalakan ulang. URL tetap butuh Named Tunnel + domain sendiri,
   dan domain umumnya berbayar.
4. Batas upload Cloudflare (paket gratis) 100 MB per request. Rekaman WAV 16 kHz mono ≈ 1.9 MB/menit,
   jadi rekaman 40 menit (±77 MB) masih aman.

---

## 1. Syarat komputer lab

| Komponen | Minimum | Catatan |
|---|---|---|
| OS | Windows 10/11 64-bit | Skrip ditulis untuk Windows |
| GPU | NVIDIA, VRAM ≥ 4 GB (RTX 4050 6 GB sudah teruji) | Tanpa GPU tetap jalan dengan `-Cpu`, tapi ASR ±5× lebih lambat |
| Driver NVIDIA | Versi baru (yang mendukung CUDA 12) | Cek: `nvidia-smi` |
| RAM | 8 GB (16 GB disarankan) | |
| Disk kosong | ±15 GB | venv + CUDA libs ±5 GB, Ollama model 1.9 GB, Whisper 0.5 GB, data |
| Internet | Saat instalasi dan selama mode tunnel | |
| Hak admin | Tidak wajib | Semua bisa dipasang per-user; admin hanya untuk firewall mode LAN |

**Tanyakan ke pengelola lab sebelum mulai:**
- **Apakah ada Deep Freeze / Reboot Restore / "clean on reboot"?** Kalau ya, semua instalasi dan data
  rekaman **hilang setiap reboot**. Minta folder/drive yang dikecualikan (mis. `D:\`), lalu pasang repo
  di sana.
- **Bolehkah komputer menyala terus dan tidak sleep?** Atur dengan PowerShell:
  ```powershell
  powercfg /change standby-timeout-ac 0      # tidak sleep saat dicolok listrik
  powercfg /change hibernate-timeout-ac 0
  ```
  Kalau komputer sleep, server dan tunnel terputus dan URL berubah setelah dinyalakan ulang.
- Apakah antivirus kampus memblokir `cloudflared.exe`?

---

## 2. Pasang prasyarat (sekali)

Buka **PowerShell** (tidak perlu Administrator) lalu jalankan:
```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.11 -e
winget install --id Ollama.Ollama -e
winget install --id Cloudflare.cloudflared -e
```
**Tutup lalu buka lagi PowerShell** supaya PATH terbarui. Cek:
```powershell
git --version; python --version; ollama --version; cloudflared --version; nvidia-smi
```

| Program | Fungsi |
|---|---|
| Git | Mengambil kode dari GitHub dan memperbaruinya (`git pull`). |
| Python 3.11 | Bahasa server dan semua model ML. Versi 3.10–3.12 didukung; 3.11 sudah teruji. |
| Ollama | Server LLM lokal yang menjalankan `qwen2.5:3b` untuk menyusun rangkuman. Berjalan otomatis di background setelah dipasang (ikon di system tray). |
| cloudflared | Membuat tunnel ke Cloudflare untuk URL publik. |
| Driver NVIDIA | Agar GPU bisa dipakai. `nvidia-smi` harus menampilkan nama GPU. |

Kalau `winget` tidak tersedia atau diblokir, unduh installer manual dari situs resminya (git-scm.com,
python.org, ollama.com, dan rilis GitHub `cloudflare/cloudflared`). Saat memasang Python, centang
**"Add python.exe to PATH"**.

---

## 3. Ambil kode dari GitHub

Repo bersifat **private**, jadi akun GitHub yang dipakai harus pemilik atau collaborator.
```powershell
cd D:\                      # folder yang TIDAK di-reset Deep Freeze; sesuaikan
git clone https://github.com/RifqiARidlo/Osce-STT.git
cd Osce-STT
```
Saat pertama kali, jendela login GitHub muncul (Git Credential Manager). Login dengan akun yang punya
akses.

**Fungsi:** semua kode, data sintetis (`data/synthetic/`), dan dokumen ikut di-clone. File yang sengaja
**tidak** ada di GitHub dan dibuat ulang oleh skrip instalasi:
- `.venv/`: lingkungan Python;
- `.env`: konfigurasi + token;
- `models/`: model ML terlatih dan Whisper;
- `runtime/`: database, profil suara, rekaman;
- `*.pem`: sertifikat HTTPS.

---

## 4. Jalankan skrip instalasi

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\setup_lab.ps1
```
Opsi: `-Cpu` (tanpa GPU), `-SkipOllama`, `-SkipTests`. Durasi ±10–20 menit, tergantung internet.
Skrip aman dijalankan ulang bila gagal di tengah jalan.

Yang dikerjakan skrip, langkah demi langkah:

| Langkah | Yang terjadi | Fungsi / alasan |
|---|---|---|
| 1. Cek Python | Mencari Python 3.10–3.12 (`py -3.11`, lalu `python`) | Versi lain belum teruji dengan dependensi. |
| 2. `.venv` | Membuat virtual environment di folder repo | Paket proyek terisolasi dari Python sistem. Tidak bentrok dengan program lab lain dan mudah dihapus. |
| 3. Paket Python | `pip install -r requirements.txt -r requirements-optional.txt` | FastAPI/uvicorn (server web), faster-whisper (ASR + Silero VAD, **tanpa** PyTorch), scikit-learn (ML-1, ML-2), numpy/scipy (fitur audio), matplotlib (grafik), pytest (test). |
| 4. CUDA libs | `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` (±2 GB) | Whisper (CTranslate2) butuh cuBLAS + cuDNN. Dipasang lewat pip agar tidak perlu instal CUDA Toolkit penuh. Folder DLL-nya didaftarkan otomatis oleh `app/pipeline/stage2_asr.py`. |
| 5. `.env` | Disalin dari `.env.example`, token acak 48 karakter dibuat | Konfigurasi server. **Token tampil sekali di layar**; nanti juga bisa dilihat di `.env`. |
| 6. Ollama | `ollama pull qwen2.5:3b` (1.9 GB) | Model LLM untuk rangkuman. 3B dipilih karena muat bergantian dengan Whisper di GPU 4–6 GB. |
| 7a. Whisper | Mengunduh Whisper `small` (±0.5 GB) ke `models/whisper` lalu memuatnya di GPU | Mengunduh model sekaligus **menguji GPU sejak awal**. Kalau ada error CUDA/DLL, ketahuan sekarang, bukan saat demo. |
| 7b. Model intent | `train_text_intent.py` pada data sintetis (±2 menit) | `models/` tidak ada di GitHub (file sklearn tidak portabel antar-versi), jadi model ML-2 dilatih ulang di mesin ini. Model ML-1 belum ada; server memakai heuristik anchor sampai data role-play terkumpul. |
| 8. pytest | 32 test CPU | Bukti instalasi utuh. Semua harus `passed`. |

---

## 5. Menyalakan server + URL publik

**Klik dua kali `start_osce.bat`** di folder repo (atau jalankan dari PowerShell: `.\start_osce.bat`).

Yang terjadi:
1. Jendela **"OSCE-STT server"** terbuka. Ini uvicorn di `http://127.0.0.1:8000` (start_server.bat).
2. Skrip menunggu sampai `/api/health` menjawab.
3. Jendela **"OSCE-STT tunnel (JANGAN DITUTUP)"** terbuka. Ini cloudflared.
4. URL `https://xxxx.trycloudflare.com` ditampilkan, disimpan ke `runtime\tunnel_url.txt`, lalu dibuka
   di browser.

**Bagikan ke anggota:** URL + token (nilai `APP_ACCESS_TOKEN` di `.env`). Saat pertama membuka halaman
yang butuh data, browser meminta token sekali, lalu menyimpannya.

**Mematikan:** tutup kedua jendela (tunnel dulu, lalu server). **Menyalakan ulang:** klik
`start_osce.bat` lagi. URL baru harus dibagikan ulang.

Cek kesehatan dari browser mana pun: `https://xxxx.trycloudflare.com/api/health` (tanpa token):
- `"status": "ok"`;
- `"llm_backend": "ollama"`;
- `"intent_model": "logreg@..."`;
- `"speaker_model": "heuristic-anchor-distance"`. Ini normal sampai ML-1 dilatih.

### Token akses = "login" sistem ini
Server sudah punya mekanisme login sederhana: **satu token bersama** (seperti password grup).
- Semua endpoint data (`/api/...`, kecuali `/api/health`) menolak request tanpa token yang benar
  (HTTP 401).
- Browser mengirim token lewat header `X-Access-Token`. Server membandingkannya dengan
  `hmac.compare_digest` (perbandingan waktu-konstan, tahan *timing attack*).
- Halaman HTML/JS dan `/docs` tetap terbuka, tapi tidak berisi data. Data hanya bisa diambil lewat API
  yang dijaga token.
- Token acak 48 karakter heksadesimal = 192 bit, jadi tidak mungkin ditebak dengan *brute force*.

Aturan pakai:
- Bagikan token hanya ke anggota kelompok, lewat chat pribadi/grup. Jangan ditulis di slide atau repo.
- **Ganti token** setelah pameran, atau bila bocor: ubah `APP_ACCESS_TOKEN` di `.env`, restart server.
  Browser anggota akan meminta token baru otomatis.

**Login per-akun (username + password per orang + session) belum ada.** Fitur itu butuh tabel user
(perubahan skema database), hashing password, cookie session, dan halaman login. Trade-off-nya:
- **Lebih baik:** jejak siapa merekam/memverifikasi apa, dan akses bisa dicabut per orang.
- **Biaya:** kode lebih banyak dan permukaan bug keamanan baru, sementara manfaatnya kecil untuk satu
  kelompok kecil dengan data fiktif.

Rekomendasi: cukup token bersama untuk mata kuliah ini. Login per-akun bisa ditambahkan bila sistem
dipakai lebih luas.

---

## 6. Operasional harian

| Kegiatan | Cara |
|---|---|
| Menyalakan | `start_osce.bat` → bagikan URL baru |
| Mematikan | Tutup jendela tunnel lalu server |
| Server saja tanpa tunnel (uji lokal di komputer lab) | `start_server.bat`, buka `http://127.0.0.1:8000` |
| Update kode dari GitHub | Matikan server → `git pull` → jalankan ulang `setup_lab.ps1` bila `requirements*.txt` berubah → nyalakan |
| Backup data | Matikan server, salin seluruh folder `runtime\` dan `models\` ke flashdisk/drive. **Folder `runtime\anchors`, `runtime\osce.sqlite3`, dan `runtime\training_data` harus dipindah bersama**, karena saling merujuk ID dokter/job |
| Lihat error | Jendela "OSCE-STT server" (log uvicorn) |
| Lihat VRAM | `nvidia-smi` |

Jangan menjalankan `eval_wer.py`, `generate_synthetic_llm.py`, atau training berat saat ada sesi yang
sedang diproses. GPU 6 GB tidak cukup untuk dua model besar sekaligus.

---

## 7. Mode LAN (tanpa internet, untuk data sensitif)

Kalau semua perangkat ada di satu jaringan dan sistem harus benar-benar offline:
1. Buat sertifikat untuk IP komputer lab dengan mkcert (README §4) → `key.pem`, `cert.pem` di folder repo.
2. `start_server.bat lan`. Server HTTPS di `https://<IP-lab>:8000`.
3. Laptop anggota memasang `rootCA.pem` dan port 8000 dibuka di firewall (`TUTORIAL_TESTING.md` §3, Opsi B).

Trade-off: data tidak keluar dari jaringan lab, tapi butuh langkah sertifikat di setiap laptop, dan
jaringan kampus sering memblokir koneksi antar-perangkat (*client isolation*).

---

## 8. Troubleshooting instalasi & tunnel

| Gejala | Penyebab | Solusi |
|---|---|---|
| `setup_lab.ps1 cannot be loaded because running scripts is disabled` | Execution policy | Jalankan persis dengan `powershell -ExecutionPolicy Bypass -File ...` |
| "Python 3.10-3.12 tidak ditemukan" | Python belum dipasang / tidak di PATH | `winget install --id Python.Python.3.11 -e`, buka PowerShell baru |
| Langkah 7a gagal: `cublas64_12.dll` / `cudnn` not found | Langkah 4 gagal atau driver terlalu lama | Ulangi skrip; update driver NVIDIA; atau pakai `-Cpu` |
| Langkah 7a gagal: CUDA out of memory | Program lain memakai GPU | Tutup program lain; cek `nvidia-smi` |
| `ollama` tidak dikenal | Ollama belum dipasang atau PowerShell belum dibuka ulang | §2; buka PowerShell baru |
| Rangkuman memakai "fallback rule-based" | Ollama tidak berjalan | Buka aplikasi Ollama (ikon tray), `ollama list` |
| `start_osce.bat`: "Server tidak menjawab" | Error saat start server | Baca pesan di jendela "OSCE-STT server" |
| `start_osce.bat`: "URL tunnel tidak muncul" | Internet mati, atau jaringan kampus memblokir Cloudflare | Cek internet; baca `runtime\tunnel.log`; coba hotspot HP; atau mode LAN (§7) |
| Browser: error 502/1033 di URL tunnel | Server mati, tunnel masih hidup | Nyalakan ulang dengan `start_osce.bat` |
| URL lama tidak bisa dibuka | Tunnel dinyalakan ulang → URL baru | Lihat `runtime\tunnel_url.txt` |
| Error 413 saat upload | File > batas (Cloudflare 100 MB, server 200 MB) | Rekam lebih pendek; jangan upload video |
| Semua data hilang setelah reboot | Deep Freeze | Pasang repo di drive yang dikecualikan (§1) |

---

## 9. Mode uji coba di laptop sendiri dengan DeepSeek API (opsional)

Untuk mencoba sistem di laptop pribadi tanpa Ollama, misalnya kalau GPU kecil atau sekadar demo cepat.
Whisper, ML-1, ML-2, dan PII scrub tetap berjalan lokal. **Hanya tahap rangkuman (S4)** yang memakai
DeepSeek, yaitu LLM cloud yang dipanggil lewat API.

1. Instal seperti §2–4. Ollama tidak perlu, jadi jalankan skrip dengan `-SkipOllama`. Tambah `-Cpu`
   bila laptop tanpa GPU NVIDIA.
2. Tambahkan API key ke `.env`. File ini tidak di-commit:
   ```
   APP_DEEPSEEK_API_KEY=sk-...
   ```
3. Klik dua kali **`start_local_deepseek.bat`**. Server menyala di `http://127.0.0.1:8000` dan browser
   terbuka otomatis. Karena alamatnya `localhost`, mikrofon diizinkan tanpa sertifikat.
4. Cek `http://127.0.0.1:8000/api/health`. Hasilnya harus `"llm_backend": "deepseek"`.

**Cara kerja:**
- Launcher hanya mengganti `APP_LLM_BACKEND` untuk proses itu. Isi `.env` (mis. Ollama untuk lab)
  tidak diubah.
- DeepSeek dipanggil dengan JSON mode. Schema dikirim di prompt, lalu output divalidasi Pydantic dan
  diulang sekali bila tidak valid.
- Aturan `sumber` dan fallback rule-based tetap berlaku.
- Hasil uji dialog fiktif: rangkuman valid untuk `soap_umum` dan `osce_respirasi`, ±2 detik per
  rangkuman.

**Batasan dan keamanan:**
- **Sistem tidak offline.** Transkrip yang sudah di-scrub PII dikirim ke server DeepSeek di luar
  negeri. Pakai data fiktif/role-play saja, dan sebutkan di laporan bila mode ini dipakai.
- API key = kunci tagihan. Jangan ditaruh di repo, slide, atau chat grup. Bila pernah terlihat orang
  lain, **revoke/ganti** di dashboard DeepSeek.
- Hasil evaluasi resmi tetap memakai Ollama lokal, karena itu konfigurasi yang dirancang.
