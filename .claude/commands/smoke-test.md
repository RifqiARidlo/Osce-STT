---
description: Smoke test server lokal tanpa GPU (CPU Whisper, tanpa LLM)
---
1. Pastikan `.env` berisi APP_WHISPER_DEVICE=cpu, APP_WHISPER_COMPUTE_TYPE=int8, APP_LLM_BACKEND=none.
2. Jalankan `pytest -q`.
3. Jalankan server di background: `uvicorn app.main:app --port 8000`.
4. Cek dengan curl: `/api/health`, `/api/templates`, `/api/doctors`, `/api/queue`, serta halaman `/`, `/enroll`, `/metrics`, `/cases` (status 200).
5. Laporkan hasil per endpoint dalam tabel, lalu matikan server. Jangan memperbaiki apa pun sebelum melapor.
