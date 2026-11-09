---
description: Kerjakan fase berikutnya yang belum dicentang di docs/NEXT_STEPS.md
---
Baca CLAUDE.md dan docs/NEXT_STEPS.md. Cek `git status`. Ambil fase PERTAMA yang masih punya item `[ ]`.
Sebelum mengeksekusi, tampilkan rencana singkat untuk fase itu dan item yang butuh persetujuan user
(install dependency berat, unduh model, ubah skema DB). Setelah disetujui, kerjakan, jalankan
`pytest -q`, centang item yang lolos, lalu usulkan commit Conventional Commits. Berhenti dan laporkan
bila ada perintah yang menggantung atau error yang tidak jelas.
