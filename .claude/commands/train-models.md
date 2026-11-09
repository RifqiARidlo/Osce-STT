---
description: Latih ulang kedua modul ML dari data yang tersedia dan ringkas metriknya
---
1. Hitung baris di data/synthetic/*.jsonl dan runtime/training_data/*.jsonl.
2. Latih intent: bila ada >= 20 baris `human_verified`, pakai `--test-source human_verified`; selain itu pakai split grup default.
3. Latih speaker role bila runtime/training_data/speaker_role.jsonl ada dan berisi >= 3 sesi.
4. Tampilkan tabel: accuracy, macro-F1, CV mean±std, perbandingan baseline, ablation. Beri catatan bila data didominasi sintetis.
5. Ingatkan user untuk klik "Muat ulang model" di /metrics (atau POST /api/ml/reload).
