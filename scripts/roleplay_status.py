"""Cek progres pengumpulan data role-play dan kesiapan training (Fase 5).

Membaca (read-only):
  runtime/training_data/speaker_role.jsonl   -> data ML-1 (terisi saat sesi diverifikasi di /review)
  runtime/training_data/text_intent.jsonl    -> data ML-2 berlabel manusia (source=human_verified)
  runtime/osce.sqlite3                       -> nama dokter + job yang selesai tapi belum diverifikasi

Menampilkan per dokter: jumlah sesi, segmen DOKTER/PASIEN, koreksi peran, lalu cek target minimum
(>= 3 dokter x >= 4 sesi; data intent >= 300 baris dengan 5 kelas) dan perintah training berikutnya.

Contoh:
    python scripts/roleplay_status.py
    python scripts/roleplay_status.py --csv runtime/status_roleplay.csv    # ekspor tabel per sesi
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.schemas.pipeline import INTENT_LABELS  # noqa: E402

MIN_DOCTORS, MIN_SESSIONS_PER_DOCTOR, MIN_INTENT_ROWS = 3, 4, 300


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _db(db_path: Path) -> tuple[dict[str, str], dict[str, dict], list[dict]]:
    """(id dokter -> nama, id job -> info job, job selesai yang belum diverifikasi)."""
    if not db_path.is_file():
        return {}, {}, []
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        names = {r["id"]: r["name"] for r in conn.execute("SELECT id, name FROM doctors")}
        jobs = {r["id"]: dict(r) for r in conn.execute(
            "SELECT id, doctor_id, template_id, status, created_at FROM jobs")}
        pending = [dict(r) for r in conn.execute(
            "SELECT id, doctor_id, template_id, created_at FROM jobs WHERE status='done' ORDER BY created_at")]
    finally:
        conn.close()
    return names, jobs, pending


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--training-dir", type=Path, default=Path("runtime/training_data"))
    ap.add_argument("--db", type=Path, default=Path("runtime/osce.sqlite3"))
    ap.add_argument("--csv", type=Path, help="ekspor ringkasan per sesi ke CSV (simpan di runtime/)")
    args = ap.parse_args(argv)

    speaker = _jsonl(args.training_dir / "speaker_role.jsonl")
    intent = [r for r in _jsonl(args.training_dir / "text_intent.jsonl") if r.get("source") == "human_verified"]
    names, jobs, pending = _db(args.db)

    # ---------------- ML-1: per sesi & per dokter ----------------
    sessions: dict[str, dict] = {}
    for r in speaker:
        s = sessions.setdefault(r["session_id"], {"doctor_id": r["doctor_id"], "labels": Counter(), "fix": 0})
        s["labels"][r["label"]] += 1
        s["fix"] += r.get("model_label") is not None and r["model_label"] != r["label"]
    intent_by_job: dict[str, Counter] = defaultdict(Counter)
    for r in intent:
        intent_by_job[r.get("job_id", "?")]["n"] += 1
        intent_by_job[r.get("job_id", "?")]["fix"] += bool(r.get("corrected"))

    per_doc: dict[str, list[str]] = defaultdict(list)
    for sid, s in sessions.items():
        per_doc[s["doctor_id"]].append(sid)

    print("=== ML-1 Speaker Role (runtime/training_data/speaker_role.jsonl)")
    if not sessions:
        print("  Belum ada sesi terverifikasi.")
    for doc, sids in sorted(per_doc.items(), key=lambda kv: names.get(kv[0], kv[0])):
        lab = Counter()
        fix = 0
        for sid in sids:
            lab += sessions[sid]["labels"]
            fix += sessions[sid]["fix"]
        n_seg = sum(lab.values())
        ok = "OK " if len(sids) >= MIN_SESSIONS_PER_DOCTOR else "-- "
        print(f"  {ok}{names.get(doc, '(dokter terhapus)')} [{doc}]: {len(sids)} sesi, {n_seg} segmen "
              f"(DOKTER {lab['DOKTER']}, PASIEN {lab['PASIEN']}), koreksi peran {fix} "
              f"({fix / max(n_seg, 1):.0%} = error model saat itu)")

    # ---------------- ML-2 ----------------
    lab_counts = Counter(r["label"] for r in intent)
    n_fix = sum(bool(r.get("corrected")) for r in intent)
    print("\n=== ML-2 Intent (human_verified di runtime/training_data/text_intent.jsonl)")
    print(f"  {len(intent)} baris dari {len(intent_by_job)} sesi, dikoreksi manusia {n_fix} "
          f"({n_fix / max(len(intent), 1):.0%})")
    print("  Distribusi:", ", ".join(f"{lab} {lab_counts.get(lab, 0)}" for lab in INTENT_LABELS))

    if pending:
        print(f"\n=== {len(pending)} job selesai tapi BELUM diverifikasi (datanya belum masuk training):")
        for j in pending:
            print(f"  /review?job={j['id']}  dokter={names.get(j['doctor_id'], '-')}  template={j['template_id']}"
                  f"  {j['created_at'][:16]}")

    # ---------------- Kesiapan ----------------
    n_doc_ok = sum(len(s) >= MIN_SESSIONS_PER_DOCTOR for s in per_doc.values())
    both_labels = all(sessions[sid]["labels"]["DOKTER"] and sessions[sid]["labels"]["PASIEN"]
                      for sid in sessions)
    checks = [
        (f">= {MIN_DOCTORS} dokter dengan >= {MIN_SESSIONS_PER_DOCTOR} sesi", n_doc_ok >= MIN_DOCTORS,
         f"{n_doc_ok} dokter memenuhi"),
        ("setiap sesi punya segmen DOKTER dan PASIEN", bool(sessions) and both_labels,
         "cek sesi yang hanya 1 peran" if sessions and not both_labels else ""),
        (f">= {MIN_INTENT_ROWS} baris intent manusia", len(intent) >= MIN_INTENT_ROWS, f"{len(intent)} baris"),
        ("kelima kelas intent ada", all(lab_counts.get(lab) for lab in INTENT_LABELS),
         "kurang: " + ", ".join(lab for lab in INTENT_LABELS if not lab_counts.get(lab))),
    ]
    print("\n=== Kesiapan training")
    for name, ok, note in checks:
        print(f"  [{'x' if ok else ' '}] {name}" + (f"  ({note})" if note else ""))
    if checks[0][1] and checks[1][1]:
        print("  -> python scripts/train_speaker_role.py --data runtime/training_data/speaker_role.jsonl --model svm")
    elif len(sessions) >= 3:
        print("  -> ML-1 sudah bisa dilatih (split per SESI), tetapi generalisasi ke dokter baru BELUM teruji.")
    if len(intent) >= 20:
        print("  -> python scripts/train_text_intent.py --data data/synthetic/text_intent.jsonl "
              "data/synthetic/text_intent_llm.jsonl runtime/training_data/text_intent.jsonl "
              "--test-source human_verified")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8-sig", newline="") as fh:
            fh.write("sep=;\n")
            w = csv.writer(fh, delimiter=";")
            w.writerow(["job_id", "dokter", "template", "dibuat", "seg_dokter", "seg_pasien", "koreksi_peran",
                        "baris_intent", "koreksi_intent"])
            for sid, s in sorted(sessions.items(), key=lambda kv: jobs.get(kv[0], {}).get("created_at", "")):
                j = jobs.get(sid, {})
                w.writerow([sid, names.get(s["doctor_id"], s["doctor_id"]), j.get("template_id", ""),
                            j.get("created_at", "")[:16], s["labels"]["DOKTER"], s["labels"]["PASIEN"], s["fix"],
                            intent_by_job[sid]["n"], intent_by_job[sid]["fix"]])
        print(f"\nCSV -> {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
