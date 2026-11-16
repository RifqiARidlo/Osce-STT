"""Audit label data sintetis LLM: ambil sampel acak, periksa manual, terapkan koreksi.

Label dari LLM belum tentu benar. Akurasi label LLM pada sampel audit = estimasi tingkat noise label,
dilaporkan di docs/LAPORAN_EVALUASI.md.

Langkah:
    python scripts/audit_llm_labels.py sample --n 50 --seed 7
    # isi "label_audit" (wajib, salah satu INTENT_LABELS) dan "catatan" (opsional) di file audit
    python scripts/audit_llm_labels.py apply
    python scripts/audit_llm_labels.py fix-ack   # balasan tanpa isi ("Ya dok", "Oke") -> CHITCHAT

`apply` menulis label hasil audit ke file data (label asli LLM disimpan di field `label_llm`) dan
idempoten: menjalankannya dua kali tidak mengubah apa pun.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.schemas.pipeline import INTENT_LABELS  # noqa: E402

DATA = Path("data/synthetic/text_intent_llm.jsonl")
AUDIT = Path("data/synthetic/audit_llm_labels.jsonl")


# Pedoman anotasi: "iya/baik dok" tanpa isi = CHITCHAT. LLM cenderung memberi label sesuai fase dialog.
ACK = re.compile(r"^(oh |ya |iya |baik |oke |ok |siap |nggih |inggih )?"
                 r"(ya|iya|baik|oke|ok|siap|nggih|inggih|terima kasih|makasih)( ya)?,? ?"
                 r"(dok|dokter|pak dokter|bu dokter)?[.!]*$", re.IGNORECASE)


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def sample(data: Path, audit: Path, n: int, seed: int) -> int:
    if audit.exists():
        print(f"{audit} sudah ada; hapus dulu bila ingin sampel baru (hasil audit lama akan hilang).")
        return 1
    rows = _read(data)
    idx = sorted(random.Random(seed).sample(range(len(rows)), min(n, len(rows))))
    _write(audit, [{"idx": i, "template_id": rows[i]["template_id"], "role": rows[i]["role"],
                    "text": rows[i]["text"], "label_llm": rows[i].get("label_llm", rows[i]["label"]),
                    "label_audit": None, "catatan": ""} for i in idx])
    print(f"{len(idx)} baris dari {len(rows)} -> {audit}. Isi 'label_audit', lalu jalankan 'apply'.")
    return 0


def apply(data: Path, audit: Path) -> int:
    rows, items = _read(data), _read(audit)
    missing = [it["idx"] for it in items if it.get("label_audit") not in INTENT_LABELS]
    if missing:
        print(f"label_audit kosong/tidak valid untuk idx {missing}. Lengkapi dulu.")
        return 1
    changed = 0
    confusion: Counter[tuple[str, str]] = Counter()
    for it in items:
        r = rows[it["idx"]]
        if r["text"] != it["text"]:
            print(f"idx {it['idx']}: teks di data tidak cocok dengan audit (file data berubah?). Dibatalkan.")
            return 1
        llm_label = r.get("label_llm", r["label"])
        if it["label_audit"] != llm_label:
            confusion[(llm_label, it["label_audit"])] += 1
        if r["label"] != it["label_audit"]:
            r["label_llm"] = llm_label
            r["label"] = it["label_audit"]
            changed += 1
        r["audited"] = True
    _write(data, rows)
    n_wrong = sum(confusion.values())
    print(f"Akurasi label LLM pada sampel audit: {len(items) - n_wrong}/{len(items)} "
          f"= {(len(items) - n_wrong) / len(items):.1%}")
    for (src, dst), k in confusion.most_common():
        print(f"  {src} -> {dst}: {k}")
    print(f"{changed} label diperbarui di {data}")
    return 0


def fix_ack(data: Path) -> int:
    rows = _read(data)
    changed = 0
    for i, r in enumerate(rows):
        if ACK.match(r["text"].strip()) and r["label"] != "CHITCHAT":
            print(f"  idx {i}: {r['label']} -> CHITCHAT | {r['text']}")
            r.setdefault("label_llm", r["label"])
            r["label"] = "CHITCHAT"
            r["relabel_rule"] = "ack"
            changed += 1
    _write(data, rows)
    print(f"{changed} balasan tanpa isi dilabel ulang CHITCHAT di {data}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["sample", "apply", "fix-ack"])
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--audit", type=Path, default=AUDIT)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    if args.action == "sample":
        return sample(args.data, args.audit, args.n, args.seed)
    if args.action == "fix-ack":
        return fix_ack(args.data)
    return apply(args.data, args.audit)


if __name__ == "__main__":
    sys.exit(main())
