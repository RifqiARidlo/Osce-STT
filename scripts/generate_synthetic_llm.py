"""Generator dialog sintetis via LLM lokal (Ollama) dengan output JSON terkunci schema.

Menghasilkan variasi leksikal yang jauh lebih kaya daripada template. Setiap dialog = satu grup
(`template_id` = "llm_<n>") sehingga split train/test tetap berbasis grup.
Label dari LLM BELUM tentu benar -> sampel acak wajib dicek manual sebelum dipakai sebagai test set.

Contoh:
    python scripts/generate_synthetic_llm.py --n-dialogs 40 --out data/synthetic/text_intent_llm.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.llm.backends import LLMError, OllamaBackend  # noqa: E402

KASUS = ["ISPA", "faringitis", "pneumonia", "asma", "TB paru", "dispepsia", "gastroenteritis", "demam tifoid",
         "DBD", "hipertensi", "diabetes melitus tipe 2", "vertigo", "migrain", "ISK", "dermatitis", "angina pektoris",
         "gagal jantung", "PPOK", "konjungtivitis", "otitis media"]
GAYA = ["pasien lansia berbahasa campur Jawa", "pasien muda santai", "ibu yang membawa anak", "pasien cemas",
        "pasien pekerja pabrik", "pasien yang singkat-singkat menjawab"]


class Line(BaseModel):
    role: Literal["DOKTER", "PASIEN"]
    text: str = Field(min_length=2)
    label: Literal["S_KELUHAN", "O_PEMERIKSAAN", "A_DIAGNOSIS", "P_EDUKASI_TERAPI", "CHITCHAT"]


class Dialog(BaseModel):
    lines: list[Line] = Field(min_length=8)


SYSTEM = """Anda pembuat data latih percakapan klinis berbahasa Indonesia sehari-hari (bukan bahasa buku).
Label: S_KELUHAN (keluhan/anamnesis, termasuk pertanyaan anamnesis dokter), O_PEMERIKSAAN (hasil/instruksi
pemeriksaan fisik & tanda vital), A_DIAGNOSIS (simpulan/diagnosis), P_EDUKASI_TERAPI (obat, edukasi, rencana,
rujukan, kontrol), CHITCHAT (basa-basi tanpa informasi medis). Satu baris = satu kalimat ucapan."""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-dialogs", type=int, default=30)
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/synthetic/text_intent_llm.jsonl"))
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    llm = OllamaBackend(args.ollama_url, args.model, num_ctx=4096, timeout_s=600, unload_after=False)
    schema = Dialog.model_json_schema()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with args.out.open("a", encoding="utf-8") as fh:
        for i in range(args.n_dialogs):
            kasus, gaya = rng.choice(KASUS), rng.choice(GAYA)
            user = (f"Buat satu percakapan konsultasi 14-24 baris di puskesmas. Kasus: {kasus}. Gaya: {gaya}. "
                    "Sertakan basa-basi pembuka & penutup, anamnesis, pemeriksaan tanda vital, diagnosis, terapi & edukasi.")
            try:
                dialog = Dialog.model_validate_json(llm.generate_json(SYSTEM, user, schema))
            except (LLMError, ValidationError) as exc:
                print(f"[{i}] gagal: {str(exc)[:120]}")
                continue
            for line in dialog.lines:
                fh.write(json.dumps({"text": line.text, "role": line.role, "label": line.label,
                                     "template_id": f"llm_{args.seed}_{i}", "source": "synthetic_llm",
                                     "kasus": kasus}, ensure_ascii=False) + "\n")
            n_ok += 1
            print(f"[{i}] {kasus}: {len(dialog.lines)} baris")
    llm.close()
    print(f"{n_ok}/{args.n_dialogs} dialog tersimpan -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
