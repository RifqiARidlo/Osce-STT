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

# Contoh few-shot (kasus lain) agar model kecil tidak mengulang satu kalimat terus-menerus.
CONTOH = ('Contoh format (kasus lain, jangan disalin):\n'
          '{"lines":[{"role":"DOKTER","text":"Selamat pagi Bu, silakan duduk.","label":"CHITCHAT"},'
          '{"role":"DOKTER","text":"Ada keluhan apa hari ini?","label":"S_KELUHAN"},'
          '{"role":"PASIEN","text":"Perut saya perih sejak tiga hari, Dok.","label":"S_KELUHAN"},'
          '{"role":"DOKTER","text":"Saya periksa tekanan darahnya dulu ya, hasilnya 120 per 80.","label":"O_PEMERIKSAAN"},'
          '{"role":"DOKTER","text":"Kemungkinan ini maag atau dispepsia.","label":"A_DIAGNOSIS"},'
          '{"role":"DOKTER","text":"Saya beri antasida, diminum sebelum makan, dan hindari makanan pedas.",'
          '"label":"P_EDUKASI_TERAPI"}]}\n'
          'Dokter dan pasien bergantian bicara. Jangan mengulang kalimat yang sama.')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-dialogs", type=int, default=30)
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="> 0 wajib: temperature 0 membuat model kecil looping mengulang kalimat tanpa henti")
    ap.add_argument("--out", type=Path, default=Path("data/synthetic/text_intent_llm.jsonl"))
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    # num_predict membatasi output: tanpa batas, Ollama terus menggenerasi (context shift) bila model looping.
    llm = OllamaBackend(args.ollama_url, args.model, num_ctx=4096, timeout_s=600, unload_after=False,
                        options={"temperature": args.temperature, "top_p": 0.9, "repeat_penalty": 1.1,
                                 "num_predict": 2500})
    schema = Dialog.model_json_schema()
    schema["properties"]["lines"]["maxItems"] = 30
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Kombinasi (kasus, gaya) diambil TANPA pengembalian agar tidak ada dua dialog dari prompt identik di
    # grup berbeda (dialog hampir kembar lintas grup = kebocoran split grup).
    combos = [(k, g) for k in KASUS for g in GAYA]
    if args.n_dialogs > len(combos):
        print(f"--n-dialogs maksimal {len(combos)} (jumlah kombinasi kasus x gaya yang unik).")
        return 1
    n_ok = 0
    with args.out.open("a", encoding="utf-8") as fh:
        for i, (kasus, gaya) in enumerate(rng.sample(combos, args.n_dialogs)):
            llm.options["seed"] = args.seed * 1000 + i  # reprodusibel per dialog
            user = (f"Buat satu percakapan konsultasi 14-24 baris di puskesmas. Kasus: {kasus}. Gaya: {gaya}. "
                    "Sertakan basa-basi pembuka & penutup, anamnesis, pemeriksaan tanda vital, diagnosis, terapi & edukasi.\n"
                    + CONTOH)
            try:
                dialog = Dialog.model_validate_json(llm.generate_json(SYSTEM, user, schema))
            except (LLMError, ValidationError) as exc:
                print(f"[{i}] gagal: {str(exc)[:120]}")
                continue
            for line in dialog.lines:
                fh.write(json.dumps({"text": line.text, "role": line.role, "label": line.label,
                                     "template_id": f"llm_{args.seed}_{i}", "source": "synthetic_llm",
                                     "kasus": kasus}, ensure_ascii=False) + "\n")
            fh.flush()  # dialog selesai tetap tersimpan bila proses dihentikan di tengah
            n_ok += 1
            print(f"[{i}] {kasus}: {len(dialog.lines)} baris")
    llm.close()
    print(f"{n_ok}/{args.n_dialogs} dialog tersimpan -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
