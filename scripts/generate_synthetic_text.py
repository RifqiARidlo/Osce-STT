"""Generator dataset sintetis (template + slot filling) untuk cold-start Text Intent Classifier.

Output JSONL: {"text", "role", "label", "template_id", "source": "synthetic_template"}

PERINGATAN METODOLOGI: data template memiliki variasi leksikal terbatas. Skor test yang tinggi
pada split acak TIDAK berarti model bagus pada percakapan nyata. Karena itu `template_id` disimpan
dan script training memakai split berbasis grup (template yang di-test tidak pernah dilihat saat
training). Evaluasi final wajib memakai transkrip nyata hasil verifikasi manusia.

Contoh:
    python scripts/generate_synthetic_text.py --n-per-template 25 --out data/synthetic/text_intent.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

SLOTS: dict[str, list[str]] = {
    "sapaan": ["Pak", "Bu", "Mas", "Mbak", "Dik"],
    "gejala": ["demam", "panas", "batuk", "batuk berdahak", "batuk kering", "pilek", "sesak napas", "pusing",
               "sakit kepala", "mual", "muntah", "mencret", "sakit perut", "nyeri ulu hati", "nyeri dada", "lemas",
               "gatal-gatal", "nyeri sendi", "kemeng", "nggliyeng", "mules", "meriang", "susah tidur",
               "bengkak di kaki", "dada ampeg", "deg-degan", "tenggorokan sakit", "kencing perih"],
    "durasi": ["dua hari", "3 hari", "tiga hari", "seminggu", "dua minggu", "sebulan", "sejak kemarin",
               "sejak tadi pagi", "5 hari", "empat hari", "kurang lebih satu minggu"],
    "waktu": ["malam hari", "pagi hari", "setelah makan", "kalau capek", "saat beraktivitas", "kalau tiduran"],
    "lokasi": ["perut kanan bawah", "ulu hati", "dada kiri", "kepala bagian belakang", "pinggang", "tenggorokan"],
    "riwayat": ["darah tinggi", "kencing manis", "asma", "maag", "TBC", "sakit jantung", "alergi obat"],
    "keluarga": ["ibu", "bapak", "kakak", "adik", "nenek", "suami", "istri"],
    "sys": ["110", "120", "130", "140", "150", "160", "100"],
    "dia": ["70", "80", "90", "100", "60"],
    "nadi": ["72", "80", "88", "96", "104", "110"],
    "suhu": ["36,8", "37,5", "38", "38,5", "39", "37,2"],
    "spo2": ["98", "97", "95", "93", "99"],
    "rr": ["18", "20", "22", "24", "28"],
    "temuan": ["terdengar ronkhi di paru kanan bawah", "ada wheezing di kedua lapang paru",
               "suara napas vesikuler, tidak ada ronkhi", "ada nyeri tekan di ulu hati", "tenggorokannya merah",
               "amandelnya membesar", "bising usus meningkat", "tidak ada pembesaran kelenjar getah bening",
               "konjungtiva tidak anemis", "ada edema di kedua tungkai", "perkusi paru sonor",
               "turgor kulit baik", "ada nyeri ketok di pinggang kanan"],
    "dx": ["ISPA", "faringitis akut", "bronkitis akut", "pneumonia", "asma eksaserbasi", "dispepsia", "gastritis",
           "gastroenteritis akut", "hipertensi", "demam tifoid", "demam berdarah", "vertigo", "migrain",
           "TB paru", "infeksi saluran kemih", "dermatitis kontak", "common cold", "tonsilitis"],
    "obat": ["paracetamol 500 mg", "amoksisilin 500 mg", "antasida sirup", "omeprazol 20 mg", "ambroxol",
             "CTM", "oralit", "salbutamol", "cetirizine 10 mg", "amlodipin 5 mg", "ibuprofen 400 mg",
             "betahistin", "zinc"],
    "frek": ["tiga kali sehari", "dua kali sehari", "sekali sehari", "tiap delapan jam", "kalau demam saja"],
    "aturan": ["sesudah makan", "sebelum makan", "sebelum tidur", "sampai habis"],
    "edukasi": ["banyak minum air putih", "istirahat yang cukup", "hindari makanan pedas dan asam",
                "jangan telat makan", "kurangi garam", "pakai masker", "berhenti merokok",
                "jaga kebersihan tangan", "makan sedikit tapi sering", "olahraga ringan teratur"],
    "kontrol": ["tiga hari lagi", "seminggu lagi", "kalau obatnya habis", "lusa", "minggu depan"],
    "lab": ["cek darah lengkap", "rontgen dada", "tes dahak", "cek gula darah", "cek urin", "tes widal"],
    "basa": ["cuacanya panas sekali ya hari ini", "tadi ke sini naik apa", "antreannya lama ya tadi",
             "rumahnya jauh dari sini", "macet sekali di jalan tadi", "hujan terus akhir-akhir ini"],
}

# (template_id, role, label, pola)
TEMPLATES: list[tuple[str, str, str, str]] = [
    # ---------------- S_KELUHAN ----------------
    ("s01", "PASIEN", "S_KELUHAN", "Saya {gejala} sudah {durasi}, dok."),
    ("s02", "PASIEN", "S_KELUHAN", "Ini dok, {gejala} terus dari {durasi}."),
    ("s03", "PASIEN", "S_KELUHAN", "Keluhannya {gejala} sama {gejala}, dok, sudah {durasi}."),
    ("s04", "PASIEN", "S_KELUHAN", "Biasanya kambuh {waktu}, dok."),
    ("s05", "PASIEN", "S_KELUHAN", "Sakitnya di {lokasi}, rasanya seperti ditusuk-tusuk."),
    ("s06", "PASIEN", "S_KELUHAN", "Iya dok, ada {gejala} juga."),
    ("s07", "PASIEN", "S_KELUHAN", "Dulu pernah {riwayat}, dok."),
    ("s08", "PASIEN", "S_KELUHAN", "{keluarga} saya juga punya {riwayat}."),
    ("s09", "PASIEN", "S_KELUHAN", "Sudah minum obat warung tapi {gejala}nya belum hilang."),
    ("s10", "PASIEN", "S_KELUHAN", "Awalnya {gejala}, terus {durasi} ini jadi {gejala}."),
    ("s11", "PASIEN", "S_KELUHAN", "Badan saya rasanya {gejala} dok, nggak enak semua."),
    ("s12", "PASIEN", "S_KELUHAN", "Tidak dok, tidak ada {gejala}."),
    ("s13", "DOKTER", "S_KELUHAN", "Keluhannya apa, {sapaan}?"),
    ("s14", "DOKTER", "S_KELUHAN", "Sudah berapa lama {gejala}nya?"),
    ("s15", "DOKTER", "S_KELUHAN", "Apakah ada {gejala} juga?"),
    ("s16", "DOKTER", "S_KELUHAN", "Ada riwayat {riwayat} sebelumnya, {sapaan}?"),
    ("s17", "DOKTER", "S_KELUHAN", "Di keluarga ada yang sakit {riwayat}?"),
    ("s18", "DOKTER", "S_KELUHAN", "Sakitnya muncul {waktu} atau terus-menerus?"),
    ("s19", "DOKTER", "S_KELUHAN", "Sudah minum obat apa saja untuk keluhan ini?"),
    ("s20", "DOKTER", "S_KELUHAN", "Ada alergi obat atau makanan, {sapaan}?"),
    ("s21", "DOKTER", "S_KELUHAN", "Sakitnya menjalar ke mana, {sapaan}?"),
    ("s22", "DOKTER", "S_KELUHAN", "Merokok tidak, {sapaan}? Sehari berapa batang?"),
    # ---------------- O_PEMERIKSAAN ----------------
    ("o01", "DOKTER", "O_PEMERIKSAAN", "Tekanan darahnya {sys} per {dia}."),
    ("o02", "DOKTER", "O_PEMERIKSAAN", "Tensinya {sys}/{dia}, nadinya {nadi} kali per menit."),
    ("o03", "DOKTER", "O_PEMERIKSAAN", "Suhunya {suhu} derajat."),
    ("o04", "DOKTER", "O_PEMERIKSAAN", "Saturasi oksigennya {spo2} persen."),
    ("o05", "DOKTER", "O_PEMERIKSAAN", "Napasnya {rr} kali per menit."),
    ("o06", "DOKTER", "O_PEMERIKSAAN", "Dari pemeriksaan, {temuan}."),
    ("o07", "DOKTER", "O_PEMERIKSAAN", "Saya dengarkan parunya ya, {temuan}."),
    ("o08", "DOKTER", "O_PEMERIKSAAN", "Hasil pemeriksaan fisiknya {temuan} dan {temuan}."),
    ("o09", "DOKTER", "O_PEMERIKSAAN", "Coba tarik napas dalam, {sapaan}. Iya, {temuan}."),
    ("o10", "DOKTER", "O_PEMERIKSAAN", "Saya tekan di sini sakit tidak? Oke, {temuan}."),
    ("o11", "DOKTER", "O_PEMERIKSAAN", "Suhu {suhu}, nadi {nadi}, tensi {sys} per {dia}."),
    ("o12", "DOKTER", "O_PEMERIKSAAN", "Coba buka mulutnya, {sapaan}, aaa. Ya, {temuan}."),
    # ---------------- A_DIAGNOSIS ----------------
    ("a01", "DOKTER", "A_DIAGNOSIS", "Dari hasil pemeriksaan, kemungkinan {sapaan} terkena {dx}."),
    ("a02", "DOKTER", "A_DIAGNOSIS", "Ini mengarah ke {dx}, {sapaan}."),
    ("a03", "DOKTER", "A_DIAGNOSIS", "Diagnosis sementaranya {dx}, tapi perlu disingkirkan {dx}."),
    ("a04", "DOKTER", "A_DIAGNOSIS", "Saya curiga ini {dx}."),
    ("a05", "DOKTER", "A_DIAGNOSIS", "Jadi kesimpulannya {sapaan} mengalami {dx}."),
    ("a06", "DOKTER", "A_DIAGNOSIS", "Kemungkinan besar {dx}, bukan {dx}."),
    ("a07", "DOKTER", "A_DIAGNOSIS", "Keluhan seperti ini biasanya karena {dx}."),
    ("a08", "PASIEN", "A_DIAGNOSIS", "Jadi saya kena {dx} ya, dok?"),
    # ---------------- P_EDUKASI_TERAPI ----------------
    ("p01", "DOKTER", "P_EDUKASI_TERAPI", "Saya resepkan {obat}, diminum {frek} {aturan}."),
    ("p02", "DOKTER", "P_EDUKASI_TERAPI", "Obatnya {obat} {frek}, ya."),
    ("p03", "DOKTER", "P_EDUKASI_TERAPI", "Jangan lupa {edukasi}."),
    ("p04", "DOKTER", "P_EDUKASI_TERAPI", "Selain obat, {sapaan} harus {edukasi} dan {edukasi}."),
    ("p05", "DOKTER", "P_EDUKASI_TERAPI", "Kontrol lagi {kontrol} ya, {sapaan}."),
    ("p06", "DOKTER", "P_EDUKASI_TERAPI", "Kalau tidak membaik, kita {lab}."),
    ("p07", "DOKTER", "P_EDUKASI_TERAPI", "Nanti sebelum pulang {lab} dulu ya."),
    ("p08", "DOKTER", "P_EDUKASI_TERAPI", "{obat} diminum {aturan}, antibiotiknya harus {aturan}."),
    ("p09", "DOKTER", "P_EDUKASI_TERAPI", "Kalau sesaknya makin berat, segera ke IGD."),
    ("p10", "DOKTER", "P_EDUKASI_TERAPI", "Saya buatkan surat rujukan ke spesialis ya."),
    ("p11", "PASIEN", "P_EDUKASI_TERAPI", "Obatnya diminum {aturan} atau bagaimana, dok?"),
    ("p12", "PASIEN", "P_EDUKASI_TERAPI", "Boleh makan pedas tidak, dok?"),
    # ---------------- CHITCHAT ----------------
    ("c01", "DOKTER", "CHITCHAT", "Selamat pagi, {sapaan}. Silakan duduk."),
    ("c02", "PASIEN", "CHITCHAT", "Pagi dok."),
    ("c03", "DOKTER", "CHITCHAT", "Wah, {basa}."),
    ("c04", "PASIEN", "CHITCHAT", "Iya dok, {basa}."),
    ("c05", "PASIEN", "CHITCHAT", "Terima kasih banyak, dok."),
    ("c06", "DOKTER", "CHITCHAT", "Sama-sama, semoga lekas sembuh ya, {sapaan}."),
    ("c07", "PASIEN", "CHITCHAT", "Baik dok."),
    ("c08", "DOKTER", "CHITCHAT", "Oke, sebentar ya saya tulis dulu."),
    ("c09", "PASIEN", "CHITCHAT", "Oh begitu ya dok."),
    ("c10", "DOKTER", "CHITCHAT", "Ada lagi yang mau ditanyakan, {sapaan}?"),
    ("c11", "PASIEN", "CHITCHAT", "Sudah dok, itu saja."),
    ("c12", "DOKTER", "CHITCHAT", "Ke sini sama siapa, {sapaan}?"),
]

FILLERS = ["eh", "ee", "anu", "hmm", "jadi", "gitu"]
_SLOT = re.compile(r"\{(\w+)\}")


def fill(pattern: str, rng: random.Random) -> str:
    return _SLOT.sub(lambda m: rng.choice(SLOTS[m.group(1)]), pattern)


def asr_noise(text: str, rng: random.Random, p_lower: float = 0.6, p_typo: float = 0.03,
              p_filler: float = 0.25) -> str:
    """Simulasi karakteristik output ASR: huruf kecil, tanpa tanda baca, filler, salah ketik ringan."""
    if rng.random() < p_lower:
        text = text.lower()
    if rng.random() < 0.4:
        text = re.sub(r"[.,?!]", "", text)
    words = text.split()
    if rng.random() < p_filler and words:
        words.insert(rng.randrange(len(words) + 1), rng.choice(FILLERS))
    out = []
    for w in words:
        if len(w) > 3 and rng.random() < p_typo:
            i = rng.randrange(len(w) - 1)
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        out.append(w)
    return " ".join(out)


def generate(n_per_template: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for tid, role, label, pattern in TEMPLATES:
        seen: set[str] = set()
        attempts = 0
        while len(seen) < n_per_template and attempts < n_per_template * 20:
            attempts += 1
            text = fill(pattern, rng)
            if rng.random() < 0.5:
                text = asr_noise(text, rng)
            if text in seen:
                continue
            seen.add(text)
            rows.append({"text": text, "role": role, "label": label, "template_id": tid,
                         "source": "synthetic_template"})
    rng.shuffle(rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-per-template", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/synthetic/text_intent.jsonl"))
    args = ap.parse_args(argv)
    rows = generate(args.n_per_template, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    print(f"{len(rows)} baris -> {args.out}  distribusi: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
