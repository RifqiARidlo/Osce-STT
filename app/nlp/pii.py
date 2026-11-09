"""PII scrubber rule-based (Regex) untuk transkrip Bahasa Indonesia.

Keterbatasan (eksplisit): deteksi NAMA berbasis aturan memiliki recall terbatas — hanya menangkap
nama setelah kata pemicu ("nama saya ...", "Pak/Bu <Kapital>") dan nama yang diisi manual di UI.
Karena itu ekspor ke Bank Kasus OSCE diwajibkan melalui verifikasi manusia (Stage 5 HITL).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

PASIEN_TAG = "[Pasien_Anonim]"
DOKTER_TAG = "[Dokter]"

_MONTHS = r"(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)"
_NAME_STOP = {
    "dokter", "dok", "bidan", "perawat", "saya", "aku", "kamu", "anda", "sudah", "belum", "ya", "iya",
    "tidak", "nggak", "enggak", "ini", "itu", "tadi", "juga", "baik", "silakan", "terima", "kasih",
    "sakit", "mau", "mohon", "boleh", "nanti", "jadi", "kalau", "sama", "dan", "atau", "yang",
}


@dataclass(frozen=True)
class _Rule:
    name: str
    pattern: re.Pattern[str]
    replacement: str


_RULES: list[_Rule] = [
    _Rule("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL_ANONIM]"),
    _Rule("nik", re.compile(r"\b\d{16}\b"), "[NIK_ANONIM]"),
    _Rule("bpjs", re.compile(r"\b\d{13}\b"), "[NO_BPJS_ANONIM]"),
    _Rule("telepon", re.compile(r"(?<!\d)(?:\+?62|0)\s?8\d{1,3}[\s-]?\d{3,4}[\s-]?\d{3,5}(?!\d)"), "[TELP_ANONIM]"),
    _Rule("rekam_medis", re.compile(r"\b(no(?:mor)?\.?\s*(?:rm|rekam\s+medis|registrasi)\s*:?\s*)[\w./-]+",
                                    re.IGNORECASE), r"\1[NO_RM_ANONIM]"),
    _Rule("tanggal_lahir", re.compile(
        rf"\b(lahir(?:\s+(?:pada\s+)?tanggal)?\s+)(\d{{1,2}}[\s/-](?:\d{{1,2}}|{_MONTHS})[\s/-]\d{{2,4}})",
        re.IGNORECASE), r"\1[TGL_LAHIR_ANONIM]"),
    _Rule("rt_rw", re.compile(r"\brt\.?\s*\d{1,3}\s*/?\s*rw\.?\s*\d{1,3}\b", re.IGNORECASE), "[ALAMAT_ANONIM]"),
    _Rule("alamat", re.compile(
        r"\b(jalan|jl\.?|jln\.?|gang|gg\.?|perumahan|perum|komplek|kompleks|dusun|desa|kelurahan|kel\.)"
        r"\s+(?:[A-Za-z0-9.'-]+\s?){1,4}?(?:no(?:mor)?\.?\s*\d+[a-z]?)?(?=[,.?!]|\s+(?:rt|rw|dok|pak|bu)\b|$)",
        re.IGNORECASE), "[ALAMAT_ANONIM]"),
]

_NAME_CUE = re.compile(
    r"\b(nama\s+(?:saya|aku|lengkap(?:\s+saya)?|pasien|anak\s+saya|ibu\s+saya|bapak\s+saya|suami\s+saya|"
    r"istri\s+saya)(?:\s+(?:adalah|itu))?|saya\s+bernama|panggil\s+saja)\s+"
    r"((?:[A-Za-z'-]+\s?){1,3})", re.IGNORECASE)
_HONORIFIC = re.compile(r"\b(Pak|Bu|Bapak|Ibu|Mas|Mbak|Mba|Dik|Adik|Sdr\.?|Sdri\.?|Tn\.?|Ny\.?|Nn\.?)\s+"
                        r"([A-Z][a-z'-]+(?:\s+[A-Z][a-z'-]+){0,2})")


def _scrub_name_cue(m: re.Match[str]) -> str:
    cue, names = m.group(1), m.group(2)
    kept: list[str] = []
    tokens = names.split()
    for tok in tokens:
        if tok.lower() in _NAME_STOP:
            break
        kept.append(tok)
    if not kept:
        return m.group(0)
    rest = " ".join(tokens[len(kept):])
    return f"{cue} {PASIEN_TAG}" + (f" {rest}" if rest else "") + (" " if names.endswith(" ") else "")


def scrub(text: str, extra_names: list[str] | None = None, doctor_names: list[str] | None = None) -> tuple[str, list[str]]:
    """Return (teks_anonim, daftar jenis PII yang ditemukan)."""
    hits: list[str] = []
    out = text
    for names, tag in ((doctor_names or [], DOKTER_TAG), (extra_names or [], PASIEN_TAG)):
        for name in sorted({n.strip() for n in names if n and len(n.strip()) >= 3}, key=len, reverse=True):
            pattern = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
            out, n = pattern.subn(tag, out)
            if n:
                hits.append(f"nama_manual:{n}")
    for rule in _RULES:
        out, n = rule.pattern.subn(rule.replacement, out)
        if n:
            hits.append(f"{rule.name}:{n}")
    out, n = _NAME_CUE.subn(_scrub_name_cue, out)
    if n:
        hits.append(f"nama_pemicu:{n}")

    honor_count = 0

    def _honor(m: re.Match[str]) -> str:
        nonlocal honor_count
        if m.group(2).split()[0].lower() in _NAME_STOP:
            return m.group(0)
        honor_count += 1
        return f"{m.group(1)} {PASIEN_TAG}"
    out = _HONORIFIC.sub(_honor, out)
    if honor_count:
        hits.append(f"nama_sapaan:{honor_count}")
    return re.sub(r"\s{2,}", " ", out).strip(), hits
