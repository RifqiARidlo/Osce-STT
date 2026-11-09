"""Pre-LLM Medical Lexicon Mapper.

Normalisasi istilah awam / bahasa daerah / salah eja ASR -> istilah medis baku sebelum masuk LLM.
- Pola regex (case-insensitive, boleh memakai grup tangkap \\1 di replacement).
- Teks asli dipertahankan dalam kurung (keep_original) agar LLM & reviewer tetap melihat sumbernya;
  mapping ke istilah diagnostik TIDAK dilakukan (mis. "nggliyeng" -> gejala "pusing berputar",
  bukan langsung "vertigo"), karena itu keputusan klinis, bukan normalisasi bahasa.
- Single pass dengan placeholder Private Use Area -> hasil satu aturan tidak dicocokkan ulang oleh
  aturan lain (mencegah penggantian berantai).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_PUA_OPEN, _PUA_CLOSE, _PUA_BASE = "", "", 0xE100


@dataclass(frozen=True)
class LexiconEntry:
    pattern: re.Pattern[str]
    replacement: str
    keep_original: bool
    category: str


class MedicalLexicon:
    def __init__(self, entries: list[LexiconEntry]) -> None:
        self.entries = entries

    @classmethod
    def from_file(cls, path: Path) -> "MedicalLexicon":
        if not path.is_file():
            log.warning("Lexicon tidak ditemukan di %s; mapper nonaktif.", path)
            return cls([])
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries: list[LexiconEntry] = []
        for i, item in enumerate(raw.get("entries", [])):
            try:
                entries.append(LexiconEntry(
                    pattern=re.compile(item["pattern"], re.IGNORECASE),
                    replacement=item["replacement"],
                    keep_original=bool(item.get("keep_original", True)),
                    category=item.get("category", "umum"),
                ))
            except (KeyError, re.error) as exc:
                log.error("Entry lexicon #%d tidak valid (%s): %r", i, exc, item)
        return cls(entries)

    def apply(self, text: str) -> tuple[str, list[str]]:
        """Return (teks ternormalisasi, daftar 'asli -> baku')."""
        if not self.entries or not text:
            return text, []
        stash: list[str] = []
        hits: list[str] = []
        working = text
        for entry in self.entries:
            def _sub(m: re.Match[str], entry: LexiconEntry = entry) -> str:
                original = m.group(0)
                replaced = m.expand(entry.replacement)
                out = f"{replaced} ({original})" if entry.keep_original and replaced.lower() != original.lower() else replaced
                stash.append(out)
                hits.append(f"{original} -> {replaced}")
                return f"{_PUA_OPEN}{chr(_PUA_BASE + len(stash) - 1)}{_PUA_CLOSE}"
            working = entry.pattern.sub(_sub, working)
        restored = re.sub(f"{_PUA_OPEN}(.){_PUA_CLOSE}", lambda m: stash[ord(m.group(1)) - _PUA_BASE], working)
        return restored, hits
