import json
from pathlib import Path

import numpy as np

from app.ml.text_intent import IntentClassifier, build_pipeline, rule_based_intent, to_frame
from app.nlp.lexicon import MedicalLexicon
from app.nlp.pii import scrub
from scripts.generate_synthetic_text import generate

ROOT = Path(__file__).resolve().parent.parent


def test_intent_pipeline_train_predict(tmp_path):
    rows = generate(8, seed=1)
    X = to_frame([r["text"] for r in rows], [r["role"] for r in rows])
    pipe = build_pipeline("logreg").fit(X, [r["label"] for r in rows])
    IntentClassifier.save_bundle(tmp_path / "m.joblib", pipe, "logreg", False)
    clf = IntentClassifier.load(tmp_path / "m.joblib")
    out = clf.predict(["Tekanan darahnya 130 per 80.", "Saya resepkan paracetamol tiga kali sehari"],
                      ["DOKTER", "DOKTER"])
    assert [o[0] for o in out] == ["O_PEMERIKSAAN", "P_EDUKASI_TERAPI"]
    assert all(abs(sum(o[2].values()) - 1) < 1e-3 for o in out)


def test_intent_fallback_rules_without_model(tmp_path):
    clf = IntentClassifier.load(tmp_path / "missing.joblib")
    assert not clf.is_trained
    assert rule_based_intent("Selamat pagi dok", "PASIEN")[0] == "CHITCHAT"
    assert clf.predict(["batuk sudah tiga hari"], ["PASIEN"])[0][0] == "S_KELUHAN"


def test_lexicon_mapping_no_chaining():
    lex = MedicalLexicon.from_file(ROOT / "data/lexicon/medical_lexicon.json")
    text, hits = lex.apply("Panas 3 hari, kepala nggliyeng, batuk berdahak, tensinya 130 per 80")
    assert "febris onset 3 hari SMRS (Panas 3 hari)" in text
    assert "pusing berputar (nggliyeng)" in text
    assert "batuk produktif (batuk berdahak)" in text
    assert "130/80 mmHg" in text
    assert "dispnea" not in text  # tidak ada penggantian berantai
    assert len(hits) == 5


def test_lexicon_file_is_valid_json_and_compiles():
    raw = json.loads((ROOT / "data/lexicon/medical_lexicon.json").read_text(encoding="utf-8"))
    lex = MedicalLexicon.from_file(ROOT / "data/lexicon/medical_lexicon.json")
    assert len(lex.entries) == len(raw["entries"])


def test_pii_scrubber():
    text = ("Nama saya Budi Santoso dok, NIK 3578123456789012, HP 0812-3456-7890. "
            "Rumah di Jalan Mawar No 5, RT 03 RW 04. Lahir tanggal 12 Maret 1980. "
            "Tadi diantar Pak Slamet. Bu dokter bilang Rina juga sakit.")
    out, hits = scrub(text, extra_names=["Rina"], doctor_names=["dr. Andi"])
    for leaked in ("Budi", "Santoso", "3578123456789012", "0812", "Mawar", "Slamet", "Rina", "1980"):
        assert leaked not in out, (leaked, out)
    assert "Bu dokter" in out
    assert "[Pasien_Anonim]" in out and "[NIK_ANONIM]" in out and "[TELP_ANONIM]" in out
    assert hits


def test_pii_keeps_clinical_numbers():
    out, _ = scrub("Tekanan darah 130/80, nadi 88, suhu 38,5, umur 45 tahun.")
    assert out == "Tekanan darah 130/80, nadi 88, suhu 38,5, umur 45 tahun."
    assert np.isfinite(1.0)
