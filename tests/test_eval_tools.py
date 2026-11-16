"""Test alat evaluasi Fase 5: WER, audit label (CSV), status role-play. CPU, tanpa model."""
from __future__ import annotations

import json

from scripts.audit_llm_labels import apply as audit_apply
from scripts.audit_llm_labels import sample as audit_sample
from scripts.eval_wer import align, normalize, wer
from scripts.roleplay_status import main as status_main


def test_wer_normalization_ignores_format_only_differences():
    ref = normalize("Tekanan darah 120/80, suhu 38,5 derajat. Batuk tiga hari, saturasi 98%.")
    hyp = normalize("tekanan darah 120 -80 suhu 38 .5 derajat batuk 3 hari saturasi 98 persen")
    assert ref == hyp
    assert wer(align(ref, hyp)) == 0.0


def test_wer_counts_substitution_deletion_insertion():
    res = align("saya beri amoksisilin tiga kali".split(), "saya amok sisi lin tiga kali".split())
    assert res["n_ref"] == 5 and res["S"] + res["D"] + res["I"] == 3   # jarak edit minimum
    assert abs(wer(res) - 3 / 5) < 1e-9
    assert wer(align([], [])) == 0.0


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_audit_csv_filter_prefill_apply_roundtrip(tmp_path):
    data = tmp_path / "llm.jsonl"
    _write_jsonl(data, [
        {"text": "Perut saya sakit.", "role": "PASIEN", "label": "S_KELUHAN", "template_id": "llm_0_0"},
        {"text": "Baik dok, trims.", "role": "PASIEN", "label": "S_KELUHAN", "template_id": "llm_0_0"},
        {"text": "Ada keluhan apa?", "role": "DOKTER", "label": "S_KELUHAN", "template_id": "llm_0_0"},
    ])
    audit = tmp_path / "rev.csv"
    assert audit_sample(data, audit, n=0, seed=1, role="PASIEN", label="S_KELUHAN", prefill=True) == 0
    lines = audit.read_text(encoding="utf-8-sig").splitlines()
    assert lines[0] == "sep=;" and len(lines) == 2 + 2          # sep + header + 2 baris PASIEN
    # simulasi edit di Excel: baris "Baik dok" -> chitchat (huruf kecil tetap diterima)
    edited = [ln.rsplit(";", 2)[0] + ";chitchat;" if "trims" in ln else ln for ln in lines]
    audit.write_text("\n".join(edited) + "\n", encoding="utf-8-sig")
    assert audit_apply(data, audit) == 0
    rows = [json.loads(x) for x in data.read_text(encoding="utf-8").splitlines()]
    assert rows[1]["label"] == "CHITCHAT" and rows[1]["label_llm"] == "S_KELUHAN"
    assert rows[0]["label"] == "S_KELUHAN" and rows[2].get("audited") is None
    before = data.read_text(encoding="utf-8")
    assert audit_apply(data, audit) == 0 and data.read_text(encoding="utf-8") == before   # idempoten


def test_roleplay_status_reports_readiness(tmp_path, capsys):
    td = tmp_path / "training_data"
    td.mkdir()
    rows = []
    for d in range(3):
        for s in range(4):
            for u, role in enumerate(["DOKTER", "PASIEN", "DOKTER", "PASIEN"]):
                rows.append({"session_id": f"j{d}{s}", "doctor_id": f"doc{d}", "utt_id": u, "label": role,
                             "model_label": role, "vector": [0.0]})
    _write_jsonl(td / "speaker_role.jsonl", rows)
    out_csv = tmp_path / "status.csv"
    assert status_main(["--training-dir", str(td), "--db", str(tmp_path / "none.sqlite3"),
                        "--csv", str(out_csv)]) == 0
    out = capsys.readouterr().out
    assert "[x] >= 3 dokter dengan >= 4 sesi" in out and "train_speaker_role.py" in out
    assert len(out_csv.read_text(encoding="utf-8-sig").splitlines()) == 2 + 12
