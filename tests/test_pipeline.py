"""Uji end-to-end CPU: ASR & LLM diganti fake (tanpa GPU/model), komponen lain asli."""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.core.audio_io import load_audio, write_wav
from app.core.job_queue import JobQueue
from app.core.model_slot import GpuModelSlot
from app.ml.speaker_role import SpeakerRoleClassifier, build_anchor
from app.ml.text_intent import IntentClassifier
from app.nlp.lexicon import MedicalLexicon
from app.pipeline.orchestrator import PipelineContext, process_job
from app.pipeline.stage1_preprocess import detect_speech, energy_vad, spectral_gate
from app.pipeline.stage2_asr import segment_words
from app.pipeline.stage4_summarize import summarize
from app.pipeline.stage5_finalize import VerificationError, verify_and_export
from app.schemas.pipeline import ReviewPayload, SpeechRegion, Utterance, Word
from app.schemas.templates import TEMPLATES, inline_json_schema, llm_json_schema
from app.storage.db import Repository
from conftest import synth_dialogue

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = [
    ("DOKTER", "Selamat pagi Pak, silakan duduk."),
    ("PASIEN", "Pagi dok. Saya Pak Budi, batuk berdahak sudah tiga hari."),
    ("DOKTER", "Ada demam atau sesak napas?"),
    ("PASIEN", "Panas 3 hari dok, kadang nggliyeng."),
    ("DOKTER", "Tekanan darahnya 120 per 80, suhunya 38 derajat."),
    ("DOKTER", "Kemungkinan Bapak terkena ISPA."),
    ("DOKTER", "Saya resepkan paracetamol tiga kali sehari sesudah makan."),
    ("PASIEN", "Terima kasih dok."),
]


class FakeASR:
    model_name = "fake-whisper"

    def __init__(self, spans):
        self.spans = spans
        self.closed = False

    def transcribe(self, audio, initial_prompt=""):
        words = []
        for (start, end), (_, text) in zip(self.spans, SCRIPT):
            toks = text.split()
            step = (end - start) / len(toks)
            words += [Word(start=start + i * step, end=start + (i + 0.9) * step, text=t, probability=0.9)
                      for i, t in enumerate(toks)]
        return words

    def close(self):
        self.closed = True


class FakeLLM:
    name = "fake-llm"

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def generate_json(self, system, user, schema):
        self.calls += 1
        return self.payload if isinstance(self.payload, str) else self.payload(self.calls)

    def close(self):
        pass


def make_settings(tmp_path: Path, **kw) -> Settings:
    s = Settings(data_dir=tmp_path, models_dir=tmp_path / "models", db_path=tmp_path / "t.sqlite3",
                 vad_backend="energy", llm_backend="ollama", **kw)
    s.ensure_dirs()
    return s


# ---------------------------------------------------------------------------------------------
def test_segment_words_splits_on_gap_and_region():
    words = [Word(start=0.0, end=0.3, text="Halo"), Word(start=0.35, end=0.6, text="dok."),
             Word(start=1.5, end=1.8, text="Iya"), Word(start=1.85, end=2.1, text="silakan")]
    utts = segment_words(words, [SpeechRegion(start=0, end=0.7), SpeechRegion(start=1.4, end=2.2)])
    assert [u.text for u in utts] == ["Halo dok.", "Iya silakan"]


def test_denoise_and_energy_vad_find_segments():
    audio, spans = synth_dialogue(["DOKTER", "PASIEN", "DOKTER"], seed=5)
    noisy = audio + 0.01 * np.random.default_rng(0).standard_normal(audio.size).astype(np.float32)
    clean = spectral_gate(noisy)
    assert clean.shape == noisy.shape
    regions = energy_vad(clean)
    assert len(regions) == 3
    for r, (s, e) in zip(regions, spans):
        assert abs(r.start - s) < 0.3 and abs(r.end - e) < 0.3
    _, backend = detect_speech(clean, backend="energy")
    assert backend == "energy"


def test_inline_schema_has_no_refs():
    for spec in TEMPLATES.values():
        assert "$ref" not in json.dumps(inline_json_schema(spec.model))


def test_llm_schema_requires_all_fields_and_is_in_prompt():
    """Regresi: tanpa `required`, LLM boleh keluarkan object minimal -> semua field jadi default."""
    def objects(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                yield node
            for v in node.values():
                yield from objects(v)
        elif isinstance(node, list):
            for v in node:
                yield from objects(v)

    for spec in TEMPLATES.values():
        schema = llm_json_schema(spec.model)
        objs = list(objects(schema))
        assert objs and all(set(o["required"]) == set(o["properties"]) for o in objs)

    seen = {}

    class SpyLLM(FakeLLM):
        def generate_json(self, system, user, schema):
            seen.update(user=user, schema=schema)
            return super().generate_json(system, user, schema)

    summarize(TEMPLATES["soap_umum"], _utts(), SpyLLM(json.dumps({"keluhan_utama": "Batuk"})))
    assert seen["schema"]["required"] and '"keluhan_utama"' in seen["user"]


def _utts():
    labels = ["CHITCHAT", "S_KELUHAN", "S_KELUHAN", "S_KELUHAN", "O_PEMERIKSAAN", "A_DIAGNOSIS",
              "P_EDUKASI_TERAPI", "CHITCHAT"]
    return [Utterance(id=i, start=i, end=i + 1, text=t, role=r, intent=lab)
            for i, ((r, t), lab) in enumerate(zip(SCRIPT, labels))]


def test_summarize_llm_valid_and_source_filtering():
    payload = json.dumps({"keluhan_utama": "Batuk produktif 3 hari", "onset": "Akut (<2 minggu)",
                          "subjective": [{"teks": "Batuk produktif", "sumber": [1, 99]}],
                          "assessment": [{"teks": "ISPA", "sumber": []}]})
    summary, source, warnings = summarize(TEMPLATES["soap_umum"], _utts(), FakeLLM(payload),
                                          locked={"sistem_organ": "Respirasi"})
    assert source == "llm"
    assert summary["subjective"][0]["sumber"] == [1]
    assert summary["sistem_organ"] == "Respirasi"  # parameter terkunci menang
    assert any("99" in w for w in warnings) and any("tanpa sumber" in w for w in warnings)


def test_summarize_retries_then_falls_back():
    llm = FakeLLM(lambda n: '{"onset": "kemarin"}')  # enum tidak valid
    summary, source, warnings = summarize(TEMPLATES["osce_respirasi"], _utts(), llm, max_retries=1)
    assert llm.calls == 2 and source == "fallback_rule"
    assert summary["plan"][0]["sumber"] == [6]
    assert summary["rubrik"]["menanyakan_demam"] is False


def test_model_slot_swaps():
    made = []

    class M:
        def __init__(self, n): self.n = n; self.closed = False; made.append(self)
        def close(self): self.closed = True

    slot = GpuModelSlot({"asr": lambda: M("asr"), "llm": lambda: M("llm")}, "swap")
    with slot.use("asr"):
        pass
    with slot.use("asr"):
        pass  # tidak dimuat ulang
    with slot.use("llm"):
        assert slot.loaded == ["llm"]
    assert len(made) == 2 and made[0].closed


def test_job_queue_sequential():
    order, active = [], []

    def proc(job_id):
        active.append(job_id)
        assert len(active) == 1, "lebih dari satu job berjalan bersamaan"
        order.append(job_id)
        active.pop()

    async def run():
        q = JobQueue(proc)
        await q.start()
        positions = [await q.submit(j) for j in ("a", "b", "c")]
        while q.snapshot()["length"] or q.snapshot()["current"]:
            await asyncio.sleep(0.01)
        await q.stop()
        return positions

    positions = asyncio.run(run())
    assert order == ["a", "b", "c"] and positions[0] >= 0


# ---------------------------------------------------------------------------------------------
def _ctx(settings, repo, spans, llm):
    return PipelineContext(
        settings=settings, repo=repo,
        slot=GpuModelSlot({"asr": lambda: FakeASR(spans), "llm": lambda: llm}, "swap"),
        speaker_clf=SpeakerRoleClassifier(None), intent_clf=IntentClassifier(None),
        lexicon=MedicalLexicon.from_file(ROOT / "data/lexicon/medical_lexicon.json"), glossary_prompt="")


@pytest.mark.parametrize("llm_ok", [True, False])
def test_end_to_end_job_and_verification(tmp_path, doctor_enroll_audio, llm_ok):
    settings = make_settings(tmp_path)
    repo = Repository(settings.db_path)
    anchor = build_anchor(doctor_enroll_audio)
    anchor.save(settings.anchors_dir / "doc1.npz")
    repo.create_doctor("dr. Andi", settings.anchors_dir / "doc1.npz", 14.0, "doc1")

    audio, spans = synth_dialogue([r for r, _ in SCRIPT], seed=7)
    wav = settings.uploads_dir / "job1.wav"
    write_wav(wav, audio)
    assert np.allclose(load_audio(wav)[:1000], audio[:1000], atol=1e-3)
    repo.create_job("job1", "doc1", "osce_respirasi", wav, ["Budi"], {"sistem_organ": "Respirasi"}, "127.0.0.1")

    good = json.dumps({"keluhan_utama": "Batuk produktif", "batuk": "Berdahak", "subjective": [{"teks": "x", "sumber": [1]}]})
    process_job(_ctx(settings, repo, spans, FakeLLM(good if llm_ok else "bukan json")), "job1")

    job = repo.get_job("job1")
    assert job["status"] == "done", job["error"]
    res = job["result"]
    assert res["summary_source"] == ("llm" if llm_ok else "fallback_rule")
    assert res["summary"]["sistem_organ"] == "Respirasi"
    roles = [u["role"] for u in res["utterances"]]
    assert roles == [r for r, _ in SCRIPT], roles
    assert all("Budi" not in u["text"] for u in res["utterances"])
    assert "pusing berputar" in res["utterances"][3]["text_normalized"]
    assert not wav.exists()  # audio mentah dihapus (privasi)

    payload = ReviewPayload.model_validate(job["result"])
    payload.utterances[0].intent = "CHITCHAT"
    payload.utterances[2].role = "DOKTER"
    case = verify_and_export(settings, repo, "job1", payload)
    assert repo.get_job("job1")["status"] == "verified"
    assert case["summary"]["sistem_organ"] == "Respirasi"
    assert len((settings.training_dir / "text_intent.jsonl").read_text().splitlines()) == len(SCRIPT)
    rows = [json.loads(x) for x in (settings.training_dir / "speaker_role.jsonl").read_text().splitlines()]
    assert len(rows) == len(SCRIPT) and len(rows[0]["vector"]) > 50
    with pytest.raises(VerificationError):
        verify_and_export(settings, repo, "job1", payload)


def test_train_speaker_script_on_active_learning_data(tmp_path, doctor_enroll_audio):
    """Simulasi beberapa sesi terverifikasi -> script training berjalan & menghasilkan metrics.json."""
    settings = make_settings(tmp_path)
    repo = Repository(settings.db_path)
    build_anchor(doctor_enroll_audio).save(settings.anchors_dir / "doc1.npz")
    repo.create_doctor("dr. Andi", settings.anchors_dir / "doc1.npz", 14.0, "doc1")
    good = json.dumps({"keluhan_utama": "Batuk"})
    for k in range(4):
        audio, spans = synth_dialogue([r for r, _ in SCRIPT], seed=20 + k)
        wav = settings.uploads_dir / f"j{k}.wav"
        write_wav(wav, audio)
        repo.create_job(f"j{k}", "doc1", "soap_umum", wav, [], {}, None)
        process_job(_ctx(settings, repo, spans, FakeLLM(good)), f"j{k}")
        verify_and_export(settings, repo, f"j{k}", ReviewPayload.model_validate(repo.get_job(f"j{k}")["result"]))
    out = tmp_path / "speaker_model"
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/train_speaker_role.py"),
                           "--data", str(settings.training_dir / "speaker_role.jsonl"),
                           "--anchors-dir", str(settings.anchors_dir), "--out", str(out), "--model", "logreg"],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    metrics = json.loads((out / "metrics.json").read_text())
    assert "confusion_matrix" in metrics["test"] and "baseline_heuristik_anchor" in metrics["comparison"]
    assert SpeakerRoleClassifier.load(out / "model.joblib").is_trained


def test_deepseek_backend_payload_and_errors(monkeypatch):
    import io
    import json as _json

    import pytest

    from app.llm import backends

    with pytest.raises(backends.LLMError):
        backends.DeepSeekBackend(api_key="")

    seen = {}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        seen["body"] = _json.loads(req.data.decode("utf-8"))
        return _Resp(_json.dumps({"choices": [{"message": {"content": '{"ok": true}'}}]}).encode())

    monkeypatch.setattr(backends.urllib.request, "urlopen", fake_urlopen)
    llm = backends.DeepSeekBackend(api_key="sk-test", model="deepseek-chat")
    assert llm.generate_json("sys", "keluarkan JSON", {"type": "object"}) == '{"ok": true}'
    assert seen["url"] == "https://api.deepseek.com/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["temperature"] == 0
    assert "sk-test" not in repr(llm.__dict__.get("name"))
