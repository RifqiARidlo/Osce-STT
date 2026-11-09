"""Uji lapisan HTTP (FastAPI) lewat TestClient. ASR & LLM di-mock seperti test_pipeline; komponen lain asli."""
import io
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.core.audio_io import write_wav
from app.core.model_slot import GpuModelSlot
from app.core.services import build_services
from conftest import synth_dialogue
from test_pipeline import SCRIPT, FakeASR, FakeLLM, make_settings

TOKEN = "token-uji-123"
AUTH = {"x-access-token": TOKEN}
GOOD_LLM = json.dumps({"keluhan_utama": "Batuk produktif", "batuk": "Berdahak",
                       "subjective": [{"teks": "Batuk berdahak 3 hari", "sumber": [1]}]})


def _wav_bytes(audio, tmp_dir: Path) -> io.BytesIO:
    path = tmp_dir / "upload_src.wav"
    write_wav(path, audio)
    return io.BytesIO(path.read_bytes())


@pytest.fixture
def client(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, access_token=TOKEN,
                             speaker_model_path=tmp_path / "models/speaker_role/model.joblib",
                             intent_model_path=tmp_path / "models/text_intent/model.joblib")
    _, spans = synth_dialogue([r for r, _ in SCRIPT], seed=7)

    def fake_build_services(s):
        svc = build_services(s)
        svc.ctx.slot = GpuModelSlot({"asr": lambda: FakeASR(spans), "llm": lambda: FakeLLM(GOOD_LLM)}, "swap")
        return svc

    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "build_services", fake_build_services)
    with TestClient(main.app) as c:
        c.settings = settings
        yield c


def _enroll(client, audio) -> str:
    r = client.post("/api/doctors", headers=AUTH, data={"name": "dr. Uji"},
                    files={"audio": ("enroll.wav", _wav_bytes(audio, client.settings.data_dir), "audio/wav")})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _submit(client, doctor_id: str) -> str:
    audio, _ = synth_dialogue([r for r, _ in SCRIPT], seed=7)
    r = client.post("/api/jobs", headers=AUTH,
                    data={"template_id": "osce_respirasi", "doctor_id": doctor_id,
                          "extra_names": json.dumps(["Budi"]), "locked": json.dumps({"sistem_organ": "Respirasi"})},
                    files={"audio": ("sesi.wav", _wav_bytes(audio, client.settings.data_dir), "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["position"] >= 0
    return body["job_id"]


def _wait_done(client, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/api/jobs/{job_id}", headers=AUTH, params={"light": "true"})
        assert r.status_code == 200, r.text
        job = r.json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.1)
    raise AssertionError(f"Job {job_id} tidak selesai dalam {timeout}s")


# ---------------------------------------------------------------------------------------------
def test_auth_token_required(client):
    assert client.get("/api/health").status_code == 200  # health dikecualikan dari auth
    assert client.get("/api/jobs").status_code == 401
    assert client.get("/api/jobs", headers={"x-access-token": "salah"}).status_code == 401
    assert client.get("/api/jobs", headers=AUTH).status_code == 200
    assert client.get("/api/jobs", params={"token": TOKEN}).status_code == 200
    client.cookies.set("access_token", TOKEN)
    assert client.get("/api/jobs").status_code == 200
    assert client.get("/").status_code == 200  # halaman statis tidak butuh token


def test_submit_rejects_invalid_input(client):
    wav = lambda: ("a.wav", _wav_bytes(synth_dialogue(["DOKTER"])[0], client.settings.data_dir), "audio/wav")  # noqa: E731
    r = client.post("/api/jobs", headers=AUTH, data={"template_id": "tidak_ada"}, files={"audio": wav()})
    assert r.status_code == 400
    r = client.post("/api/jobs", headers=AUTH, data={"doctor_id": "tidak_ada"}, files={"audio": wav()})
    assert r.status_code == 404
    r = client.post("/api/jobs", headers=AUTH, data={"locked": json.dumps({"bukan_field": "x"})}, files={"audio": wav()})
    assert r.status_code == 400
    r = client.post("/api/jobs", headers=AUTH, files={"audio": ("a.exe", io.BytesIO(b"xx"), "application/octet-stream")})
    assert r.status_code == 415
    assert client.get("/api/jobs/tidak_ada", headers=AUTH).status_code == 404


def test_job_flow_submit_review_verify(client, doctor_enroll_audio):
    doctor_id = _enroll(client, doctor_enroll_audio)
    assert [d["id"] for d in client.get("/api/doctors", headers=AUTH).json()] == [doctor_id]

    job_id = _submit(client, doctor_id)
    light = _wait_done(client, job_id)
    assert light["status"] == "done", light.get("error")
    assert "result" not in light and "review" not in light
    assert "audio_path" not in light and "client_ip" not in light

    job = client.get(f"/api/jobs/{job_id}", headers=AUTH).json()
    res = job["result"]
    assert res["summary_source"] == "llm"
    assert [u["role"] for u in res["utterances"]] == [r for r, _ in SCRIPT]
    assert all("Budi" not in u["text"] for u in res["utterances"])  # PII ter-scrub

    # Review: tukar peran + ganti kelas, simpan draft.
    payload = {"utterances": res["utterances"], "summary": res["summary"]}
    payload["utterances"][0]["intent"] = "CHITCHAT"
    payload["utterances"][2]["role"] = "PASIEN"
    r = client.put(f"/api/jobs/{job_id}/review", headers=AUTH, json=payload)
    assert r.status_code == 200, r.text
    review = client.get(f"/api/jobs/{job_id}", headers=AUTH).json()["review"]
    assert review["utterances"][2]["role"] == "PASIEN"
    assert client.put("/api/jobs/tidak_ada/review", headers=AUTH, json=payload).status_code == 404

    # Verifikasi -> kasus di bank + data active learning.
    r = client.post(f"/api/jobs/{job_id}/verify", headers=AUTH, json=payload)
    assert r.status_code == 200, r.text
    case_id = r.json()["case_id"]
    assert client.get(f"/api/jobs/{job_id}", headers=AUTH, params={"light": "true"}).json()["status"] == "verified"
    assert case_id in [c["id"] for c in client.get("/api/cases", headers=AUTH).json()]
    assert client.get(f"/api/cases/{case_id}", headers=AUTH).json()["summary"]["sistem_organ"] == "Respirasi"
    training = client.settings.training_dir
    assert len((training / "text_intent.jsonl").read_text(encoding="utf-8").splitlines()) == len(SCRIPT)
    assert len((training / "speaker_role.jsonl").read_text(encoding="utf-8").splitlines()) == len(SCRIPT)

    # Setelah verified: tidak bisa diedit/diverifikasi ulang.
    assert client.put(f"/api/jobs/{job_id}/review", headers=AUTH, json=payload).status_code == 409
    assert client.post(f"/api/jobs/{job_id}/verify", headers=AUTH, json=payload).status_code == 422
