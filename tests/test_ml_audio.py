import numpy as np

from app.ml.audio_features import delta, extract_many, extract_segment_features, feature_dim, feature_names
from app.ml.augment import add_noise, pitch_speed_perturb, random_augment
from app.ml.speaker_role import (DoctorAnchor, SpeakerRoleClassifier, build_anchor, build_role_features,
                                 make_estimator, role_feature_names)
from conftest import DOCTOR_VOICE, PATIENT_VOICE, SR, synth_dialogue, synth_voice


def test_feature_vector_shape_and_f0():
    vec = extract_segment_features(synth_voice(150.0, (700, 1200, 2600), 1.5))
    assert vec.shape == (feature_dim(),) == (len(feature_names()),)
    assert np.isfinite(vec).all()
    f0 = vec[feature_names().index("f0_mean")]
    assert abs(f0 - 150.0) < 10.0


def test_short_and_silent_segments_do_not_crash():
    assert extract_segment_features(np.zeros(50, np.float32)).shape == (feature_dim(),)
    assert np.isfinite(extract_segment_features(np.zeros(SR, np.float32))).all()


def test_delta_of_linear_ramp_is_constant():
    feat = np.arange(20, dtype=float)[:, None] * 2.0
    d = delta(feat)
    assert np.allclose(d[3:-3], 2.0)


def test_anchor_roundtrip(tmp_path, doctor_enroll_audio):
    anchor = build_anchor(doctor_enroll_audio)
    assert anchor.n_chunks >= 10 and (anchor.scale > 0).all()
    anchor.save(tmp_path / "a.npz")
    loaded = DoctorAnchor.load(tmp_path / "a.npz")
    assert np.allclose(loaded.mean, anchor.mean) and loaded.embedding is None


def test_heuristic_separates_roles(doctor_enroll_audio):
    roles = ["DOKTER", "PASIEN"] * 5
    audio, spans = synth_dialogue(roles, seed=3)
    preds = SpeakerRoleClassifier(None).predict(extract_many(audio, spans), build_anchor(doctor_enroll_audio))
    assert [p[0] for p in preds] == roles
    assert all(p[2] == "heuristic" for p in preds)


def test_trained_model_generalizes_to_unseen_doctor(tmp_path):
    """Latih pada dokter A & B, uji pada dokter C (suara berbeda) -> fitur relatif harus tetap bekerja."""
    doctors = {"A": (110.0, (680, 1150, 2500)), "B": (135.0, (720, 1250, 2650)), "C": (170.0, (760, 1350, 2800))}
    patients = [(220.0, (880, 1750, 3050)), (95.0, (600, 1050, 2400)), (190.0, (820, 1600, 2950))]
    X, y, test = [], [], None
    for di, (name, voice) in enumerate(doctors.items()):
        anchor = build_anchor(synth_voice(*voice, dur=12.0, seed=50 + di))
        for k, patient in enumerate(patients):
            roles = ["DOKTER", "PASIEN", "PASIEN", "DOKTER", "PASIEN", "DOKTER", "DOKTER", "PASIEN"]
            audio, spans = synth_dialogue(roles, seed=10 * di + k, doctor=voice, patient=patient)
            vecs = extract_many(audio, spans)
            if name == "C":
                test = (vecs, anchor, roles) if test is None else test
                continue
            X.append(build_role_features(vecs, anchor))
            y += roles
    model = make_estimator("svm").fit(np.vstack(X), y)
    path = tmp_path / "m.joblib"
    SpeakerRoleClassifier.save_bundle(path, model, "svm", False)
    clf = SpeakerRoleClassifier.load(path)
    vecs, anchor, roles = test
    preds = clf.predict(vecs, anchor)
    acc = np.mean([p[0] == r for p, r in zip(preds, roles)])
    assert clf.is_trained and acc >= 0.85, preds
    assert len(role_feature_names(False)) == build_role_features(vecs, anchor).shape[1]


def test_augmentations_preserve_length_and_range():
    rng = np.random.default_rng(0)
    x = synth_voice(*PATIENT_VOICE, dur=1.0)
    assert add_noise(x, rng, 10).shape == x.shape
    assert abs(pitch_speed_perturb(x, 1.08).size - x.size / 1.08) < 5
    out = random_augment(x, rng)
    assert np.isfinite(out).all() and np.max(np.abs(out)) <= 1.0
    _ = DOCTOR_VOICE
