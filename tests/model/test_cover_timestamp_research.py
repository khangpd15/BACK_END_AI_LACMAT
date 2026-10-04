import json

import joblib
import numpy as np
import pytest

from research.cover_test_timestamp.features import (
    FEATURE_NAMES, _motion, downsample, extract_features, predict_recording, sampling_quality,
    split_clock_segments,
)
from research.cover_test_timestamp.train import (
    SOURCE, audit, build_rows, train, verified_groups, verify_permission,
)


def series(hz=60, label="NORMAL", offset=0.0):
    samples = []
    for i in range(hz * 4):
        t = i / hz
        x = 0.003 * np.sin(2 * np.pi * t) + 0.0007 * np.sin(7 * t)
        y = 0.002 * np.cos(3 * t)
        factor = 1.1 if label == "NORMAL" else 2.4
        samples.append({"t": t, "leftX": 0.2 + x + offset, "leftY": 0.5 + y,
                        "rightX": 0.6 + factor * x + offset, "rightY": 0.5 + factor * y,
                        "leftValid": True, "rightValid": True, "phase": "UNKNOWN"})
    return samples


def records():
    return [{"id": str(i), "sha256": str(i), "payload": {"source": SOURCE,
             "label": "NORMAL" if i < 4 else "STRABISMUS",
             "samples": series(label="NORMAL" if i < 4 else "STRABISMUS", offset=i * 0.001)}}
            for i in range(8)]


@pytest.mark.parametrize("bad_times", [[0, 0], [1, 0], [0, float("nan")]])
def test_rejects_duplicate_reversed_nonfinite_timestamps(bad_times):
    samples = series()[:2]
    for s, t in zip(samples, bad_times):
        s["t"] = t
    with pytest.raises(ValueError, match="strictly increasing"):
        sampling_quality(samples)


def test_timestamp_unit_must_be_explicit_seconds():
    with pytest.raises(ValueError, match="seconds"):
        extract_features(series(), "milliseconds")


def test_derivatives_use_timestamp_duration_not_frame_number():
    a = extract_features(series())
    slower = series()
    for s in slower:
        s["t"] *= 2
    b = extract_features(slower)
    for base in (0, 6, 12):
        np.testing.assert_allclose(a[base:base + 3], b[base:base + 3])
        np.testing.assert_allclose(a[base + 3:base + 6] / 2, b[base + 3:base + 6])


def test_no_upsampling_or_invented_observations():
    samples = series()
    reduced = downsample(samples, 30)
    assert len(reduced) <= 121
    assert all(any(s is original for original in samples) for s in reduced)
    with pytest.raises(ValueError, match="higher frame rate"):
        downsample(series(30), 60)


def test_missing_points_are_not_zeros_or_bridged():
    samples = series()
    for s in samples[70:90]:
        s.update(leftValid=False, leftX=None, leftY=None)
    assert np.isfinite(extract_features(samples)).all()
    times = np.arange(80) / 60
    coords = np.column_stack((np.sin(times), np.cos(times)))
    valid = np.ones(80, bool)
    valid[20:30] = False
    _, velocities = _motion(coords, times, valid)
    assert len(velocities) == 79 - 11
    times[40:] += 0.5
    _, with_gap = _motion(coords, times, valid)
    assert len(with_gap) == len(velocities) - 1


def test_rejects_invalid_flags_and_constant_motion():
    samples = series()
    samples[0]["leftValid"] = "false"
    with pytest.raises(ValueError, match="booleans"):
        extract_features(samples)
    samples = series()
    for s in samples:
        s.update(leftX=0.2, leftY=0.5)
    with pytest.raises(ValueError, match="undefined"):
        extract_features(samples)


def test_spatial_translation_does_not_change_features():
    np.testing.assert_allclose(extract_features(series()), extract_features(series(offset=0.15)), atol=1e-10)


def test_permission_requires_existing_evidence(tmp_path):
    path = tmp_path / "permission.json"
    path.write_text(json.dumps({"source_domain": SOURCE, "training_permitted": True,
                               "consent_verified": True, "verified_by": "test fixture",
                               "evidence_file": "absent.txt"}))
    with pytest.raises(ValueError, match="evidence"):
        verify_permission(path)
    with pytest.raises(ValueError, match="not permitted"):
        train(records(), {}, tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()


def test_patient_mapping_cannot_be_inferred_or_incomplete(tmp_path):
    path = tmp_path / "groups.json"
    path.write_text(json.dumps({"identity_basis": "filename", "recordings": {}}))
    with pytest.raises(ValueError, match="verified source mapping"):
        verified_groups(records(), path)


def test_audit_never_reports_accuracy_or_fake_patient_count():
    result = audit(records())
    assert result["verified_patient_count"] is None
    assert result["segments_with_cover_phase_labels"] == 0
    assert result["status"] == "AUDIT_ONLY_NO_TRAINING"


def test_clock_resets_split_without_changing_timestamps_or_recording_owner():
    samples = series() + series()
    segments = split_clock_segments(samples)
    assert len(segments) == 2
    assert segments[1][0]["t"] == 0
    assert segments[1][0] is samples[240]
    data = records()
    data[0]["payload"]["samples"] = samples
    x, y, owners, rejected = build_rows(data)
    assert len(x) == len(y) == 27
    assert (owners == 0).sum() == 6
    assert not rejected
    result = audit(data)
    assert result["recordings_with_clock_boundaries"] == 1
    assert result["valid_monotonic_segments"] == 9


def test_synthetic_training_serialization_and_source_rejection(tmp_path):
    data = records()
    output = tmp_path / "synthetic-run"
    summary = train(data, {"training_permitted": True, "consent_verified": True,
                           "scope": "SYNTHETIC_TEST_ONLY"}, output)
    assert summary["recordings_used"] == 8
    assert summary["training_rows_rate_variants"] == 24
    assert summary["evaluation"]["accuracy_claim"] is None
    bundle = joblib.load(output / "model.joblib")
    assert len(bundle["feature_names"]) == len(FEATURE_NAMES)
    result = predict_recording(bundle, series(30), timestamp_unit="seconds", source=SOURCE)
    assert result["status"] == "RESEARCH_PREDICTION"
    assert result["cover_response_trained"] is False
    assert predict_recording(bundle, series(), timestamp_unit="seconds", source="WEBCAM")["reason"] == "SOURCE_DOMAIN_MISMATCH"
    with pytest.raises(FileExistsError):
        train(data, {"training_permitted": True, "consent_verified": True}, output)


def test_rate_variants_remain_in_same_grouped_fold(tmp_path):
    data = records()
    mapping = tmp_path / "groups.json"
    mapping.write_text(json.dumps({"identity_basis": "verified_source_patient_mapping",
        "verified_by": "SYNTHETIC_TEST_FIXTURE", "recordings": {str(i): f"synthetic-person-{i // 2}" for i in range(8)}}))
    summary = train(data, {"training_permitted": True, "consent_verified": True}, tmp_path / "grouped", mapping)
    evaluation = summary["evaluation"]
    assert evaluation["recording_count"] == 8
    assert evaluation["verified_patient_count"] == 4
    assert all(len(m["ci95_patient_bootstrap"]) == 2 for m in evaluation["metrics"].values())
    for fold in evaluation["splits"]:
        train_groups = {int(i) // 2 for i in fold["train_sha256"]}
        test_groups = {int(i) // 2 for i in fold["test_sha256"]}
        assert train_groups.isdisjoint(test_groups)
