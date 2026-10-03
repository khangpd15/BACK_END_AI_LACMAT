import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# pyrefly: ignore [missing-import]
from research_dataset_pipeline import (  # noqa: E402
    ParticipantRecord,
    build_dataset_summary,
    stable_patient_split,
    training_eligibility,
    validate_ground_truth,
)
# pyrefly: ignore [missing-import]
from train_research_model import (  # noqa: E402
    TRAINING_DISABLED_REASON,
    assert_training_may_start,
    build_training_plan,
)


def _record(participant_id, domain="hirschberg_cover_protocol_v1", ground_truth=None):
    return ParticipantRecord(
        participant_id=participant_id,
        domain=domain,
        consent_research=True,
        consent_raw_video_or_landmarks=False,
        age_years=9,
        glasses_on=False,
        distance_bucket="TARGET_20_25_CM",
        hirschberg={"quality": {"reflexCountPerEye": {"OD": 1, "OS": 1}}, "measurements": {"delta_h": 0.01}},
        cover={"samples_path": "cover.json", "quality": {"validFrameRatio": 0.9}},
        ground_truth=ground_truth
        or {
            "participant_id": participant_id,
            "label_source": "doctor",
            "exam_date": "2026-10-03",
            "horizontal_strabismus_type": "orthophoria",
            "intermittent": "false",
            "glasses_group": "none",
            "pseudostrabismus": "false",
            "exam_method": "cover_test_by_doctor",
        },
    )


def test_patient_split_keeps_participant_in_one_split():
    participant_ids = ["p1", "p1", "p2", "p3", "p3"]
    split = stable_patient_split(participant_ids, seed=123)

    assert set(split) == {"p1", "p2", "p3"}
    assert all(value in {"train", "validation", "test"} for value in split.values())


def test_legacy_non_hirschberg_is_not_training_eligible():
    record = _record("legacy-1", domain="legacy_non_hirschberg")

    result = training_eligibility(record)

    assert result["eligible"] is False
    assert "LEGACY_NON_HIRSCHBERG_FORBIDDEN_FOR_THRESHOLD_OR_TRAINING" in result["reasons"]


def test_model_output_cannot_be_ground_truth_label():
    issues = validate_ground_truth(
        {
            "participant_id": "p1",
            "label_source": "doctor",
            "exam_date": "2026-10-03",
            "horizontal_strabismus_type": "orthophoria",
            "intermittent": "false",
            "glasses_group": "none",
            "pseudostrabismus": "false",
            "exam_method": "cover_test_by_doctor",
            "model_prediction": "NORMAL",
        }
    )

    assert "MODEL_OUTPUT_MUST_NOT_BE_LABEL" in issues


def test_dataset_summary_reports_distance_experiment_without_selecting_threshold():
    summary = build_dataset_summary([_record("p1"), _record("p2")], seed=123)

    assert summary["datasetVersion"] == "remicare-hirschberg-cover-dataset-v0.1"
    assert summary["trainingDisabledReason"] == TRAINING_DISABLED_REASON
    assert "does not choose a clinical standard distance" in summary["distanceExperiment"]["note"]


def test_training_plan_is_guarded_by_default():
    summary = build_dataset_summary([_record("p1")], seed=123)
    plan = build_training_plan(summary, seed=123)

    assert plan["status"] == "PLAN_ONLY_NO_TRAINING"
    with pytest.raises(RuntimeError) as exc:
        assert_training_may_start(summary, allow_train=False)

    assert str(exc.value) == TRAINING_DISABLED_REASON
