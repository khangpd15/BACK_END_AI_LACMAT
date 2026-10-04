import json
import uuid
from pathlib import Path
from PIL import Image
import pytest

from scripts.verify_dataset_governance import (
    DiagnosisLabel,
    ImageSampleRecord,
    analyze_patient_leakage,
    scan_image_duplicates,
    validate_records,
    verify_dataset,
)


def _make_sample(
    participant_id: str,
    split: str,
    label: str = "normal",
    image_path: str = "dummy.jpg",
):
    return {
        "sample_id": str(uuid.uuid4()),
        "participant_id": participant_id,
        "image_path": image_path,
        "label": label,
        "clinician_confirmed": True,
        "diagnosis_source": "orthoptist_cover_test",
        "age_group": "preschool_3_5",
        "capture_condition": {"flash_present": True, "ambient": "standard_clinic"},
        "head_pose_status": "optimal",
        "license": "RemiCare_Proprietary_Clinical",
        "consent_status": "guardian_consented_full",
        "split": split,
    }


def test_schema_validation_success():
    records = [
        _make_sample("PT_01", "train", "normal"),
        _make_sample("PT_02", "val", "esotropia"),
        _make_sample("PT_03", "test", "pseudostrabismus"),
    ]
    valid, errors = validate_records(records)
    assert len(valid) == 3
    assert len(errors) == 0


def test_schema_validation_rejects_invalid_fields():
    records = [
        {
            "sample_id": "not-a-uuid",
            "participant_id": "PT_01",
            "image_path": "a.jpg",
            "label": "invalid_label",
            "clinician_confirmed": True,
            "diagnosis_source": "doctor",
            "age_group": "preschool_3_5",
            "capture_condition": {"flash_present": True},
            "head_pose_status": "optimal",
            "license": "RemiCare",
            "consent_status": "guardian_consented_full",
            "split": "train",
        }
    ]
    valid, errors = validate_records(records)
    assert len(valid) == 0
    assert len(errors) == 1


def test_patient_leakage_detection():
    # PT_01 is in both train and val -> LEAKAGE
    records = [
        ImageSampleRecord(**_make_sample("PT_01", "train")),
        ImageSampleRecord(**_make_sample("PT_01", "val")),
        ImageSampleRecord(**_make_sample("PT_02", "val")),
        ImageSampleRecord(**_make_sample("PT_03", "test")),
    ]
    result = analyze_patient_leakage(records)
    assert result["is_strictly_disjoint"] is False
    assert result["leakage_summary"]["train_val_overlap_count"] == 1
    assert result["violating_participants"][0]["participant_id"] == "PT_01"


def test_patient_disjoint_split_passes():
    records = [
        ImageSampleRecord(**_make_sample("PT_01", "train")),
        ImageSampleRecord(**_make_sample("PT_01", "train")),
        ImageSampleRecord(**_make_sample("PT_02", "val")),
        ImageSampleRecord(**_make_sample("PT_03", "test")),
    ]
    result = analyze_patient_leakage(records)
    assert result["is_strictly_disjoint"] is True
    assert result["leakage_summary"]["total_violating_participants"] == 0


def test_image_hash_cross_split_duplicate(tmp_path: Path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_a = img_dir / "a.png"
    img_b = img_dir / "b.png"

    # Create identical white image
    im = Image.new("RGB", (64, 64), color="white")
    im.save(img_a)
    im.save(img_b)

    records = [
        ImageSampleRecord(**_make_sample("PT_01", "train", image_path=str(img_a))),
        ImageSampleRecord(**_make_sample("PT_02", "test", image_path=str(img_b))),
    ]
    hash_result = scan_image_duplicates(records, phash_threshold=2)
    assert hash_result["is_hash_clean"] is False
    assert hash_result["exact_sha256_cross_split_duplicates_count"] == 1
