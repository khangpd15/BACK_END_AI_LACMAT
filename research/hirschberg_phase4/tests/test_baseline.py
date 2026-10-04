from copy import deepcopy

import numpy as np
import pytest

from research.hirschberg_phase4.evaluate import evaluate, metrics
from research.hirschberg_phase4.features import canthal_distance, displacement_mm, extract, to_native
from research.hirschberg_phase4.import_legacy import convert
from research.hirschberg_phase4.schema import validate
from research.hirschberg_phase4.synthetic import fixture_records
from research.hirschberg_phase4.train import fit_baseline


def test_fixture_reproducible():
    assert fixture_records() == fixture_records()


@pytest.mark.parametrize("change", ["rights", "patient", "hash", "nan", "bounds", "legacy", "mirror"])
def test_invalid_contract_rejected(change):
    rows = fixture_records()
    if change == "rights":
        rows[0]["rights_status"] = "hold"
    elif change == "patient":
        rows[-1]["patient_id"] = rows[0]["patient_id"]
    elif change == "hash":
        rows[0]["image_sha256"] = rows[-1]["image_sha256"] = "same"
    elif change == "nan":
        rows[0]["landmarks"]["OD"]["glint"][0] = float("nan")
    elif change == "bounds":
        rows[0]["landmarks"]["OD"]["glint"][0] = 800
    elif change == "legacy":
        rows[0]["coordinate_space"] = "legacy_crop"
    else:
        rows[0]["mirrored"] = True
    with pytest.raises(ValueError):
        validate(rows)


def test_human_requires_documented_scope_and_clinician():
    rows = fixture_records()
    rows[0]["source_kind"] = "human"
    with pytest.raises(ValueError, match="permission"):
        validate(rows)
    for key in ("permission_record_id", "consent_scope_id", "license_record_id"):
        rows[0][key] = "reviewed-record"
    with pytest.raises(ValueError, match="clinical"):
        validate(rows)


def test_missing_glint_abstains_no_fake_point():
    row = fixture_records()[0]
    row["landmarks"]["OD"]["glint"] = None
    assert extract(row) == (None, "MISSING_GEOMETRY")


def test_scale_requires_individual_measurement():
    assert displacement_mm([0, 0], [3, 0], 100) is None
    assert displacement_mm([0, 0], [3, 0], 100, 12) == [0.36, 0.0]
    with pytest.raises(ValueError):
        displacement_mm([0, 0], [3, 0], 0, 12)


def test_transform_native_resize_and_canthus_missing():
    assert to_native([3, 4], np.diag([4, 4, 1])) == [12, 16]
    assert canthal_distance(None, [5, 6]) is None
    assert canthal_distance([0, 0], [3, 4]) == 5


def test_od_os_signs():
    row = fixture_records()[0]
    values, _ = extract(row)
    assert values[0] > 0 and values[2] > 0 and values[4] > 0


def test_abstain_denominators_and_refer_all_not_success():
    report = metrics(["esotropia", "exotropia", "pseudostrabismus", "normal"], ["abstain"] * 4)
    assert report["coverage"] == 0
    assert report["macro_recall_all"] == 0
    assert report["referral_sensitivity"] == 1
    assert report["referral_specificity"] == 0
    assert report["per_class"]["normal"]["recall_covered"] is None


def test_missing_class_metric_is_na():
    assert metrics(["normal"], ["normal"])["macro_recall_all"] is None


def test_fit_only_training_split_and_evaluator_never_fits(monkeypatch):
    rows = fixture_records()
    bundle = fit_baseline(rows)
    monkeypatch.setattr(bundle["model"], "fit", lambda *a, **k: pytest.fail("Evaluator fit"))
    val = [r for r in rows if r["split"] == "val"]
    report = evaluate(bundle, val)
    assert report["n"] == 16
    assert report["source_kind"] == "synthetic" and not report["clinical_evidence"]
    changed = deepcopy(rows)
    for row in changed:
        if row["split"] != "train":
            row["label"] = "normal"
    second = fit_baseline(changed)
    np.testing.assert_allclose(bundle["model"][-1].coef_, second["model"][-1].coef_)


def test_evaluation_rejects_training_people():
    rows = fixture_records()
    with pytest.raises(ValueError, match="overlap"):
        evaluate(fit_baseline(rows), [rows[0]])


def test_training_rejects_missing_pseudo_and_duplicate_patient():
    rows = [r for r in fixture_records() if r["label"] != "pseudostrabismus"]
    with pytest.raises(ValueError, match="four classes"):
        fit_baseline(rows)
    rows = fixture_records()
    rows[1]["patient_id"] = rows[0]["patient_id"]
    with pytest.raises(ValueError, match="preselected"):
        fit_baseline(rows)


def test_evaluate_direct_api_enforces_permissions():
    rows = fixture_records()
    bundle = fit_baseline(rows)
    val = [r for r in rows if r["split"] == "val"]
    val[0]["rights_status"] = "hold"
    with pytest.raises(ValueError, match="Rights"):
        evaluate(bundle, val)


def test_missing_glint_evaluation_keeps_denominator():
    rows = fixture_records()
    bundle = fit_baseline(rows)
    val = [r for r in rows if r["split"] == "val"]
    val[0]["landmarks"]["OD"]["glint"] = None
    report = evaluate(bundle, val)
    assert report["n"] == 16 and report["coverage"] == 15 / 16
    assert report["confusion"][0][4] == 1


def test_legacy_import_quarantines_without_reading_images(tmp_path):
    path = tmp_path / "old.csv"
    path.write_text("relative_path,class_label,od_pupil_x,od_pupil_y,od_reflex_x,od_reflex_y,"
                    "os_pupil_x,os_pupil_y,os_reflex_x,os_reflex_y,notes\n"
                    "missing.jpg,normal,1,2,3,4,5,6,7,8,old\n", encoding="utf-8")
    result = convert(path)
    assert result[0]["rights_status"] == "hold"
    assert result[0]["patient_id"] is None
    assert len(result[0]["source_csv_sha256"]) == 64
    assert result[0]["coordinate_space"] == "legacy_crop"
    with pytest.raises(ValueError):
        validate(result)


def test_legacy_owner_confirmation_does_not_certify_coordinates(tmp_path):
    path = tmp_path / "old.csv"
    path.write_text("relative_path,class_label,od_pupil_x,od_pupil_y,od_reflex_x,od_reflex_y,"
                    "os_pupil_x,os_pupil_y,os_reflex_x,os_reflex_y,notes\n"
                    "missing.jpg,normal,1,2,3,4,5,6,7,8,old\n", encoding="utf-8")
    attestation = {"record_id": "owner-review", "evidence_type": "project_owner_statement",
                   "approves_training_rights": True,
                   "clinical_label_confirmation": "reported_by_project_owner"}
    row = convert(path, attestation)[0]
    assert row["rights_status"] == "approved" and row["clinician_confirmed"] is True
    assert row["landmarks_clinician_confirmed"] is False
    assert row["patient_id"] is None and row["training_ready"] is False
    assert row["coordinate_space"] == "legacy_crop"
    with pytest.raises(ValueError):
        validate([row])
    attestation["approves_training_rights"] = False
    with pytest.raises(ValueError, match="Attestation"):
        convert(path, attestation)
