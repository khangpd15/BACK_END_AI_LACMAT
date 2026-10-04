import numpy as np
import pytest

from research.hirschberg_phase4.exploratory import CLASSES, evaluate_oof, legacy_features
from research.hirschberg_phase4.grouping import cluster


def record(name, dhash, phash, sha=None, label="normal"):
    return {"image_id": name, "dhash": dhash, "phash": phash,
            "sha256": sha or name, "label": label}


def test_transitive_groups_keep_patient_unknown():
    rows = [record("a", 0, 0), record("b", 1, 31), record("c", 3, 63)]
    assigned, _, summary = cluster(rows, 1)
    assert len({r["provisional_group_id"] for r in assigned}) == 1
    assert all(r["patient_id"] is None for r in assigned)
    assert summary["patient_independent"] is False


def test_grouping_not_separated_by_label():
    rows = [record("a", 0, 0, "same", "normal"), record("b", 31, 31, "same", "esotropia")]
    _, _, summary = cluster(rows, 0)
    assert summary["provisional_groups"] == 1
    assert summary["mixed_label_groups"] == 1


def test_old_inferred_links_are_not_upgraded_to_patient_ids():
    rows = [record("a", 0, 0), record("b", 31, 31)]
    assigned, edges, _ = cluster(rows, 0, {"a": "old-cluster", "b": "old-cluster"})
    assert assigned[0]["provisional_group_id"] == assigned[1]["provisional_group_id"]
    assert "existing_inferred_cluster_not_patient" in edges[0]["reasons"]


def test_threshold_sensitivity_coarsens_not_splits_groups():
    rows = [record("a", 0, 0), record("b", 255, 255)]
    assert cluster(rows, 4)[2]["provisional_groups"] == 2
    assert cluster(rows, 8)[2]["provisional_groups"] == 1


def test_legacy_features_are_dimensionless_not_fake_iris_or_mm():
    row = {"od_pupil_x": "10", "od_pupil_y": "20", "os_pupil_x": "110", "os_pupil_y": "20",
           "od_reflex_x": "12", "od_reflex_y": "20", "os_reflex_x": "108", "os_reflex_y": "20"}
    np.testing.assert_allclose(legacy_features(row), [0.02, 0, -0.02, 0, 0.04, 0])
    row["os_pupil_x"] = "10"
    with pytest.raises(ValueError, match="Degenerate"):
        legacy_features(row)


def test_oof_keeps_provisional_groups_together():
    x, y, groups = [], [], []
    for i, label in enumerate(CLASSES):
        for g in range(6):
            for repeat in range(2):
                x.append([i, repeat / 10])
                y.append(label)
                groups.append(f"{label}-{g}")
    result = evaluate_oof(x, y, groups)
    for group in set(groups):
        assert len({fold for fold, g in zip(result["fold_ids"], groups) if g == group}) == 1
    assert result["model"]["balanced_accuracy_3class"] == 1.0


def test_too_few_groups_fail_instead_of_image_random_split():
    with pytest.raises(ValueError, match="Not enough"):
        evaluate_oof([[0], [1], [2]], list(CLASSES), ["a", "b", "c"])
