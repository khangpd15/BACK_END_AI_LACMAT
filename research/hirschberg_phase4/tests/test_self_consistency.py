import csv

from research.hirschberg_phase4.compare_self_consistency import compare, summarize
from research.hirschberg_phase4.select_self_consistency import stratified_sample


def test_stratified_sample_is_seeded_and_approximately_proportional():
    rows = [
        {"relative_path": f"{label}/{index}.jpg", "class_label": label}
        for label, count in (("eso", 33), ("exo", 19), ("normal", 10))
        for index in range(count)
    ]
    first = stratified_sample(rows, n=20, seed=20261004)
    assert first == stratified_sample(rows, n=20, seed=20261004)
    counts = {label: sum(row["class_label"] == label for row in first)
              for label in ("eso", "exo", "normal")}
    assert counts == {"eso": 11, "exo": 6, "normal": 3}
    assert len({row["relative_path"] for row in first}) == 20


def test_summary_does_not_turn_missing_points_into_zero():
    result = summarize([2.0, 4.0], expected=4, caution_threshold=3.0)
    assert result["n"] == 2
    assert result["missing"] == 2
    assert result["mean"] == 3.0
    assert result["mean_exceeds_approx_3px"] is False
    assert result["p90_exceeds_approx_3px"] is True


def test_compare_reports_per_eye_px_and_normalized_errors(tmp_path):
    csv_path = tmp_path / "old.csv"
    fields = ["relative_path", "class_label", "od_pupil_x", "od_pupil_y",
              "od_reflex_x", "od_reflex_y", "os_pupil_x", "os_pupil_y",
              "os_reflex_x", "os_reflex_y"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"relative_path": "eye.jpg", "class_label": "eso",
                         "od_pupil_x": 10, "od_pupil_y": 10, "od_reflex_x": 12,
                         "od_reflex_y": 10, "os_pupil_x": 30, "os_pupil_y": 10,
                         "os_reflex_x": 32, "os_reflex_y": 10})
    sample_path = tmp_path / "sample.json"
    sample_path.write_text(
        '{"samples":[{"sample_id":"S001","relative_path":"eye.jpg"}],'
        '"strata_counts":{"eso":1}}', encoding="utf-8"
    )
    annotations_path = tmp_path / "repeat.json"
    annotations_path.write_text(
        '{"annotations":[{"sample_id":"S001","points":{'
        '"OD pupil":[13,14],"OD reflex":[12,10],"OS pupil":[30,10],'
        '"OS reflex":[32,10]}}]}', encoding="utf-8"
    )
    report = compare(csv_path, sample_path, annotations_path)
    pupil_px = report["results"]["pupil"]["px"]
    assert pupil_px["n"] == 2
    assert pupil_px["mean"] == 2.5
    assert pupil_px["missing"] == 0
    assert report["results"]["pupil"]["fraction_of_inter_pupil_distance"]["mean"] == 0.125
