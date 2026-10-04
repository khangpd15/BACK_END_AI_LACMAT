"""Compare repeated human landmark clicks with the legacy coordinates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


OLD_COLUMNS = {
    "OD pupil": ("od_pupil_x", "od_pupil_y"),
    "OD reflex": ("od_reflex_x", "od_reflex_y"),
    "OS pupil": ("os_pupil_x", "os_pupil_y"),
    "OS reflex": ("os_reflex_x", "os_reflex_y"),
}


def summarize(
    errors: list[float], expected: int, missing_normalizers: int = 0,
    caution_threshold: float | None = None,
) -> dict:
    values = np.asarray(errors, dtype=float)
    if values.size == 0:
        return {"n": 0, "expected": expected, "missing": expected,
                "mean": None, "median": None, "p90": None,
                "normalized_missing": missing_normalizers}
    result = {
        "n": int(values.size),
        "expected": expected,
        "missing": expected - int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "normalized_missing": missing_normalizers,
    }
    if caution_threshold is not None:
        result.update({
            "mean_exceeds_approx_3px": bool(values.mean() > caution_threshold),
            "median_exceeds_approx_3px": bool(np.median(values) > caution_threshold),
            "p90_exceeds_approx_3px": bool(np.percentile(values, 90) > caution_threshold),
        })
    return result


def compare(csv_path: Path, sample_path: Path, annotations_path: Path) -> dict:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        old_rows = {row["relative_path"]: row for row in csv.DictReader(handle)}
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))["annotations"]
    selected = {row["sample_id"]: row for row in sample["samples"]}
    by_sample = {row["sample_id"]: row for row in annotations}
    errors_px = {"pupil": [], "reflex": []}
    errors_norm = {"pupil": [], "reflex": []}
    normalization_missing = {"pupil": 0, "reflex": 0}
    expected = {"pupil": 0, "reflex": 0}

    for sample_id, selected_row in selected.items():
        annotation = by_sample.get(sample_id)
        if annotation is None:
            for target in expected:
                expected[target] += 2
            continue
        old = old_rows[selected_row["relative_path"]]
        old_od = [float(old["od_pupil_x"]), float(old["od_pupil_y"])]
        old_os = [float(old["os_pupil_x"]), float(old["os_pupil_y"])]
        inter_pupil_px = math.dist(old_od, old_os)
        for point_name, columns in OLD_COLUMNS.items():
            target = "pupil" if point_name.endswith("pupil") else "reflex"
            expected[target] += 1
            xy = annotation.get("points", {}).get(point_name)
            if not isinstance(xy, list) or len(xy) != 2:
                continue
            old_xy = [float(old[columns[0]]), float(old[columns[1]])]
            error = math.dist(old_xy, [float(xy[0]), float(xy[1])])
            errors_px[target].append(error)
            if inter_pupil_px > 0 and math.isfinite(inter_pupil_px):
                errors_norm[target].append(error / inter_pupil_px)
            else:
                normalization_missing[target] += 1

    return {
        "purpose": "same-annotator agreement with legacy clicks; not detector error or clinical ground truth",
        "sample_n": len(selected),
        "annotated_n": len(by_sample),
        "strata_counts": sample.get("strata_counts", {}),
        "pixel_coordinate_space": "stored legacy crop pixels; no original-image transform is available",
        "normalization": "per-point Euclidean error divided by that image's legacy OD-OS pupil-center distance",
        "approximately_3px_caution": "If repeat-click disagreement exceeds about 3 px, these labels cannot support a 3 px detector benchmark. This is only a practical warning, not a statistical test.",
        "results": {
            target: {
                "px": summarize(errors_px[target], expected[target], caution_threshold=3.0),
                "fraction_of_inter_pupil_distance": summarize(
                    errors_norm[target], expected[target], normalization_missing[target]
                ),
            }
            for target in ("pupil", "reflex")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="processed/hirschberg_manual_annotations.csv")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = compare(Path(args.csv), Path(args.sample), Path(args.annotations))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
