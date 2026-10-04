"""Export old CSV as quarantined provenance, not an approved native manifest."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def convert(path):
    csv_digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    converted = []
    for row in rows:
        converted.append({"legacy_relative_path": row["relative_path"],
                          "legacy_folder_label": row["class_label"], "annotation_origin": "legacy",
                          "source_csv_sha256": csv_digest,
                          "rights_status": "hold", "patient_id": None,
                          "coordinate_space": "legacy_crop", "clinician_confirmed": False,
                          "landmarks": {eye: {target: [float(row[f"{eye.lower()}_{target}_x"]),
                                                        float(row[f"{eye.lower()}_{target}_y"])]
                                               for target in ("pupil", "reflex")}
                                        for eye in ("OD", "OS")}, "notes": row.get("notes", "")})
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = convert(args.csv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
