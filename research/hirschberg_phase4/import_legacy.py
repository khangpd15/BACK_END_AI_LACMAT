"""Import legacy coordinates with explicit provenance, never fake native/patient IDs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def convert(path, attestation=None):
    if attestation is not None:
        if not (
            attestation.get("evidence_type") == "project_owner_statement"
            and attestation.get("approves_training_rights") is True
            and attestation.get("clinical_label_confirmation") == "reported_by_project_owner"
            and attestation.get("record_id")
        ):
            raise ValueError("Attestation must explicitly confirm research use and clinical labels")
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
        if attestation is not None:
            converted[-1].update({
                "rights_status": "approved",
                "permission_record_id": attestation["record_id"],
                "clinician_confirmed": True,
                "clinical_confirmation_basis": "project_owner_statement",
                "clinical_review_record_id": attestation["record_id"],
                "landmarks_clinician_confirmed": False,
                "allowed_use": "noncommercial_research",
                "public_redistribution_permission": "not_granted",
                "training_ready": False,
                "pending_requirements": ["real_patient_linkage", "native_coordinates", "limbus_labels"]
            })
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--attestation", help="Explicit owner confirmation scoped to supplied data")
    args = parser.parse_args()
    attestation = None
    if args.attestation:
        with Path(args.attestation).open(encoding="utf-8") as handle:
            attestation = json.load(handle)
    rows = convert(args.csv, attestation)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
