"""Validate permission, coordinate contracts and patient-independent splits."""

import json
import math
from pathlib import Path

LABELS = ("esotropia", "exotropia", "pseudostrabismus", "normal")
TARGETS = ("pupil", "limbus", "glint")


def validate(records, require_approved=True):
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must contain a nonempty record list")
    ids, patients, hashes = set(), {}, {}
    for row in records:
        for key in ("image_id", "patient_id", "source_id", "split"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing {key}")
        if row["image_id"] in ids:
            raise ValueError("Duplicate image_id")
        ids.add(row["image_id"])
        split = row["split"]
        if split not in ("train", "val", "test"):
            raise ValueError("Invalid split")
        patient = row["patient_id"]
        if patient in patients and patients[patient] != split:
            raise ValueError("Patient overlap across splits")
        patients[patient] = split
        digest = row.get("image_sha256")
        if digest:
            if digest in hashes and hashes[digest] != split:
                raise ValueError("Image hash overlap across splits")
            hashes[digest] = split
        kind = row.get("source_kind")
        if kind not in ("human", "synthetic"):
            raise ValueError("Unknown source_kind")
        if require_approved and row.get("rights_status") != "approved":
            raise ValueError("Rights not approved; quarantine, do not train/evaluate")
        if require_approved and kind == "human":
            for key in ("permission_record_id", "consent_scope_id", "license_record_id"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"Missing human permission field: {key}")
            if row.get("clinician_confirmed") is not True:
                raise ValueError("Human clinical label not confirmed")
            if row.get("patient_id_basis") != "verified":
                raise ValueError("Verified patient linkage required; provisional groups are exploratory only")
        if row.get("label") not in LABELS:
            raise ValueError("Unsupported label; indeterminate needs a separate cohort")
        if row.get("coordinate_space") != "native":
            raise ValueError("Native coordinates required; legacy crops are not native")
        for key in ("width", "height"):
            if type(row.get(key)) is not int or row[key] <= 0:
                raise ValueError("Invalid native dimensions")
        if row.get("mirrored") is not False:
            raise ValueError("Unmirrored orientation must be explicitly confirmed")
        landmarks = row.get("landmarks")
        if not isinstance(landmarks, dict):
            raise ValueError("Missing landmarks")
        for eye in ("OD", "OS"):
            if not isinstance(landmarks.get(eye), dict):
                raise ValueError("Missing eye landmark dictionary")
            for target in TARGETS:
                if target not in landmarks[eye]:
                    raise ValueError("Explicit target or null required")
                point = landmarks[eye][target]
                if point is None:
                    continue
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError("Point must be [x, y] or null")
                if any(type(v) not in (int, float) or not math.isfinite(v) for v in point):
                    raise ValueError("Nonfinite/nonnumeric coordinate")
                if not (0 <= point[0] < row["width"] and 0 <= point[1] < row["height"]):
                    raise ValueError("Point outside native image")
            diameter = landmarks[eye].get("limbus_diameter_px")
            if diameter is not None and (
                type(diameter) not in (int, float) or not math.isfinite(diameter) or diameter <= 0
            ):
                raise ValueError("Invalid limbus diameter")
    return records


def load_manifest(path):
    with Path(path).open(encoding="utf-8") as handle:
        return validate(json.load(handle))


def assert_independent(rows, training):
    for key in ("patient_id", "image_id", "image_sha256"):
        old = {r.get(key) for r in training if r.get(key)}
        new = {r.get(key) for r in rows if r.get(key)}
        if old & new:
            raise ValueError(f"Evaluation overlap with training: {key}")
