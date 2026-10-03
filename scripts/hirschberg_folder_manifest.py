"""Build a labeled manifest from a folder-organized Hirschberg image dataset.

The folder name is treated as the label source. This script does not invent
participant IDs; it creates a conservative group_id from label + filename stem
so repeated/augmented variants do not cross the generated split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image


LABEL_MAP = {
    "ESOTROPIA": "esotropia",
    "EXOTROPIA": "exotropia",
    "NORMAL": "normal",
}

DATASET_VERSION = "hirschberg-folder-labels-v0.1"
LABEL_SOURCE = "user_attested_doctor_confirmed_folder_label"
DOMAIN = "legacy_or_unknown_hirschberg_crop"
SEED = 20261003


def deterministic_group_split(group_id: str, seed: int = SEED) -> str:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < 0.70:
        return "train"
    if bucket < 0.85:
        return "val"
    return "test"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            yield path


def build_manifest(root: Path, seed: int = SEED) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in iter_images(root):
        rel = path.relative_to(root)
        parts = rel.parts
        if len(parts) < 3:
            continue

        original_split = parts[0]
        raw_label = parts[1].upper()
        if raw_label not in LABEL_MAP:
            continue

        class_label = LABEL_MAP[raw_label]
        group_id = f"{class_label}:{path.stem.lower()}"
        generated_split = deterministic_group_split(group_id, seed=seed)
        digest = sha256_file(path)

        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format
            mode = image.mode

        rows.append(
            {
                "dataset_version": DATASET_VERSION,
                "relative_path": str(rel).replace("\\", "/"),
                "absolute_path": str(path),
                "original_split": original_split,
                "generated_split": generated_split,
                "raw_folder_label": raw_label,
                "class_label": class_label,
                "is_horizontal_strabismus": class_label in {"esotropia", "exotropia"},
                "participant_id": "UNKNOWN",
                "group_id": group_id,
                "domain": DOMAIN,
                "label_source": LABEL_SOURCE,
                "label_provenance_note": "User states labels were previously confirmed by doctors at an organization; no separate clinician manifest was present in the folder.",
                "consent_status": "UNKNOWN",
                "protocol_capture_status": "UNKNOWN_CROPPED_224",
                "exam_date": "UNKNOWN",
                "exam_method": "UNKNOWN",
                "intermittent": "UNKNOWN",
                "glasses_group": "UNKNOWN",
                "pseudostrabismus": "UNKNOWN",
                "sha256": digest,
                "width": width,
                "height": height,
                "image_format": image_format,
                "mode": mode,
                "training_use": "exploratory_only_until_participant_id_and_protocol_are_confirmed",
            }
        )
    return rows


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    label_counts = Counter(row["class_label"] for row in rows)
    original_split_counts = Counter(row["original_split"] for row in rows)
    generated_counts = Counter(row["generated_split"] for row in rows)
    generated_label_counts = Counter((row["generated_split"], row["class_label"]) for row in rows)
    group_to_original_splits = defaultdict(set)
    group_to_generated_splits = defaultdict(set)
    sha_counts = Counter(row["sha256"] for row in rows)
    for row in rows:
        group_to_original_splits[row["group_id"]].add(row["original_split"])
        group_to_generated_splits[row["group_id"]].add(row["generated_split"])

    original_cross_split_groups = {
        group: sorted(splits)
        for group, splits in group_to_original_splits.items()
        if len(splits) > 1
    }
    generated_cross_split_groups = {
        group: sorted(splits)
        for group, splits in group_to_generated_splits.items()
        if len(splits) > 1
    }

    return {
        "datasetVersion": DATASET_VERSION,
        "domain": DOMAIN,
        "labelSource": LABEL_SOURCE,
        "recordCount": len(rows),
        "labelCounts": dict(label_counts),
        "originalSplitCounts": dict(original_split_counts),
        "generatedSplitCounts": dict(generated_counts),
        "generatedSplitLabelCounts": {
            f"{split}/{label}": count
            for (split, label), count in sorted(generated_label_counts.items())
        },
        "uniqueGroupCount": len(group_to_generated_splits),
        "originalCrossSplitGroupCount": len(original_cross_split_groups),
        "generatedCrossSplitGroupCount": len(generated_cross_split_groups),
        "duplicateShaCount": sum(1 for count in sha_counts.values() if count > 1),
        "limitations": [
            "participant_id is UNKNOWN because no participant manifest was present",
            "generated split prevents same label+basename group crossing splits but cannot prove patient-level independence",
            "consent_status is UNKNOWN",
            "protocol capture status is UNKNOWN_CROPPED_224",
            "training use remains exploratory until participant IDs and protocol metadata are confirmed",
        ],
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a labeled manifest from folder labels.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rows = build_manifest(args.root, seed=args.seed)
    if not rows:
        raise SystemExit("No labeled images found.")

    write_csv(args.csv, rows)
    write_jsonl(args.jsonl, rows)
    summary = summarize(rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
