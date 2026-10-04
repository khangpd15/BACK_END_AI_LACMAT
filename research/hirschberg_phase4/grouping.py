"""Conservative duplicate groups are NOT verified patient identities."""

import hashlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DATA_DIRS = {
    "harschberg_data_detect/esotropia_harschberg": "esotropia",
    "harschberg_data_detect/exotropia_harschberg": "exotropia",
    "harschberg_data_detect/normal_harschberg": "normal",
    "dataset_by_pedseye_manifest/esotropia_patient_before": "esotropia",
    "dataset_by_pedseye_manifest/exotropia_patient_before": "exotropia",
    "dataset_by_pedseye_manifest/normal_patient_minifest_after": "normal",
}


def fingerprint(path, root):
    with Image.open(path) as image:
        gray = image.convert("L")
        small = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
        diff = small[:, 1:] > small[:, :-1]
        dhash = sum(int(v) << i for i, v in enumerate(diff.ravel()))
        resized = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
        low = cv2.dct(resized)[:8, :8].ravel()[1:]
        phash = sum(int(v) << i for i, v in enumerate(low > np.median(low)))
        width, height = image.size
    return {"image_id": path.relative_to(root).as_posix(), "width": width, "height": height,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "dhash": dhash, "phash": phash}


def collect(root):
    root = Path(root).resolve()
    rows = []
    for directory, label in DATA_DIRS.items():
        path = root / directory
        if not path.is_dir():
            raise ValueError(f"Expected supplied-data directory missing: {directory}")
        for image in sorted(path.rglob("*")):
            if image.suffix.lower() in (".jpg", ".jpeg", ".png"):
                rows.append({**fingerprint(image, root), "label": label,
                             "patient_id": None, "identity_basis": "unverified"})
    if not rows:
        raise ValueError("No supplied images found")
    return rows


def cluster(rows, distance=8, existing_links=None):
    if not 0 <= distance <= 63:
        raise ValueError("Hash threshold outside [0,63]")
    ids = [r["image_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate image ID in grouping inventory")
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a, b):
        parents[find(b)] = find(a)

    edges = []
    for i, a in enumerate(rows):
        for j in range(i + 1, len(rows)):
            b = rows[j]
            reasons = []
            if a["sha256"] == b["sha256"]:
                reasons.append("exact_sha256")
            if (a["dhash"] ^ b["dhash"]).bit_count() <= distance:
                reasons.append("dhash_candidate")
            if (a["phash"] ^ b["phash"]).bit_count() <= distance:
                reasons.append("phash_candidate")
            if existing_links and existing_links.get(a["image_id"]) is not None and (
                existing_links.get(a["image_id"]) == existing_links.get(b["image_id"])
            ):
                reasons.append("existing_inferred_cluster_not_patient")
            if reasons:
                union(i, j)
                edges.append({"a": a["image_id"], "b": b["image_id"], "reasons": reasons})
    members = {}
    for i, row in enumerate(rows):
        members.setdefault(find(i), []).append(row["image_id"])
    group_ids = {key: "PROVISIONAL_" + hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()[:16]
                 for key, values in members.items()}
    assigned = [{**row, "provisional_group_id": group_ids[find(i)],
                 "patient_id": None, "identity_basis": "inferred_similarity_not_patient"}
                for i, row in enumerate(rows)]
    label_sets = {}
    for row in assigned:
        label_sets.setdefault(row["provisional_group_id"], set()).add(row.get("label"))
    sizes = Counter(r["provisional_group_id"] for r in assigned)
    summary = {"images": len(rows), "provisional_groups": len(members), "candidate_edges": len(edges),
               "largest_group": max(sizes.values(), default=0),
               "mixed_label_groups": sum(len(v) > 1 for v in label_sets.values()),
               "hash_distance_threshold": distance, "verified_patient_ids": 0,
               "patient_independent": False}
    return assigned, edges, summary
