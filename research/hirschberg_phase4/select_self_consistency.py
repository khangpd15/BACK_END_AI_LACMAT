"""Create a blinded, stratified image sample for landmark self-consistency review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


DEFAULT_SEED = 20261004


def stratified_sample(rows: list[dict], n: int = 20, seed: int = DEFAULT_SEED) -> list[dict]:
    if n <= 0 or n > len(rows):
        raise ValueError("Sample size must be between 1 and the number of rows")
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["class_label"], []).append(row)
    if n < len(groups):
        raise ValueError("Sample size must include at least one item per class")

    exact = {label: n * len(group) / len(rows) for label, group in groups.items()}
    allocation = {label: int(value) for label, value in exact.items()}
    for label in sorted(groups):
        if allocation[label] == 0:
            allocation[label] = 1
    while sum(allocation.values()) > n:
        candidates = [label for label in groups if allocation[label] > 1]
        label = min(candidates, key=lambda name: (exact[name] - allocation[name], name))
        allocation[label] -= 1
    remainder_order = sorted(groups, key=lambda name: (-(exact[name] - int(exact[name])), name))
    while sum(allocation.values()) < n:
        label = next(name for name in remainder_order if allocation[name] < len(groups[name]))
        allocation[label] += 1

    rng = random.Random(seed)
    chosen = []
    for label in sorted(groups):
        group = sorted(groups[label], key=lambda row: row["relative_path"])
        chosen.extend(rng.sample(group, allocation[label]))
    chosen.sort(key=lambda row: (row["class_label"], row["relative_path"]))
    return [
        {
            "sample_id": f"S{index:03d}",
            "relative_path": row["relative_path"],
            "class_label": row["class_label"],
        }
        for index, row in enumerate(chosen, start=1)
    ]


def build_manifest(csv_path: Path, repo_root: Path, n: int = 20, seed: int = DEFAULT_SEED) -> dict:
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 62:
        raise ValueError(f"Expected the 62 legacy annotations, found {len(rows)}")
    selected = stratified_sample(rows, n=n, seed=seed)
    for row in selected:
        image_path = (repo_root / row["relative_path"]).resolve()
        if not image_path.is_file() or not image_path.is_relative_to(repo_root.resolve()):
            raise ValueError(f"Missing or out-of-repository image: {row['relative_path']}")
    return {
        "format_version": 1,
        "purpose": "same-annotator landmark repeatability; not detector validation",
        "seed": seed,
        "source_csv_sha256": digest,
        "source_row_count": len(rows),
        "sample_count": n,
        "strata_counts": dict(Counter(row["class_label"] for row in selected)),
        "samples": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="processed/hirschberg_manual_annotations.csv")
    parser.add_argument("--root", default=".")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(Path(args.csv), Path(args.root), args.n, args.seed)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(output), "n": args.n, "seed": args.seed,
                      "strata_counts": manifest["strata_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
