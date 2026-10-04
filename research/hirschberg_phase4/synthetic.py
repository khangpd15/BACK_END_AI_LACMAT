"""Contract fixtures, not simulated patients or evidence of clinical accuracy."""

import argparse
import json
from pathlib import Path

import numpy as np

from .schema import LABELS, validate


def fixture_records(seed=42):
    rng = np.random.default_rng(seed)
    records = []
    for split, count in (("train", 12), ("val", 4), ("test", 4)):
        for label in LABELS:
            for i in range(count):
                uid = f"fixture-{split}-{label}-{i}"
                # Artificial toy codes intentionally have no clinical semantics.
                offsets = {"esotropia": (12, -8), "exotropia": (-12, 8),
                           "normal": (0, 0), "pseudostrabismus": (0, 0)}[label]
                eyes = {}
                for eye, x, offset in zip(("OD", "OS"), (200, 600), offsets):
                    noise = rng.normal(0, 0.4, 2)
                    eyes[eye] = {"pupil": [x, 200], "limbus": [x, 200],
                                 "limbus_diameter_px": 160,
                                 "glint": [float(x + offset + noise[0]), float(200 + noise[1])]}
                records.append({"image_id": uid, "patient_id": uid, "source_id": "generated-fixture-v1",
                                "source_kind": "synthetic", "rights_status": "approved", "split": split,
                                "label": label, "width": 800, "height": 400, "mirrored": False,
                                "coordinate_space": "native", "landmarks": eyes})
    return validate(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(fixture_records(), handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
