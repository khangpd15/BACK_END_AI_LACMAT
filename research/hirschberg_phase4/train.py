"""A0 logistic baseline on approved landmarks, never images or private test."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, extract
from .schema import LABELS, load_manifest


def fit_baseline(records, seed=42, min_confidence=0.0):
    from .schema import validate
    validate(records)
    if any(r["source_kind"] == "human" and r["split"] == "test" for r in records):
        raise ValueError("Keep private human test records in a separate evaluator-only manifest")
    if not 0 <= min_confidence <= 1:
        raise ValueError("Confidence threshold outside [0,1]")
    training = [r for r in records if r["split"] == "train"]
    if not training or len({r["patient_id"] for r in training}) != len(training):
        raise ValueError("One preselected record per training patient required")
    if len({r["source_kind"] for r in training}) != 1:
        raise ValueError("Do not mix human and synthetic cohorts")
    usable = []
    for row in training:
        values, _ = extract(row)
        if values is not None:
            usable.append((row, values))
    if {r["label"] for r, _ in usable} != set(LABELS):
        raise ValueError("All four classes need usable training records; no invented pseudo")
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, random_state=seed))
    model.fit(np.stack([x for _, x in usable]), [r["label"] for r, _ in usable])
    identifiers = [{k: r.get(k) for k in ("patient_id", "image_id", "image_sha256")} for r in training]
    return {"model": model, "feature_names": list(FEATURE_NAMES),
            "training_records": identifiers, "source_kind": training[0]["source_kind"],
            "seed": seed, "min_confidence": min_confidence,
            "training_count": len(training), "training_usable": len(usable)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True, help="New, nonexistent run directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Predeclared development threshold, not selected on test")
    args = parser.parse_args()
    records = load_manifest(args.manifest)
    bundle = fit_baseline(records, args.seed, args.min_confidence)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    joblib.dump(bundle, output / "model.joblib")
    metadata = {k: v for k, v in bundle.items() if k != "model"}
    metadata.update({"research_only": True, "clinical_evidence": False,
                     "python": platform.python_version(), "numpy": np.__version__,
                     "sklearn": sklearn.__version__,
                     "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                     "model_sha256": hashlib.sha256((output / "model.joblib").read_bytes()).hexdigest()})
    with (output / "metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
