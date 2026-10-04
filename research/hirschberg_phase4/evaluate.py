"""Evaluation only: no fitting, tuning or implicit exclusion of abstentions."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from .features import FEATURE_NAMES, extract
from .schema import LABELS, assert_independent, load_manifest, validate


def metrics(truth, predictions):
    if len(truth) != len(predictions) or not truth:
        raise ValueError("Nonempty aligned predictions required")
    matrix = np.zeros((4, 5), dtype=int)
    for actual, predicted in zip(truth, predictions):
        if actual not in LABELS or predicted not in (*LABELS, "abstain"):
            raise ValueError("Unknown metric label")
        matrix[LABELS.index(actual), 4 if predicted == "abstain" else LABELS.index(predicted)] += 1
    per_class = {}
    for i, label in enumerate(LABELS):
        n, tp, predicted_n = int(matrix[i].sum()), int(matrix[i, i]), int(matrix[:, i].sum())
        recall = tp / n if n else None
        precision = tp / predicted_n if predicted_n else None
        f1 = 2 * tp / (n + predicted_n) if n else None
        covered_n = int(matrix[i, :4].sum())
        per_class[label] = {"n": n, "precision": precision, "recall_all": recall,
                            "f1_all": f1, "recall_covered": tp / covered_n if covered_n else None}
    positives = {"esotropia", "exotropia"}
    positive_n = sum(t in positives for t in truth)
    negative_n = len(truth) - positive_n
    referred = [p in positives or p == "abstain" for p in predictions]
    sensitivity = sum(t in positives and r for t, r in zip(truth, referred)) / positive_n if positive_n else None
    specificity = sum(t not in positives and not r for t, r in zip(truth, referred)) / negative_n if negative_n else None
    recalls = [v["recall_all"] for v in per_class.values()]
    return {"n": len(truth), "confusion_columns": [*LABELS, "abstain"],
            "confusion_rows": list(LABELS), "confusion": matrix.tolist(), "per_class": per_class,
            "coverage": sum(p != "abstain" for p in predictions) / len(truth),
            "macro_recall_all": float(np.mean(recalls)) if all(v is not None for v in recalls) else None,
            "referral_sensitivity": sensitivity, "referral_specificity": specificity,
            "referral_rate": sum(referred) / len(truth)}


def evaluate(bundle, rows):
    validate(rows)
    assert_independent(rows, bundle["training_records"])
    if bundle["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Feature contract mismatch")
    if {r["source_kind"] for r in rows} != {bundle["source_kind"]}:
        raise ValueError("Synthetic and human evaluation must remain separate")
    # Primary endpoint is one explicitly preselected record per patient.
    if len({r["patient_id"] for r in rows}) != len(rows):
        raise ValueError("One preselected record per evaluation patient required")
    outputs = []
    for row in rows:
        values, reason = extract(row)
        prediction, confidence = "abstain", None
        if values is not None:
            probs = bundle["model"].predict_proba(values.reshape(1, -1))[0]
            confidence = float(probs.max())
            if confidence >= bundle["min_confidence"]:
                prediction = str(bundle["model"].classes_[int(probs.argmax())])
            else:
                reason = "UNCERTAIN_CLASS"
        outputs.append({"image_id": row["image_id"], "patient_id": row["patient_id"],
                        "truth": row["label"], "prediction": prediction,
                        "confidence": confidence, "reason": reason})
    report = metrics([r["truth"] for r in outputs], [r["prediction"] for r in outputs])
    report.update({"research_only": True, "clinical_evidence": False,
                   "source_kind": bundle["source_kind"], "predictions": outputs})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True, help="Trusted local joblib only; pickle can execute code")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [r for r in load_manifest(args.manifest) if r["split"] == args.split]
    report = evaluate(joblib.load(args.model), rows)
    report.update({"evaluation_split": args.split,
                   "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                   "model_sha256": hashlib.sha256(Path(args.model).read_bytes()).hexdigest()})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
