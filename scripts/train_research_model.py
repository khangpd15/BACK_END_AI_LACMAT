"""Reproducible training scaffold for future RemiCare research models.

Phase 5 intentionally does NOT train a model. This script only records the
minimum reproducibility contract and exits unless an explicit future approval
flag is provided with an eligible dataset summary.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from research_dataset_pipeline import (
    RESEARCH_DATASET_VERSION,
    RESEARCH_FEATURE_VERSION,
    TRAINING_DISABLED_REASON,
)


DEFAULT_SEED = 20261003


def build_training_plan(
    dataset_summary: Dict[str, Any],
    *,
    seed: int = DEFAULT_SEED,
    model_family: str = "TODO_APPROVED_AFTER_CLINICAL_DATASET",
) -> Dict[str, Any]:
    return {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "status": "PLAN_ONLY_NO_TRAINING",
        "trainingDisabledReason": TRAINING_DISABLED_REASON,
        "seed": seed,
        "datasetVersion": dataset_summary.get("datasetVersion", RESEARCH_DATASET_VERSION),
        "featureVersion": RESEARCH_FEATURE_VERSION,
        "modelFamily": model_family,
        "hyperparameters": None,
        "artifactPath": None,
        "metrics": {
            "sensitivity": None,
            "specificity": None,
            "ppv": None,
            "npv": None,
            "roc_auc": None,
            "pr_auc": None,
            "confusion_matrix": None,
            "mae": None,
            "rmse": None,
            "bias": None,
            "bland_altman": None,
        },
        "splitPolicy": "patient_level_only",
        "participantSplit": dataset_summary.get("participantSplit", {}),
        "trainingEligibleParticipantCount": dataset_summary.get("trainingEligibleParticipantCount", 0),
        "notes": [
            "Do not use legacy_non_hirschberg data for threshold selection or training.",
            "Do not use model output as ground truth.",
            "Only horizontal strabismus protocol data with clinician labels may be eligible.",
        ],
    }


def assert_training_may_start(dataset_summary: Dict[str, Any], allow_train: bool) -> None:
    if not allow_train:
        raise RuntimeError(TRAINING_DISABLED_REASON)
    if dataset_summary.get("trainingEligibleParticipantCount", 0) <= 0:
        raise RuntimeError("NO_ELIGIBLE_PARTICIPANTS_WITH_APPROVED_CLINICAL_LABELS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a research training plan. Does not train by default.")
    parser.add_argument("--dataset-summary", required=True, type=Path)
    parser.add_argument("--plan-output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--allow-train",
        action="store_true",
        help="Future explicit approval gate. In Phase 5 this should remain unset.",
    )
    args = parser.parse_args()

    dataset_summary = json.loads(args.dataset_summary.read_text(encoding="utf-8"))
    plan = build_training_plan(dataset_summary, seed=args.seed)
    args.plan_output.parent.mkdir(parents=True, exist_ok=True)
    args.plan_output.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    assert_training_may_start(dataset_summary, allow_train=args.allow_train)
    raise RuntimeError("TRAINING_IMPLEMENTATION_NOT_PRESENT_IN_PHASE_5")


if __name__ == "__main__":
    raise SystemExit(main())
