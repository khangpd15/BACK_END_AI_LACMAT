"""Shared Feature ML Model Pipeline for Korean -> RemiCare Cross-Domain Transfer.

Trains on Korean Training Data using strictly the 26 SHARED features:
    Korean TRAIN (297 samples)
        ↓
    Shared Features Only (26 features)
        ↓
    Train Models (Random Forest, Logistic Regression, SVM)
        ↓
    Korean TEST (41 samples: 20 Normal, 21 Strabismus)
        ↓
    Evaluate (Accuracy, Precision, Recall, F1, Confusion Matrix)
        ↓
    Save Artifact to app/models/shared_strabismus_model.joblib
        ↓
    RemiCare sample.json Inference using identical Shared Features
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.schemas import ScreeningRequest
from app.services.feature_contract import (
    REQUIRED_SHARED_FEATURES,
    extract_contract_features,
)
from app.services.korean_adapter import parse_korean_csv

__test__ = False


def load_dataset_shared_features(file_tasks: List[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract only the 26 REQUIRED_SHARED_FEATURES from a list of (file_path, label) tuples."""
    X_rows: List[List[float]] = []
    y_labels: List[int] = []
    sample_ids: List[str] = []

    for f_path, label in file_tasks:
        record, errs = parse_korean_csv(f_path, label=label)
        if record is None:
            continue

        feats = extract_contract_features(record.samples)
        # Order values strictly according to REQUIRED_SHARED_FEATURES
        feature_vector = [feats[f_name] for f_name in REQUIRED_SHARED_FEATURES]
        X_rows.append(feature_vector)
        y_labels.append(1 if label == "STRABISMUS" else 0)
        sample_ids.append(record.sampleId)

    return np.array(X_rows, dtype=float), np.array(y_labels, dtype=int), sample_ids


def get_training_tasks(repo_data_dir: str) -> List[Tuple[str, str]]:
    """Collect (file_path, label) for training dataset."""
    train_tasks: List[Tuple[str, str]] = []
    normal_dir = os.path.join(repo_data_dir, "normal")
    if os.path.exists(normal_dir):
        for f in Path(normal_dir).rglob("*.csv"):
            train_tasks.append((str(f), "NORMAL"))

    for sub_dir in ["esotropia", "exotropia", "hypertropia"]:
        s_path = os.path.join(repo_data_dir, sub_dir)
        if os.path.exists(s_path):
            for f in Path(s_path).rglob("*.csv"):
                train_tasks.append((str(f), "STRABISMUS"))

    return train_tasks


def get_testing_tasks(test_normal_dir: str, test_strabismus_dir: str) -> List[Tuple[str, str]]:
    """Collect (file_path, label) for independent test dataset."""
    test_tasks: List[Tuple[str, str]] = []
    if os.path.exists(test_normal_dir):
        for f in Path(test_normal_dir).rglob("*.csv"):
            test_tasks.append((str(f), "NORMAL"))

    if os.path.exists(test_strabismus_dir):
        for f in Path(test_strabismus_dir).rglob("*.csv"):
            test_tasks.append((str(f), "STRABISMUS"))

    return test_tasks


def run_shared_feature_pipeline() -> Dict[str, Any]:
    """Execute complete training, evaluation, and RemiCare transfer pipeline."""
    repo_data_dir = os.path.join(PROJECT_ROOT, "data", "korean_repo", "data")
    test_norm_dir = os.path.join(PROJECT_ROOT, "data", "test_normal")
    test_strab_dir = os.path.join(PROJECT_ROOT, "data", "test_strabismus")

    train_tasks = get_training_tasks(repo_data_dir)
    test_tasks = get_testing_tasks(test_norm_dir, test_strab_dir)

    if not train_tasks:
        print("Only test data available; training cannot be performed without data leakage.")
        return {"error": "NO_TRAINING_DATA"}

    print("=" * 65)
    print("PHASE 3: SHARED FEATURE ML MODEL PIPELINE")
    print(f"Contract Features: {len(REQUIRED_SHARED_FEATURES)} shared technical metrics")
    print("=" * 65)
    print(f"Training tasks   : {len(train_tasks)} files")
    print(f"Testing tasks    : {len(test_tasks)} files")
    print("-" * 65)

    print("Extracting shared features from training set...")
    X_train, y_train, _ = load_dataset_shared_features(train_tasks)

    print("Extracting shared features from test set...")
    X_test, y_test, _ = load_dataset_shared_features(test_tasks)

    print(f"Train matrix shape: {X_train.shape} (NORMAL: {sum(y_train == 0)}, STRABISMUS: {sum(y_train == 1)})")
    print(f"Test matrix shape : {X_test.shape} (NORMAL: {sum(y_test == 0)}, STRABISMUS: {sum(y_test == 1)})")
    print("-" * 65)

    # Clean NaNs or infs if any
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1.0, neginf=-1.0)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        "Support Vector Machine": make_pipeline(StandardScaler(), SVC(probability=True, kernel="rbf", random_state=42)),
    }

    eval_results = {}
    best_model_name = "Random Forest"
    best_f1 = -1.0
    best_fitted_model = None

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        eval_results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "confusionMatrix": cm,
        }

        print(f"Model: {name}")
        print(f"  Accuracy : {acc * 100:.2f}% | Precision: {prec * 100:.2f}%")
        print(f"  Recall   : {rec * 100:.2f}% | F1 Score : {f1:.4f}")
        print(f"  Confusion Matrix (TN, FP / FN, TP): {cm}")
        print("-" * 65)

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_fitted_model = clf

    # Save best shared feature model artifact
    models_dir = os.path.join(PROJECT_ROOT, "app", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_artifact_path = os.path.join(models_dir, "shared_strabismus_model.joblib")

    model_metadata = {
        "model": best_fitted_model,
        "name": best_model_name,
        "version": "korean-shared-v0.1.0",
        "feature_names": REQUIRED_SHARED_FEATURES,
        "metrics": eval_results[best_model_name],
        "training_samples": len(X_train),
        "test_samples": len(X_test),
    }

    joblib.dump(model_metadata, model_artifact_path)
    print(f"Saved best model ({best_model_name}, F1: {best_f1:.4f}) to {model_artifact_path}")

    # Save evaluation report JSON
    eval_report_path = os.path.join(PROJECT_ROOT, "data", "processed", "shared_feature_eval_report.json")
    with open(eval_report_path, "w", encoding="utf-8") as fp:
        json.dump({
            "featureCount": len(REQUIRED_SHARED_FEATURES),
            "featureList": REQUIRED_SHARED_FEATURES,
            "bestModel": best_model_name,
            "results": eval_results,
        }, fp, indent=2)
    print(f"Saved evaluation report to {eval_report_path}")

    # =========================================================================
    # INFERENCE EXPERIMENT ON REMICARE SAMPLE.JSON
    # =========================================================================
    remicare_sample_path = os.path.join(PROJECT_ROOT, "data", "raw", "sample.json")
    print("\n" + "=" * 65)
    print("REMICARE INFERENCE EXPERIMENT USING SHARED FEATURE MODEL")
    print(f"Loading RemiCare sample: {remicare_sample_path}")
    print("=" * 65)

    with open(remicare_sample_path, "r", encoding="utf-8") as f:
        remicare_raw = json.load(f)

    req = ScreeningRequest.model_validate(remicare_raw)
    all_samples = []
    for cycle in req.cycles:
        for s in cycle.samples:
            all_samples.append({
                "t": s.t,
                "leftX": s.leftX,
                "leftY": s.leftY,
                "leftValid": s.leftValid,
                "rightX": s.rightX,
                "rightY": s.rightY,
                "rightValid": s.rightValid,
            })

    remicare_features = extract_contract_features(all_samples)
    remicare_vector = np.array([remicare_features[f] for f in REQUIRED_SHARED_FEATURES], dtype=float).reshape(1, -1)
    remicare_vector = np.nan_to_num(remicare_vector, nan=0.0)

    pred_label = int(best_fitted_model.predict(remicare_vector)[0])
    pred_prob = None
    if hasattr(best_fitted_model, "predict_proba"):
        probs = best_fitted_model.predict_proba(remicare_vector)[0]
        pred_prob = float(probs[pred_label])

    screening_outcome = "SCREENING_ATTENTION" if pred_label == 1 else "SCREENING_NORMAL"

    print(f"RemiCare Sample ID       : {req.sampleId}")
    print(f"Features Evaluated       : {len(REQUIRED_SHARED_FEATURES)} shared technical features")
    print(f"Raw Model Prediction     : {pred_label} ({'STRABISMUS' if pred_label == 1 else 'NORMAL'})")
    if pred_prob is not None:
        print(f"Model Confidence / Prob  : {pred_prob * 100:.2f}%")
    print(f"Mapped Screening Status  : {screening_outcome}")
    print("Clinical Disclaimer      : Screening result only — not a diagnosis.")
    print("=" * 65)

    return {
        "bestModel": best_model_name,
        "evalResults": eval_results,
        "remicarePrediction": {
            "sampleId": req.sampleId,
            "prediction": pred_label,
            "status": screening_outcome,
            "confidence": pred_prob,
        },
    }


if __name__ == "__main__":
    run_shared_feature_pipeline()
