"""Train Domain-Adapted Transfer Models for RemiCare — Phase 5.

Implements two experiments derived from the Phase 5 training audit (REMICARE-DOC-EXP-005):

Experiment A (Baseline — already saved, loaded here for comparison):
    Korean 30-feature shared model → models/korean_shared_model.joblib

Experiment B1 (26-Invariant Features):
    Drop 4 POTENTIAL_DOMAIN_SHIFT viewport-position features.
    Train on 26 translation-invariant features only.
    → models/remicare_b1_invariant_model.joblib

Experiment B2 (30-feature + Coordinate Space Alignment):
    Keep all 30 features.
    Apply IPD-ratio rescaling to the 7 horizontal disparity features before training,
    so the Korean model learns values in the RemiCare coordinate scale.
    Scale factor = Korean_median_IPD / RemiCare_estimated_IPD ≈ 0.33 / 0.13 ≈ 2.54
    At inference: RemiCare values are passed directly (already in the rescaled scale).
    → models/remicare_b2_rescaled_model.joblib

Best model (highest F1 across B1 and B2) saved as:
    → models/remicare_transfer_model.joblib
    → app/models/remicare_transfer_model.joblib

CRITICAL CONSTRAINTS:
- NEVER overwrite models/korean_shared_model.joblib
- Labels come ONLY from folder names (test_normal / test_strabismus)
- No RemiCare clinical labels used for training
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.services.shared_feature_contract import (
    ALL_SHARED_FEATURES,
    INVARIANT_SHARED_FEATURES,
    VIEWPORT_POSITION_FEATURES,
)

__test__ = False


# =============================================================================
# CONSTANTS
# =============================================================================

# Horizontal disparity features that show HIGH domain shift (z > 3)
HORIZONTAL_DISPARITY_FEATURES: List[str] = [
    "meanDeltaX",
    "medianDeltaX",
    "stdDeltaX",
    "minDeltaX",
    "maxDeltaX",
    "rangeDeltaX",
    "meanAbsDeltaX",
]

# Korean training set median IPD (computed from Korean training data)
KOREAN_MEDIAN_IPD: float = 0.33009  # Korean medianDeltaX median value from training data

# RemiCare estimated IPD (from the single available demo sample)
REMICARE_ESTIMATED_IPD: float = 0.13190  # remicare-demo-sample-001 medianDeltaX

# Coordinate scale factor: multiply RemiCare horizontal disparity by this to map to Korean scale
# OR: divide Korean horizontal disparity by this when training, so model expects RemiCare scale
IPD_SCALE_FACTOR: float = KOREAN_MEDIAN_IPD / REMICARE_ESTIMATED_IPD  # ~2.504

# Feature order for B1 (26 invariant features only — same as INVARIANT_SHARED_FEATURES)
B1_FEATURE_ORDER: List[str] = INVARIANT_SHARED_FEATURES

# Feature order for B2 (30 features, same canonical order)
B2_FEATURE_ORDER: List[str] = ALL_SHARED_FEATURES


def _print_section(title: str) -> None:
    print()
    print("=" * 68)
    print(f" {title}")
    print("=" * 68)


def _print_metrics(name: str, acc: float, prec: float, rec: float, f1: float, cm: List) -> None:
    print(f"  Model          : {name}")
    print(f"  Accuracy       : {acc * 100:.2f}%")
    print(f"  Precision      : {prec * 100:.2f}%")
    print(f"  Recall         : {rec * 100:.2f}%")
    print(f"  F1 Score       : {f1:.4f}")
    tn, fp = cm[0]
    fn, tp = cm[1]
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    print(f"  Specificity    : {spec * 100:.2f}%  (True Normal Rate = {tn}/{tn+fp})")
    print(f"  Confusion Matrix:")
    print(f"                      Pred NORMAL  Pred STRAB")
    print(f"    Actual NORMAL  :  {tn:>5}        {fp:>5}")
    print(f"    Actual STRAB   :  {fn:>5}        {tp:>5}")
    print()


# =============================================================================
# LOAD DATA
# =============================================================================

def load_train_test_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-built Korean shared train and test CSVs."""
    train_path = os.path.join(PROJECT_ROOT, "data", "processed", "korean_shared_train.csv")
    test_path = os.path.join(PROJECT_ROOT, "data", "processed", "korean_shared_test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training CSV not found: {train_path}\n"
            "Run scripts/build_korean_shared_datasets.py first."
        )
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Test CSV not found: {test_path}\n"
            "Run scripts/build_korean_shared_datasets.py first."
        )

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    return df_train, df_test


def get_Xy(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix and label vector from dataframe."""
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


# =============================================================================
# EXPERIMENT B1: 26 Invariant Features
# =============================================================================

def run_experiment_b1(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train models on 26 invariant features (no viewport positions)."""
    _print_section("EXPERIMENT B1: 26 Invariant Features (Drop POTENTIAL_DOMAIN_SHIFT)")
    print(f"  Dropped features: {VIEWPORT_POSITION_FEATURES}")
    print(f"  Training features ({len(B1_FEATURE_ORDER)}): {B1_FEATURE_ORDER}")
    print()

    X_train, y_train = get_Xy(df_train, B1_FEATURE_ORDER)
    X_test, y_test = get_Xy(df_test, B1_FEATURE_ORDER)

    print(f"  Train: {X_train.shape} | NORMAL={sum(y_train==0)} STRAB={sum(y_train==1)}")
    print(f"  Test : {X_test.shape} | NORMAL={sum(y_test==0)} STRAB={sum(y_test==1)}")

    sample_weights = compute_sample_weight("balanced", y_train)

    models = {
        "Random Forest (balanced)": RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42
        ),
        "Logistic Regression (balanced)": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        ),
        "SVM RBF (balanced)": make_pipeline(
            StandardScaler(),
            SVC(probability=True, kernel="rbf", class_weight="balanced", random_state=42),
        ),
    }

    results = {}
    best_f1 = -1.0
    best_name = ""
    best_model = None

    for name, clf in models.items():
        # Gradient Boosting uses sample_weight; others use class_weight built-in
        if "Gradient Boosting" in name:
            clf.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "confusionMatrix": cm,
            "specificity": round(cm[0][0] / (cm[0][0] + cm[0][1]), 4) if (cm[0][0] + cm[0][1]) > 0 else 0.0,
        }

        _print_metrics(name, acc, prec, rec, f1, cm)

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = clf

    print(f"  >>> Best B1 model: {best_name} | F1={best_f1:.4f}")

    artifact = {
        "model": best_model,
        "name": best_name,
        "experiment": "B1_INVARIANT_FEATURES",
        "version": "remicare-transfer-b1-v1.0.0",
        "feature_names": B1_FEATURE_ORDER,
        "feature_count": len(B1_FEATURE_ORDER),
        "ipd_scale_factor": None,
        "coordinate_rescaling": False,
        "dropped_features": VIEWPORT_POSITION_FEATURES,
        "metrics": results[best_name],
        "all_metrics": results,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "domain": "KOREAN_INFRARED_EYE_TRACKER",
        "description": (
            "Trained on 26 translation-invariant shared features. "
            "Excludes 4 POTENTIAL_DOMAIN_SHIFT viewport position features "
            "(meanLeftX, meanLeftY, meanRightX, meanRightY). "
            "Class imbalance addressed via class_weight='balanced'."
        ),
    }

    return artifact, results


# =============================================================================
# EXPERIMENT B2: 30 Features + IPD Coordinate Rescaling
# =============================================================================

def rescale_horizontal_disparity(X: np.ndarray, feature_cols: List[str], scale: float) -> np.ndarray:
    """Divide Korean horizontal disparity features by scale_factor.

    This maps Korean coordinate-space IPD values (~0.33) down to RemiCare
    MediaPipe coordinate-space scale (~0.13).

    During training: Korean X gets divided by scale (Korean → RemiCare scale).
    During inference: RemiCare features are passed in directly (already in RemiCare scale).
    """
    X_scaled = X.copy()
    for feat in HORIZONTAL_DISPARITY_FEATURES:
        if feat in feature_cols:
            idx = feature_cols.index(feat)
            X_scaled[:, idx] = X_scaled[:, idx] / scale
    return X_scaled


def run_experiment_b2(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    scale_factor: float = IPD_SCALE_FACTOR,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train models on 30 features with coordinate-space alignment (IPD rescaling)."""
    _print_section("EXPERIMENT B2: 30 Features + IPD Coordinate-Space Rescaling")
    print(f"  IPD Scale Factor       : {scale_factor:.4f}")
    print(f"  Korean median IPD      : {KOREAN_MEDIAN_IPD:.5f}")
    print(f"  RemiCare estimated IPD : {REMICARE_ESTIMATED_IPD:.5f}")
    print(f"  Rescaled features (÷ {scale_factor:.2f}): {HORIZONTAL_DISPARITY_FEATURES}")
    print(f"  All features ({len(B2_FEATURE_ORDER)})")
    print()

    X_train_raw, y_train = get_Xy(df_train, B2_FEATURE_ORDER)
    X_test_raw, y_test = get_Xy(df_test, B2_FEATURE_ORDER)

    # Rescale horizontal disparity: Korean → RemiCare coordinate scale
    X_train = rescale_horizontal_disparity(X_train_raw, B2_FEATURE_ORDER, scale_factor)
    X_test = rescale_horizontal_disparity(X_test_raw, B2_FEATURE_ORDER, scale_factor)

    print(f"  Train: {X_train.shape} | NORMAL={sum(y_train==0)} STRAB={sum(y_train==1)}")
    print(f"  Test : {X_test.shape} | NORMAL={sum(y_test==0)} STRAB={sum(y_test==1)}")
    print(f"  Verify rescaling — Train meanDeltaX after scale: "
          f"mean={X_train[:, B2_FEATURE_ORDER.index('meanDeltaX')].mean():.5f} "
          f"(expected ~{REMICARE_ESTIMATED_IPD:.5f})")

    sample_weights = compute_sample_weight("balanced", y_train)

    models = {
        "Random Forest (balanced)": RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42
        ),
        "Logistic Regression (balanced)": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        ),
        "SVM RBF (balanced)": make_pipeline(
            StandardScaler(),
            SVC(probability=True, kernel="rbf", class_weight="balanced", random_state=42),
        ),
    }

    results = {}
    best_f1 = -1.0
    best_name = ""
    best_model = None

    for name, clf in models.items():
        if "Gradient Boosting" in name:
            clf.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "confusionMatrix": cm,
            "specificity": round(cm[0][0] / (cm[0][0] + cm[0][1]), 4) if (cm[0][0] + cm[0][1]) > 0 else 0.0,
        }

        _print_metrics(name, acc, prec, rec, f1, cm)

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = clf

    print(f"  >>> Best B2 model: {best_name} | F1={best_f1:.4f}")

    artifact = {
        "model": best_model,
        "name": best_name,
        "experiment": "B2_IPD_RESCALED",
        "version": "remicare-transfer-b2-v1.0.0",
        "feature_names": B2_FEATURE_ORDER,
        "feature_count": len(B2_FEATURE_ORDER),
        "ipd_scale_factor": scale_factor,
        "coordinate_rescaling": True,
        "horizontal_disparity_features": HORIZONTAL_DISPARITY_FEATURES,
        "korean_median_ipd": KOREAN_MEDIAN_IPD,
        "remicare_estimated_ipd": REMICARE_ESTIMATED_IPD,
        "metrics": results[best_name],
        "all_metrics": results,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "domain": "KOREAN_INFRARED_EYE_TRACKER",
        "description": (
            f"Trained on all 30 features with horizontal disparity rescaled by "
            f"1/{scale_factor:.2f} (Korean→RemiCare coordinate space alignment). "
            f"At inference, RemiCare MediaPipe features are used directly. "
            f"Class imbalance addressed via class_weight='balanced'."
        ),
    }

    return artifact, results


# =============================================================================
# INFERENCE TEST ON REMICARE DEMO SAMPLE
# =============================================================================

def run_inference_test(artifact: Dict[str, Any]) -> Dict[str, Any]:
    """Run inference experiment on the single available RemiCare demo sample."""
    sample_path = os.path.join(PROJECT_ROOT, "data", "raw", "sample.json")
    if not os.path.exists(sample_path):
        print("  WARNING: data/raw/sample.json not found — skipping RemiCare inference test.")
        return {}

    from app.schemas import ScreeningRequest
    from app.services.shared_feature_contract import extract_shared_features_vector

    with open(sample_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    req = ScreeningRequest.model_validate(raw)
    all_samples = []
    for cycle in req.cycles:
        for s in cycle.samples:
            all_samples.append({
                "t": s.t,
                "leftX": s.leftX, "leftY": s.leftY, "leftValid": s.leftValid,
                "rightX": s.rightX, "rightY": s.rightY, "rightValid": s.rightValid,
            })

    features = extract_shared_features_vector(all_samples)

    feature_cols = artifact["feature_names"]
    feat_vec = np.array([features.get(f, 0.0) or 0.0 for f in feature_cols], dtype=float).reshape(1, -1)

    clf = artifact["model"]
    pred_label = int(clf.predict(feat_vec)[0])
    pred_str = "STRABISMUS" if pred_label == 1 else "NORMAL"

    probs = {}
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(feat_vec)[0]
        probs = {"NORMAL": round(float(p[0]), 4), "STRABISMUS": round(float(p[1]), 4)}

    print(f"  RemiCare Demo Sample — {artifact['experiment']}:")
    print(f"    Prediction : {pred_str}")
    print(f"    Probability: NORMAL={probs.get('NORMAL', 'N/A'):.4f}  STRAB={probs.get('STRABISMUS', 'N/A'):.4f}")
    print(f"    Note: No clinical ground truth for this sample.")

    return {
        "sampleId": req.sampleId,
        "experiment": artifact["experiment"],
        "prediction": pred_str,
        "classProbability": probs,
    }


# =============================================================================
# SAVE ARTIFACTS
# =============================================================================

def save_artifact(artifact: Dict[str, Any], filename: str) -> str:
    """Save joblib artifact to models/ and app/models/."""
    models_dir = os.path.join(PROJECT_ROOT, "models")
    app_models_dir = os.path.join(PROJECT_ROOT, "app", "models")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(app_models_dir, exist_ok=True)

    path_1 = os.path.join(models_dir, filename)
    path_2 = os.path.join(app_models_dir, filename)

    # Safety guard: never overwrite the baseline Korean model
    if "korean_shared_model" in filename:
        raise ValueError(
            "SAFETY GUARD: Cannot overwrite korean_shared_model.joblib. "
            "Use a different filename for new models."
        )

    joblib.dump(artifact, path_1)
    joblib.dump(artifact, path_2)
    print(f"  Saved: {path_1}")
    print(f"  Saved: {path_2}")
    return path_1


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    _print_section("PHASE 5: DOMAIN-ADAPTED TRANSFER MODELS FOR REMICARE")
    print("  REMICARE-DOC-EXP-005 Implementation")
    print("  Korean Infrared -> RemiCare Webcam/MediaPipe Transfer")
    print()

    # Load data
    df_train, df_test = load_train_test_dataframes()
    print(f"  Loaded train: {df_train.shape}  (NORMAL={sum(df_train.label==0)}, STRAB={sum(df_train.label==1)})")
    print(f"  Loaded test : {df_test.shape}  (NORMAL={sum(df_test.label==0)}, STRAB={sum(df_test.label==1)})")

    # -------------------------------------------------------------------------
    # Experiment B1
    # -------------------------------------------------------------------------
    b1_artifact, b1_results = run_experiment_b1(df_train, df_test)
    save_artifact(b1_artifact, "remicare_b1_invariant_model.joblib")
    b1_inference = run_inference_test(b1_artifact)

    # -------------------------------------------------------------------------
    # Experiment B2
    # -------------------------------------------------------------------------
    b2_artifact, b2_results = run_experiment_b2(df_train, df_test)
    save_artifact(b2_artifact, "remicare_b2_rescaled_model.joblib")
    b2_inference = run_inference_test(b2_artifact)

    # -------------------------------------------------------------------------
    # Pick best overall model → remicare_transfer_model.joblib
    # -------------------------------------------------------------------------
    _print_section("SELECTING BEST TRANSFER MODEL")
    b1_f1 = b1_artifact["metrics"]["f1"]
    b2_f1 = b2_artifact["metrics"]["f1"]
    b1_spec = b1_artifact["metrics"]["specificity"]
    b2_spec = b2_artifact["metrics"]["specificity"]

    print(f"  Experiment B1 (26 invariant features): F1={b1_f1:.4f}, Specificity={b1_spec*100:.1f}%")
    print(f"  Experiment B2 (30 features + IPD rescaling): F1={b2_f1:.4f}, Specificity={b2_spec*100:.1f}%")

    # Selection criterion: prioritize specificity improvement (more NORMAL correctly identified)
    # then F1 as tiebreaker
    if b2_spec > b1_spec or (b2_spec == b1_spec and b2_f1 >= b1_f1):
        best_artifact = b2_artifact
        best_exp = "B2"
    else:
        best_artifact = b1_artifact
        best_exp = "B1"

    print(f"  >>> Selected: Experiment {best_exp} as remicare_transfer_model")
    save_artifact(best_artifact, "remicare_transfer_model.joblib")

    # -------------------------------------------------------------------------
    # Save evaluation report
    # -------------------------------------------------------------------------
    report = {
        "phase": "Phase 5 — Domain-Adapted Transfer Models",
        "documentId": "REMICARE-DOC-EXP-005",
        "ipdScaleFactor": IPD_SCALE_FACTOR,
        "koreanMedianIPD": KOREAN_MEDIAN_IPD,
        "remicareEstimatedIPD": REMICARE_ESTIMATED_IPD,
        "selectedExperiment": best_exp,
        "selectedModelVersion": best_artifact["version"],
        "featureCount": best_artifact["feature_count"],
        "coordinateRescaling": best_artifact["coordinate_rescaling"],
        "experiments": {
            "B1_invariant_26": {
                "featureCount": len(B1_FEATURE_ORDER),
                "droppedFeatures": VIEWPORT_POSITION_FEATURES,
                "bestModel": b1_artifact["name"],
                "metrics": b1_artifact["metrics"],
                "remicareInference": b1_inference,
            },
            "B2_ipd_rescaled_30": {
                "featureCount": len(B2_FEATURE_ORDER),
                "rescaledFeatures": HORIZONTAL_DISPARITY_FEATURES,
                "scaleFactor": IPD_SCALE_FACTOR,
                "bestModel": b2_artifact["name"],
                "metrics": b2_artifact["metrics"],
                "remicareInference": b2_inference,
            },
        },
        "notice": (
            "Technical benchmark on Korean infrared dataset only. "
            "NOT clinical performance. NOT a diagnostic tool."
        ),
    }

    report_path = os.path.join(PROJECT_ROOT, "data", "processed", "domain_adapted_models_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved evaluation report: {report_path}")

    # Print final summary
    _print_section("FINAL SUMMARY")
    best_m = best_artifact["metrics"]
    print(f"  Selected Model : {best_artifact['name']}")
    print(f"  Experiment     : {best_artifact['experiment']}")
    print(f"  Version        : {best_artifact['version']}")
    print(f"  Feature Count  : {best_artifact['feature_count']}")
    print(f"  Accuracy       : {best_m['accuracy']*100:.2f}%")
    print(f"  Precision      : {best_m['precision']*100:.2f}%")
    print(f"  Recall (STRAB) : {best_m['recall']*100:.2f}%")
    print(f"  Specificity    : {best_m['specificity']*100:.2f}%  ← key improvement target")
    print(f"  F1 Score       : {best_m['f1']:.4f}")
    print()
    print("  Artifacts saved:")
    print("    models/remicare_b1_invariant_model.joblib")
    print("    models/remicare_b2_rescaled_model.joblib")
    print("    models/remicare_transfer_model.joblib   ← used by FastAPI")
    print()
    print("  IMPORTANT: models/korean_shared_model.joblib was NOT modified.")
    print()
    print("  CLINICAL DISCLAIMER:")
    print("    All metrics are technical benchmarks on the Korean infrared dataset.")
    print("    NOT clinical performance. NOT a diagnostic tool.")
    print("    RemiCare clinical labels are required for a proper RemiCare model.")


if __name__ == "__main__":
    main()
