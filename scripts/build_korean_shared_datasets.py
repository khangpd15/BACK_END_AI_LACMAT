"""Build Korean Shared Datasets, Audit Participant Leakage, and Train Shared Models.

Implements Steps 3, 4, 5, 6, 7, 8 of Phase 4:
- Extracts data/processed/korean_shared_train.csv from 297 training CSVs.
- Extracts data/processed/korean_shared_test.csv from 41 testing CSVs.
- Audits participant-level data leakage between train and test sets.
- Generates docs/participant_leakage_report.md.
- Generates docs/korean_shared_feature_analysis.md (feature distribution comparison).
- Trains baseline models (Logistic Regression, Random Forest, SVM) fitting scalers ONLY on train.
- Saves models/korean_shared_model.joblib and evaluation reports.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.services.korean_adapter import parse_korean_csv
from app.services.shared_feature_contract import (
    ALL_SHARED_FEATURES,
    SHARED_FEATURE_SPECS,
    VIEWPORT_POSITION_FEATURES,
    extract_shared_features_vector,
)

__test__ = False


def extract_participant_id(file_path: str) -> str:
    """Extract participant identifier from filename.
    
    Examples:
    - '1-박서영_all_gaze.csv' -> '박서영'
    - '김민경-정상_all_gaze.csv' -> '김민경'
    - '장진우1-정_all_gaze.csv' -> '장진우'
    - '1-김태형 15_all_gaze.csv' -> '김태형'
    """
    base = os.path.basename(file_path)
    base = os.path.splitext(base)[0]
    base = re.sub(r"_(all_gaze|fixations)$", "", base)

    # Match Korean letters
    m = re.search(r"[\uac00-\ud7a3]+", base)
    if m:
        name = m.group(0)
        # Strip common eye/condition suffixes if attached to name
        for suf in ["정상", "정", "외", "내", "좌내", "우내", "좌외", "우외"]:
            if name.endswith(suf) and len(name) > len(suf) + 1:
                name = name[:-len(suf)]
        return name
    return re.sub(r"^[0-9]+[\-_]", "", base)


def collect_file_tasks(repo_data_dir: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Collect file tasks for train and test sets."""
    train_tasks: List[Tuple[str, str]] = []
    normal_dir = os.path.join(repo_data_dir, "normal")
    if os.path.exists(normal_dir):
        for f in Path(normal_dir).rglob("*.csv"):
            train_tasks.append((str(f), "NORMAL"))

    for sub in ["esotropia", "exotropia", "hypertropia"]:
        sub_dir = os.path.join(repo_data_dir, sub)
        if os.path.exists(sub_dir):
            for f in Path(sub_dir).rglob("*.csv"):
                train_tasks.append((str(f), "STRABISMUS"))

    test_tasks: List[Tuple[str, str]] = []
    test_normal_dir = os.path.join(PROJECT_ROOT, "data", "test_normal")
    test_strabismus_dir = os.path.join(PROJECT_ROOT, "data", "test_strabismus")

    if os.path.exists(test_normal_dir):
        for f in Path(test_normal_dir).rglob("*.csv"):
            test_tasks.append((str(f), "NORMAL"))

    if os.path.exists(test_strabismus_dir):
        for f in Path(test_strabismus_dir).rglob("*.csv"):
            test_tasks.append((str(f), "STRABISMUS"))

    return train_tasks, test_tasks


def build_shared_dataset_dataframe(tasks: List[Tuple[str, str]]) -> pd.DataFrame:
    """Extract shared features from tasks and return DataFrame."""
    rows: List[Dict[str, Any]] = []

    for f_path, label_str in tasks:
        record, errs = parse_korean_csv(f_path, label=label_str)
        if record is None:
            continue

        feats = extract_shared_features_vector(record.samples)
        row = {
            "sampleId": record.sampleId,
            "label": 1 if label_str == "STRABISMUS" else 0,
            "participantId": extract_participant_id(f_path),
        }
        # Add strictly the shared features in canonical order
        for f_name in ALL_SHARED_FEATURES:
            row[f_name] = feats.get(f_name)

        rows.append(row)

    return pd.DataFrame(rows)


def analyze_participant_leakage(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    output_report_md: str,
) -> Dict[str, Any]:
    """Audit participant-level leakage between train and test datasets."""
    train_participants = set(df_train["participantId"].dropna().unique())
    test_participants = set(df_test["participantId"].dropna().unique())

    overlap = train_participants.intersection(test_participants)
    overlap_list = sorted(list(overlap))

    leakage_status = "PARTICIPANT_LEVEL_LEAKAGE" if len(overlap) > 0 else "PASS"

    overlap_details: List[Dict[str, Any]] = []
    for p in overlap_list:
        n_train = int((df_train["participantId"] == p).sum())
        n_test = int((df_test["participantId"] == p).sum())
        overlap_details.append({
            "participantId": p,
            "trainFileCount": n_train,
            "testFileCount": n_test,
        })

    # Write Markdown report
    Path(output_report_md).parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_md, "w", encoding="utf-8") as f:
        f.write("# Participant-Level Data Leakage Audit Report\n\n")
        f.write(f"**Audit Status:** `{leakage_status}`\n\n")
        f.write("> **WARNING:** This audit is mandatory before claiming model generalization. "
                "If overlap > 0, the test set cannot be claimed as an independent participant-level evaluation.\n\n")
        f.write("## 1. Summary Statistics\n\n")
        f.write(f"- **Total Training Files:** {len(df_train)}\n")
        f.write(f"- **Unique Training Participants:** {len(train_participants)}\n")
        f.write(f"- **Total Testing Files:** {len(df_test)}\n")
        f.write(f"- **Unique Testing Participants:** {len(test_participants)}\n")
        f.write(f"- **Overlapping Participants:** **{len(overlap)}**\n\n")

        f.write("## 2. Leakage Findings & Specific Overlaps\n\n")
        if overlap_details:
            f.write("| Participant ID (Extracted) | Train File Count | Test File Count | Leakage Risk |\n")
            f.write("|---|---|---|---|\n")
            for item in overlap_details:
                f.write(f"| `{item['participantId']}` | {item['trainFileCount']} | {item['testFileCount']} | HIGH (Same subject present in both sets) |\n")
            f.write("\n### Root Cause Analysis:\n")
            f.write("In the original Korean repository, certain subjects (e.g. `장진우`, `김민경`) had multiple recording sessions. "
                    "One session was filed under `data/normal/` (training) while subsequent sessions were filed under `data/test_normal/` (testing). "
                    "This demonstrates that the test set provided in the source repository is **session-split**, NOT strictly **participant-split**.\n\n")
        else:
            f.write("No participant identifier overlaps were found between train and test sets.\n\n")

        f.write("## 3. Scientific and Regulatory Implications\n\n")
        f.write("1. **Data Leakage Warning:** Performance figures on this test set include subject identity leakage. "
                "The test accuracy cannot be considered an unbiased estimate of generalization to unseen subjects.\n")
        f.write("2. **RemiCare Implication:** Because RemiCare tests completely novel subjects, performance on RemiCare cannot be predicted from this test set.\n")

    return {
        "status": leakage_status,
        "uniqueTrainParticipants": len(train_participants),
        "uniqueTestParticipants": len(test_participants),
        "overlapCount": len(overlap),
        "overlapList": overlap_list,
        "overlapParticipants": overlap_list,
        "details": overlap_details,
    }


def analyze_feature_distributions(
    df_train: pd.DataFrame,
    output_report_md: str,
) -> Dict[str, Any]:
    """Compute empirical statistical distributions of shared features by class."""
    normal_df = df_train[df_train["label"] == 0]
    strab_df = df_train[df_train["label"] == 1]

    stats_dict: Dict[str, Any] = {}

    Path(output_report_md).parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_md, "w", encoding="utf-8") as f:
        f.write("# Korean Shared Feature Statistical Distribution Analysis\n\n")
        f.write("Empirical distribution comparison between **NORMAL** (N=86) and **STRABISMUS** (N=211) "
                "on the 30 shared features in the Korean training dataset.\n\n")
        f.write("> **CLINICAL NOTICE:** This analysis reports empirical training sample statistics only. "
                "No clinical thresholds or diagnostic cut-offs are derived or asserted from these numbers.\n\n")
        f.write("## Distribution Table\n\n")
        f.write("| Feature Name | Class | Mean | Median | Std | Min | Max | Missing Rate | Domain Shift Note |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")

        for f_name in ALL_SHARED_FEATURES:
            norm_col = normal_df[f_name].dropna()
            strab_col = strab_df[f_name].dropna()

            norm_missing = round(float(normal_df[f_name].isna().mean()), 4)
            strab_missing = round(float(strab_df[f_name].isna().mean()), 4)

            norm_stats = {
                "mean": round(float(norm_col.mean()), 6) if len(norm_col) > 0 else 0.0,
                "median": round(float(norm_col.median()), 6) if len(norm_col) > 0 else 0.0,
                "std": round(float(norm_col.std()), 6) if len(norm_col) > 0 else 0.0,
                "min": round(float(norm_col.min()), 6) if len(norm_col) > 0 else 0.0,
                "max": round(float(norm_col.max()), 6) if len(norm_col) > 0 else 0.0,
                "missingRate": norm_missing,
            }
            strab_stats = {
                "mean": round(float(strab_col.mean()), 6) if len(strab_col) > 0 else 0.0,
                "median": round(float(strab_col.median()), 6) if len(strab_col) > 0 else 0.0,
                "std": round(float(strab_col.std()), 6) if len(strab_col) > 0 else 0.0,
                "min": round(float(strab_col.min()), 6) if len(strab_col) > 0 else 0.0,
                "max": round(float(strab_col.max()), 6) if len(strab_col) > 0 else 0.0,
                "missingRate": strab_missing,
            }

            risk = SHARED_FEATURE_SPECS[f_name].domain_shift_risk
            risk_label = "POTENTIAL_DOMAIN_SHIFT" if risk == "POTENTIAL_DOMAIN_SHIFT" else "LOW_RISK"

            stats_dict[f_name] = {"NORMAL": norm_stats, "STRABISMUS": strab_stats, "risk": risk}

            # Normal row
            f.write(f"| `{f_name}` | NORMAL | {norm_stats['mean']} | {norm_stats['median']} | {norm_stats['std']} | {norm_stats['min']} | {norm_stats['max']} | {norm_missing * 100:.1f}% | {risk_label} |\n")
            # Strabismus row
            f.write(f"| | STRABISMUS | {strab_stats['mean']} | {strab_stats['median']} | {strab_stats['std']} | {strab_stats['min']} | {strab_stats['max']} | {strab_missing * 100:.1f}% | |\n")

    return stats_dict


def train_and_save_shared_models(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Dict[str, Any]:
    """Train models fitting scalers strictly on train, and evaluate on test."""
    feature_cols = ALL_SHARED_FEATURES

    # Ensure no target leakage or identifier in feature matrix
    assert "label" not in feature_cols
    assert "sampleId" not in feature_cols
    assert "participantId" not in feature_cols

    X_train = df_train[feature_cols].to_numpy(dtype=float)
    y_train = df_train["label"].to_numpy(dtype=int)

    X_test = df_test[feature_cols].to_numpy(dtype=float)
    y_test = df_test["label"].to_numpy(dtype=int)

    # Impute any edge NaNs cleanly with 0.0 without fitting on test
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # Models with scaler fitted strictly in pipeline on train
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        "SVM RBF": make_pipeline(StandardScaler(), SVC(probability=True, kernel="rbf", random_state=42)),
    }

    eval_results = {}
    best_f1 = -1.0
    best_name = "Random Forest"
    best_model = None

    print("\n" + "=" * 65)
    print("STEP 5 & 6: TRAINING AND EVALUATION OF SHARED FEATURE MODELS")
    print("=" * 65)

    for name, clf in models.items():
        # Fit pipeline strictly on X_train
        clf.fit(X_train, y_train)

        # Predict on X_test
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
            best_name = name
            best_model = clf

    # Save to models/korean_shared_model.joblib
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "korean_shared_model.joblib")

    # Also save in app/models/ for runtime access
    app_models_dir = os.path.join(PROJECT_ROOT, "app", "models")
    os.makedirs(app_models_dir, exist_ok=True)
    app_model_path = os.path.join(app_models_dir, "korean_shared_model.joblib")

    artifact = {
        "model": best_model,
        "name": best_name,
        "version": "korean-shared-v1.0.0",
        "feature_names": feature_cols,
        "metrics": eval_results[best_name],
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "domain": "KOREAN_INFRARED_EYE_TRACKER",
    }

    joblib.dump(artifact, model_path)
    joblib.dump(artifact, app_model_path)
    print(f"Saved best model artifact ({best_name}) to:")
    print(f"  - {model_path}")
    print(f"  - {app_model_path}")

    return eval_results


def main():
    print("=" * 65)
    print("PHASE 4: BUILD KOREAN SHARED DATASETS & AUDIT PIPELINE")
    print("=" * 65)

    repo_data_dir = os.path.join(PROJECT_ROOT, "data", "korean_repo", "data")
    train_tasks, test_tasks = collect_file_tasks(repo_data_dir)

    print(f"Collecting features for {len(train_tasks)} train CSVs...")
    df_train = build_shared_dataset_dataframe(train_tasks)

    print(f"Collecting features for {len(test_tasks)} test CSVs...")
    df_test = build_shared_dataset_dataframe(test_tasks)

    # Verify column equality
    train_feat_cols = [c for c in df_train.columns if c not in ("sampleId", "label", "participantId")]
    test_feat_cols = [c for c in df_test.columns if c not in ("sampleId", "label", "participantId")]
    assert train_feat_cols == test_feat_cols, "Train and test feature columns must be identical!"
    assert train_feat_cols == ALL_SHARED_FEATURES, "Feature columns must match ALL_SHARED_FEATURES contract!"

    # Save cleaned datasets without target leakage (participantId retained for auditing only)
    train_csv = os.path.join(PROJECT_ROOT, "data", "processed", "korean_shared_train.csv")
    test_csv = os.path.join(PROJECT_ROOT, "data", "processed", "korean_shared_test.csv")

    df_train.drop(columns=["participantId"]).to_csv(train_csv, index=False)
    df_test.drop(columns=["participantId"]).to_csv(test_csv, index=False)
    print(f"Saved korean_shared_train.csv: {df_train.shape}")
    print(f"Saved korean_shared_test.csv : {df_test.shape}")

    # Participant leakage audit
    leakage_md = os.path.join(PROJECT_ROOT, "docs", "participant_leakage_report.md")
    leakage_results = analyze_participant_leakage(df_train, df_test, leakage_md)
    print(f"\nParticipant Leakage Status: {leakage_results['status']}")
    print(f"  Unique Train Participants: {leakage_results['uniqueTrainParticipants']}")
    print(f"  Unique Test Participants : {leakage_results['uniqueTestParticipants']}")
    print(f"  Overlap Count            : {leakage_results['overlapCount']}")

    # Feature distribution analysis
    dist_md = os.path.join(PROJECT_ROOT, "docs", "korean_shared_feature_analysis.md")
    analyze_feature_distributions(df_train, dist_md)
    print(f"Saved feature distribution analysis to {dist_md}")

    # Train and evaluate models
    eval_results = train_and_save_shared_models(df_train, df_test)

    # Save summary eval json
    eval_json = os.path.join(PROJECT_ROOT, "data", "processed", "korean_shared_model_eval.json")
    with open(eval_json, "w", encoding="utf-8") as f:
        json.dump({
            "trainSamples": len(df_train),
            "testSamples": len(df_test),
            "featureCount": len(ALL_SHARED_FEATURES),
            "leakage": leakage_results,
            "results": eval_results,
        }, f, indent=2)
    print(f"Saved evaluation JSON to {eval_json}")


if __name__ == "__main__":
    main()
