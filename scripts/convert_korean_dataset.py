"""Script to convert Korean Strabismus Eye-Tracker Dataset and run ML evaluation.

Processes all CSVs in:
    data/test_normal/    -> NORMAL (label: 0)
    data/test_strabismus/ -> STRABISMUS (label: 1)

Generates:
    data/normalized/korean/NORMAL/*.json
    data/normalized/korean/STRABISMUS/*.json
    data/normalized/korean/conversion_report.json
    data/processed/korean_features.csv

If training data is available in data/korean_repo/data:
    Trains: Logistic Regression, Random Forest, SVM
    Evaluates strictly on test_normal + test_strabismus without data leakage.
Otherwise:
    Reports: 'Only test data available; training cannot be performed without data leakage.'
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
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

from app.services.korean_adapter import (
    convert_korean_test_folders,
    extract_features_from_korean_record,
    parse_korean_csv,
)

__test__ = False


def check_and_extract_training_data(repo_data_dir: str) -> Optional[pd.DataFrame]:
    """Check if repository contains training data and extract features into DataFrame."""
    train_normal_dir = os.path.join(repo_data_dir, "normal")
    train_esotropia_dir = os.path.join(repo_data_dir, "esotropia")
    train_exotropia_dir = os.path.join(repo_data_dir, "exotropia")
    train_hypertropia_dir = os.path.join(repo_data_dir, "hypertropia")

    has_train_data = os.path.exists(train_normal_dir) and (
        os.path.exists(train_esotropia_dir) or os.path.exists(train_exotropia_dir)
    )

    if not has_train_data:
        return None

    train_tasks: List[tuple] = []
    # Normal files
    if os.path.exists(train_normal_dir):
        for f in Path(train_normal_dir).rglob("*.csv"):
            train_tasks.append((str(f), "NORMAL"))

    # Strabismus subtypes mapped strictly to binary STRABISMUS
    for s_dir in [train_esotropia_dir, train_exotropia_dir, train_hypertropia_dir]:
        if os.path.exists(s_dir):
            for f in Path(s_dir).rglob("*.csv"):
                train_tasks.append((str(f), "STRABISMUS"))

    if not train_tasks:
        return None

    print(f"Extracting features from {len(train_tasks)} training CSVs (Binary: NORMAL vs STRABISMUS)...")
    train_features: List[Dict[str, Any]] = []
    for f_path, label in train_tasks:
        record, errs = parse_korean_csv(f_path, label=label)
        if record:
            feats = extract_features_from_korean_record(record)
            train_features.append(feats)

    if train_features:
        return pd.DataFrame(train_features)
    return None


def train_and_evaluate_models(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """Train baseline models on df_train and evaluate on df_test."""
    feature_cols = [
        c for c in df_train.columns
        if c not in ("sampleId", "label_name", "label")
    ]

    # Ensure clean numeric values
    X_train = df_train[feature_cols].fillna(0.0).to_numpy()
    y_train = df_train["label"].to_numpy().astype(int)

    X_test = df_test[feature_cols].fillna(0.0).to_numpy()
    y_test = df_test["label"].to_numpy().astype(int)

    print("\n" + "=" * 60)
    print("BASELINE ML MODELS EVALUATION (Trained on Train, Evaluated on Test)")
    print("Notice: Technical benchmark only — not a clinical performance rating.")
    print("=" * 60)
    print(f"Training samples: {len(X_train)} (NORMAL: {sum(y_train == 0)}, STRABISMUS: {sum(y_train == 1)})")
    print(f"Testing samples : {len(X_test)} (NORMAL: {sum(y_test == 0)}, STRABISMUS: {sum(y_test == 1)})")
    print("-" * 60)

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine (SVM)": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
    }

    results = {}

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "confusionMatrix": cm,
        }

        print(f"\nModel: {name}")
        print(f"  Accuracy : {acc * 100:.2f}%")
        print(f"  Precision: {prec * 100:.2f}%")
        print(f"  Recall   : {rec * 100:.2f}%")
        print(f"  F1 Score : {f1:.4f}")
        print(f"  Confusion Matrix (TN, FP / FN, TP):")
        print(f"    [ {cm[0][0]:>3}, {cm[0][1]:>3} ]")
        print(f"    [ {cm[1][0]:>3}, {cm[1][1]:>3} ]")

    return results


def main():
    parser = argparse.ArgumentParser(description="Convert Korean Strabismus Dataset")
    parser.add_argument(
        "--normal-dir",
        default=os.path.join(PROJECT_ROOT, "data", "test_normal"),
        help="Path to test_normal folder",
    )
    parser.add_argument(
        "--strabismus-dir",
        default=os.path.join(PROJECT_ROOT, "data", "test_strabismus"),
        help="Path to test_strabismus folder",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "data", "normalized", "korean"),
        help="Output directory for normalized JSONs",
    )
    parser.add_argument(
        "--features-csv",
        default=os.path.join(PROJECT_ROOT, "data", "processed", "korean_features.csv"),
        help="Output CSV for test features",
    )
    parser.add_argument(
        "--report-json",
        default=os.path.join(PROJECT_ROOT, "data", "normalized", "korean", "conversion_report.json"),
        help="Output JSON for conversion report",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Korean Strabismus Dataset Conversion & Analysis Pipeline")
    print("=" * 60)
    print(f"Normal Dir    : {args.normal_dir}")
    print(f"Strabismus Dir: {args.strabismus_dir}")
    print(f"Normalized Out: {args.output_dir}")
    print(f"Features CSV  : {args.features_csv}")
    print(f"Report JSON   : {args.report_json}")
    print("-" * 60)

    # 1. Convert test folders
    report = convert_korean_test_folders(
        normal_dir=args.normal_dir,
        strabismus_dir=args.strabismus_dir,
        normalized_out_dir=args.output_dir,
        features_out_csv=args.features_csv,
        report_out_json=args.report_json,
    )

    print("\nCONVERSION REPORT SUMMARY:")
    print("-" * 60)
    print(f"Total Files       : {report['totalFiles']}")
    print(f"Successful Files  : {report['successfulFiles']}")
    print(f"Failed Files      : {report['failedFiles']}")
    print(f"NORMAL Files      : {report['normalFiles']}")
    print(f"STRABISMUS Files  : {report['strabismusFiles']}")
    print(f"Total Samples     : {report['totalSamples']}")
    print(f"Invalid Rows      : {report['invalidRows']}")
    print(f"Missing Columns   : {report['missingColumns']}")
    print(f"Errors Logged     : {len(report['errors'])}")
    print("=" * 60)

    # 2. Check for training data
    repo_data_dir = os.path.join(PROJECT_ROOT, "data", "korean_repo", "data")
    df_train = check_and_extract_training_data(repo_data_dir)

    if df_train is not None and os.path.exists(args.features_csv):
        df_test = pd.read_csv(args.features_csv)
        results = train_and_evaluate_models(df_train, df_test)

        # Save evaluation report
        eval_report_path = os.path.join(PROJECT_ROOT, "data", "processed", "korean_eval_report.json")
        with open(eval_report_path, "w", encoding="utf-8") as fp:
            json.dump(results, fp, indent=2)
        print(f"\nSaved evaluation metrics to: {eval_report_path}")
    else:
        print("\n" + "=" * 60)
        print("MODEL TRAINING STATUS:")
        print("Only test data available; training cannot be performed without data leakage.")
        print("=" * 60)


if __name__ == "__main__":
    main()
