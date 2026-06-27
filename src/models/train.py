"""
Train Random Forest and XGBoost classifiers for pipeline incident severity.
Run: python -m src.models.train
Outputs: models/random_forest_pipeline_risk.pkl
         models/xgboost_pipeline_risk.pkl
         data/processed/risk_scores.csv  (predictions on test set with lat/lon)
"""

import time
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

TARGET = "high_severity"


def load_features() -> tuple[pd.DataFrame, pd.Series]:
    path = PROCESSED_DIR / "pipeline_features.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Run src/features/build_features.py first to generate pipeline_features.csv"
        )
    df = pd.read_csv(path, low_memory=False)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in feature matrix.")

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    # Drop any remaining non-numeric columns
    obj_cols = X.select_dtypes(include=["object", "datetime64"]).columns.tolist()
    if obj_cols:
        print(f"Dropping non-numeric columns from features: {obj_cols}")
        X = X.drop(columns=obj_cols)

    # Fill any remaining NaNs with median
    X = X.fillna(X.median(numeric_only=True))

    print(f"Feature matrix: {X.shape}  |  Target distribution:")
    print(y.value_counts().to_string())
    return X, y


def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "name": name,
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "f1_weighted": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }

    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"{'─' * 50}")
    print(f"  ROC-AUC  : {metrics['roc_auc']}")
    print(f"  F1 (wtd) : {metrics['f1_weighted']}")
    print(f"  Precision: {metrics['precision']}")
    print(f"  Recall   : {metrics['recall']}")
    print(f"\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Low", "High"]))

    return metrics


def train() -> dict:
    X, y = load_features()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\nTrain: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    print(f"Class ratio neg/pos: {scale_pos_weight:.2f} → scale_pos_weight for XGBoost")

    # ── Random Forest ──────────────────────────────────────────────────────────
    print("\nTraining Random Forest...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_time = time.time() - t0
    print(f"  Trained in {rf_time:.1f}s")

    rf_metrics = evaluate_model("Random Forest", rf, X_test, y_test)
    rf_metrics["training_time_s"] = round(rf_time, 2)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("\nTraining XGBoost...")
    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
        use_label_encoder=False,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_time = time.time() - t0
    print(f"  Trained in {xgb_time:.1f}s")

    xgb_metrics = evaluate_model("XGBoost", xgb, X_test, y_test)
    xgb_metrics["training_time_s"] = round(xgb_time, 2)

    # ── Save models ───────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, MODELS_DIR / "random_forest_pipeline_risk.pkl")
    joblib.dump(xgb, MODELS_DIR / "xgboost_pipeline_risk.pkl")
    print(f"\nModels saved to {MODELS_DIR}/")

    # ── Save test predictions (for Folium map) ─────────────────────────────────
    best_model = rf if rf_metrics["roc_auc"] >= xgb_metrics["roc_auc"] else xgb
    best_proba = best_model.predict_proba(X_test)[:, 1]

    test_results = X_test.copy()
    test_results["high_severity_actual"] = y_test.values
    test_results["risk_score"] = best_proba
    test_results["risk_label"] = pd.cut(
        best_proba,
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"],
    )

    out_path = PROCESSED_DIR / "risk_scores.csv"
    test_results.to_csv(out_path, index=False)
    print(f"Risk scores saved → {out_path}")

    return {
        "rf": rf_metrics,
        "xgb": xgb_metrics,
        "feature_names": X.columns.tolist(),
        "rf_model": rf,
        "xgb_model": xgb,
        "X_test": X_test,
        "y_test": y_test,
    }


if __name__ == "__main__":
    results = train()
    print("\n✓ Training complete.")
