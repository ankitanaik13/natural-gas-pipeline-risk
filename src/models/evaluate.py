"""
Generate all model evaluation plots and SHAP explainability outputs.
Run: python -m src.models.evaluate
Outputs saved to outputs/
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

PALETTE = {"rf": "#2196F3", "xgb": "#FF5722"}
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 150})


def load_artifacts():
    features_path = PROCESSED_DIR / "pipeline_features.csv"
    df = pd.read_csv(features_path, low_memory=False)

    y = df["high_severity"].astype(int)
    X = df.drop(columns=["high_severity"])
    obj_cols = X.select_dtypes(include=["object", "datetime64"]).columns.tolist()
    X = X.drop(columns=obj_cols)
    X = X.fillna(X.median(numeric_only=True))

    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    rf = joblib.load(MODELS_DIR / "random_forest_pipeline_risk.pkl")
    xgb = joblib.load(MODELS_DIR / "xgboost_pipeline_risk.pkl")
    return rf, xgb, X_test, y_test, X.columns.tolist()


# ── 1. ROC-AUC Curve (both models) ────────────────────────────────────────────

def plot_roc_curves(rf, xgb, X_test, y_test, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for name, model, color in [("Random Forest", rf, PALETTE["rf"]),
                                ("XGBoost", xgb, PALETTE["xgb"])]:
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-AUC Curve — Pipeline Incident Severity")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    path = out_dir / "roc_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


# ── 2. Confusion Matrix ────────────────────────────────────────────────────────

def plot_confusion_matrix(model, X_test, y_test, model_name: str, out_dir: Path) -> None:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Low Severity", "High Severity"],
        yticklabels=["Low Severity", "High Severity"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")

    path = out_dir / "confusion_matrix.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


# ── 3. SHAP Summary Plot ───────────────────────────────────────────────────────

def plot_shap_summary(model, X_test: pd.DataFrame, out_dir: Path) -> None:
    print("Computing SHAP values (this may take a minute)...")
    explainer = shap.TreeExplainer(model)
    # Use a sample for speed
    sample = X_test.sample(min(500, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample)

    # For binary classifiers, shap_values is a list [class0, class1]
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(sv, sample, max_display=15, show=False)
    plt.title("SHAP Feature Importance — Top 15 Features")
    plt.tight_layout()

    path = out_dir / "shap_summary.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


# ── 4. SHAP Waterfall Plot ─────────────────────────────────────────────────────

def plot_shap_waterfall(model, X_test: pd.DataFrame, out_dir: Path) -> None:
    explainer = shap.TreeExplainer(model)
    sample = X_test.iloc[[0]]
    shap_values = explainer(sample)

    fig, ax = plt.subplots(figsize=(9, 5))
    # shap_values shape for binary RF is (n_samples, n_features, n_classes)
    # Use class-1 (high severity) slice: shap_values[:, :, 1]
    sv = shap_values[:, :, 1] if shap_values.values.ndim == 3 else shap_values
    shap.plots.waterfall(sv[0], max_display=12, show=False)
    plt.title("SHAP Waterfall — Single Prediction Explanation")
    plt.tight_layout()

    path = out_dir / "shap_waterfall.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


# ── 5. Precision-Recall Curve ──────────────────────────────────────────────────

def plot_precision_recall(rf, xgb, X_test, y_test, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for name, model, color in [("Random Forest", rf, PALETTE["rf"]),
                                ("XGBoost", xgb, PALETTE["xgb"])]:
        y_proba = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax.plot(rec, prec, color=color, lw=2,
                label=f"{name} (AP = {ap:.3f})")

    baseline = y_test.mean()
    ax.axhline(baseline, color="gray", linestyle="--", label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Pipeline Incident Severity")
    ax.legend()
    ax.grid(alpha=0.3)

    path = out_dir / "precision_recall_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


# ── 6. Feature Importance Bar Chart ───────────────────────────────────────────

def plot_feature_importance(rf, feature_names: list, out_dir: Path) -> None:
    importances = pd.Series(rf.feature_importances_, index=feature_names)
    top20 = importances.nlargest(20).sort_values()

    fig, ax = plt.subplots(figsize=(8, 7))
    top20.plot(kind="barh", ax=ax, color="#2196F3", edgecolor="white")
    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)")
    ax.set_title("Top 20 Features — Random Forest")
    ax.grid(axis="x", alpha=0.3)

    path = out_dir / "feature_importance.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    rf, xgb, X_test, y_test, feature_names = load_artifacts()

    # Determine best model by AUC
    rf_auc = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
    xgb_auc = roc_curve(y_test, xgb.predict_proba(X_test)[:, 1])
    best_model = rf  # default; can swap to xgb if needed

    print("Generating evaluation plots...")
    plot_roc_curves(rf, xgb, X_test, y_test, OUTPUTS_DIR)
    plot_confusion_matrix(best_model, X_test, y_test, "Random Forest", OUTPUTS_DIR)
    plot_precision_recall(rf, xgb, X_test, y_test, OUTPUTS_DIR)
    plot_feature_importance(rf, feature_names, OUTPUTS_DIR)

    print("\nGenerating SHAP plots...")
    plot_shap_summary(best_model, X_test, OUTPUTS_DIR)
    plot_shap_waterfall(best_model, X_test, OUTPUTS_DIR)

    print(f"\n✓ All evaluation plots saved to {OUTPUTS_DIR}/")


if __name__ == "__main__":
    main()
