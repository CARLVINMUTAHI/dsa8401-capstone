"""
05_evaluate.py
==============
Full model evaluation report with publication-quality figures.

Reads    : models/best_model_*.pkl
           data/features_test.csv
           models/shap_values_*.csv
Outputs  : reports/figures/eval_*.png
           reports/model_evaluation.txt

Usage
-----
    python 05_evaluate.py
"""

import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve,
    roc_auc_score, roc_curve,
)

warnings.filterwarnings("ignore")

MODEL_DIR   = "models"
DATA_DIR    = "data"
REPORT_DIR  = "reports"
FIGURES_DIR = os.path.join(REPORT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

OUTCOMES = ["stunted", "wasted", "underweight"]
OUTCOME_LABELS = {
    "stunted": "Stunting",
    "wasted": "Wasting",
    "underweight": "Underweight",
}
COLORS = {"stunted": "#d32f2f", "wasted": "#f57c00", "underweight": "#1565c0"}

FEATURE_COLS = [
    "child_age_months", "child_sex", "birth_order", "prev_birth_interval",
    "size_at_birth", "birth_weight_kg", "diarrhea_2wk", "fever_2wk",
    "breastfeeding", "maternal_edu", "maternal_bmi",
    "wealth_index", "under5_in_hh", "hh_head_sex", "hh_head_age",
    "electricity", "residence_urban", "wash_score", "county",
]


def load_model(outcome: str) -> dict:
    path = os.path.join(MODEL_DIR, f"best_model_{outcome}.pkl")
    return joblib.load(path)


def load_test() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "features_test.csv"))


# ── FIGURE 1: ROC curves (all 3 outcomes on one plot) ────────────────────────

def fig_roc_curves(test_df: pd.DataFrame) -> dict:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    aucs = {}
    for outcome in OUTCOMES:
        try:
            obj    = load_model(outcome)
            pipe   = obj["pipeline"]
            X      = test_df[FEATURE_COLS].values
            y      = test_df[outcome].values
            y_prob = pipe.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_prob)
            roc_auc = auc(fpr, tpr)
            aucs[outcome] = roc_auc
            ax.plot(fpr, tpr, lw=2, color=COLORS[outcome],
                    label=f"{OUTCOME_LABELS[outcome]} (AUC = {roc_auc:.3f})")
        except FileNotFoundError:
            print(f"  SKIP {outcome}: model not found")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Best Models per Outcome", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, "eval_01_roc_curves.png")
    return aucs


# ── FIGURE 2: Precision-Recall curves ────────────────────────────────────────

def fig_pr_curves(test_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for outcome in OUTCOMES:
        try:
            obj    = load_model(outcome)
            pipe   = obj["pipeline"]
            X      = test_df[FEATURE_COLS].values
            y      = test_df[outcome].values
            y_prob = pipe.predict_proba(X)[:, 1]
            prec, rec, _ = precision_recall_curve(y, y_prob)
            ap = average_precision_score(y, y_prob)
            ax.plot(rec, prec, lw=2, color=COLORS[outcome],
                    label=f"{OUTCOME_LABELS[outcome]} (AP = {ap:.3f})")
        except FileNotFoundError:
            pass
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, "eval_02_pr_curves.png")


# ── FIGURE 3: Confusion matrices ─────────────────────────────────────────────

def fig_confusion_matrices(test_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, outcome in zip(axes, OUTCOMES):
        try:
            obj    = load_model(outcome)
            pipe   = obj["pipeline"]
            X      = test_df[FEATURE_COLS].values
            y      = test_df[outcome].values
            y_pred = pipe.predict(X)
            cm     = confusion_matrix(y, y_pred)
            cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
            im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)",
                            ha="center", va="center", fontsize=11,
                            color="white" if cm_pct[i, j] > 50 else "black")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred: Normal", "Pred: Malnourished"], fontsize=9)
            ax.set_yticklabels(["True: Normal", "True: Malnourished"], fontsize=9)
            ax.set_title(OUTCOME_LABELS[outcome], fontsize=11, fontweight="bold")
        except FileNotFoundError:
            ax.set_title(f"{outcome} — model not found")
    fig.suptitle("Confusion Matrices (Test Set)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "eval_03_confusion_matrices.png")


# ── FIGURE 4: Calibration curves ─────────────────────────────────────────────

def fig_calibration(test_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for outcome in OUTCOMES:
        try:
            obj    = load_model(outcome)
            pipe   = obj["pipeline"]
            X      = test_df[FEATURE_COLS].values
            y      = test_df[outcome].values
            y_prob = pipe.predict_proba(X)[:, 1]
            prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10)
            ax.plot(prob_pred, prob_true, marker="o", lw=2,
                    color=COLORS[outcome], label=OUTCOME_LABELS[outcome])
        except FileNotFoundError:
            pass
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curves", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, "eval_04_calibration.png")


# ── FIGURE 5: SHAP global bar plot ───────────────────────────────────────────

def fig_shap_global_importance() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, outcome in zip(axes, OUTCOMES):
        path = os.path.join(MODEL_DIR, f"shap_values_{outcome}.csv")
        if not os.path.exists(path):
            ax.set_title(f"{outcome} — SHAP not found")
            continue
        shap_df  = pd.read_csv(path)
        feat_cols = [c for c in shap_df.columns if c != "county"]
        importance = shap_df[feat_cols].abs().mean().sort_values(ascending=True)

        bars = ax.barh(importance.index, importance.values,
                       color=COLORS[outcome], alpha=0.85)
        ax.set_xlabel("Mean |SHAP| value", fontsize=10)
        ax.set_title(f"{OUTCOME_LABELS[outcome]}\nGlobal Feature Importance",
                     fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("SHAP Global Feature Importance (Test Set)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "eval_05_shap_global_importance.png")


# ── FIGURE 6: Metric comparison bar chart ────────────────────────────────────

def fig_metric_comparison(test_df: pd.DataFrame) -> None:
    results_path = os.path.join(MODEL_DIR, "results_summary.csv")
    if not os.path.exists(results_path):
        print("  SKIP metric comparison: results_summary.csv not found")
        return

    df = pd.read_csv(results_path)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, outcome in zip(axes, OUTCOMES):
        sub = df[df["outcome"] == outcome].copy()
        x   = np.arange(len(sub))
        w   = 0.25
        ax.bar(x - w, sub["test_auc"],    width=w, label="AUC-ROC", color="#1565c0", alpha=0.85)
        ax.bar(x,     sub["test_recall"], width=w, label="Recall",  color="#d32f2f", alpha=0.85)
        ax.bar(x + w, sub["test_f1"],     width=w, label="F1",      color="#2e7d32", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["model"], rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_title(OUTCOME_LABELS[outcome], fontsize=11, fontweight="bold")
        ax.axhline(0.7, color="gray", linestyle="--", lw=1, alpha=0.6)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Model Comparison — AUC-ROC, Recall, F1 (Test Set)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "eval_06_model_comparison.png")


# ── TEXT REPORT ──────────────────────────────────────────────────────────────

def write_text_report(test_df: pd.DataFrame) -> None:
    lines = [
        "=" * 65,
        "MODEL EVALUATION REPORT — Kenya Child Malnutrition Risk",
        "=" * 65,
    ]
    for outcome in OUTCOMES:
        try:
            obj    = load_model(outcome)
            pipe   = obj["pipeline"]
            name   = obj.get("model_name", "Unknown")
            X      = test_df[FEATURE_COLS].values
            y      = test_df[outcome].values
            y_pred = pipe.predict(X)
            y_prob = pipe.predict_proba(X)[:, 1]

            lines += [
                f"\n── {OUTCOME_LABELS[outcome].upper()} (best: {name}) ──",
                f"  AUC-ROC   : {roc_auc_score(y, y_prob):.4f}",
                f"  Recall    : {f1_score(y, y_pred, pos_label=1, average='binary'):.4f}",
                f"  F1-score  : {f1_score(y, y_pred):.4f}",
                f"  Avg Prec  : {average_precision_score(y, y_prob):.4f}",
                "",
                "  Classification Report:",
            ]
            report = classification_report(y, y_pred,
                                           target_names=["Normal", "Malnourished"])
            for line in report.strip().split("\n"):
                lines.append("    " + line)
        except FileNotFoundError:
            lines.append(f"\n── {outcome} — model not found ──")

    lines.append("\n" + "=" * 65)
    out_path = os.path.join(REPORT_DIR, "model_evaluation.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n✓ Report saved → {out_path}")


# ── HELPER ────────────────────────────────────────────────────────────────────

def _save(fig, name: str) -> None:
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading test data …")
    test_df = load_test()
    print(f"  {len(test_df):,} test observations\n")

    print("Generating evaluation figures …")
    fig_roc_curves(test_df)
    fig_pr_curves(test_df)
    fig_confusion_matrices(test_df)
    fig_calibration(test_df)
    fig_shap_global_importance()
    fig_metric_comparison(test_df)
    write_text_report(test_df)

    print("\n✓ Evaluation complete. All figures in reports/figures/")


if __name__ == "__main__":
    main()
