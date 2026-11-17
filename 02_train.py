"""
02_train.py
===========
Model training script

Reads    : data/features_train.csv, data/features_test.csv
Outputs  : models/best_model_<outcome>.pkl   (serialised pipeline per outcome)
           models/results_summary.csv        (AUC-ROC, Recall, F1 per model)
           models/shap_values_<outcome>.csv  (SHAP values on test set)

Technique justification:
  - Logistic Regression  : Works as the statistical baseline and is directly comparable to prior
                           Kenya-specific regression studies.
  - Random Forest        : Is the ensemble baseline as it handles non-linearity. It is also robust to
                           missing values and high-cardinality categoricals.
  - XGBoost              : Sequential boosting, empirically strongest on tabular
                           DHS data in prior SSA studies.
  - SMOTE on training set only: addresses class imbalance without leaking
                           synthetic minority samples into test evaluation.
  - 5-fold stratified CV : preferred over simple hold-out given 8k training
                           rows. Stratification preserves malnutrition class ratio.
  - AUC-ROC primary      : appropriate for imbalanced outcomes. Recall co-primary: cost of missing a
                           malnourished child exceeds cost of a false alert.
  - SHAP TreeExplainer   : enables county-level feature importance — methodological novelty.

Usage
-----
    python 02_train.py [--outcome stunted|wasted|underweight]
    python 02_train.py          # trains all three outcomes sequentially
"""

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, recall_score, f1_score,
    precision_score, classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import shap
import pandas as pd
import numpy as np
import mlflow.sklearn
import mlflow
import joblib
import warnings
import json
import argparse
import os
# ── DAGSHUB credentials ───────────────
os.environ["MLFLOW_TRACKING_USERNAME"] = "CARLVINMUTAHI"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "ef2023d70c0fa36be9c0bef33ac3b941b81459ce"


warnings.filterwarnings("ignore")

# ── DAGSHUB / MLFLOW SETUP ────────────────────────────────────────────────────
mlflow.set_tracking_uri("https://dagshub.com/CARLVINMUTAHI/dsa8401-capstone.mlflow")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

RANDOM_SEED = 42
N_FOLDS = 5
DATA_DIR = "data"
MODEL_DIR = "models"

FEATURE_COLS = [
    "child_age_months", "child_sex", "birth_order", "prev_birth_interval",
    "size_at_birth", "birth_weight_kg", "diarrhea_2wk", "fever_2wk",
    "breastfeeding", "maternal_edu", "maternal_bmi",
    "wealth_index", "under5_in_hh", "hh_head_sex", "hh_head_age",
    "electricity", "residence_urban", "wash_score", "county",
]

OUTCOMES = ["stunted", "wasted", "underweight"]

# ── MODEL DEFINITIONS ─────────────────────────────────────────────────────────


def build_models() -> dict:
    """Return dict of name → sklearn-compatible estimator."""
    return {
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_SEED, n_jobs=-1,
        ),
    }

# ── TRAINING ──────────────────────────────────────────────────────────────────


def train_outcome(outcome: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Train all classifiers for one outcome, evaluate, and save best model.
    Returns a dict of results for the summary table.
    """
    print(f"\n{'='*60}")
    print(f"  Outcome: {outcome.upper()}")
    print(f"{'='*60}")

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[outcome].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df[outcome].values

    print(f"  Train positives: {y_train.sum()} / {len(y_train)} "
          f"({y_train.mean():.1%})")
    print(f"  Test  positives: {y_test.sum()}  / {len(y_test)} "
          f"({y_test.mean():.1%})")

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    models = build_models()
    results = []
    best_auc = -1
    best_model = None
    best_name = ""

    for name, clf in models.items():
        print(f"\n  -- {name} --")

        # Build imbalanced-learn pipeline: scaler → SMOTE → classifier
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote",  SMOTE(random_state=RANDOM_SEED, k_neighbors=5)),
            ("clf",    clf),
        ])

        # 5-fold CV AUC-ROC
        cv_aucs = cross_val_score(
            pipe, X_train, y_train, cv=cv,
            scoring="roc_auc", n_jobs=-1,
        )
        print(f"     CV AUC-ROC : {cv_aucs.mean():.4f} +/- {cv_aucs.std():.4f}")

        # Fit on full training set, evaluate on held-out test set
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = pipe.predict(X_test)

        test_auc = roc_auc_score(y_test, y_prob)
        test_rec = recall_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred)

        print(f"     Test AUC   : {test_auc:.4f}")
        print(f"     Recall     : {test_rec:.4f}")
        print(f"     F1-score   : {test_f1:.4f}")

        # ── LOG TO MLFLOW / DAGSHUB ───────────────────────────────────────────
        with mlflow.start_run(run_name=f"{outcome}__{name}"):
            # Parameters
            mlflow.log_param("outcome",      outcome)
            mlflow.log_param("model",        name)
            mlflow.log_param("n_folds",      N_FOLDS)
            mlflow.log_param("smote_k",      5)
            mlflow.log_param("random_seed",  RANDOM_SEED)
            mlflow.log_param("train_size",   len(X_train))
            mlflow.log_param("test_size",    len(X_test))
            mlflow.log_param("pos_rate_train", round(float(y_train.mean()), 4))

            # CV metrics (all 5 folds + summary)
            for i, fold_auc in enumerate(cv_aucs):
                mlflow.log_metric(f"cv_fold_{i+1}_auc", round(float(fold_auc), 4))
            mlflow.log_metric("cv_auc_mean", round(float(cv_aucs.mean()), 4))
            mlflow.log_metric("cv_auc_std",  round(float(cv_aucs.std()),  4))

            # Test-set metrics
            mlflow.log_metric("test_auc",       round(float(test_auc),  4))
            mlflow.log_metric("test_recall",    round(float(test_rec),  4))
            mlflow.log_metric("test_f1",        round(float(test_f1),   4))
            mlflow.log_metric("test_precision", round(float(test_prec), 4))

            # Note: mlflow.sklearn.log_model skipped — imblearn pipeline is
            # serialised locally as models/best_model_<outcome>.pkl instead.
            # Params + metrics above are sufficient for experiment comparison.

        results.append({
            "outcome": outcome,
            "model":   name,
            "cv_auc_mean": round(cv_aucs.mean(), 4),
            "cv_auc_std":  round(cv_aucs.std(),  4),
            "test_auc":    round(test_auc,  4),
            "test_recall": round(test_rec,  4),
            "test_f1":     round(test_f1,   4),
            "test_precision": round(test_prec, 4),
        })

        if test_auc > best_auc:
            best_auc = test_auc
            best_model = pipe
            best_name = name

    # ── Save best model ───────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"best_model_{outcome}.pkl")
    joblib.dump({"pipeline": best_model, "features": FEATURE_COLS,
                 "outcome": outcome, "model_name": best_name}, model_path)
    print(f"\n  ✓ Best model ({best_name}, AUC={best_auc:.4f}) saved → {model_path}")

    # ── SHAP values for best model ────────────────────────────────────────────
    print(f"  Computing SHAP values …")
    # Extract the fitted classifier from pipeline for SHAP
    clf_fitted = best_model.named_steps["clf"]
    scaler = best_model.named_steps["scaler"]
    X_test_scaled = scaler.transform(X_test)

    if best_name == "XGBoost" or best_name == "RandomForest":
        explainer = shap.TreeExplainer(clf_fitted)
        shap_values = explainer.shap_values(X_test_scaled)
    else:
        explainer = shap.LinearExplainer(clf_fitted, X_test_scaled)
        shap_values = explainer.shap_values(X_test_scaled)

    # Normalise to 2-D array regardless of SHAP version
    import numpy as np
    if isinstance(shap_values, list):
        # Older SHAP: list of [neg_class, pos_class]
        shap_values = shap_values[1]
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        # Newer SHAP: shape (n_samples, n_features, n_classes) — take positive class
        shap_values = shap_values[:, :, 1]
    if shap_values.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape after normalisation: {shap_values.shape}")

    shap_df = pd.DataFrame(shap_values, columns=FEATURE_COLS)
    shap_df["county"] = test_df["county"].values
    shap_df.to_csv(
        os.path.join(MODEL_DIR, f"shap_values_{outcome}.csv"), index=False
    )
    print(f"  ✓ SHAP values saved → models/shap_values_{outcome}.csv")

    # Global feature importance (mean |SHAP|)
    importance = pd.Series(
        np.abs(shap_values).mean(axis=0), index=FEATURE_COLS
    ).sort_values(ascending=False)
    print(f"\n  Top 5 features ({outcome}):")
    for feat, val in importance.head(5).items():
        print(f"     {feat:30s} {val:.4f}")

    return results


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main(outcomes: list[str]) -> None:
    train_df = pd.read_csv(os.path.join(DATA_DIR, "features_train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "features_test.csv"))

    all_results = []
    for outcome in outcomes:
        res = train_outcome(outcome, train_df, test_df)
        all_results.extend(res)

    summary = pd.DataFrame(all_results)
    summary.to_csv(os.path.join(MODEL_DIR, "results_summary.csv"), index=False)
    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(summary.to_string(index=False))
    print(f"\n✓ Results saved → {MODEL_DIR}/results_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KDHS 2022 — model training")
    parser.add_argument(
        "--outcome",
        choices=OUTCOMES,
        default=None,
        help="Single outcome to train (default: all three)",
    )
    args = parser.parse_args()
    outcomes = [args.outcome] if args.outcome else OUTCOMES
    main(outcomes)
