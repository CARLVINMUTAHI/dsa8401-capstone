"""Quick SHAP shape test — run once to verify fix."""
import joblib
import shap
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "child_age_months", "child_sex", "birth_order", "prev_birth_interval",
    "size_at_birth", "birth_weight_kg", "diarrhea_2wk", "fever_2wk",
    "breastfeeding", "maternal_edu", "maternal_bmi", "wealth_index",
    "under5_in_hh", "hh_head_sex", "hh_head_age", "electricity",
    "residence_urban", "wash_score", "county",
]

obj = joblib.load("models/best_model_stunted.pkl")
df = pd.read_csv("data/features_test.csv")
X = df[FEATURE_COLS].values
clf = obj["pipeline"].named_steps["clf"]
scaler = obj["pipeline"].named_steps["scaler"]
Xs = scaler.transform(X)

sv = shap.TreeExplainer(clf).shap_values(Xs)
if isinstance(sv, list):
    sv = sv[1]
sv = np.array(sv)
if sv.ndim == 3:
    sv = sv[:, :, 1]

print(f"SHAP shape: {sv.shape}")
if sv.ndim == 2 and sv.shape[1] == 19:
    print("OK — fix works, safe to run 02_train.py")
else:
    print("UNEXPECTED shape — please share this output")
