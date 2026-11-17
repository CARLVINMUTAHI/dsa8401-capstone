"""
01_pipeline.py
==============
End-to-end data pipeline for the DSA 8401 Capstone Project.

Problem : Predict child malnutrition risk (stunting, wasting, underweight)
          among children under five in Kenya using the 2022 KDHS.
Data    : 2022 KDHS Children's Recode (KEKR8CFL.DTA)
Output  : data/features_train.csv, data/features_test.csv
          data/processed_full.csv  (for EDA / county profiling)

Usage
-----
    python 01_pipeline.py --kr_path <path/to/KEKR8CFL.DTA>

The script is fully reproducible from scratch; no manual steps required.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pyreadstat
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── CONFIGURATION ────────────────────────────────────────────────────────────

RANDOM_SEED = 42
TEST_SIZE    = 0.20   # 80 / 20 stratified split (stratify on stunting)

# WHO anthropometric flag bounds (children outside these are excluded)
HAZ_MIN, HAZ_MAX = -600, 600   # stored as x100 in DHS
WHZ_MIN, WHZ_MAX = -500, 500
WAZ_MIN, WAZ_MAX = -600, 500

# Variables to load from KR file (keeps memory footprint small)
KR_VARS = [
    # identifiers / weights
    "v001", "v002", "v005",
    # geography
    "v024", "v025",
    # child
    "b4", "b19", "bord", "b11", "m18", "m19",
    # maternal
    "v106", "v149", "v437", "v438", "v404",
    # household
    "v113", "v116", "v119", "v137", "v190", "v191",
    "v151", "v152",
    # morbidity
    "h11", "h22",
    # anthropometry (z-scores x100)
    "hw70", "hw71", "hw72", "hw13",
]

OUTPUT_DIR = "data"

# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_kr(path: str) -> pd.DataFrame:
    """Load selected columns from KR Stata file."""
    print(f"[1/6] Loading KR file: {path}")
    df, meta = pyreadstat.read_dta(
        path,
        usecols=KR_VARS,
        apply_value_formats=False,   # keep numeric codes
    )
    print(f"      Raw shape: {df.shape}")
    return df, meta


def filter_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only children under 5 with valid anthropometric measurements.
    WHO protocol: flag and exclude implausible z-scores.
    """
    print("[2/6] Filtering sample …")

    # hw13 == 0 means measurement was taken (not missing/refused)
    df = df[df["hw13"] == 0].copy()

    # Exclude implausible z-scores (DHS stores as integer x100)
    df = df[
        df["hw70"].between(HAZ_MIN, HAZ_MAX) &
        df["hw71"].between(WAZ_MIN, WAZ_MAX) &
        df["hw72"].between(WHZ_MIN, WHZ_MAX)
    ].copy()

    # Keep children under 60 months (5 years)
    df = df[df["b19"] < 60].copy()

    print(f"      After filtering: {len(df)} children")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create outcome variables and engineered features.
    All feature engineering uses training-set statistics applied to both sets,
    so this function creates raw features only; scaling happens in 02_train.py.
    """
    print("[3/6] Engineering features …")

    # ── OUTCOMES (z-scores stored as integer x100) ──────────────────────────
    df["stunted"]     = (df["hw70"] < -200).astype(int)   # HAZ < -2 SD
    df["wasted"]      = (df["hw72"] < -200).astype(int)   # WHZ < -2 SD
    df["underweight"] = (df["hw71"] < -200).astype(int)   # WAZ < -2 SD

    # ── CHILD FEATURES ───────────────────────────────────────────────────────
    df["child_age_months"] = df["b19"].fillna(df["b19"].median())
    df["child_sex"]        = (df["b4"] == 2).astype(int)   # 1=female, 0=male
    df["birth_order"]      = df["bord"].clip(upper=6)       # cap at 6+
    df["prev_birth_interval"] = df["b11"].fillna(df["b11"].median())
    df["size_at_birth"]    = df["m18"].fillna(df["m18"].mode()[0])  # 1-5 scale
    df["birth_weight_kg"]  = df["m19"].replace(9.998, np.nan)       # 9.998 = missing code
    df["birth_weight_kg"]  = df["birth_weight_kg"].fillna(df["birth_weight_kg"].median())
    df["diarrhea_2wk"]     = (df["h11"] == 1).astype(int)
    df["fever_2wk"]        = (df["h22"] == 1).astype(int)
    df["breastfeeding"]    = (df["v404"] == 1).astype(int)

    # ── MATERNAL FEATURES ────────────────────────────────────────────────────
    df["maternal_edu"]     = df["v106"].fillna(0)           # 0=no edu, 1=primary, 2=secondary, 3=higher
    df["maternal_bmi"]     = (df["v437"] / 10) / ((df["v438"] / 1000) ** 2)
    df["maternal_bmi"]     = df["maternal_bmi"].replace([np.inf, -np.inf], np.nan)
    df["maternal_bmi"]     = df["maternal_bmi"].clip(10, 50)
    df["maternal_bmi"]     = df["maternal_bmi"].fillna(df["maternal_bmi"].median())

    # ── HOUSEHOLD FEATURES ───────────────────────────────────────────────────
    df["wealth_index"]     = df["v190"].fillna(df["v190"].mode()[0])   # 1-5 quintile
    df["under5_in_hh"]     = df["v137"].clip(upper=5)
    df["hh_head_sex"]      = (df["v151"] == 2).astype(int)             # 1=female head
    df["hh_head_age"]      = df["v152"].fillna(df["v152"].median())
    df["electricity"]      = (df["v119"] == 1).astype(int)
    df["residence_urban"]  = (df["v025"] == 1).astype(int)

    # WASH composite score (0-3): improved water + improved toilet + electricity proxy
    improved_water  = df["v113"].isin([10,11,12,13,14,21,31,41,51,61,71]).astype(int)
    improved_toilet = df["v116"].isin([10,11,12,13,14,15,21,22,41]).astype(int)
    df["wash_score"] = improved_water + improved_toilet + df["electricity"]

    # ── GEOGRAPHY (county identifier) ────────────────────────────────────────
    df["county"] = df["v024"].astype(int)

    # ── SAMPLE WEIGHT ────────────────────────────────────────────────────────
    df["survey_weight"] = df["v005"] / 1_000_000

    print(f"      Stunting prevalence : {df['stunted'].mean():.1%}")
    print(f"      Wasting prevalence  : {df['wasted'].mean():.1%}")
    print(f"      Underweight prev.   : {df['underweight'].mean():.1%}")
    return df


def select_features(df: pd.DataFrame) -> tuple[list, pd.DataFrame]:
    """Return the final feature list and cleaned modelling dataframe."""
    print("[4/6] Selecting final feature set …")

    FEATURE_COLS = [
        # child
        "child_age_months", "child_sex", "birth_order", "prev_birth_interval",
        "size_at_birth", "birth_weight_kg", "diarrhea_2wk", "fever_2wk",
        "breastfeeding",
        # maternal
        "maternal_edu", "maternal_bmi",
        # household
        "wealth_index", "under5_in_hh", "hh_head_sex", "hh_head_age",
        "electricity", "residence_urban", "wash_score",
        # geography
        "county",
    ]

    OUTCOME_COLS = ["stunted", "wasted", "underweight"]
    META_COLS    = ["v001", "v002", "survey_weight", "county"]

    keep = FEATURE_COLS + OUTCOME_COLS + META_COLS
    df_model = df[keep].dropna(subset=FEATURE_COLS).copy()
    print(f"      Final sample: {len(df_model)} children, {len(FEATURE_COLS)} features")
    return FEATURE_COLS, df_model


def split_and_save(df: pd.DataFrame, feature_cols: list) -> None:
    """Stratified train/test split (on stunting) and save CSVs."""
    print("[5/6] Splitting train / test …")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["stunted"],
    )

    train.to_csv(os.path.join(OUTPUT_DIR, "features_train.csv"), index=False)
    test.to_csv(os.path.join(OUTPUT_DIR, "features_test.csv"),  index=False)
    df.to_csv(os.path.join(OUTPUT_DIR, "processed_full.csv"),   index=False)

    print(f"      Train : {len(train)} rows  |  Test : {len(test)} rows")
    print(f"      Saved to {OUTPUT_DIR}/")


def main(kr_path: str) -> None:
    df, meta   = load_kr(kr_path)
    df         = filter_sample(df)
    df         = engineer_features(df)
    feat_cols, df_model = select_features(df)
    split_and_save(df_model, feat_cols)
    print("[6/6] Pipeline complete. ✓")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KDHS 2022 — data pipeline")
    parser.add_argument(
        "--kr_path",
        required=True,
        help="Path to KEKR8CFL.DTA (Children's Recode Stata file)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.kr_path):
        sys.exit(f"ERROR: file not found — {args.kr_path}")

    main(args.kr_path)
