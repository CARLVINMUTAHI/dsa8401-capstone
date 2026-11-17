"""
pipeline_helpers.py
===================
Importable wrappers around the core pipeline logic, used by unit tests.
All logic is identical to 01_pipeline.py; this module exposes it
without triggering argparse or file I/O when imported.
"""

import numpy as np
import pandas as pd

# ── WHO anthropometric bounds ─────────────────────────────────────────────────
HAZ_MIN, HAZ_MAX = -600, 600
WHZ_MIN, WHZ_MAX = -500, 500
WAZ_MIN, WAZ_MAX = -600, 500


def filter_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only children under 5 with valid anthropometric measurements.
    WHO protocol: flag and exclude implausible z-scores.
    """
    df = df[df["hw13"] == 0].copy()
    df = df[
        df["hw70"].between(HAZ_MIN, HAZ_MAX) &
        df["hw71"].between(WAZ_MIN, WAZ_MAX) &
        df["hw72"].between(WHZ_MIN, WHZ_MAX)
    ].copy()
    df = df[df["b19"] < 60].copy()
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create outcome variables and engineered features.
    Scaling is deferred to 02_train.py (inside the pipeline).
    """
    df = df.copy()

    # ── OUTCOMES ─────────────────────────────────────────────────────────────
    df["stunted"]     = (df["hw70"] < -200).astype(int)
    df["wasted"]      = (df["hw72"] < -200).astype(int)
    df["underweight"] = (df["hw71"] < -200).astype(int)

    # ── CHILD ─────────────────────────────────────────────────────────────────
    df["child_age_months"]     = df["b19"].fillna(df["b19"].median())
    df["child_sex"]            = (df["b4"] == 2).astype(int)
    df["birth_order"]          = df["bord"].clip(upper=6)
    df["prev_birth_interval"]  = df["b11"].fillna(df["b11"].median())
    df["size_at_birth"]        = df["m18"].fillna(df["m18"].mode()[0])
    df["birth_weight_kg"]      = df["m19"].replace(9.998, np.nan)
    df["birth_weight_kg"]      = df["birth_weight_kg"].fillna(df["birth_weight_kg"].median())
    df["diarrhea_2wk"]         = (df["h11"] == 1).astype(int)
    df["fever_2wk"]            = (df["h22"] == 1).astype(int)
    df["breastfeeding"]        = (df["v404"] == 1).astype(int)

    # ── MATERNAL ─────────────────────────────────────────────────────────────
    df["maternal_edu"]  = df["v106"].fillna(0)
    df["maternal_bmi"]  = (df["v437"] / 10) / ((df["v438"] / 1000) ** 2)
    df["maternal_bmi"]  = df["maternal_bmi"].replace([np.inf, -np.inf], np.nan)
    df["maternal_bmi"]  = df["maternal_bmi"].clip(10, 50)
    df["maternal_bmi"]  = df["maternal_bmi"].fillna(df["maternal_bmi"].median())

    # ── HOUSEHOLD ─────────────────────────────────────────────────────────────
    df["wealth_index"]    = df["v190"].fillna(df["v190"].mode()[0])
    df["under5_in_hh"]    = df["v137"].clip(upper=5)
    df["hh_head_sex"]     = (df["v151"] == 2).astype(int)
    df["hh_head_age"]     = df["v152"].fillna(df["v152"].median())
    df["electricity"]     = (df["v119"] == 1).astype(int)
    df["residence_urban"] = (df["v025"] == 1).astype(int)

    # WASH composite score (0-3)
    improved_water  = df["v113"].isin([10,11,12,13,14,21,31,41,51,61,71]).astype(int)
    improved_toilet = df["v116"].isin([10,11,12,13,14,15,21,22,41]).astype(int)
    df["wash_score"] = improved_water + improved_toilet + df["electricity"]

    # ── GEOGRAPHY & WEIGHT ────────────────────────────────────────────────────
    df["county"]        = df["v024"].astype(int)
    df["survey_weight"] = df["v005"] / 1_000_000

    return df


def select_features(df: pd.DataFrame) -> tuple:
    """Return (feature_col_list, model_dataframe) with no nulls in features."""
    FEATURE_COLS = [
        "child_age_months", "child_sex", "birth_order", "prev_birth_interval",
        "size_at_birth", "birth_weight_kg", "diarrhea_2wk", "fever_2wk",
        "breastfeeding",
        "maternal_edu", "maternal_bmi",
        "wealth_index", "under5_in_hh", "hh_head_sex", "hh_head_age",
        "electricity", "residence_urban", "wash_score",
        "county",
    ]
    OUTCOME_COLS = ["stunted", "wasted", "underweight"]
    META_COLS    = ["v001", "v002", "survey_weight", "county"]

    keep = list(dict.fromkeys(FEATURE_COLS + OUTCOME_COLS + META_COLS))
    # only keep columns that actually exist in df
    keep = [c for c in keep if c in df.columns]
    feat_in_df = [c for c in FEATURE_COLS if c in df.columns]

    df_model = df[keep].dropna(subset=feat_in_df).copy()
    return feat_in_df, df_model
