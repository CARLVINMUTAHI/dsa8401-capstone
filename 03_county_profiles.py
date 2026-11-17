"""
03_county_profiles.py
=====================
Generates county-level malnutrition risk profiles from SHAP values.

This is the methodological novelty of this study: aggregating individual
SHAP values within each of Kenya's 47 counties to produce county-specific
feature importance rankings that answer different policy questions per county.

Reads    : models/shap_values_*.csv
           data/processed_full.csv
Outputs  : reports/county_risk_profiles.csv      (prevalence + mean risk score per county)
           reports/county_shap_rankings.csv       (top 5 risk drivers per county per outcome)
           reports/county_summary.csv             (wide table: all metrics per county)
           reports/figures/county_heatmap_*.png   (SHAP heatmap per outcome)

Usage
-----
    python 03_county_profiles.py
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

MODEL_DIR   = "models"
DATA_DIR    = "data"
REPORT_DIR  = "reports"
FIGURES_DIR = os.path.join(REPORT_DIR, "figures")
OUTCOMES    = ["stunted", "wasted", "underweight"]

# DHS county codes → county names (2022 KDHS v024 mapping)
COUNTY_NAMES = {
    1: "Mombasa", 2: "Kwale", 3: "Kilifi", 4: "Tana River", 5: "Lamu",
    6: "Taita/Taveta", 7: "Garissa", 8: "Wajir", 9: "Mandera",
    10: "Marsabit", 11: "Isiolo", 12: "Meru", 13: "Tharaka-Nithi",
    14: "Embu", 15: "Kitui", 16: "Machakos", 17: "Makueni",
    18: "Nyandarua", 19: "Nyeri", 20: "Kirinyaga", 21: "Murang'a",
    22: "Kiambu", 23: "Turkana", 24: "West Pokot", 25: "Samburu",
    26: "Trans Nzoia", 27: "Uasin Gishu", 28: "Elgeyo/Marakwet",
    29: "Nandi", 30: "Baringo", 31: "Laikipia", 32: "Nakuru",
    33: "Narok", 34: "Kajiado", 35: "Kericho", 36: "Bomet",
    37: "Kakamega", 38: "Vihiga", 39: "Bungoma", 40: "Busia",
    41: "Siaya", 42: "Kisumu", 43: "Homa Bay", 44: "Migori",
    45: "Kisii", 46: "Nyamira", 47: "Nairobi",
}

FEATURE_COLS = [
    "child_age_months", "child_sex", "birth_order", "prev_birth_interval",
    "size_at_birth", "birth_weight_kg", "diarrhea_2wk", "fever_2wk",
    "breastfeeding", "maternal_edu", "maternal_bmi",
    "wealth_index", "under5_in_hh", "hh_head_sex", "hh_head_age",
    "electricity", "residence_urban", "wash_score", "county",
]

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORT_DIR,  exist_ok=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_full_data() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "processed_full.csv")
    df = pd.read_csv(path)
    df["county_name"] = df["county"].map(COUNTY_NAMES)
    return df


def load_shap(outcome: str) -> pd.DataFrame:
    path = os.path.join(MODEL_DIR, f"shap_values_{outcome}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"SHAP file not found: {path}\n"
            "Run 02_train.py first to generate SHAP values."
        )
    return pd.read_csv(path)


# ── COUNTY PREVALENCE TABLE ───────────────────────────────────────────────────

def build_prevalence_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute observed malnutrition prevalence per county."""
    print("[1/4] Building county prevalence table …")
    agg = (
        df.groupby(["county", "county_name"])
        .agg(
            n_children=("stunted", "count"),
            stunting_pct=("stunted",     lambda x: round(x.mean() * 100, 1)),
            wasting_pct=("wasted",       lambda x: round(x.mean() * 100, 1)),
            underweight_pct=("underweight", lambda x: round(x.mean() * 100, 1)),
            mean_wealth=("wealth_index", "mean"),
            pct_urban=("residence_urban", lambda x: round(x.mean() * 100, 1)),
            pct_electricity=("electricity", lambda x: round(x.mean() * 100, 1)),
            mean_wash=("wash_score", "mean"),
            pct_maternal_edu_none=("maternal_edu", lambda x: round((x == 0).mean() * 100, 1)),
        )
        .reset_index()
        .sort_values("stunting_pct", ascending=False)
    )
    agg["mean_wealth"]  = agg["mean_wealth"].round(2)
    agg["mean_wash"]    = agg["mean_wash"].round(2)
    return agg


# ── COUNTY SHAP PROFILES ──────────────────────────────────────────────────────

def build_shap_profiles(outcome: str, shap_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each county, compute mean absolute SHAP per feature and rank them.
    Returns a long-format table: county × feature × mean_abs_shap × rank.
    """
    feature_cols = [c for c in shap_df.columns if c != "county"]
    rows = []
    for county_code, grp in shap_df.groupby("county"):
        mean_abs = grp[feature_cols].abs().mean()
        ranked   = mean_abs.sort_values(ascending=False).reset_index()
        ranked.columns = ["feature", "mean_abs_shap"]
        ranked["rank"]        = range(1, len(ranked) + 1)
        ranked["county"]      = county_code
        ranked["county_name"] = COUNTY_NAMES.get(county_code, str(county_code))
        ranked["outcome"]     = outcome
        rows.append(ranked)
    return pd.concat(rows, ignore_index=True)


def top_drivers_per_county(shap_profiles: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Return top N risk drivers per county per outcome (wide format)."""
    top = (
        shap_profiles[shap_profiles["rank"] <= top_n]
        .pivot_table(
            index=["county", "county_name", "outcome"],
            columns="rank",
            values="feature",
            aggfunc="first",
        )
        .reset_index()
    )
    top.columns = (
        ["county", "county_name", "outcome"] +
        [f"top_{i}_driver" for i in range(1, top_n + 1)]
    )
    return top


# ── SHAP HEATMAP ─────────────────────────────────────────────────────────────

def plot_shap_heatmap(outcome: str, shap_profiles: pd.DataFrame) -> None:
    """
    Heatmap: counties (rows) × features (cols) = mean |SHAP|.
    Rows sorted by overall malnutrition risk (sum of mean |SHAP|).
    """
    feature_cols = [c for c in shap_profiles["feature"].unique() if c != "county"]

    pivot = (
        shap_profiles[shap_profiles["outcome"] == outcome]
        .pivot_table(index="county_name", columns="feature", values="mean_abs_shap")
    )
    pivot["_total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("_total", ascending=False).drop(columns="_total")

    fig, ax = plt.subplots(figsize=(16, max(8, len(pivot) * 0.35)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    plt.colorbar(im, ax=ax, label="Mean |SHAP| value")
    ax.set_title(
        f"County-Level SHAP Feature Importance — {outcome.capitalize()}\n"
        f"(counties sorted by total predicted risk, highest first)",
        fontsize=12, fontweight="bold", pad=15,
    )
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, f"county_shap_heatmap_{outcome}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved → {out}")


# ── TOP-10 HIGH-RISK COUNTIES BAR CHART ──────────────────────────────────────

def plot_top_counties(prevalence: pd.DataFrame) -> None:
    """Horizontal bar chart: top 10 counties by stunting prevalence."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    outcomes_map = {
        "stunting_pct": "Stunting",
        "wasting_pct":  "Wasting",
        "underweight_pct": "Underweight",
    }
    colors = ["#d32f2f", "#f57c00", "#1565c0"]
    for ax, (col, label), color in zip(axes, outcomes_map.items(), colors):
        top10 = prevalence.nlargest(10, col)[["county_name", col]].copy()
        bars = ax.barh(top10["county_name"][::-1], top10[col][::-1], color=color, alpha=0.85)
        ax.set_xlabel("Prevalence (%)", fontsize=10)
        ax.set_title(f"Top 10 Counties\n{label} Prevalence", fontsize=11, fontweight="bold")
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
        ax.set_xlim(0, top10[col].max() * 1.25)
        ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle("2022 KDHS — Highest Malnutrition Burden Counties", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "top10_counties_prevalence.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved → {out}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    df = load_full_data()

    # 1. Prevalence table
    prevalence = build_prevalence_table(df)
    prevalence.to_csv(os.path.join(REPORT_DIR, "county_risk_profiles.csv"), index=False)
    print(f"      Saved → {REPORT_DIR}/county_risk_profiles.csv")

    print("\nTop 10 counties by stunting:")
    print(prevalence[["county_name", "stunting_pct", "wasting_pct",
                       "underweight_pct", "n_children"]].head(10).to_string(index=False))

    # 2. SHAP profiles per outcome
    print("\n[2/4] Building SHAP county profiles …")
    all_shap_profiles = []
    all_top_drivers   = []

    for outcome in OUTCOMES:
        try:
            shap_df = load_shap(outcome)
        except FileNotFoundError as e:
            print(f"  SKIP {outcome}: {e}")
            continue

        profiles = build_shap_profiles(outcome, shap_df)
        all_shap_profiles.append(profiles)

        top_drivers = top_drivers_per_county(profiles, top_n=5)
        all_top_drivers.append(top_drivers)

        print(f"  {outcome}: {len(shap_df)} test observations, "
              f"{shap_df['county'].nunique()} counties covered")

    if all_shap_profiles:
        all_profiles_df = pd.concat(all_shap_profiles, ignore_index=True)
        all_profiles_df.to_csv(
            os.path.join(REPORT_DIR, "county_shap_rankings.csv"), index=False
        )
        print(f"      Saved → {REPORT_DIR}/county_shap_rankings.csv")

    if all_top_drivers:
        top_df = pd.concat(all_top_drivers, ignore_index=True)
        top_df.to_csv(
            os.path.join(REPORT_DIR, "county_top_drivers.csv"), index=False
        )
        print(f"      Saved → {REPORT_DIR}/county_top_drivers.csv")

    # 3. County summary (merge prevalence + top driver per outcome)
    print("\n[3/4] Building county summary table …")
    summary = prevalence.copy()
    for outcome in OUTCOMES:
        try:
            top_df_outcome = pd.concat(all_top_drivers)[
                pd.concat(all_top_drivers)["outcome"] == outcome
            ][["county", "top_1_driver", "top_2_driver", "top_3_driver"]].copy()
            top_df_outcome.columns = [
                "county",
                f"{outcome}_driver1",
                f"{outcome}_driver2",
                f"{outcome}_driver3",
            ]
            summary = summary.merge(top_df_outcome, on="county", how="left")
        except Exception:
            pass

    summary.to_csv(os.path.join(REPORT_DIR, "county_summary.csv"), index=False)
    print(f"      Saved → {REPORT_DIR}/county_summary.csv")

    # 4. Figures
    print("\n[4/4] Generating figures …")
    plot_top_counties(prevalence)

    for outcome in OUTCOMES:
        if all_shap_profiles:
            all_prof = pd.concat(all_shap_profiles)
            if outcome in all_prof["outcome"].values:
                plot_shap_heatmap(outcome, all_prof)

    print("\n✓ County profiling complete.")
    print(f"  Outputs in: {REPORT_DIR}/")


if __name__ == "__main__":
    main()
