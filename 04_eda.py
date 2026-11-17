"""
04_eda.py
=========
Exploratory Data Analysis for the DSA 8401 Capstone Project.

Reads    : data/processed_full.csv
Outputs  : reports/figures/eda_*.png
           reports/eda_summary.txt

Run this AFTER 01_pipeline.py and BEFORE modelling, to understand the data.

Usage
-----
    python 04_eda.py
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

DATA_DIR    = "data"
REPORT_DIR  = "reports"
FIGURES_DIR = os.path.join(REPORT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

OUTCOMES = ["stunted", "wasted", "underweight"]
OUTCOME_LABELS = {
    "stunted": "Stunting (HAZ < −2SD)",
    "wasted": "Wasting (WHZ < −2SD)",
    "underweight": "Underweight (WAZ < −2SD)",
}
COLORS = {"stunted": "#d32f2f", "wasted": "#f57c00", "underweight": "#1565c0"}

COUNTY_NAMES = {
    1:"Mombasa",2:"Kwale",3:"Kilifi",4:"Tana River",5:"Lamu",
    6:"Taita/Taveta",7:"Garissa",8:"Wajir",9:"Mandera",10:"Marsabit",
    11:"Isiolo",12:"Meru",13:"Tharaka-Nithi",14:"Embu",15:"Kitui",
    16:"Machakos",17:"Makueni",18:"Nyandarua",19:"Nyeri",20:"Kirinyaga",
    21:"Murang'a",22:"Kiambu",23:"Turkana",24:"West Pokot",25:"Samburu",
    26:"Trans Nzoia",27:"Uasin Gishu",28:"Elgeyo/Marakwet",29:"Nandi",
    30:"Baringo",31:"Laikipia",32:"Nakuru",33:"Narok",34:"Kajiado",
    35:"Kericho",36:"Bomet",37:"Kakamega",38:"Vihiga",39:"Bungoma",
    40:"Busia",41:"Siaya",42:"Kisumu",43:"Homa Bay",44:"Migori",
    45:"Kisii",46:"Nyamira",47:"Nairobi",
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "processed_full.csv"))
    df["county_name"] = df["county"].map(COUNTY_NAMES)
    return df


# ── FIGURE 1: National prevalence overview ───────────────────────────────────

def fig_prevalence_overview(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, outcome in zip(axes, OUTCOMES):
        counts = df[outcome].value_counts()
        pct    = df[outcome].mean() * 100
        wedges, texts, autotexts = ax.pie(
            [counts.get(1, 0), counts.get(0, 0)],
            labels=["Malnourished", "Normal"],
            colors=[COLORS[outcome], "#e0e0e0"],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        ax.set_title(
            f"{OUTCOME_LABELS[outcome]}\n{pct:.1f}% prevalence",
            fontsize=11, fontweight="bold"
        )
    fig.suptitle(
        "2022 KDHS — National Malnutrition Prevalence (Children Under 5)",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    _save(fig, "eda_01_national_prevalence.png")


# ── FIGURE 2: Class imbalance bar (imbalance ratio) ──────────────────────────

def fig_class_imbalance(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    irs = {}
    for outcome in OUTCOMES:
        majority = (df[outcome] == 0).sum()
        minority = (df[outcome] == 1).sum()
        irs[outcome] = round(majority / minority, 2)

    bars = ax.bar(
        [OUTCOME_LABELS[o] for o in OUTCOMES],
        list(irs.values()),
        color=[COLORS[o] for o in OUTCOMES],
        alpha=0.85, edgecolor="white"
    )
    ax.bar_label(bars, labels=[f"IR = {v}" for v in irs.values()], padding=5, fontsize=11)
    ax.set_ylabel("Imbalance Ratio (majority:minority)", fontsize=11)
    ax.set_title("Class Imbalance Ratio per Outcome\n(justifies SMOTE application)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(irs.values()) * 1.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, "eda_02_class_imbalance.png")


# ── FIGURE 3: Age distribution by outcome ────────────────────────────────────

def fig_age_by_outcome(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, outcome in zip(axes, OUTCOMES):
        for label, val, lc in [("Normal", 0, "#90a4ae"), ("Malnourished", 1, COLORS[outcome])]:
            subset = df[df[outcome] == val]["child_age_months"]
            ax.hist(subset, bins=24, alpha=0.6, color=lc, label=label, density=True)
        ax.set_xlabel("Child age (months)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(outcome.capitalize(), fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
    fig.suptitle("Age Distribution by Malnutrition Status", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "eda_03_age_by_outcome.png")


# ── FIGURE 4: Stunting by wealth quintile ────────────────────────────────────

def fig_wealth_gradient(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    wealth_labels = {1: "Poorest", 2: "Poor", 3: "Middle", 4: "Rich", 5: "Richest"}
    for ax, outcome in zip(axes, OUTCOMES):
        grp = df.groupby("wealth_index")[outcome].mean() * 100
        grp.index = [wealth_labels.get(i, i) for i in grp.index]
        bars = ax.bar(grp.index, grp.values, color=COLORS[outcome], alpha=0.85, edgecolor="white")
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
        ax.set_ylabel("Prevalence (%)", fontsize=10)
        ax.set_title(f"{outcome.capitalize()}\nby Wealth Quintile", fontsize=11, fontweight="bold")
        ax.set_ylim(0, grp.max() * 1.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Malnutrition Prevalence by DHS Wealth Index", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "eda_04_wealth_gradient.png")


# ── FIGURE 5: County-level stunting map (bar chart proxy) ────────────────────

def fig_county_stunting(df: pd.DataFrame) -> None:
    county_prev = (
        df.groupby("county_name")
        .agg(stunting_pct=("stunted", lambda x: x.mean() * 100),
             n=("stunted", "count"))
        .reset_index()
        .sort_values("stunting_pct", ascending=True)
    )
    nat_avg = df["stunted"].mean() * 100

    fig, ax = plt.subplots(figsize=(10, 16))
    colors = ["#d32f2f" if v > nat_avg else "#42a5f5" for v in county_prev["stunting_pct"]]
    bars = ax.barh(county_prev["county_name"], county_prev["stunting_pct"],
                   color=colors, alpha=0.85)
    ax.axvline(nat_avg, color="black", linestyle="--", linewidth=1.5,
               label=f"National avg {nat_avg:.1f}%")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=7.5)
    ax.set_xlabel("Stunting Prevalence (%)", fontsize=11)
    ax.set_title("Stunting Prevalence by County — 2022 KDHS\n"
                 "(red = above national average)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, "eda_05_county_stunting.png")


# ── FIGURE 6: Correlation matrix of features ─────────────────────────────────

def fig_correlation(df: pd.DataFrame) -> None:
    num_cols = [
        "child_age_months", "birth_order", "prev_birth_interval",
        "birth_weight_kg", "maternal_bmi", "maternal_edu",
        "wealth_index", "wash_score", "under5_in_hh",
        "stunted", "wasted", "underweight",
    ]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    _save(fig, "eda_06_correlation_matrix.png")


# ── FIGURE 7: Maternal education vs outcomes ─────────────────────────────────

def fig_maternal_edu(df: pd.DataFrame) -> None:
    edu_labels = {0: "None", 1: "Primary", 2: "Secondary", 3: "Higher"}
    fig, axes  = plt.subplots(1, 3, figsize=(14, 4))
    for ax, outcome in zip(axes, OUTCOMES):
        grp = df.groupby("maternal_edu")[outcome].mean() * 100
        grp.index = [edu_labels.get(i, i) for i in grp.index]
        bars = ax.bar(grp.index, grp.values, color=COLORS[outcome], alpha=0.85, edgecolor="white")
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
        ax.set_ylabel("Prevalence (%)", fontsize=10)
        ax.set_title(f"{outcome.capitalize()}\nvs Maternal Education", fontsize=11, fontweight="bold")
        ax.set_ylim(0, grp.max() * 1.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Malnutrition Prevalence by Maternal Education Level", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "eda_07_maternal_education.png")


# ── FIGURE 8: WASH score distribution ───────────────────────────────────────

def fig_wash(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for outcome in OUTCOMES:
        grp = df.groupby("wash_score")[outcome].mean() * 100
        ax.plot(grp.index, grp.values, marker="o", label=OUTCOME_LABELS[outcome],
                color=COLORS[outcome], linewidth=2, markersize=7)
    ax.set_xlabel("WASH Score (0 = worst, 3 = best)", fontsize=11)
    ax.set_ylabel("Malnutrition Prevalence (%)", fontsize=11)
    ax.set_title("WASH Score vs Malnutrition Prevalence", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xticks([0, 1, 2, 3])
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, "eda_08_wash_score.png")


# ── SUMMARY STATS ────────────────────────────────────────────────────────────

def write_summary(df: pd.DataFrame) -> None:
    lines = [
        "=" * 60,
        "EDA SUMMARY — 2022 KDHS Child Malnutrition",
        "=" * 60,
        f"Total children: {len(df):,}",
        f"Counties covered: {df['county'].nunique()}",
        "",
        "NATIONAL PREVALENCE",
        f"  Stunting    : {df['stunted'].mean()*100:.1f}%  (n={int(df['stunted'].sum()):,})",
        f"  Wasting     : {df['wasted'].mean()*100:.1f}%   (n={int(df['wasted'].sum()):,})",
        f"  Underweight : {df['underweight'].mean()*100:.1f}%  (n={int(df['underweight'].sum()):,})",
        "",
        "CLASS IMBALANCE RATIOS",
    ]
    for outcome in OUTCOMES:
        maj = (df[outcome] == 0).sum()
        mi  = (df[outcome] == 1).sum()
        lines.append(f"  {outcome:15s}: {maj/mi:.2f}:1 → SMOTE justified")

    lines += [
        "",
        "TOP 5 COUNTIES — STUNTING",
    ]
    top5 = (
        df.groupby("county_name")["stunted"].mean() * 100
    ).nlargest(5)
    for county, pct in top5.items():
        lines.append(f"  {county:20s}: {pct:.1f}%")

    lines += ["", "=" * 60]
    summary_path = os.path.join(REPORT_DIR, "eda_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n✓ EDA summary saved -> {summary_path}")
    print("\n".join(lines))


# ── HELPER ────────────────────────────────────────────────────────────────────

def _save(fig, name: str) -> None:
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    df = load_data()
    print(f"  {len(df):,} children, {df['county'].nunique()} counties\n")

    print("Generating EDA figures …")
    fig_prevalence_overview(df)
    fig_class_imbalance(df)
    fig_age_by_outcome(df)
    fig_wealth_gradient(df)
    fig_county_stunting(df)
    fig_correlation(df)
    fig_maternal_edu(df)
    fig_wash(df)

    write_summary(df)
    print("\n✓ EDA complete. All figures saved to reports/figures/")


if __name__ == "__main__":
    main()
