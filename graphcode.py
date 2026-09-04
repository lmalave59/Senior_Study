# -*- coding: utf-8 -*-
"""
Impostor Phenomenon Analysis - CSC 450 Senior Research
Lucas Malave, Eastern Connecticut State University

Scores 20-item Clance Impostor Phenomenon Scale (CIPS) responses, compares
CS students against student-athletes, and generates six figures.

Usage:  python graphcode.py
Output: figures/
"""

import os
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

DATA_PATH = "responses.csv"
FIGURE_DIR = "figures"

# CIPS severity thresholds (Clance, 1985)
SEVERITY_BINS = [0, 40, 60, 80, 100]
SEVERITY_LABELS = ["Few", "Moderate", "Frequent", "Intense"]

# Column holding the scores totalled by hand in the spreadsheet, used only
# to verify this script reproduces them. Not required for the analysis.
CHECK_COL = "Column 1"

os.makedirs(FIGURE_DIR, exist_ok=True)


def save(fig_name):
    """Write the current figure to the output directory at print resolution."""
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, fig_name), dpi=300, bbox_inches="tight")
    plt.close()


# ---------- Load ----------
df = pd.read_csv(DATA_PATH)

# The 20 CIPS items are matched on their leading item number, so this works
# whether the Google Form export carries the full question text as the header
# or an abbreviated "1." through "20.".
q_cols = sorted(
    [c for c in df.columns if re.match(r"^\s*\d{1,2}\.", str(c))],
    key=lambda c: int(re.match(r"^\s*(\d{1,2})\.", str(c)).group(1)),
)
if len(q_cols) != 20:
    raise ValueError(f"Expected 20 CIPS items, matched {len(q_cols)}: {q_cols}")

df["IP_Score"] = df[q_cols].sum(axis=1)

# Confirm the computed totals agree with the ones totalled in the spreadsheet.
if CHECK_COL in df.columns:
    mismatches = (df["IP_Score"] != pd.to_numeric(df[CHECK_COL], errors="coerce")).sum()
    if mismatches:
        raise ValueError(f"{mismatches} scored rows disagree with '{CHECK_COL}'")

# Major names arrive with inconsistent trailing spaces ("Health Science " vs
# "Health Science"), so normalise before grouping.
df["Major"] = df["Major field of study"].str.strip()
df["Group"] = np.where(
    df["Major"].str.contains("Computer Science", case=False),
    "CS Seniors",
    "Baseball Athletes",
)

# Some GPA entries can arrive as "NA" or "N/A"; coerce rather than fail.
df["GPA"] = pd.to_numeric(df["Overall GPA"], errors="coerce")

df["Severity"] = pd.cut(
    df["IP_Score"], bins=SEVERITY_BINS, labels=SEVERITY_LABELS, include_lowest=True
)

# Sample sizes are derived, never hardcoded, so figure labels stay correct
# if the dataset changes.
n_total = len(df)
valid = df.dropna(subset=["GPA"])
n_gpa = len(valid)

# ---------- Figure 1: score distribution ----------
plt.figure(figsize=(9, 6))
plt.hist(df["IP_Score"], bins=10, edgecolor="black", color="#3498db")
plt.title(
    f"Figure 1. Distribution of Impostor Phenomenon Scores (n={n_total})",
    fontsize=14, fontweight="bold",
)
plt.xlabel("Total IP Score (20-100)")
plt.ylabel("Number of Participants")
plt.grid(axis="y", alpha=0.3)
save("Figure1_Histogram.png")

# ---------- Figure 2: GPA vs score ----------
r, p = stats.pearsonr(valid["GPA"], valid["IP_Score"])
plt.figure(figsize=(9, 6))
sns.regplot(x="GPA", y="IP_Score", data=valid, scatter_kws={"s": 80})
plt.title(
    f"Figure 2. GPA vs Impostor Score (n={n_gpa})\nr = {r:.2f}, p = {p:.3f}",
    fontsize=14, fontweight="bold",
)
plt.xlabel("Overall GPA")
plt.ylabel("Total IP Score")
save("Figure2_Scatter.png")

# ---------- Figure 3: group means ----------
# Welch's t-test (equal_var=False) because the two groups cannot be assumed
# to share a variance.
means = df.groupby("Group")["IP_Score"].mean()
stds = df.groupby("Group")["IP_Score"].std()
order = ["Baseball Athletes", "CS Seniors"]
tstat, pval = stats.ttest_ind(
    df[df["Group"] == "CS Seniors"]["IP_Score"],
    df[df["Group"] == "Baseball Athletes"]["IP_Score"],
    equal_var=False,
)
plt.figure(figsize=(8, 6))
plt.bar(order, [means[g] for g in order], yerr=[stds[g] for g in order],
        capsize=10, color=["#3498db", "#e67e22"])
plt.title(
    f"Figure 3. Mean IP Score by Group\nt = {tstat:.2f}, p = {pval:.3f}",
    fontsize=14, fontweight="bold",
)
plt.ylabel("Mean Total IP Score")
save("Figure3_Means.png")

# ---------- Figure 4: distribution by group ----------
plt.figure(figsize=(8, 6))
sns.boxplot(x="Group", y="IP_Score", hue="Group", data=df,
            palette=["#3498db", "#e67e22"], legend=False)
plt.title("Figure 4. IP Score Distribution by Group", fontsize=14, fontweight="bold")
plt.ylabel("Total IP Score")
save("Figure4_Boxplot.png")

# ---------- Figure 5: severity mix by group ----------
counts = pd.crosstab(df["Group"], df["Severity"])[SEVERITY_LABELS]
perc = counts.div(counts.sum(axis=1), axis=0) * 100
perc.plot(kind="barh", stacked=True, figsize=(10, 5),
          color=["#d5dbdb", "#85c1e2", "#3498db", "#1a5276"])
plt.title("Figure 5. Impostor Severity Category by Group", fontsize=14, fontweight="bold")
plt.xlabel("Percentage of Participants (%)")
plt.legend(title="Severity", bbox_to_anchor=(1, 1))
save("Figure5_Severity.png")

# ---------- Figure 6: largest per-item gaps ----------
means_q = df.groupby("Group")[q_cols].mean()
diff = means_q.loc["CS Seniors"] - means_q.loc["Baseball Athletes"]
top_diff = diff[diff.abs().nlargest(5).index]
# Long question text is trimmed to the item number for a readable axis.
top_diff.index = [re.match(r"^\s*(\d{1,2})\.", str(c)).group(1) for c in top_diff.index]
top_diff.index = ["Item " + i for i in top_diff.index]
plt.figure(figsize=(10, 6))
top_diff.sort_values().plot(kind="barh", color="#e74c3c", alpha=0.8)
plt.title("Figure 6. Top 5 Items Separating CS Students from Athletes",
          fontsize=14, fontweight="bold")
plt.xlabel("Mean Difference (CS - Athletes)")
for i, (q, v) in enumerate(top_diff.sort_values().items()):
    plt.text(v + 0.02 if v > 0 else v - 0.02, i, f"{v:+.2f}",
             va="center", fontweight="bold")
save("Figure6_TopQuestions.png")

n_cs = int((df["Group"] == "CS Seniors").sum())
print(f"Analyzed {n_total} responses ({n_cs} CS, {n_total - n_cs} athletes; "
      f"{n_gpa} with usable GPA).")
print(f"Six figures written to ./{FIGURE_DIR}/")
