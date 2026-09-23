#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Settings
# ============================================================

RESULTS_DIR = Path(
    "/data/analysis/combinations/models/in-house/regulons/"
    "regulon_results_ulm"
)

ACTIVITY_FILE = (
    RESULTS_DIR /
    "TF_activity_contextualized_collectri_ULM.csv"
)

NAMING_FILE = Path(
    "/data/analysis/combinations/input/cell_lines/depmap/"
    "naming.csv"
)

OUTDIR = RESULTS_DIR / "aging_regulon_QC"
OUTDIR.mkdir(parents=True, exist_ok=True)

N_RANDOM = 20
RANDOM_SEED = 42


# ============================================================
# Aging-related regulons
# ============================================================

REGULON_GROUPS = {

    "NF-kB": [
        "RELA",
        "RELB",
        "REL",
        "NFKB1",
        "NFKB2",
        "NFKB",
    ],

    "CEBPB": [
        "CEBPB",
    ],

    "GATA4": [
        "GATA4",
    ],

    "E2F": [
        "E2F1",
        "E2F2",
        "E2F3",
        "E2F4",
        "E2F5",
        "E2F6",
        "E2F7",
        "E2F8",
    ],
}


# ============================================================
# Helper: shorten CCLE names for plotting
# ============================================================

def shorten_ccle_name(name):

    if pd.isna(name):
        return name

    name = str(name)

    replacements = {
        "_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE": "_HAEMATOPOIETIC",
        "_CENTRAL_NERVOUS_SYSTEM": "_CNS",
        "_AUTONOMIC_GANGLIA": "_AUTONOMIC",
        "_LARGE_INTESTINE": "_COLON",
        "_UPPER_AERODIGESTIVE_TRACT": "_UADT",
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    return name


# ============================================================
# Load ULM activities
# ============================================================

print("\n========================================")
print("Loading contextualized ULM activities")
print("========================================")

activity = pd.read_csv(
    ACTIVITY_FILE,
    index_col="ModelID",
)

print("Activity matrix:", activity.shape)
print("Unique ModelIDs:", activity.index.nunique())


# ============================================================
# Load DepMap naming map
# ============================================================

print("\n========================================")
print("Loading DepMap naming map")
print("========================================")

naming = pd.read_csv(
    NAMING_FILE
)

print("Naming columns:")
print(naming.columns.tolist())

required_cols = {
    "broad_id",
    "canonical_ccle_name",
}

missing_cols = (
    required_cols
    - set(naming.columns)
)

if missing_cols:
    raise RuntimeError(
        f"Missing required columns in naming.csv: {missing_cols}"
    )


# One canonical CCLE name per ACH ModelID
name_table = (
    naming[
        [
            "broad_id",
            "canonical_ccle_name",
        ]
    ]
    .dropna(
        subset=[
            "broad_id",
            "canonical_ccle_name",
        ]
    )
    .drop_duplicates(
        subset="broad_id",
        keep="first",
    )
    .copy()
)


name_map = (
    name_table
    .set_index("broad_id")[
        "canonical_ccle_name"
    ]
    .to_dict()
)


# How many activity samples can be named?
n_named = sum(
    model_id in name_map
    for model_id in activity.index
)

print(
    f"Activity samples with canonical CCLE name: "
    f"{n_named} / {len(activity)}"
)


# ============================================================
# Determine which requested TFs are available
# ============================================================

available_groups = {}

print("\n========================================")
print("Available aging-related regulons")
print("========================================")

for group, tfs in REGULON_GROUPS.items():

    available = [
        tf
        for tf in tfs
        if tf in activity.columns
    ]

    available_groups[group] = available

    print(
        f"{group:8s}: "
        f"{available}"
    )


# Flatten TF list while preserving group order
selected_tfs = []

for group in [
    "NF-kB",
    "CEBPB",
    "GATA4",
    "E2F",
]:

    selected_tfs.extend(
        available_groups[group]
    )


if len(selected_tfs) == 0:
    raise RuntimeError(
        "None of the aging-related TFs were found."
    )


print(
    "\nTFs used:",
    selected_tfs,
)


# ============================================================
# Random 20 cell lines
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

n_select = min(
    N_RANDOM,
    len(activity),
)
#
#selected_ids = rng.choice(
#    activity.index.to_numpy(),
#    size=n_select,
#    replace=False,
#)

# ============================================================
# Specific 20 cell lines
# ============================================================

selected_ids = np.array([
    "ACH-001500",
    "ACH-000903",
    "ACH-000897",
    "ACH-000574",
    "ACH-000633",
    "ACH-000082",
    "ACH-000572",
    "ACH-000096",
    "ACH-000375",
    "ACH-000162",
    "ACH-000098",
    "ACH-000738",
    "ACH-000835",
    "ACH-000081",
    "ACH-000756",
    "ACH-001344",
    "ACH-000102",
    "ACH-000027",
    "ACH-001716",
    "ACH-000982",
])

# Check that all requested IDs exist in the activity matrix
missing_ids = [
    x for x in selected_ids
    if x not in activity.index
]

if missing_ids:
    print("\nWARNING: These requested ModelIDs are missing:")
    for x in missing_ids:
        print("  ", x)

# Keep only those actually available
selected_ids = np.array([
    x for x in selected_ids
    if x in activity.index
])

print(
    f"\nUsing {len(selected_ids)} requested cell lines."
)

# ============================================================
# Convert ModelIDs to CCLE names
# ============================================================

selected_names_full = [
    name_map.get(
        model_id,
        model_id,
    )
    for model_id in selected_ids
]

selected_names_plot = [
    shorten_ccle_name(name)
    for name in selected_names_full
]


selected_metadata = pd.DataFrame(
    {
        "ModelID": selected_ids,
        "canonical_ccle_name": selected_names_full,
        "plot_label": selected_names_plot,
    }
)

selected_metadata.to_csv(
    OUTDIR /
    "random20_cell_line_names.csv",
    index=False,
)


print("\n========================================")
print("Selected cell lines")
print("========================================")

for model_id, name in zip(
    selected_ids,
    selected_names_full,
):
    print(
        f"{model_id:12s}  {name}"
    )


# ============================================================
# Raw TF-level ULM matrix
# ============================================================

sub_raw = activity.loc[
    selected_ids,
    selected_tfs,
].copy()

# Save with ModelIDs first
sub_raw_with_ids = sub_raw.copy()
sub_raw_with_ids.insert(
    0,
    "canonical_ccle_name",
    selected_names_full,
)

sub_raw_with_ids.to_csv(
    OUTDIR /
    "aging_regulon_ULM_random20_raw.csv"
)


# Use readable names for plotting
sub_raw_plot = sub_raw.copy()
sub_raw_plot.index = selected_names_plot


# ============================================================
# Z-score TF activity for visualization
# ============================================================
#
# IMPORTANT:
#
# Each TF is standardized across ALL 1719 DepMap cell lines.
#
# Interpretation:
#   positive = higher TF activity than DepMap average
#   negative = lower TF activity than DepMap average
#
# We do NOT z-score only the selected 20 samples.
# ============================================================

aging_all = activity[
    selected_tfs
].copy()

mu = aging_all.mean(
    axis=0
)

sd = (
    aging_all
    .std(
        axis=0,
        ddof=0,
    )
    .replace(
        0,
        np.nan,
    )
)

aging_z = (
    (aging_all - mu)
    / sd
).fillna(0)


sub_z = aging_z.loc[
    selected_ids
].copy()


# Save z-score matrix with names
sub_z_save = sub_z.copy()

sub_z_save.insert(
    0,
    "canonical_ccle_name",
    selected_names_full,
)

sub_z_save.to_csv(
    OUTDIR /
    "aging_regulon_ULM_random20_zscore.csv"
)


# Plot version
sub_z_plot = sub_z.copy()
sub_z_plot.index = selected_names_plot


# ============================================================
# TF-level heatmap
# ============================================================

plt.figure(
    figsize=(
        max(
            10,
            len(selected_tfs) * 0.7,
        ),
        11,
    )
)

sns.heatmap(
    sub_z_plot,
    cmap="RdBu_r",
    center=0,
    linewidths=0.3,
    linecolor="white",
    cbar_kws={
        "label":
        "ULM activity\n(z-score across DepMap)"
    },
)

plt.title(
    "Aging-related regulon activity across random DepMap cell lines",
    fontsize=14,
    pad=15,
)

plt.xlabel(
    "Transcription factor / regulon"
)

plt.ylabel(
    "Cell line"
)

plt.xticks(
    rotation=45,
    ha="right",
)

plt.yticks(
    rotation=0,
)

plt.tight_layout()

plt.savefig(
    OUTDIR /
    "aging_regulon_heatmap_random20.pdf",
    bbox_inches="tight",
)

plt.savefig(
    OUTDIR /
    "aging_regulon_heatmap_random20.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Aggregate activity into four biological programs
# ============================================================
#
# NF-kB = mean standardized activity of:
#         RELA / RELB / REL / NFKB1 / NFKB2
#
# CEBPB = CEBPB
#
# GATA4 = GATA4
#
# E2F = mean standardized activity of:
#       E2F1-E2F7 (whatever is available)
#
# IMPORTANT:
# This aggregation is ONLY for visualization / QC.
#
# For the actual model, keep individual ULM TF activities.
# ============================================================

program_scores = pd.DataFrame(
    index=activity.index
)

for group, tfs in available_groups.items():

    if len(tfs) == 0:
        continue

    program_scores[group] = (
        aging_z[
            tfs
        ]
        .mean(
            axis=1
        )
    )


program_sub = program_scores.loc[
    selected_ids
].copy()


# Save program scores with readable names
program_sub_save = program_sub.copy()

program_sub_save.insert(
    0,
    "canonical_ccle_name",
    selected_names_full,
)

program_sub_save.to_csv(
    OUTDIR /
    "aging_program_scores_random20.csv"
)


# Plot version
program_sub_plot = program_sub.copy()
program_sub_plot.index = selected_names_plot


# ============================================================
# Program-level heatmap
# ============================================================

plt.figure(
    figsize=(
        7,
        11,
    )
)

sns.heatmap(
    program_sub_plot,
    cmap="RdBu_r",
    center=0,
    linewidths=0.4,
    linecolor="white",
    cbar_kws={
        "label":
        "Mean standardized ULM activity"
    },
)

plt.title(
    "Aging-related transcriptional programs",
    fontsize=14,
    pad=15,
)

plt.xlabel(
    "Regulatory program"
)

plt.ylabel(
    "Cell line"
)

plt.xticks(
    rotation=0,
)

plt.yticks(
    rotation=0,
)

plt.tight_layout()

plt.savefig(
    OUTDIR /
    "aging_program_heatmap_random20.pdf",
    bbox_inches="tight",
)

plt.savefig(
    OUTDIR /
    "aging_program_heatmap_random20.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Senescence-like regulatory score
# ============================================================
#
# Positive side:
#     NF-kB
#     CEBPB
#     GATA4
#
# Negative side:
#     E2F
#
# score =
# mean(NF-kB, CEBPB, GATA4) - E2F
#
# IMPORTANT:
# This is ONLY a biological sanity-check / visualization score.
# It is NOT a validated senescence score.
# ============================================================

positive_programs = [
    x
    for x in [
        "NF-kB",
        "CEBPB",
        "GATA4",
    ]
    if x in program_scores.columns
]


if (
    len(positive_programs) > 0
    and "E2F" in program_scores.columns
):

    program_scores[
        "senescence_like_regulon_score"
    ] = (
        program_scores[
            positive_programs
        ]
        .mean(
            axis=1
        )
        -
        program_scores[
            "E2F"
        ]
    )


    score_sub = (
        program_scores
        .loc[
            selected_ids,
            "senescence_like_regulon_score",
        ]
        .copy()
    )


    # Replace ACH IDs by CCLE names
    score_sub.index = selected_names_plot


    score_sub = (
        score_sub
        .sort_values()
    )


    score_sub.to_csv(
        OUTDIR /
        "senescence_like_score_random20.csv",
        header=[
            "senescence_like_regulon_score"
        ],
    )


    # --------------------------------------------------------
    # Horizontal bar plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(
            9,
            10,
        )
    )

    score_sub.plot(
        kind="barh"
    )

    plt.axvline(
        0,
        linewidth=1,
        linestyle="--",
    )

    plt.xlabel(
        "Senescence-like regulon score"
    )

    plt.ylabel(
        "Cell line"
    )

    plt.title(
        "Relative aging/senescence-like regulatory state"
    )

    plt.tight_layout()

    plt.savefig(
        OUTDIR /
        "senescence_like_score_random20.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        OUTDIR /
        "senescence_like_score_random20.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Also save all DepMap program scores
# ============================================================
#
# This may be useful later if we want:
# - highest / lowest senescence-like cell lines
# - tissue comparisons
# - select representative examples
# ============================================================

all_program_scores = (
    program_scores
    .copy()
)

all_program_scores.insert(
    0,
    "canonical_ccle_name",
    [
        name_map.get(
            model_id,
            model_id,
        )
        for model_id in all_program_scores.index
    ],
)

all_program_scores.to_csv(
    OUTDIR /
    "aging_program_scores_all_DepMap.csv"
)


# ============================================================
# Finish
# ============================================================

print("\n========================================")
print("DONE")
print("========================================")

print(
    "\nRandomly selected cell lines:"
)

for model_id, name in zip(
    selected_ids,
    selected_names_full,
):
    print(
        f"{model_id:12s}  {name}"
    )


print(
    "\nResults written to:",
    OUTDIR.resolve(),
)

print(
    "\nMain plots:"
)

print(
    "  aging_regulon_heatmap_random20.png"
)

print(
    "  aging_program_heatmap_random20.png"
)

print(
    "  senescence_like_score_random20.png"
)
