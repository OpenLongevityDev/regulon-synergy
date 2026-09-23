#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


# ============================================================
# Settings
# ============================================================

BASE_DIR = Path(
    "/data/analysis/combinations/models/in-house/regulons"
)

RESULTS_DIR = (
    BASE_DIR /
    "regulon_results_ulm"
)

ACTIVITY_FILE = (
    RESULTS_DIR /
    "TF_activity_contextualized_collectri_ULM.csv"
)

EXPR_FILE = Path(
    "/data/analysis/combinations/input/cell_lines/depmap/"
    "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
)

NAMING_FILE = Path(
    "/data/analysis/combinations/input/cell_lines/depmap/"
    "naming.csv"
)

OUTDIR = (
    RESULTS_DIR /
    "senescence_validation"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

N_PER_GROUP = 20


# ============================================================
# Independent genes used ONLY to define senescence-like state
# ============================================================
#
# IMPORTANT:
# These genes are used for selecting the groups.
#
# They are NOT the TF/regulon activities that we later test.
#
# Senescence / growth-arrest side:
#   CDKN2A, CDKN1A, SERPINE1
#
# Proliferation side:
#   MKI67, PCNA, TOP2A, MCM2, MCM5, CCNB1
#
# Composite selection score:
#
# mean(senescence markers)
# -
# mean(proliferation markers)
#
# ============================================================

SENESCENCE_MARKERS = [
    "CDKN2A",
    "CDKN1A",
    "SERPINE1",
]

PROLIFERATION_MARKERS = [
    "MKI67",
    "PCNA",
    "TOP2A",
    "MCM2",
    "MCM5",
    "CCNB1",
]


# ============================================================
# Aging-related regulons to validate
# ============================================================

REGULON_GROUPS = {

    "NF-kB": [
        "RELA",
        "RELB",
        "REL",
        "NFKB1",
        "NFKB2",
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
# Helpers
# ============================================================

def detect_gene_mapping(columns):
    """
    Convert DepMap column names such as:

        TP53 (7157)

    into:

        TP53 -> original column
    """

    mapping = {}

    for col in columns:

        col = str(col)

        if " (" in col and col.endswith(")"):
            symbol = col.split(" (")[0]
        else:
            symbol = col

        if symbol not in mapping:
            mapping[symbol] = col

    return mapping


def shorten_ccle_name(name):

    name = str(name)

    replacements = {
        "_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE":
            "_HAEMATOPOIETIC",

        "_CENTRAL_NERVOUS_SYSTEM":
            "_CNS",

        "_AUTONOMIC_GANGLIA":
            "_AUTONOMIC",

        "_LARGE_INTESTINE":
            "_COLON",

        "_UPPER_AERODIGESTIVE_TRACT":
            "_UADT",
    }

    for old, new in replacements.items():
        name = name.replace(
            old,
            new,
        )

    return name


# ============================================================
# 1. Load contextualized ULM activities
# ============================================================

print("\n========================================")
print("Loading contextualized ULM activities")
print("========================================")

activity = pd.read_csv(
    ACTIVITY_FILE,
    index_col="ModelID",
)

print(
    "Activity matrix:",
    activity.shape,
)


# ============================================================
# 2. Load CCLE names
# ============================================================

print("\n========================================")
print("Loading CCLE names")
print("========================================")

naming = pd.read_csv(
    NAMING_FILE
)

name_map = (
    naming[
        [
            "broad_id",
            "canonical_ccle_name",
        ]
    ]
    .dropna()
    .drop_duplicates(
        subset="broad_id",
        keep="first",
    )
    .set_index(
        "broad_id"
    )[
        "canonical_ccle_name"
    ]
    .to_dict()
)


# ============================================================
# 3. Load only genes needed to define senescence
# ============================================================

print("\n========================================")
print("Loading independent senescence markers")
print("========================================")

needed_markers = (
    SENESCENCE_MARKERS
    +
    PROLIFERATION_MARKERS
)


# Read header first
header = pd.read_csv(
    EXPR_FILE,
    nrows=0,
).columns.tolist()


metadata_columns = {
    "Unnamed: 0",
    "SequencingID",
    "ModelConditionID",
    "ModelID",
    "IsDefaultEntryForMC",
    "IsDefaultEntryForModel",
}


gene_columns = [
    x
    for x in header
    if x not in metadata_columns
]


gene_mapping = detect_gene_mapping(
    gene_columns
)


available_markers = [
    gene
    for gene in needed_markers
    if gene in gene_mapping
]


missing_markers = [
    gene
    for gene in needed_markers
    if gene not in gene_mapping
]


print(
    "Available markers:",
    available_markers,
)

if missing_markers:
    print(
        "Missing markers:",
        missing_markers,
    )


original_columns = [
    gene_mapping[gene]
    for gene in available_markers
]


expr = pd.read_csv(
    EXPR_FILE,
    usecols=[
        "ModelID",
        "IsDefaultEntryForModel",
    ]
    + original_columns,
)


# Same canonical-profile filtering as regulon pipeline
expr = expr[
    expr["IsDefaultEntryForModel"]
    == "Yes"
].copy()


expr = expr.drop(
    columns=[
        "IsDefaultEntryForModel"
    ]
)


expr = expr.set_index(
    "ModelID"
)


reverse_mapping = {
    gene_mapping[gene]: gene
    for gene in available_markers
}


expr = expr.rename(
    columns=reverse_mapping
)


expr = expr.apply(
    pd.to_numeric,
    errors="coerce",
)


print(
    "Marker expression matrix:",
    expr.shape,
)


# ============================================================
# 4. Restrict to cell lines present in BOTH datasets
# ============================================================

shared_ids = (
    activity.index
    .intersection(
        expr.index
    )
)


activity = activity.loc[
    shared_ids
].copy()


expr = expr.loc[
    shared_ids
].copy()


print(
    "Shared cell lines:",
    len(shared_ids),
)


# ============================================================
# 5. Z-score expression markers across all DepMap cell lines
# ============================================================
#
# Each marker receives equal weight.
#
# This is used ONLY for group definition.
# ============================================================

expr_mean = expr.mean(
    axis=0
)

expr_sd = (
    expr
    .std(
        axis=0,
        ddof=0,
    )
    .replace(
        0,
        np.nan,
    )
)


expr_z = (
    (expr - expr_mean)
    / expr_sd
)


# ============================================================
# 6. Construct independent senescence-selection score
# ============================================================

senescence_available = [
    gene
    for gene in SENESCENCE_MARKERS
    if gene in expr_z.columns
]


proliferation_available = [
    gene
    for gene in PROLIFERATION_MARKERS
    if gene in expr_z.columns
]


if len(senescence_available) == 0:
    raise RuntimeError(
        "No senescence markers available."
    )


if len(proliferation_available) == 0:
    raise RuntimeError(
        "No proliferation markers available."
    )


print(
    "\nSenescence markers used:",
    senescence_available,
)

print(
    "Proliferation markers used:",
    proliferation_available,
)


selection_score = (
    expr_z[
        senescence_available
    ].mean(
        axis=1
    )
    -
    expr_z[
        proliferation_available
    ].mean(
        axis=1
    )
)


selection_score.name = (
    "expression_senescence_score"
)


# ============================================================
# 7. Select top 20 and bottom 20
# ============================================================

selection_score = (
    selection_score
    .dropna()
    .sort_values()
)


non_senescent_ids = (
    selection_score
    .head(
        N_PER_GROUP
    )
    .index
)


senescent_ids = (
    selection_score
    .tail(
        N_PER_GROUP
    )
    .index
)


selected_ids = (
    list(non_senescent_ids)
    +
    list(senescent_ids)
)


group = pd.Series(
    index=selected_ids,
    dtype="object",
)


group.loc[
    non_senescent_ids
] = "Non-senescent-like"


group.loc[
    senescent_ids
] = "Senescence-like"


print("\n========================================")
print("Selected groups")
print("========================================")

print(
    "\nSenescence-like:",
    len(senescent_ids),
)

print(
    "Non-senescent-like:",
    len(non_senescent_ids),
)


# ============================================================
# 8. Save selected cell lines
# ============================================================

selected_metadata = pd.DataFrame(
    {
        "ModelID":
            selected_ids,

        "CCLE_name":
            [
                name_map.get(
                    x,
                    x,
                )
                for x in selected_ids
            ],

        "group":
            group.loc[
                selected_ids
            ].values,

        "expression_senescence_score":
            selection_score.loc[
                selected_ids
            ].values,
    }
)


selected_metadata.to_csv(
    OUTDIR /
    "selected_senescence_cell_lines.csv",
    index=False,
)


print(
    "\nSelected lines:"
)

print(
    selected_metadata[
        [
            "ModelID",
            "CCLE_name",
            "group",
            "expression_senescence_score",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 9. Determine available aging TFs
# ============================================================

available_groups = {}


print("\n========================================")
print("Available regulons")
print("========================================")


for program, tfs in REGULON_GROUPS.items():

    present = [
        tf
        for tf in tfs
        if tf in activity.columns
    ]

    available_groups[
        program
    ] = present

    print(
        f"{program}: {present}"
    )


selected_tfs = []

for program in [
    "NF-kB",
    "CEBPB",
    "GATA4",
    "E2F",
]:

    selected_tfs.extend(
        available_groups[
            program
        ]
    )


# ============================================================
# 10. Standardize ULM activity across ALL shared DepMap lines
# ============================================================
#
# Same logic as scores.py.
# ============================================================

aging_activity = activity[
    selected_tfs
].copy()


activity_mean = aging_activity.mean(
    axis=0
)


activity_sd = (
    aging_activity
    .std(
        axis=0,
        ddof=0,
    )
    .replace(
        0,
        np.nan,
    )
)


activity_z = (
    (aging_activity - activity_mean)
    / activity_sd
).fillna(0)


# ============================================================
# 11. Construct program-level activity
# ============================================================

program_scores = pd.DataFrame(
    index=activity.index
)


for program, tfs in available_groups.items():

    if len(tfs) == 0:
        continue

    program_scores[
        program
    ] = (
        activity_z[
            tfs
        ]
        .mean(
            axis=1
        )
    )


# ============================================================
# 12. Combined aging / senescence regulon score
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


if "E2F" not in program_scores.columns:
    raise RuntimeError(
        "E2F activity not available."
    )


program_scores[
    "Combined score"
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


# ============================================================
# 13. Selected 40 lines
# ============================================================

selected_program_scores = (
    program_scores
    .loc[
        selected_ids
    ]
    .copy()
)


selected_program_scores[
    "Group"
] = (
    group.loc[
        selected_ids
    ].values
)


selected_program_scores[
    "CCLE_name"
] = [
    name_map.get(
        x,
        x,
    )
    for x in selected_ids
]


selected_program_scores[
    "expression_senescence_score"
] = (
    selection_score.loc[
        selected_ids
    ].values
)


selected_program_scores.to_csv(
    OUTDIR /
    "selected40_regulon_scores.csv"
)


# ============================================================
# 14. Statistics
# ============================================================
#
# Mann-Whitney U:
# robust non-parametric comparison between the 20 + 20 lines.
#
# We test:
#   NF-kB
#   CEBPB
#   GATA4
#   E2F
#   Combined score
#
# BH correction across these five comparisons.
# ============================================================

metrics = [
    x
    for x in [
        "NF-kB",
        "CEBPB",
        "GATA4",
        "E2F",
        "Combined score",
    ]
    if x in selected_program_scores.columns
]


stats_rows = []


for metric in metrics:

    sen = (
        selected_program_scores.loc[
            selected_program_scores["Group"]
            == "Senescence-like",
            metric,
        ]
        .dropna()
        .values
    )


    non = (
        selected_program_scores.loc[
            selected_program_scores["Group"]
            == "Non-senescent-like",
            metric,
        ]
        .dropna()
        .values
    )


    u_stat, p_value = mannwhitneyu(
        sen,
        non,
        alternative="two-sided",
    )


    # --------------------------------------------
    # Rank-biserial effect size
    #
    # positive =
    # higher score in senescence-like group
    # --------------------------------------------

    n1 = len(sen)
    n2 = len(non)

    rank_biserial = (
        2 * u_stat
        / (n1 * n2)
        - 1
    )


    stats_rows.append(
        {
            "metric":
                metric,

            "n_senescence":
                n1,

            "n_non_senescent":
                n2,

            "median_senescence":
                np.median(sen),

            "median_non_senescent":
                np.median(non),

            "U":
                u_stat,

            "p_value":
                p_value,

            "rank_biserial":
                rank_biserial,
        }
    )


stats_df = pd.DataFrame(
    stats_rows
)


# BH multiple-testing correction
stats_df[
    "p_adj_BH"
] = multipletests(
    stats_df[
        "p_value"
    ],
    method="fdr_bh",
)[1]


stats_df.to_csv(
    OUTDIR /
    "regulon_group_statistics.csv",
    index=False,
)


print("\n========================================")
print("Regulon statistics")
print("========================================")

print(
    stats_df.to_string(
        index=False
    )
)


# ============================================================
# 15. Heatmap
# ============================================================
#
# Rows:
#   20 non-senescent-like
#   20 senescence-like
#
# Columns:
#   NF-kB
#   CEBPB
#   GATA4
#   E2F
#   Combined score
#
# ============================================================

heatmap_df = (
    selected_program_scores[
        metrics
    ]
    .copy()
)


# Sort within each group by expression-based selection score
non_order = (
    selection_score
    .loc[
        non_senescent_ids
    ]
    .sort_values()
    .index
)


sen_order = (
    selection_score
    .loc[
        senescent_ids
    ]
    .sort_values()
    .index
)


heatmap_order = (
    list(non_order)
    +
    list(sen_order)
)


heatmap_df = heatmap_df.loc[
    heatmap_order
]


heatmap_labels = [
    shorten_ccle_name(
        name_map.get(
            model_id,
            model_id,
        )
    )
    for model_id in heatmap_order
]


heatmap_df.index = heatmap_labels


plt.figure(
    figsize=(
        8,
        15,
    )
)


ax = sns.heatmap(
    heatmap_df,
    cmap="RdBu_r",
    center=0,
    linewidths=0.35,
    linecolor="white",
    cbar_kws={
        "label":
        "Standardized regulon activity"
    },
)


# Separation between groups
plt.axhline(
    y=N_PER_GROUP,
    color="black",
    linewidth=2.5,
)


plt.text(
    len(metrics) + 0.15,
    N_PER_GROUP / 2,
    "Non-senescent-like",
    rotation=90,
    va="center",
    fontsize=10,
)


plt.text(
    len(metrics) + 0.15,
    N_PER_GROUP + N_PER_GROUP / 2,
    "Senescence-like",
    rotation=90,
    va="center",
    fontsize=10,
)


plt.title(
    "Aging-related regulon activity in DepMap\n"
    "senescence-like vs non-senescent-like cell lines",
    fontsize=14,
    pad=14,
)


plt.xlabel(
    "Regulatory program"
)

plt.ylabel(
    "Cell line"
)


plt.xticks(
    rotation=30,
    ha="right",
)


plt.yticks(
    rotation=0,
    fontsize=8,
)


plt.tight_layout()


plt.savefig(
    OUTDIR /
    "senescence_regulon_heatmap.png",
    dpi=300,
    bbox_inches="tight",
)


plt.savefig(
    OUTDIR /
    "senescence_regulon_heatmap.pdf",
    bbox_inches="tight",
)


plt.close()


# ============================================================
# 16. Boxplot for all regulon metrics
# ============================================================

long_df = (
    selected_program_scores[
        metrics
        +
        ["Group"]
    ]
    .reset_index()
    .melt(
        id_vars=[
            "ModelID",
            "Group",
        ],
        value_vars=metrics,
        var_name="Regulatory program",
        value_name="Activity score",
    )
)


plt.figure(
    figsize=(
        11,
        7,
    )
)


ax = sns.boxplot(
    data=long_df,
    x="Regulatory program",
    y="Activity score",
    hue="Group",
    order=metrics,
    hue_order=[
        "Non-senescent-like",
        "Senescence-like",
    ],
    showfliers=False,
)


# Add individual observations
sns.stripplot(
    data=long_df,
    x="Regulatory program",
    y="Activity score",
    hue="Group",
    order=metrics,
    hue_order=[
        "Non-senescent-like",
        "Senescence-like",
    ],
    dodge=True,
    alpha=0.65,
    size=4,
    ax=ax,
)


# Remove duplicate legend entries produced by box + strip
handles, labels = ax.get_legend_handles_labels()

ax.legend(
    handles[:2],
    labels[:2],
    title="Group",
    frameon=False,
)


# ============================================================
# Add p-values
# ============================================================

y_min = long_df[
    "Activity score"
].min()

y_max = long_df[
    "Activity score"
].max()

y_range = (
    y_max - y_min
)

annotation_base = (
    y_max
    +
    0.08 * y_range
)


for i, metric in enumerate(metrics):

    row = stats_df[
        stats_df["metric"]
        == metric
    ].iloc[0]

    p = row[
        "p_value"
    ]

    q = row[
        "p_adj_BH"
    ]

    if p < 0.001:
        p_text = "p<0.001"
    else:
        p_text = f"p={p:.3f}"

    if q < 0.001:
        q_text = "q<0.001"
    else:
        q_text = f"q={q:.3f}"

    ax.text(
        i,
        annotation_base,
        f"{p_text}\n{q_text}",
        ha="center",
        va="bottom",
        fontsize=9,
    )


ax.set_ylim(
    y_min - 0.1 * y_range,
    annotation_base + 0.18 * y_range,
)


plt.axhline(
    0,
    color="grey",
    linestyle="--",
    linewidth=0.8,
)


plt.title(
    "Aging-related regulon activity:\n"
    "senescence-like vs non-senescent-like DepMap cell lines",
    fontsize=14,
    pad=14,
)


plt.xlabel(
    ""
)

plt.ylabel(
    "Standardized regulon activity"
)


plt.xticks(
    rotation=20,
    ha="right",
)


plt.tight_layout()


plt.savefig(
    OUTDIR /
    "senescence_regulon_boxplot.png",
    dpi=300,
    bbox_inches="tight",
)


plt.savefig(
    OUTDIR /
    "senescence_regulon_boxplot.pdf",
    bbox_inches="tight",
)


plt.close()


# ============================================================
# 17. Additional diagnostic plot:
# expression-based selection score
# ============================================================

selection_plot_df = (
    selected_metadata
    .copy()
)


plt.figure(
    figsize=(
        5,
        6,
    )
)


ax = sns.boxplot(
    data=selection_plot_df,
    x="group",
    y="expression_senescence_score",
    order=[
        "Non-senescent-like",
        "Senescence-like",
    ],
    showfliers=False,
)


sns.stripplot(
    data=selection_plot_df,
    x="group",
    y="expression_senescence_score",
    order=[
        "Non-senescent-like",
        "Senescence-like",
    ],
    size=5,
    alpha=0.7,
    ax=ax,
)


plt.xlabel(
    ""
)

plt.ylabel(
    "Independent expression-based\nsenescence selection score"
)


plt.xticks(
    rotation=15,
)


plt.title(
    "Cell lines used for validation"
)


plt.tight_layout()


plt.savefig(
    OUTDIR /
    "selection_score_groups.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# Done
# ============================================================

print("\n========================================")
print("DONE")
print("========================================")

print(
    "\nResults written to:"
)

print(
    OUTDIR.resolve()
)

print(
    "\nMain outputs:"
)

print(
    "  senescence_regulon_heatmap.png"
)

print(
    "  senescence_regulon_boxplot.png"
)

print(
    "  regulon_group_statistics.csv"
)

print(
    "  selected_senescence_cell_lines.csv"
)

print(
    "  selected40_regulon_scores.csv"
)
