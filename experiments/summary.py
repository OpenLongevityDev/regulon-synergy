import hashlib
import json
from pathlib import Path
import os

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

# Repository root. Override with DRUG_SYNERGY_ROOT when data live elsewhere.
ROOT = Path(
    os.environ.get("DRUG_SYNERGY_ROOT", Path(__file__).resolve().parents[1])
).resolve()

# External source data are expected under input/ and are not distributed here.
INPUT_ROOT = Path(
    os.environ.get("DRUG_SYNERGY_INPUT_ROOT", ROOT / "input")
).resolve()

INPUT = INPUT_ROOT / "experiments" / "DrugComb_summary.tsv"
OUTDIR = ROOT / "experiments" / "drugcomb_master"

MASTER_FILE = OUTDIR / "drugcomb_training_master.tsv"
PAIR_CONTEXT_FILE = OUTDIR / "drugcomb_pair_context_master.tsv"
SUMMARY_FILE = OUTDIR / "run_summary.json"


# ============================================================
# Settings
# ============================================================

ZIP_THRESHOLD = 10.0

SCORE_COLS = [
    "synergy_zip",
    "synergy_loewe",
    "synergy_hsa",
    "synergy_bliss",
]


# ============================================================
# Helper functions
# ============================================================

def zip_class(x, threshold=10.0):

    if pd.isna(x):
        return np.nan

    if x < -threshold:
        return "antagonism"

    if x > threshold:
        return "synergy"

    return "no_interaction"


def stable_hash(*values, prefix=""):

    text = "||".join(
        str(x)
        for x in values
    )

    digest = hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()[:16]

    if prefix:
        return f"{prefix}_{digest}"

    return digest


def unique_join(values):

    vals = (
        pd.Series(values)
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return ";".join(
        sorted(vals)
    )


# ============================================================
# Load
# ============================================================

print("Loading DrugComb summary...")

df = pd.read_csv(
    INPUT,
    sep="\t",
    na_values=[
        "\\N",
        "NA",
        "NaN",
        "",
    ],
    low_memory=False,
)

print(
    "Raw shape:",
    df.shape
)


# ============================================================
# Numeric score conversion
# ============================================================

for col in SCORE_COLS:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


if df["synergy_zip"].isna().any():

    raise ValueError(
        "Some rows are missing ZIP scores. "
        "ZIP is required for the primary label."
    )


# ============================================================
# Preserve original drug ordering
# ============================================================

df["drug_1_original"] = (
    df["drug_1"]
    .astype(str)
    .str.strip()
)

df["drug_2_original"] = (
    df["drug_2"]
    .astype(str)
    .str.strip()
)


# ============================================================
# Canonicalize drug pair orientation
#
# For now this is NAME-based.
#
# Later, after mapping to DrugComb IDs, we will re-canonicalize
# using stable numeric IDs.
# ============================================================

d1 = df["drug_1_original"]
d2 = df["drug_2_original"]

swap = (
    d1.str.casefold()
    >
    d2.str.casefold()
)

df["drug_a_name"] = np.where(
    swap,
    d2,
    d1
)

df["drug_b_name"] = np.where(
    swap,
    d1,
    d2
)


# ============================================================
# Raw ZIP class
#
# This is used only to measure replicate class consistency.
# Final label will come from median ZIP per experimental unit.
# ============================================================

df["raw_zip_class"] = (
    df["synergy_zip"]
    .apply(
        lambda x: zip_class(
            x,
            threshold=ZIP_THRESHOLD
        )
    )
)


# ============================================================
# Build observational units
#
# one unit =
# drug A × drug B × cell line × study
# ============================================================

GROUP_COLS = [
    "drug_a_name",
    "drug_b_name",
    "cell_line",
    "study_name",
]


print(
    "\nCollapsing replicate measurements "
    "within pair × cell line × study..."
)


master = (
    df.groupby(
        GROUP_COLS,
        dropna=False,
        sort=False,
    )
    .agg(

        # --------------------------------------------
        # Tissue
        # --------------------------------------------

        tissue=(
            "tissue",
            unique_join
        ),

        n_tissues=(
            "tissue",
            "nunique"
        ),

        # --------------------------------------------
        # Replication
        # --------------------------------------------

        n_measurements=(
            "synergy_zip",
            "size"
        ),

        # --------------------------------------------
        # ZIP
        # --------------------------------------------

        zip_median=(
            "synergy_zip",
            "median"
        ),

        zip_mean=(
            "synergy_zip",
            "mean"
        ),

        zip_std=(
            "synergy_zip",
            "std"
        ),

        zip_min=(
            "synergy_zip",
            "min"
        ),

        zip_max=(
            "synergy_zip",
            "max"
        ),

        # --------------------------------------------
        # Other metrics retained only as metadata/QC
        # --------------------------------------------

        loewe_median=(
            "synergy_loewe",
            "median"
        ),

        hsa_median=(
            "synergy_hsa",
            "median"
        ),

        bliss_median=(
            "synergy_bliss",
            "median"
        ),

        # --------------------------------------------
        # Replicate class consistency
        # --------------------------------------------

        n_raw_zip_classes=(
            "raw_zip_class",
            "nunique"
        ),

    )
    .reset_index()
)


# ============================================================
# Derived QC variables
# ============================================================

master["zip_range"] = (
    master["zip_max"]
    -
    master["zip_min"]
)

master["zip_std"] = (
    master["zip_std"]
    .fillna(0.0)
)

master["replicate_class_consistent"] = (
    master["n_raw_zip_classes"] == 1
)

master["tissue_consistent"] = (
    master["n_tissues"] == 1
)


# ============================================================
# Final ZIP class
# ============================================================

master["label_3class"] = (
    master["zip_median"]
    .apply(
        lambda x: zip_class(
            x,
            threshold=ZIP_THRESHOLD
        )
    )
)


# ============================================================
# Stable IDs
# ============================================================

# pair_context_id:
# same drug pair + cell line regardless of study

master["pair_context_id"] = [
    stable_hash(
        a,
        b,
        c,
        prefix="PC"
    )
    for a, b, c
    in zip(
        master["drug_a_name"],
        master["drug_b_name"],
        master["cell_line"],
    )
]


# sample_id:
# specific observational unit including study

master["sample_id"] = [
    stable_hash(
        a,
        b,
        c,
        s,
        prefix="S"
    )
    for a, b, c, s
    in zip(
        master["drug_a_name"],
        master["drug_b_name"],
        master["cell_line"],
        master["study_name"],
    )
]


# ============================================================
# Add study-count information
#
# Number of distinct studies represented for this
# biological drug pair × context.
# ============================================================

n_studies = (
    master.groupby(
        "pair_context_id"
    )["study_name"]
    .nunique()
)

master["n_studies_for_pair_context"] = (
    master["pair_context_id"]
    .map(n_studies)
)


# ============================================================
# Column order
# ============================================================

cols = [

    # IDs
    "sample_id",
    "pair_context_id",

    # Drug pair
    "drug_a_name",
    "drug_b_name",

    # Biological / experimental context
    "cell_line",
    "study_name",
    "tissue",

    # Primary label
    "zip_median",
    "label_3class",

    # Replicate QC
    "n_measurements",
    "zip_mean",
    "zip_std",
    "zip_min",
    "zip_max",
    "zip_range",
    "replicate_class_consistent",

    # Study/context information
    "n_studies_for_pair_context",

    # Tissue QC
    "n_tissues",
    "tissue_consistent",

    # Alternative synergy metrics — metadata only
    "loewe_median",
    "hsa_median",
    "bliss_median",
]

master = master[
    cols
].copy()


# ============================================================
# Sanity checks
# ============================================================

assert (
    master["sample_id"]
    .is_unique
)

assert (
    master["sample_id"]
    .notna()
    .all()
)

assert (
    master["pair_context_id"]
    .notna()
    .all()
)

assert (
    master["zip_median"]
    .notna()
    .all()
)

assert (
    master["label_3class"]
    .notna()
    .all()
)


# ============================================================
# Pair-context summary table
#
# One row = drug pair × cell line
#
# IMPORTANT:
# This is NOT a training-label table.
# We do not collapse study-specific labels here.
#
# It is mainly useful for:
# - grouped CV
# - modality mapping
# - counting contexts
# ============================================================

pair_context = (
    master.groupby(
        [
            "pair_context_id",
            "drug_a_name",
            "drug_b_name",
            "cell_line",
        ],
        as_index=False,
        sort=False,
    )
    .agg(

        tissue=(
            "tissue",
            unique_join
        ),

        n_studies=(
            "study_name",
            "nunique"
        ),

        n_samples=(
            "sample_id",
            "size"
        ),

        total_raw_measurements=(
            "n_measurements",
            "sum"
        ),

        zip_median_min=(
            "zip_median",
            "min"
        ),

        zip_median_max=(
            "zip_median",
            "max"
        ),

        n_study_level_classes=(
            "label_3class",
            "nunique"
        ),
    )
)


pair_context[
    "study_label_consistent"
] = (
    pair_context[
        "n_study_level_classes"
    ]
    == 1
)

pair_context[
    "zip_range_across_studies"
] = (
    pair_context[
        "zip_median_max"
    ]
    -
    pair_context[
        "zip_median_min"
    ]
)


# ============================================================
# Save
# ============================================================

OUTDIR.mkdir(parents=True, exist_ok=True)

master.to_csv(
    MASTER_FILE,
    sep="\t",
    index=False,
)

pair_context.to_csv(
    PAIR_CONTEXT_FILE,
    sep="\t",
    index=False,
)


# ============================================================
# Summary
# ============================================================

label_counts = (
    master["label_3class"]
    .value_counts()
)

label_percent = (
    master["label_3class"]
    .value_counts(
        normalize=True
    )
    * 100
)


summary = {

    "input_file":
        str(INPUT.relative_to(ROOT))
        if INPUT.is_relative_to(ROOT)
        else str(INPUT),

    "zip_threshold":
        ZIP_THRESHOLD,

    "raw_rows":
        int(len(df)),

    "training_samples":
        int(len(master)),

    "unique_pair_contexts":
        int(
            master[
                "pair_context_id"
            ].nunique()
        ),

    "unique_drug_a_names":
        int(
            master[
                "drug_a_name"
            ].nunique()
        ),

    "unique_drug_b_names":
        int(
            master[
                "drug_b_name"
            ].nunique()
        ),

    "unique_drugs_total":
        int(
            len(
                set(
                    master[
                        "drug_a_name"
                    ]
                )
                |
                set(
                    master[
                        "drug_b_name"
                    ]
                )
            )
        ),

    "unique_cell_lines":
        int(
            master[
                "cell_line"
            ].nunique()
        ),

    "unique_studies":
        int(
            master[
                "study_name"
            ].nunique()
        ),

    "label_counts": {
        str(k): int(v)
        for k, v
        in label_counts.items()
    },

    "label_percent": {
        str(k): float(v)
        for k, v
        in label_percent.items()
    },

    "replicated_samples":
        int(
            (
                master[
                    "n_measurements"
                ]
                > 1
            ).sum()
        ),

    "replicate_class_inconsistent":
        int(
            (
                ~master[
                    "replicate_class_consistent"
                ]
            ).sum()
        ),

    "pair_contexts_multi_study":
        int(
            (
                pair_context[
                    "n_studies"
                ]
                > 1
            ).sum()
        ),

    "multi_study_pair_contexts_with_label_conflict":
        int(
            (
                (
                    pair_context[
                        "n_studies"
                    ]
                    > 1
                )
                &
                (
                    ~pair_context[
                        "study_label_consistent"
                    ]
                )
            ).sum()
        ),
}


with open(
    SUMMARY_FILE,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# Print results
# ============================================================

print(
    "\n========================================"
)

print(
    "DRUGCOMB MASTER TABLE COMPLETE"
)

print(
    "========================================"
)


print(
    "\nRaw rows:",
    f"{len(df):,}"
)

print(
    "Training samples:",
    f"{len(master):,}"
)

print(
    "Unique pair × cell contexts:",
    f"{master['pair_context_id'].nunique():,}"
)

print(
    "Unique drugs:",
    f"{summary['unique_drugs_total']:,}"
)

print(
    "Unique cell lines:",
    f"{master['cell_line'].nunique():,}"
)

print(
    "Unique studies:",
    f"{master['study_name'].nunique():,}"
)


print(
    "\n3-class ZIP label distribution:"
)

label_table = pd.DataFrame({
    "n": label_counts,
    "percent": label_percent.round(2),
})

print(
    label_table
)


print(
    "\nReplicated observational samples:",
    f"{summary['replicated_samples']:,}"
)

print(
    "Replicated samples with class disagreement:",
    f"{summary['replicate_class_inconsistent']:,}"
)


print(
    "\nMulti-study pair-contexts:",
    f"{summary['pair_contexts_multi_study']:,}"
)

print(
    "Multi-study pair-contexts with label disagreement:",
    f"{summary['multi_study_pair_contexts_with_label_conflict']:,}"
)


print(
    "\nSaved:"
)

print(
    MASTER_FILE
)

print(
    PAIR_CONTEXT_FILE
)

print(
    SUMMARY_FILE
)
