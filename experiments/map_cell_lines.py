from pathlib import Path
import os
import json
import re

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

OUTDIR = ROOT / "experiments" / "drugcomb_master"
MASTER_FILE = OUTDIR / "drugcomb_training_master_drug_ids.tsv"
NAMING_FILE = INPUT_ROOT / "cell_lines" / "depmap" / "naming.csv"

OUT_MASTER = OUTDIR / "drugcomb_training_master_drug_ids_cell_ids.tsv"
OUT_LOOKUP = OUTDIR / "cell_line_to_depmap_model.tsv"
OUT_UNMAPPED = OUTDIR / "unmapped_cell_lines.tsv"
OUT_AMBIGUOUS = OUTDIR / "ambiguous_cell_lines.tsv"
OUT_SUMMARY = OUTDIR / "cell_line_mapping_summary.json"


# ============================================================
# Helpers
# ============================================================

def normalize_cell_name(x):
    """
    Conservative normalization:
      MDA-MB-231 -> mdamb231
      MDA MB 231 -> mdamb231
      LOX IMVI   -> loximvi
    """

    if pd.isna(x):
        return np.nan

    x = str(x).strip().casefold()

    # Remove superficial separators only
    x = re.sub(
        r"[\s\-_./]+",
        "",
        x
    )

    return x


def ccle_base_name(x):
    """
    Extract approximate cell-line portion from CCLE-style name.

    Examples:
      NIHOVCAR3_OVARY -> NIHOVCAR3
      CACO2_LARGE_INTESTINE -> CACO2
      A101D_SKIN -> A101D

    This uses the first underscore-separated token as a candidate.
    We keep the full CCLE name as an independent candidate too.
    """

    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    if "_" in x:
        return x.split("_")[0]

    return x


# ============================================================
# Load training master
# ============================================================

print("Loading training master...")

master = pd.read_csv(
    MASTER_FILE,
    sep="\t",
    low_memory=False
)

print(
    "Training master:",
    master.shape
)


# ============================================================
# Load DepMap naming table
# ============================================================

print("\nLoading DepMap naming table...")

naming = pd.read_csv(
    NAMING_FILE,
    low_memory=False
)

print(
    "Naming table:",
    naming.shape
)

print(
    "\nNaming columns:"
)

print(
    naming.columns.tolist()
)


required = {
    "ccle_name",
    "canonical_ccle_name",
    "broad_id",
}

missing = required - set(
    naming.columns
)

if missing:

    raise ValueError(
        f"naming.csv missing required columns: {missing}"
    )


# ============================================================
# Clean naming table
# ============================================================

for col in [
    "ccle_name",
    "canonical_ccle_name",
    "broad_id",
]:

    naming[col] = (
        naming[col]
        .astype(str)
        .str.strip()
    )


# ============================================================
# Build candidate aliases
# ============================================================

candidate_rows = []


def add_candidate(name, broad_id, source):

    if pd.isna(name):
        return

    name = str(name).strip()

    if not name:
        return

    candidate_rows.append({
        "depmap_name": name,
        "normalized_name": normalize_cell_name(name),
        "broad_id": broad_id,
        "name_source": source,
    })


for _, row in naming.iterrows():

    broad_id = row[
        "broad_id"
    ]

    # full CCLE alias
    add_candidate(
        row["ccle_name"],
        broad_id,
        "ccle_name_full"
    )

    # full canonical alias
    add_candidate(
        row["canonical_ccle_name"],
        broad_id,
        "canonical_full"
    )

    # base name from CCLE alias
    add_candidate(
        ccle_base_name(
            row["ccle_name"]
        ),
        broad_id,
        "ccle_base"
    )

    # base name from canonical alias
    add_candidate(
        ccle_base_name(
            row["canonical_ccle_name"]
        ),
        broad_id,
        "canonical_base"
    )


candidates = (
    pd.DataFrame(
        candidate_rows
    )
    .dropna(
        subset=[
            "normalized_name",
            "broad_id",
        ]
    )
    .drop_duplicates()
)


print(
    "\nTotal candidate aliases:",
    f"{len(candidates):,}"
)


# ============================================================
# Find ambiguous aliases
# ============================================================

alias_counts = (
    candidates.groupby(
        "normalized_name"
    )["broad_id"]
    .nunique()
)

ambiguous_aliases = set(
    alias_counts[
        alias_counts > 1
    ].index
)

print(
    "Ambiguous normalized aliases:",
    f"{len(ambiguous_aliases):,}"
)


# ============================================================
# Keep only unambiguous aliases
# ============================================================

unique_candidates = (
    candidates[
        ~candidates[
            "normalized_name"
        ].isin(
            ambiguous_aliases
        )
    ]
    .copy()
)


# Prioritize base names slightly over full names
# for matching simple DrugComb names.
priority = {
    "canonical_base": 0,
    "ccle_base": 1,
    "canonical_full": 2,
    "ccle_name_full": 3,
}

unique_candidates[
    "_priority"
] = (
    unique_candidates[
        "name_source"
    ]
    .map(priority)
    .fillna(99)
)


unique_candidates = (
    unique_candidates
    .sort_values(
        [
            "normalized_name",
            "_priority",
        ]
    )
    .drop_duplicates(
        subset=[
            "normalized_name"
        ]
    )
)


lookup_id = dict(
    zip(
        unique_candidates[
            "normalized_name"
        ],
        unique_candidates[
            "broad_id"
        ],
    )
)

lookup_name = dict(
    zip(
        unique_candidates[
            "normalized_name"
        ],
        unique_candidates[
            "depmap_name"
        ],
    )
)

lookup_source = dict(
    zip(
        unique_candidates[
            "normalized_name"
        ],
        unique_candidates[
            "name_source"
        ],
    )
)


# ============================================================
# Unique DrugComb cell lines
# ============================================================

cell_lookup = pd.DataFrame({
    "drugcomb_cell_line":
        sorted(
            master[
                "cell_line"
            ]
            .dropna()
            .astype(str)
            .unique()
        )
})


cell_lookup[
    "normalized_name"
] = (
    cell_lookup[
        "drugcomb_cell_line"
    ]
    .map(
        normalize_cell_name
    )
)


# ============================================================
# Map
# ============================================================

cell_lookup[
    "ModelID"
] = (
    cell_lookup[
        "normalized_name"
    ]
    .map(
        lookup_id
    )
)

cell_lookup[
    "depmap_matched_name"
] = (
    cell_lookup[
        "normalized_name"
    ]
    .map(
        lookup_name
    )
)

cell_lookup[
    "depmap_name_source"
] = (
    cell_lookup[
        "normalized_name"
    ]
    .map(
        lookup_source
    )
)


# ============================================================
# Status
# ============================================================

cell_lookup[
    "mapping_status"
] = "unmapped"


cell_lookup.loc[
    cell_lookup[
        "ModelID"
    ].notna(),
    "mapping_status"
] = "mapped"


cell_lookup.loc[
    cell_lookup[
        "normalized_name"
    ].isin(
        ambiguous_aliases
    ),
    "mapping_status"
] = "ambiguous"


# ============================================================
# Save diagnostics
# ============================================================

OUTDIR.mkdir(parents=True, exist_ok=True)

cell_lookup.to_csv(
    OUT_LOOKUP,
    sep="\t",
    index=False
)


cell_lookup[
    cell_lookup[
        "mapping_status"
    ]
    == "unmapped"
].to_csv(
    OUT_UNMAPPED,
    sep="\t",
    index=False
)


ambiguous_details = (
    candidates[
        candidates[
            "normalized_name"
        ].isin(
            ambiguous_aliases
        )
    ]
    .merge(
        cell_lookup[
            [
                "drugcomb_cell_line",
                "normalized_name",
            ]
        ],
        on="normalized_name",
        how="inner"
    )
    .sort_values(
        [
            "drugcomb_cell_line",
            "broad_id",
        ]
    )
)


ambiguous_details.to_csv(
    OUT_AMBIGUOUS,
    sep="\t",
    index=False
)


# ============================================================
# Add IDs to master
# ============================================================

cell_to_model = dict(
    zip(
        cell_lookup[
            "drugcomb_cell_line"
        ],
        cell_lookup[
            "ModelID"
        ],
    )
)


cell_to_status = dict(
    zip(
        cell_lookup[
            "drugcomb_cell_line"
        ],
        cell_lookup[
            "mapping_status"
        ],
    )
)


master[
    "ModelID"
] = (
    master[
        "cell_line"
    ]
    .map(
        cell_to_model
    )
)


master[
    "cell_line_mapping_status"
] = (
    master[
        "cell_line"
    ]
    .map(
        cell_to_status
    )
)


master[
    "context_mapped"
] = (
    master[
        "ModelID"
    ].notna()
)


master[
    "core_ids_available"
] = (
    master[
        "both_drugs_mapped"
    ]
    &
    master[
        "context_mapped"
    ]
)


master.to_csv(
    OUT_MASTER,
    sep="\t",
    index=False
)


# ============================================================
# Summary
# ============================================================

n_total = len(
    cell_lookup
)

n_mapped = (
    cell_lookup[
        "mapping_status"
    ]
    .eq(
        "mapped"
    )
    .sum()
)

n_unmapped = (
    cell_lookup[
        "mapping_status"
    ]
    .eq(
        "unmapped"
    )
    .sum()
)

n_ambiguous = (
    cell_lookup[
        "mapping_status"
    ]
    .eq(
        "ambiguous"
    )
    .sum()
)


samples_total = len(
    master
)

samples_context = int(
    master[
        "context_mapped"
    ].sum()
)

samples_core = int(
    master[
        "core_ids_available"
    ].sum()
)


summary = {

    "unique_drugcomb_cell_lines":
        int(
            n_total
        ),

    "mapped_cell_lines":
        int(
            n_mapped
        ),

    "unmapped_cell_lines":
        int(
            n_unmapped
        ),

    "ambiguous_cell_lines":
        int(
            n_ambiguous
        ),

    "cell_line_mapping_percent":
        float(
            100
            *
            n_mapped
            /
            n_total
        ),

    "training_samples":
        int(
            samples_total
        ),

    "samples_with_context_mapped":
        int(
            samples_context
        ),

    "sample_context_mapping_percent":
        float(
            100
            *
            samples_context
            /
            samples_total
        ),

    "samples_with_both_drugs_and_context":
        int(
            samples_core
        ),

    "core_id_coverage_percent":
        float(
            100
            *
            samples_core
            /
            samples_total
        ),
}


with open(
    OUT_SUMMARY,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# Report
# ============================================================

print(
    "\n========================================"
)

print(
    "CELL-LINE MAPPING COMPLETE"
)

print(
    "========================================"
)


print(
    "\nUnique DrugComb cell lines:",
    f"{n_total:,}"
)

print(
    "Mapped:",
    f"{n_mapped:,}",
    f"({100*n_mapped/n_total:.2f}%)"
)

print(
    "Unmapped:",
    f"{n_unmapped:,}"
)

print(
    "Ambiguous:",
    f"{n_ambiguous:,}"
)


print(
    "\nTraining samples:",
    f"{samples_total:,}"
)

print(
    "Samples with context mapped:",
    f"{samples_context:,}",
    f"({100*samples_context/samples_total:.2f}%)"
)

print(
    "Samples with both drugs + context mapped:",
    f"{samples_core:,}",
    f"({100*samples_core/samples_total:.2f}%)"
)


print(
    "\nMatched alias sources:"
)

print(
    cell_lookup[
        "depmap_name_source"
    ]
    .value_counts(
        dropna=False
    )
)


print(
    "\nSaved:"
)

for x in [
    OUT_MASTER,
    OUT_LOOKUP,
    OUT_UNMAPPED,
    OUT_AMBIGUOUS,
    OUT_SUMMARY,
]:

    print(
        x
    )
