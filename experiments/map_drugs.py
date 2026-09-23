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
MASTER_FILE = OUTDIR / "drugcomb_training_master.tsv"
PAIR_CONTEXT_FILE = OUTDIR / "drugcomb_pair_context_master.tsv"
DRUG_FILE = INPUT_ROOT / "drugs" / "DrugComb_drug_identifiers.tsv"


OUT_MASTER = (
    OUTDIR
    / "drugcomb_training_master_drug_ids.tsv"
)

OUT_PAIR_CONTEXT = (
    OUTDIR
    / "drugcomb_pair_context_master_drug_ids.tsv"
)

OUT_DRUG_LOOKUP = (
    OUTDIR
    / "drug_name_to_drugcomb_id.tsv"
)

OUT_UNMAPPED = (
    OUTDIR
    / "unmapped_drug_names.tsv"
)

OUT_AMBIGUOUS = (
    OUTDIR
    / "ambiguous_drug_names.tsv"
)

OUT_SUMMARY = (
    OUTDIR
    / "drug_mapping_summary.json"
)


# ============================================================
# Helpers
# ============================================================

def normalize_name(x):
    """
    Conservative normalization.

    We intentionally do NOT:
      - fuzzy match
      - remove punctuation broadly
      - rewrite chemical names
      - infer synonyms

    This normalization only handles superficial formatting.
    """

    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    # Unicode/case normalization
    x = x.casefold()

    # Collapse repeated whitespace
    x = re.sub(
        r"\s+",
        " ",
        x
    )

    # Normalize spacing around commas
    x = re.sub(
        r"\s*,\s*",
        ",",
        x
    )

    # Normalize spacing around hyphens
    x = re.sub(
        r"\s*-\s*",
        "-",
        x
    )

    return x


# ============================================================
# Load
# ============================================================

print(
    "Loading training master..."
)

master = pd.read_csv(
    MASTER_FILE,
    sep="\t",
    low_memory=False
)

print(
    "Training master:",
    master.shape
)


print(
    "\nLoading DrugComb drug identifiers..."
)

drug_ref = pd.read_csv(
    DRUG_FILE,
    sep="\t",
    low_memory=False
)

print(
    "Drug reference:",
    drug_ref.shape
)

print(
    "\nOriginal drug reference columns:"
)

print(
    drug_ref.columns.tolist()
)


# ============================================================
# Standardize DrugComb ID column name
# ============================================================

if "drugcomb_id" not in drug_ref.columns:

    if "id" in drug_ref.columns:

        drug_ref = drug_ref.rename(
            columns={
                "id": "drugcomb_id"
            }
        )

        print(
            "\nUsing column 'id' as 'drugcomb_id'."
        )

    else:

        raise ValueError(
            "Could not find DrugComb ID column. "
            "Expected either 'drugcomb_id' or 'id'."
        )


print(
    "\nStandardized drug reference columns:"
)

print(
    drug_ref.columns.tolist()
)

# ============================================================
# Validate required columns
# ============================================================

required_master = {
    "drug_a_name",
    "drug_b_name",
}

missing = (
    required_master
    - set(master.columns)
)

if missing:

    raise ValueError(
        f"Training master missing columns: {missing}"
    )


required_drug = {
    "drugcomb_id",
    "dname",
}

missing = (
    required_drug
    - set(drug_ref.columns)
)

if missing:

    raise ValueError(
        f"Drug reference missing columns: {missing}"
    )


# ============================================================
# Clean DrugComb reference
# ============================================================

drug_ref = (
    drug_ref[
        [
            "drugcomb_id",
            "dname",
        ]
    ]
    .dropna(
        subset=[
            "drugcomb_id",
            "dname",
        ]
    )
    .copy()
)


drug_ref["dname"] = (
    drug_ref["dname"]
    .astype(str)
    .str.strip()
)

drug_ref[
    "normalized_name"
] = (
    drug_ref["dname"]
    .map(normalize_name)
)


# ============================================================
# Inspect ID/name structure
# ============================================================

print(
    "\nUnique DrugComb IDs:",
    f"{drug_ref['drugcomb_id'].nunique():,}"
)

print(
    "Unique drug names:",
    f"{drug_ref['dname'].nunique():,}"
)

print(
    "Unique normalized names:",
    f"{drug_ref['normalized_name'].nunique():,}"
)


# ============================================================
# Detect names mapping to multiple IDs
# ============================================================

name_id_counts = (
    drug_ref.groupby(
        "normalized_name"
    )["drugcomb_id"]
    .nunique()
)

ambiguous_norm_names = set(
    name_id_counts[
        name_id_counts > 1
    ].index
)

print(
    "\nNormalized names mapping to >1 DrugComb ID:",
    f"{len(ambiguous_norm_names):,}"
)


# ============================================================
# Build unambiguous lookup
# ============================================================

unambiguous_ref = (
    drug_ref[
        ~drug_ref[
            "normalized_name"
        ].isin(
            ambiguous_norm_names
        )
    ]
    .drop_duplicates(
        subset=[
            "normalized_name"
        ]
    )
    .copy()
)


lookup_id = dict(
    zip(
        unambiguous_ref[
            "normalized_name"
        ],
        unambiguous_ref[
            "drugcomb_id"
        ],
    )
)

lookup_canonical_name = dict(
    zip(
        unambiguous_ref[
            "normalized_name"
        ],
        unambiguous_ref[
            "dname"
        ],
    )
)


# ============================================================
# All unique drugs present in labelled dataset
# ============================================================

all_training_drugs = sorted(
    set(
        master[
            "drug_a_name"
        ].dropna()
    )
    |
    set(
        master[
            "drug_b_name"
        ].dropna()
    )
)

drug_lookup = pd.DataFrame({
    "training_name":
        all_training_drugs
})

drug_lookup[
    "normalized_name"
] = (
    drug_lookup[
        "training_name"
    ]
    .map(
        normalize_name
    )
)


# ============================================================
# Map names
# ============================================================

drug_lookup[
    "drugcomb_id"
] = (
    drug_lookup[
        "normalized_name"
    ]
    .map(
        lookup_id
    )
)

drug_lookup[
    "drugcomb_reference_name"
] = (
    drug_lookup[
        "normalized_name"
    ]
    .map(
        lookup_canonical_name
    )
)


# ============================================================
# Mapping status
# ============================================================

drug_lookup[
    "mapping_status"
] = "unmapped"


drug_lookup.loc[
    drug_lookup[
        "drugcomb_id"
    ].notna(),
    "mapping_status"
] = "mapped"


drug_lookup.loc[
    drug_lookup[
        "normalized_name"
    ].isin(
        ambiguous_norm_names
    ),
    "mapping_status"
] = "ambiguous"


# ============================================================
# Distinguish exact from normalized matching
# ============================================================

exact_pairs = set(
    zip(
        drug_ref[
            "dname"
        ],
        drug_ref[
            "drugcomb_id"
        ],
    )
)


def determine_match_type(row):

    if row[
        "mapping_status"
    ] != "mapped":

        return row[
            "mapping_status"
        ]

    pair = (
        row[
            "training_name"
        ],
        row[
            "drugcomb_id"
        ],
    )

    if pair in exact_pairs:
        return "exact"

    return "normalized"


drug_lookup[
    "match_type"
] = (
    drug_lookup.apply(
        determine_match_type,
        axis=1
    )
)


# ============================================================
# Save lookup tables
# ============================================================

OUTDIR.mkdir(parents=True, exist_ok=True)

drug_lookup.to_csv(
    OUT_DRUG_LOOKUP,
    sep="\t",
    index=False
)


drug_lookup[
    drug_lookup[
        "mapping_status"
    ]
    == "unmapped"
].to_csv(
    OUT_UNMAPPED,
    sep="\t",
    index=False
)


# Full details for ambiguous names
ambiguous_details = (
    drug_ref[
        drug_ref[
            "normalized_name"
        ].isin(
            ambiguous_norm_names
        )
    ]
    .merge(
        drug_lookup[
            [
                "training_name",
                "normalized_name",
            ]
        ],
        on="normalized_name",
        how="inner",
    )
    .sort_values(
        [
            "training_name",
            "drugcomb_id",
        ]
    )
)


ambiguous_details.to_csv(
    OUT_AMBIGUOUS,
    sep="\t",
    index=False
)


# ============================================================
# Mapping dictionaries
# ============================================================

training_to_id = dict(
    zip(
        drug_lookup[
            "training_name"
        ],
        drug_lookup[
            "drugcomb_id"
        ],
    )
)

training_to_refname = dict(
    zip(
        drug_lookup[
            "training_name"
        ],
        drug_lookup[
            "drugcomb_reference_name"
        ],
    )
)

training_to_status = dict(
    zip(
        drug_lookup[
            "training_name"
        ],
        drug_lookup[
            "mapping_status"
        ],
    )
)


# ============================================================
# Add IDs to master table
# ============================================================

master[
    "drug_a_id"
] = (
    master[
        "drug_a_name"
    ]
    .map(
        training_to_id
    )
)

master[
    "drug_b_id"
] = (
    master[
        "drug_b_name"
    ]
    .map(
        training_to_id
    )
)


master[
    "drug_a_mapping_status"
] = (
    master[
        "drug_a_name"
    ]
    .map(
        training_to_status
    )
)

master[
    "drug_b_mapping_status"
] = (
    master[
        "drug_b_name"
    ]
    .map(
        training_to_status
    )
)


# ============================================================
# Sample-level mapping status
# ============================================================

master[
    "both_drugs_mapped"
] = (
    master[
        "drug_a_id"
    ].notna()
    &
    master[
        "drug_b_id"
    ].notna()
)


# ============================================================
# Re-canonicalize pair using stable IDs
#
# IMPORTANT:
# Only where both IDs are mapped.
#
# This gives us the permanent orientation that feature matrices
# should eventually use.
# ============================================================

both = (
    master[
        "both_drugs_mapped"
    ]
)


a_id = (
    master.loc[
        both,
        "drug_a_id"
    ]
)

b_id = (
    master.loc[
        both,
        "drug_b_id"
    ]
)


# Try numeric comparison if DrugComb IDs are numeric.
a_num = pd.to_numeric(
    a_id,
    errors="coerce"
)

b_num = pd.to_numeric(
    b_id,
    errors="coerce"
)


if (
    a_num.notna().all()
    and
    b_num.notna().all()
):

    swap_id = (
        a_num > b_num
    )

else:

    swap_id = (
        a_id.astype(str)
        >
        b_id.astype(str)
    )


idx = (
    master.index[
        both
    ]
)


swap_idx = (
    idx[
        swap_id.to_numpy()
    ]
)


# Store temporary copies
tmp_a_id = (
    master.loc[
        swap_idx,
        "drug_a_id"
    ].copy()
)

tmp_b_id = (
    master.loc[
        swap_idx,
        "drug_b_id"
    ].copy()
)

tmp_a_name = (
    master.loc[
        swap_idx,
        "drug_a_name"
    ].copy()
)

tmp_b_name = (
    master.loc[
        swap_idx,
        "drug_b_name"
    ].copy()
)


master.loc[
    swap_idx,
    "drug_a_id"
] = tmp_b_id.values

master.loc[
    swap_idx,
    "drug_b_id"
] = tmp_a_id.values

master.loc[
    swap_idx,
    "drug_a_name"
] = tmp_b_name.values

master.loc[
    swap_idx,
    "drug_b_name"
] = tmp_a_name.values


# ============================================================
# Save updated training master
# ============================================================

master.to_csv(
    OUT_MASTER,
    sep="\t",
    index=False
)


# ============================================================
# Update pair-context table too
# ============================================================

pair_context = pd.read_csv(
    PAIR_CONTEXT_FILE,
    sep="\t",
    low_memory=False
)


pair_context[
    "drug_a_id"
] = (
    pair_context[
        "drug_a_name"
    ]
    .map(
        training_to_id
    )
)

pair_context[
    "drug_b_id"
] = (
    pair_context[
        "drug_b_name"
    ]
    .map(
        training_to_id
    )
)

pair_context[
    "both_drugs_mapped"
] = (
    pair_context[
        "drug_a_id"
    ].notna()
    &
    pair_context[
        "drug_b_id"
    ].notna()
)


pair_context.to_csv(
    OUT_PAIR_CONTEXT,
    sep="\t",
    index=False
)


# ============================================================
# Summary statistics
# ============================================================

n_unique = len(
    drug_lookup
)

n_mapped = (
    drug_lookup[
        "mapping_status"
    ]
    .eq(
        "mapped"
    )
    .sum()
)

n_exact = (
    drug_lookup[
        "match_type"
    ]
    .eq(
        "exact"
    )
    .sum()
)

n_normalized = (
    drug_lookup[
        "match_type"
    ]
    .eq(
        "normalized"
    )
    .sum()
)

n_unmapped = (
    drug_lookup[
        "mapping_status"
    ]
    .eq(
        "unmapped"
    )
    .sum()
)

n_ambiguous = (
    drug_lookup[
        "mapping_status"
    ]
    .eq(
        "ambiguous"
    )
    .sum()
)


samples_both = int(
    master[
        "both_drugs_mapped"
    ].sum()
)

samples_total = len(
    master
)


summary = {

    "training_unique_drugs":
        int(
            n_unique
        ),

    "mapped_unique_drugs":
        int(
            n_mapped
        ),

    "exact_matches":
        int(
            n_exact
        ),

    "normalized_matches":
        int(
            n_normalized
        ),

    "unmapped_unique_drugs":
        int(
            n_unmapped
        ),

    "ambiguous_unique_drugs":
        int(
            n_ambiguous
        ),

    "unique_drug_mapping_percent":
        float(
            100
            *
            n_mapped
            /
            n_unique
        ),

    "training_samples":
        int(
            samples_total
        ),

    "samples_with_both_drugs_mapped":
        int(
            samples_both
        ),

    "sample_mapping_percent":
        float(
            100
            *
            samples_both
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
    "DRUG MAPPING COMPLETE"
)

print(
    "========================================"
)


print(
    "\nUnique drugs in labelled dataset:",
    f"{n_unique:,}"
)

print(
    "Mapped:",
    f"{n_mapped:,}",
    f"({100*n_mapped/n_unique:.2f}%)"
)

print(
    "  exact:",
    f"{n_exact:,}"
)

print(
    "  normalized:",
    f"{n_normalized:,}"
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
    "Samples with BOTH drugs mapped:",
    f"{samples_both:,}",
    f"({100*samples_both/samples_total:.2f}%)"
)


print(
    "\nMapping status counts:"
)

print(
    drug_lookup[
        "match_type"
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
    OUT_PAIR_CONTEXT,
    OUT_DRUG_LOOKUP,
    OUT_UNMAPPED,
    OUT_AMBIGUOUS,
    OUT_SUMMARY,
]:

    print(
        x
    )
