#!/usr/bin/env python3

"""
synergy_dataset.py

Build the model-ready DrugComb observation table and provide the PyTorch Dataset
used by the publication experiments.

The large model-ready tables are generated artifacts and do not need to be
distributed with the repository. Running this module reconstructs them from the
mapped DrugComb observation master and the generated drug/context feature files.

Inputs
------
Drug features:
- Morgan fingerprints
- direct target binary profiles
- fixed HuRI RWR profiles
- target/PPI availability state

Biological context:
- basal L1000 landmark expression
- Hallmark pathway activity
- regulon activity

Labels:
- DrugComb 3-class ZIP labels

Generated tables
----------------
- model_ready/drugcomb_model_ready_audit.tsv
- model_ready/drugcomb_model_ready_v1.tsv
- model_ready/run_summary.json

The audit table retains all observations plus feature-availability flags. The
strict V1 table contains only observations with every required feature.

Each Dataset sample returns tensors with keys expected by synergy_model.py:

    drug_a_morgan
    drug_a_targets
    drug_a_ppi
    drug_a_availability

    drug_b_morgan
    drug_b_targets
    drug_b_ppi
    drug_b_availability

    context_expression
    context_pathways
    context_regulons

    label

plus metadata:
    observation_id
    pair_context_id
    drug_a_id
    drug_b_id
    depmap_id
    study

IMPORTANT
---------
This v1 is intentionally strict:
- both drugs must have Morgan
- both drugs must exist in target/PPI matrices
- both drugs must have availability status
- cell line must exist in expression/pathway/regulon matrices
- no missing modality is silently zero-filled
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


# ============================================================
# Paths
# ============================================================

# Repository root. By default this is the directory containing this script.
# Set DRUG_SYNERGY_ROOT to point to another checkout/data root if needed.
ROOT = Path(
    os.environ.get("DRUG_SYNERGY_ROOT", Path(__file__).resolve().parent)
).resolve()

MASTER_PATH = (
    ROOT
    / "experiments/drugcomb_master/"
    / "drugcomb_training_master_drug_ids.tsv"
)

CELL_MAPPED_MASTER_PATH = (
    ROOT
    / "experiments/drugcomb_master/"
    / "drugcomb_training_master_drug_ids_cell_ids.tsv"
)


# ------------------------------------------------------------
# Drug features
# ------------------------------------------------------------

MORGAN_PATH = (
    ROOT
    / "smiles/smiles_results_morgan/"
    / "Morgan_ECFP4_2048.npz"
)

TARGET_DIR = (
    ROOT
    / "targets/target_ppi_results"
)

DIRECT_TARGET_PATH = (
    TARGET_DIR
    / "direct_target_binary.npz"
)

PPI_PATH = (
    TARGET_DIR
    / "ppi_propagated_rwr.npz"
)

TARGET_LOOKUP_PATH = (
    TARGET_DIR
    / "drug_target_ppi_lookup.tsv"
)


# ------------------------------------------------------------
# Biological context
# ------------------------------------------------------------

EXPRESSION_PATH = (
    ROOT
    / "expression/expression_results_l1000/"
    / "L1000_expression_TPMLogp1.csv"
)

PATHWAY_PATH = (
    ROOT
    / "pathways/pathway_results_hallmark_ulm/"
    / "Hallmark_activity_ULM.csv"
)

REGULON_PATH = (
    ROOT
    / "regulons/regulon_results_ulm/"
    / "TF_activity_fixed_collectri_ULM.csv"
)


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_DIR = (
    ROOT
    / "model_ready"
)

# OUTPUT_DIR is created only when the model-ready tables are generated.
# Importing this module for training therefore has no filesystem side effects.


# ============================================================
# Labels
# ============================================================

LABEL_TO_INT = {
    "antagonism": 0,
    "no_interaction": 1,
    "synergy": 2,
}

INT_TO_LABEL = {
    v: k
    for k, v in LABEL_TO_INT.items()
}


# ============================================================
# Utilities
# ============================================================

def print_section(title: str):

    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def decode_array(
    arr: np.ndarray,
) -> np.ndarray:

    arr = np.asarray(arr)

    if arr.dtype.kind in {
        "S",
        "O",
    }:

        return np.array(
            [
                (
                    x.decode("utf-8")
                    if isinstance(
                        x,
                        bytes,
                    )
                    else str(x)
                )
                for x in arr
            ],
            dtype=object,
        )

    return arr.astype(str)


def normalize_id(x):
    """
    Canonicalize IDs across pandas tables and NPZ archives.

    Important for DrugComb IDs because pandas may represent
    integer IDs as floats when missing values are present:

        1234 -> 1234.0

    while NPZ files may contain:

        "1234"

    This function converts integer-like numeric IDs to the same
    canonical string form while leaving IDs such as ACH-000681
    unchanged.
    """

    if x is None:
        return None

    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    # --------------------------------------------------------
    # Native integer types
    # --------------------------------------------------------

    if isinstance(
        x,
        (
            int,
            np.integer,
        ),
    ):

        return str(
            int(x)
        )


    # --------------------------------------------------------
    # Native float types
    # --------------------------------------------------------

    if isinstance(
        x,
        (
            float,
            np.floating,
        ),
    ):

        if not np.isfinite(x):
            return None

        if float(x).is_integer():

            return str(
                int(x)
            )

        return str(x).strip()


    # --------------------------------------------------------
    # String / everything else
    # --------------------------------------------------------

    x = str(x).strip()

    if (
        x == ""
        or x.casefold()
        in {
            "nan",
            "none",
            "null",
        }
    ):

        return None


    # --------------------------------------------------------
    # Integer-like decimal strings
    #
    # "1234.0" -> "1234"
    # --------------------------------------------------------

    try:

        numeric = float(x)

        if (
            np.isfinite(numeric)
            and numeric.is_integer()
            and "." in x
        ):

            return str(
                int(numeric)
            )

    except (
        ValueError,
        TypeError,
        OverflowError,
    ):

        pass


    return x


def find_column(
    df: pd.DataFrame,
    candidates: List[str],
    required: bool = False,
    label: Optional[str] = None,
) -> Optional[str]:

    lookup = {
        c.casefold(): c
        for c in df.columns
    }

    for candidate in candidates:

        if candidate.casefold() in lookup:

            return lookup[
                candidate.casefold()
            ]

    if required:

        raise ValueError(
            f"Could not find {label or 'column'}.\n"
            f"Tried: {candidates}\n"
            f"Available: {df.columns.tolist()}"
        )

    return None


# ============================================================
# NPZ loaders
# ============================================================

def load_morgan_npz(
    path: Path,
):

    data = np.load(
        path,
        allow_pickle=True,
    )

    if "X" not in data.files:

        raise ValueError(
            f"Morgan NPZ missing X. "
            f"Keys: {data.files}"
        )

    X = data[
        "X"
    ]

    drug_ids = decode_array(
        data[
            "drug_id"
        ]
    )

    if X.shape[0] != len(
        drug_ids
    ):

        raise ValueError(
            "Morgan row count != drug_id count."
        )

    index = {
        normalize_id(
            drug_id
        ): i
        for i, drug_id
        in enumerate(
            drug_ids
        )
        if normalize_id(
            drug_id
        ) is not None
    }

    return (
        X,
        drug_ids,
        index,
    )


def load_manual_csr_npz(
    path: Path,
):

    data = np.load(
        path,
        allow_pickle=True,
    )

    required = {
        "data",
        "indices",
        "indptr",
        "shape",
        "drug_id",
    }

    missing = (
        required.difference(
            data.files
        )
    )

    if missing:

        raise ValueError(
            f"CSR archive missing keys: {missing}"
        )

    shape = tuple(
        int(x)
        for x in data[
            "shape"
        ]
    )

    X = sp.csr_matrix(
        (
            data[
                "data"
            ],
            data[
                "indices"
            ],
            data[
                "indptr"
            ],
        ),
        shape=shape,
    )

    drug_ids = decode_array(
        data[
            "drug_id"
        ]
    )

    if X.shape[0] != len(
        drug_ids
    ):

        raise ValueError(
            "CSR row count != drug_id count."
        )

    index = {
        normalize_id(
            drug_id
        ): i
        for i, drug_id
        in enumerate(
            drug_ids
        )
        if normalize_id(
            drug_id
        ) is not None
    }

    return (
        X,
        drug_ids,
        index,
    )


def load_dense_npz(
    path: Path,
):

    data = np.load(
        path,
        allow_pickle=True,
    )

    if "X" not in data.files:

        raise ValueError(
            f"Dense NPZ missing X. "
            f"Keys: {data.files}"
        )

    X = data[
        "X"
    ]

    drug_ids = decode_array(
        data[
            "drug_id"
        ]
    )

    if X.shape[0] != len(
        drug_ids
    ):

        raise ValueError(
            "Dense row count != drug_id count."
        )

    index = {
        normalize_id(
            drug_id
        ): i
        for i, drug_id
        in enumerate(
            drug_ids
        )
        if normalize_id(
            drug_id
        ) is not None
    }

    return (
        X,
        drug_ids,
        index,
    )


# ============================================================
# Context loader
# ============================================================

def load_context_matrix(
    path: Path,
    label: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Dict[str, int],
    List[str],
]:

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    print(
        f"{label}: raw shape",
        df.shape,
    )

    model_id_col = find_column(
        df,
        [
            "ModelID",
            "model_id",
            "broad_id",
            "depmap_id",
        ],
    )


    if model_id_col is not None:

        ids = (
            df[
                model_id_col
            ]
            .map(
                normalize_id
            )
            .to_numpy()
        )

        feature_df = df.drop(
            columns=[
                model_id_col
            ]
        )


    else:

        first_col = (
            df.columns[
                0
            ]
        )

        if (
            first_col.startswith(
                "Unnamed"
            )
            or first_col
            in {
                "index",
                "",
            }
        ):

            ids = (
                df[
                    first_col
                ]
                .map(
                    normalize_id
                )
                .to_numpy()
            )

            feature_df = df.drop(
                columns=[
                    first_col
                ]
            )

        else:

            raise ValueError(
                f"Could not identify row ID column "
                f"for {label}.\n"
                f"First columns: "
                f"{df.columns[:10].tolist()}"
            )


    feature_names = (
        feature_df.columns
        .astype(str)
        .tolist()
    )


    X = feature_df.to_numpy(
        dtype=np.float32
    )


    if np.isnan(
        X
    ).any():

        n_nan = int(
            np.isnan(
                X
            ).sum()
        )

        raise ValueError(
            f"{label} contains "
            f"{n_nan:,} NaN values."
        )


    index = {
        model_id: i
        for i, model_id
        in enumerate(
            ids
        )
        if model_id is not None
    }


    n_nonmissing_ids = len(
        [
            x
            for x in ids
            if x is not None
        ]
    )


    if len(
        index
    ) != n_nonmissing_ids:

        raise ValueError(
            f"{label}: duplicated context IDs detected."
        )


    print(
        f"{label}: matrix",
        X.shape,
        "| IDs:",
        len(
            index
        ),
    )


    return (
        X,
        ids,
        index,
        feature_names,
    )


# ============================================================
# Target/PPI availability
# ============================================================

def build_availability_lookup(
    path: Path,
) -> Dict[
    str,
    np.ndarray,
]:

    lookup = pd.read_csv(
        path,
        sep="\t",
        low_memory=False,
    )


    print(
        "Target/PPI lookup:",
        lookup.shape,
    )


    id_col = find_column(
        lookup,
        [
            "drugcomb_id",
            "drug_id",
            "id",
        ],
        required=True,
        label="target lookup drug ID",
    )


    status_col = find_column(
        lookup,
        [
            "drug_target_status",
            "availability_status",
            "target_availability_status",
            "status",
            "mapping_status",
        ],
    )


    if status_col is None:

        likely = [
            c
            for c in lookup.columns
            if (
                "status"
                in c.casefold()
                or "availability"
                in c.casefold()
            )
        ]

        raise ValueError(
            "Could not identify target availability "
            "status column.\n"
            f"Possible columns: {likely}\n"
            f"All columns: "
            f"{lookup.columns.tolist()}"
        )


    print(
        "Drug ID column:",
        id_col,
    )

    print(
        "Availability status column:",
        status_col,
    )


    print(
        "\nAvailability status counts:"
    )

    print(
        lookup[
            status_col
        ]
        .value_counts(
            dropna=False
        )
    )


    status_to_vector = {

        "no_known_targets":
            np.array(
                [
                    1,
                    0,
                    0,
                ],
                dtype=np.float32,
            ),

        "targets_known_but_none_in_huri":
            np.array(
                [
                    0,
                    1,
                    0,
                ],
                dtype=np.float32,
            ),

        "targets_mapped_to_huri":
            np.array(
                [
                    0,
                    0,
                    1,
                ],
                dtype=np.float32,
            ),
    }


    result = {}

    unknown_statuses = set()


    for _, row in lookup.iterrows():

        drug_id = normalize_id(
            row[
                id_col
            ]
        )

        status = normalize_id(
            row[
                status_col
            ]
        )


        if drug_id is None:

            continue


        if status is None:

            raise ValueError(
                f"Drug {drug_id} has missing "
                "drug_target_status."
            )


        if status not in status_to_vector:

            unknown_statuses.add(
                status
            )

            continue


        result[
            drug_id
        ] = status_to_vector[
            status
        ]


    if unknown_statuses:

        raise ValueError(
            "Unknown drug_target_status values:\n"
            + "\n".join(
                sorted(
                    unknown_statuses
                )
            )
        )


    return result


# ============================================================
# Observation master
# ============================================================

def load_observation_master(
    path: Path,
) -> pd.DataFrame:

    df = pd.read_csv(
        path,
        sep="\t",
        low_memory=False,
    )


    print(
        "Observation master:",
        df.shape,
    )

    print(
        "Columns:"
    )

    print(
        df.columns.tolist()
    )


    return df


def detect_master_columns(
    df: pd.DataFrame,
):

    drug_a_col = find_column(
        df,
        [
            "drug_a_id",
            "drug1_id",
            "drug_a_drugcomb_id",
            "drug1_drugcomb_id",
            "drug_row_id",
            "drug1",
        ],
        required=True,
        label="Drug A ID",
    )


    drug_b_col = find_column(
        df,
        [
            "drug_b_id",
            "drug2_id",
            "drug_b_drugcomb_id",
            "drug2_drugcomb_id",
            "drug_col_id",
            "drug2",
        ],
        required=True,
        label="Drug B ID",
    )


    depmap_col = find_column(
        df,
        [
            "ModelID",
            "model_id",
            "depmap_id",
            "broad_id",
        ],
        required=True,
        label="DepMap cell ID",
    )


    label_col = find_column(
        df,
        [
            "label_3class",
            "label",
            "response_class",
            "zip_class",
            "class",
        ],
        required=True,
        label="3-class label",
    )


    pair_context_col = find_column(
        df,
        [
            "pair_context_id",
            "pair_cell_id",
        ],
        required=True,
        label="pair_context_id",
    )


    study_col = find_column(
        df,
        [
            "study_name",
            "study",
        ],
    )


    observation_col = find_column(
        df,
        [
            "sample_id",
            "observation_id",
            "row_id",
        ],
    )


    return {

        "drug_a":
            drug_a_col,

        "drug_b":
            drug_b_col,

        "depmap":
            depmap_col,

        "label":
            label_col,

        "pair_context":
            pair_context_col,

        "study":
            study_col,

        "observation":
            observation_col,
    }


# ============================================================
# Feature store
# ============================================================

@dataclass
class FeatureStore:

    morgan: np.ndarray
    morgan_index: Dict[
        str,
        int,
    ]

    direct_targets: sp.csr_matrix
    direct_index: Dict[
        str,
        int,
    ]

    ppi: np.ndarray
    ppi_index: Dict[
        str,
        int,
    ]

    availability: Dict[
        str,
        np.ndarray,
    ]

    expression: np.ndarray
    expression_index: Dict[
        str,
        int,
    ]

    pathways: np.ndarray
    pathways_index: Dict[
        str,
        int,
    ]

    regulons: np.ndarray
    regulons_index: Dict[
        str,
        int,
    ]


# ============================================================
# Drug ID diagnostic
# ============================================================

def print_drug_id_diagnostic(
    master: pd.DataFrame,
    columns: Dict[
        str,
        str,
    ],
    features: FeatureStore,
):

    print_section(
        "DRUG ID MATCHING DIAGNOSTIC"
    )


    a_raw = (
        master[
            columns[
                "drug_a"
            ]
        ]
        .dropna()
        .head(
            10
        )
        .tolist()
    )


    b_raw = (
        master[
            columns[
                "drug_b"
            ]
        ]
        .dropna()
        .head(
            10
        )
        .tolist()
    )


    print(
        "Example master Drug A IDs (raw):"
    )

    print(
        a_raw
    )


    print(
        "\nExample master Drug A IDs (normalized):"
    )

    print(
        [
            normalize_id(
                x
            )
            for x in a_raw
        ]
    )


    print(
        "\nExample master Drug B IDs (raw):"
    )

    print(
        b_raw
    )


    print(
        "\nExample master Drug B IDs (normalized):"
    )

    print(
        [
            normalize_id(
                x
            )
            for x in b_raw
        ]
    )


    master_drug_ids = (
        set(
            master[
                columns[
                    "drug_a"
                ]
            ]
            .map(
                normalize_id
            )
            .dropna()
        )
        |
        set(
            master[
                columns[
                    "drug_b"
                ]
            ]
            .map(
                normalize_id
            )
            .dropna()
        )
    )


    morgan_ids = set(
        features.morgan_index
    )

    direct_ids = set(
        features.direct_index
    )

    ppi_ids = set(
        features.ppi_index
    )

    availability_ids = set(
        features.availability
    )


    print(
        "\nUnique normalized DrugComb IDs in master:",
        len(
            master_drug_ids
        ),
    )


    print(
        "Overlap with Morgan:",
        len(
            master_drug_ids
            & morgan_ids
        ),
    )


    print(
        "Overlap with direct target matrix:",
        len(
            master_drug_ids
            & direct_ids
        ),
    )


    print(
        "Overlap with PPI matrix:",
        len(
            master_drug_ids
            & ppi_ids
        ),
    )


    print(
        "Overlap with availability lookup:",
        len(
            master_drug_ids
            & availability_ids
        ),
    )


    print(
        "\nExample Morgan IDs:"
    )

    print(
        list(
            morgan_ids
        )[:10]
    )


    print(
        "\nExample target/PPI IDs:"
    )

    print(
        list(
            direct_ids
        )[:10]
    )


# ============================================================
# Build model-ready master
# ============================================================

def build_model_ready_master(
    master: pd.DataFrame,
    columns: Dict[
        str,
        str,
    ],
    features: FeatureStore,
) -> pd.DataFrame:

    df = master.copy()


    # --------------------------------------------------------
    # Normalize IDs
    # --------------------------------------------------------

    df[
        "drug_a_id_std"
    ] = df[
        columns[
            "drug_a"
        ]
    ].map(
        normalize_id
    )


    df[
        "drug_b_id_std"
    ] = df[
        columns[
            "drug_b"
        ]
    ].map(
        normalize_id
    )


    df[
        "depmap_id_std"
    ] = df[
        columns[
            "depmap"
        ]
    ].map(
        normalize_id
    )


    # --------------------------------------------------------
    # Feature availability
    # --------------------------------------------------------

    df[
        "has_morgan_a"
    ] = df[
        "drug_a_id_std"
    ].isin(
        features.morgan_index
    )


    df[
        "has_morgan_b"
    ] = df[
        "drug_b_id_std"
    ].isin(
        features.morgan_index
    )


    df[
        "has_direct_a"
    ] = df[
        "drug_a_id_std"
    ].isin(
        features.direct_index
    )


    df[
        "has_direct_b"
    ] = df[
        "drug_b_id_std"
    ].isin(
        features.direct_index
    )


    df[
        "has_ppi_a"
    ] = df[
        "drug_a_id_std"
    ].isin(
        features.ppi_index
    )


    df[
        "has_ppi_b"
    ] = df[
        "drug_b_id_std"
    ].isin(
        features.ppi_index
    )


    df[
        "has_availability_a"
    ] = df[
        "drug_a_id_std"
    ].isin(
        features.availability
    )


    df[
        "has_availability_b"
    ] = df[
        "drug_b_id_std"
    ].isin(
        features.availability
    )


    df[
        "has_expression"
    ] = df[
        "depmap_id_std"
    ].isin(
        features.expression_index
    )


    df[
        "has_pathways"
    ] = df[
        "depmap_id_std"
    ].isin(
        features.pathways_index
    )


    df[
        "has_regulons"
    ] = df[
        "depmap_id_std"
    ].isin(
        features.regulons_index
    )


    required_flags = [

        "has_morgan_a",
        "has_morgan_b",

        "has_direct_a",
        "has_direct_b",

        "has_ppi_a",
        "has_ppi_b",

        "has_availability_a",
        "has_availability_b",

        "has_expression",
        "has_pathways",
        "has_regulons",
    ]


    df[
        "usable_v1"
    ] = df[
        required_flags
    ].all(
        axis=1
    )


    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    raw_label = df[
        columns[
            "label"
        ]
    ]


    if pd.api.types.is_numeric_dtype(
        raw_label
    ):

        values = set(
            raw_label
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )


        if not values.issubset(
            {
                0,
                1,
                2,
            }
        ):

            raise ValueError(
                "Numeric label column contains values "
                f"outside 0/1/2: {sorted(values)}"
            )


        df[
            "label_int"
        ] = raw_label.astype(
            int
        )


    else:

        cleaned = (
            raw_label
            .astype(str)
            .str.strip()
            .str.casefold()
        )


        mapping = {

            "antagonism":
                0,

            "antagonistic":
                0,

            "no_interaction":
                1,

            "no interaction":
                1,

            "none":
                1,

            "neutral":
                1,

            "synergy":
                2,

            "synergistic":
                2,
        }


        df[
            "label_int"
        ] = cleaned.map(
            mapping
        )


        if df[
            "label_int"
        ].isna().any():

            bad = (
                raw_label[
                    df[
                        "label_int"
                    ].isna()
                ]
                .value_counts()
                .head(
                    20
                )
            )

            raise ValueError(
                "Unrecognized label values:\n"
                + bad.to_string()
            )


        df[
            "label_int"
        ] = df[
            "label_int"
        ].astype(
            int
        )


    # --------------------------------------------------------
    # Feature row indices
    # --------------------------------------------------------

    df[
        "morgan_a_idx"
    ] = df[
        "drug_a_id_std"
    ].map(
        features.morgan_index
    )


    df[
        "morgan_b_idx"
    ] = df[
        "drug_b_id_std"
    ].map(
        features.morgan_index
    )


    df[
        "direct_a_idx"
    ] = df[
        "drug_a_id_std"
    ].map(
        features.direct_index
    )


    df[
        "direct_b_idx"
    ] = df[
        "drug_b_id_std"
    ].map(
        features.direct_index
    )


    df[
        "ppi_a_idx"
    ] = df[
        "drug_a_id_std"
    ].map(
        features.ppi_index
    )


    df[
        "ppi_b_idx"
    ] = df[
        "drug_b_id_std"
    ].map(
        features.ppi_index
    )


    df[
        "expression_idx"
    ] = df[
        "depmap_id_std"
    ].map(
        features.expression_index
    )


    df[
        "pathways_idx"
    ] = df[
        "depmap_id_std"
    ].map(
        features.pathways_index
    )


    df[
        "regulons_idx"
    ] = df[
        "depmap_id_std"
    ].map(
        features.regulons_index
    )


    return df


# ============================================================
# PyTorch Dataset
# ============================================================

class DrugCombinationDataset(
    Dataset
):

    def __init__(
        self,
        master: pd.DataFrame,
        features: FeatureStore,
        pair_context_col: str,
        study_col: Optional[
            str
        ] = None,
        observation_col: Optional[
            str
        ] = None,
        strict_only: bool = True,
    ):

        if strict_only:

            master = master[
                master[
                    "usable_v1"
                ]
            ].copy()


        self.master = (
            master
            .reset_index(
                drop=True
            )
        )

        self.features = (
            features
        )

        self.pair_context_col = (
            pair_context_col
        )

        self.study_col = (
            study_col
        )

        self.observation_col = (
            observation_col
        )


    def __len__(
        self,
    ):

        return len(
            self.master
        )


    def _drug_features(
        self,
        drug_id: str,
        morgan_idx: int,
        direct_idx: int,
        ppi_idx: int,
    ):

        morgan = (
            self.features
            .morgan[
                morgan_idx
            ]
            .astype(
                np.float32,
                copy=False,
            )
        )


        direct = (
            self.features
            .direct_targets[
                direct_idx
            ]
            .toarray()
            .ravel()
            .astype(
                np.float32,
                copy=False,
            )
        )


        ppi = (
            self.features
            .ppi[
                ppi_idx
            ]
            .astype(
                np.float32,
                copy=False,
            )
        )


        availability = (
            self.features
            .availability[
                drug_id
            ]
            .astype(
                np.float32,
                copy=False,
            )
        )


        return (
            morgan,
            direct,
            ppi,
            availability,
        )


    def __getitem__(
        self,
        idx: int,
    ):

        row = self.master.iloc[
            idx
        ]


        drug_a_id = row[
            "drug_a_id_std"
        ]

        drug_b_id = row[
            "drug_b_id_std"
        ]


        # ----------------------------------------------------
        # Drug A
        # ----------------------------------------------------

        (
            morgan_a,
            direct_a,
            ppi_a,
            availability_a,
        ) = self._drug_features(

            drug_id=
                drug_a_id,

            morgan_idx=
                int(
                    row[
                        "morgan_a_idx"
                    ]
                ),

            direct_idx=
                int(
                    row[
                        "direct_a_idx"
                    ]
                ),

            ppi_idx=
                int(
                    row[
                        "ppi_a_idx"
                    ]
                ),
        )


        # ----------------------------------------------------
        # Drug B
        # ----------------------------------------------------

        (
            morgan_b,
            direct_b,
            ppi_b,
            availability_b,
        ) = self._drug_features(

            drug_id=
                drug_b_id,

            morgan_idx=
                int(
                    row[
                        "morgan_b_idx"
                    ]
                ),

            direct_idx=
                int(
                    row[
                        "direct_b_idx"
                    ]
                ),

            ppi_idx=
                int(
                    row[
                        "ppi_b_idx"
                    ]
                ),
        )


        # ----------------------------------------------------
        # Context
        # ----------------------------------------------------

        expression = (
            self.features
            .expression[
                int(
                    row[
                        "expression_idx"
                    ]
                )
            ]
            .astype(
                np.float32,
                copy=False,
            )
        )


        pathways = (
            self.features
            .pathways[
                int(
                    row[
                        "pathways_idx"
                    ]
                )
            ]
            .astype(
                np.float32,
                copy=False,
            )
        )


        regulons = (
            self.features
            .regulons[
                int(
                    row[
                        "regulons_idx"
                    ]
                )
            ]
            .astype(
                np.float32,
                copy=False,
            )
        )


        result = {

            # Drug A
            "drug_a_morgan":
                torch.from_numpy(
                    morgan_a
                ),

            "drug_a_targets":
                torch.from_numpy(
                    direct_a
                ),

            "drug_a_ppi":
                torch.from_numpy(
                    ppi_a
                ),

            "drug_a_availability":
                torch.from_numpy(
                    availability_a
                ),


            # Drug B
            "drug_b_morgan":
                torch.from_numpy(
                    morgan_b
                ),

            "drug_b_targets":
                torch.from_numpy(
                    direct_b
                ),

            "drug_b_ppi":
                torch.from_numpy(
                    ppi_b
                ),

            "drug_b_availability":
                torch.from_numpy(
                    availability_b
                ),


            # Context
            "context_expression":
                torch.from_numpy(
                    expression
                ),

            "context_pathways":
                torch.from_numpy(
                    pathways
                ),

            "context_regulons":
                torch.from_numpy(
                    regulons
                ),


            # Label
            "label":
                torch.tensor(
                    int(
                        row[
                            "label_int"
                        ]
                    ),
                    dtype=torch.long,
                ),


            # Metadata
            "drug_a_id":
                drug_a_id,

            "drug_b_id":
                drug_b_id,

            "depmap_id":
                row[
                    "depmap_id_std"
                ],

            "pair_context_id":
                str(
                    row[
                        self.pair_context_col
                    ]
                ),
        }


        if self.study_col is not None:

            result[
                "study"
            ] = str(
                row[
                    self.study_col
                ]
            )


        if self.observation_col is not None:

            result[
                "observation_id"
            ] = str(
                row[
                    self.observation_col
                ]
            )


        return result


# ============================================================
# Load all feature matrices
# ============================================================

def load_all_features() -> FeatureStore:

    print_section(
        "LOADING DRUG FEATURES"
    )


    (
        morgan,
        _,
        morgan_index,
    ) = load_morgan_npz(
        MORGAN_PATH
    )

    print(
        "Morgan:",
        morgan.shape,
    )


    (
        direct,
        direct_ids,
        direct_index,
    ) = load_manual_csr_npz(
        DIRECT_TARGET_PATH
    )

    print(
        "Direct targets:",
        direct.shape,
        "| nnz:",
        direct.nnz,
    )


    (
        ppi,
        ppi_ids,
        ppi_index,
    ) = load_dense_npz(
        PPI_PATH
    )

    print(
        "PPI:",
        ppi.shape,
    )


    # --------------------------------------------------------
    # Drug row-order QC
    # --------------------------------------------------------

    direct_ids_norm = [
        normalize_id(
            x
        )
        for x in direct_ids
    ]

    ppi_ids_norm = [
        normalize_id(
            x
        )
        for x in ppi_ids
    ]


    if direct_ids_norm != ppi_ids_norm:

        raise ValueError(
            "Direct-target and PPI drug row order differs."
        )


    print(
        "Direct target / PPI drug order: MATCH ✓"
    )


    availability = (
        build_availability_lookup(
            TARGET_LOOKUP_PATH
        )
    )


    print(
        "Availability lookup:",
        len(
            availability
        ),
    )


    print_section(
        "LOADING BIOLOGICAL CONTEXT"
    )


    (
        expression,
        _,
        expression_index,
        _,
    ) = load_context_matrix(
        EXPRESSION_PATH,
        "Expression",
    )


    (
        pathways,
        _,
        pathways_index,
        _,
    ) = load_context_matrix(
        PATHWAY_PATH,
        "Pathways",
    )


    (
        regulons,
        _,
        regulons_index,
        _,
    ) = load_context_matrix(
        REGULON_PATH,
        "Regulons",
    )


    # --------------------------------------------------------
    # Context overlap
    # --------------------------------------------------------

    expression_ids = set(
        expression_index
    )

    pathway_ids = set(
        pathways_index
    )

    regulon_ids = set(
        regulons_index
    )


    common_context = (
        expression_ids
        & pathway_ids
        & regulon_ids
    )


    print(
        "\nExpression IDs:",
        len(
            expression_ids
        ),
    )

    print(
        "Pathway IDs:",
        len(
            pathway_ids
        ),
    )

    print(
        "Regulon IDs:",
        len(
            regulon_ids
        ),
    )

    print(
        "Common context IDs:",
        len(
            common_context
        ),
    )


    return FeatureStore(

        morgan=
            morgan,

        morgan_index=
            morgan_index,

        direct_targets=
            direct,

        direct_index=
            direct_index,

        ppi=
            ppi,

        ppi_index=
            ppi_index,

        availability=
            availability,

        expression=
            expression,

        expression_index=
            expression_index,

        pathways=
            pathways,

        pathways_index=
            pathways_index,

        regulons=
            regulons,

        regulons_index=
            regulons_index,
    )


# ============================================================
# QC summary
# ============================================================

def print_model_ready_summary(
    df: pd.DataFrame,
):

    print_section(
        "MODEL-READY INTERSECTION"
    )


    print(
        "Total observations:",
        f"{len(df):,}",
    )


    checks = [

        (
            "Both Morgan",
            df[
                "has_morgan_a"
            ]
            & df[
                "has_morgan_b"
            ],
        ),

        (
            "Both direct target rows",
            df[
                "has_direct_a"
            ]
            & df[
                "has_direct_b"
            ],
        ),

        (
            "Both PPI rows",
            df[
                "has_ppi_a"
            ]
            & df[
                "has_ppi_b"
            ],
        ),

        (
            "Both availability states",
            df[
                "has_availability_a"
            ]
            & df[
                "has_availability_b"
            ],
        ),

        (
            "Expression context",
            df[
                "has_expression"
            ],
        ),

        (
            "Pathway context",
            df[
                "has_pathways"
            ],
        ),

        (
            "Regulon context",
            df[
                "has_regulons"
            ],
        ),

        (
            "ALL REQUIRED V1 FEATURES",
            df[
                "usable_v1"
            ],
        ),
    ]


    for name, mask in checks:

        n = int(
            mask.sum()
        )

        pct = (
            n
            / len(
                df
            )
            * 100
            if len(
                df
            )
            else 0
        )

        print(
            f"{name:<32}"
            f"{n:>10,} "
            f"({pct:6.2f}%)"
        )


    usable = df[
        df[
            "usable_v1"
        ]
    ]


    print(
        "\nFinal usable observations:",
        f"{len(usable):,}",
    )


    unique_drugs = (
        set(
            usable[
                "drug_a_id_std"
            ]
        )
        |
        set(
            usable[
                "drug_b_id_std"
            ]
        )
    )


    print(
        "Unique drugs:",
        len(
            unique_drugs
        ),
    )


    print(
        "Unique DepMap contexts:",
        usable[
            "depmap_id_std"
        ].nunique(),
    )


    if (
        "pair_context_id"
        in usable.columns
    ):

        print(
            "Unique pair_context_id:",
            usable[
                "pair_context_id"
            ].nunique(),
        )


    print(
        "\nClass counts:"
    )


    class_counts = (
        usable[
            "label_int"
        ]
        .map(
            INT_TO_LABEL
        )
        .value_counts()
    )


    print(
        class_counts
    )


    print(
        "\nClass fractions:"
    )


    if class_counts.sum() > 0:

        print(
            (
                class_counts
                / class_counts.sum()
            )
            .round(
                4
            )
        )

    else:

        print(
            class_counts.astype(
                float
            )
        )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print_section(
        "LOAD FEATURES"
    )


    features = (
        load_all_features()
    )


    # --------------------------------------------------------
    # Choose observation master
    # --------------------------------------------------------

    if CELL_MAPPED_MASTER_PATH.exists():

        master_path = (
            CELL_MAPPED_MASTER_PATH
        )


    elif MASTER_PATH.exists():

        master_path = (
            MASTER_PATH
        )


        print(
            "\nWARNING:"
        )

        print(
            "Cell-mapped master file not found:"
        )

        print(
            CELL_MAPPED_MASTER_PATH
        )

        print(
            "\nFalling back to:"
        )

        print(
            MASTER_PATH
        )

        print(
            "\nThis fallback only works if that table "
            "already contains a DepMap ID column."
        )


    else:

        raise FileNotFoundError(
            "Could not find DrugComb observation master.\n"
            f"Tried:\n"
            f"{CELL_MAPPED_MASTER_PATH}\n"
            f"{MASTER_PATH}"
        )


    print_section(
        "LOAD DRUGCOMB MASTER"
    )


    master = (
        load_observation_master(
            master_path
        )
    )


    columns = (
        detect_master_columns(
            master
        )
    )


    print(
        "\nDetected columns:"
    )


    for key, value in columns.items():

        print(
            f"{key}: {value}"
        )


    # --------------------------------------------------------
    # Explicit drug ID diagnostic
    # --------------------------------------------------------

    print_drug_id_diagnostic(
        master=
            master,

        columns=
            columns,

        features=
            features,
    )


    print_section(
        "BUILD MODEL-READY MASTER"
    )


    model_master = (
        build_model_ready_master(
            master=
                master,

            columns=
                columns,

            features=
                features,
        )
    )


    # --------------------------------------------------------
    # Standardized aliases
    # --------------------------------------------------------

    if (
        columns[
            "pair_context"
        ]
        != "pair_context_id"
    ):

        model_master[
            "pair_context_id"
        ] = model_master[
            columns[
                "pair_context"
            ]
        ]


    if (
        columns[
            "observation"
        ]
        is not None
        and columns[
            "observation"
        ]
        != "observation_id"
    ):

        model_master[
            "observation_id"
        ] = model_master[
            columns[
                "observation"
            ]
        ]


    print_model_ready_summary(
        model_master
    )


    # --------------------------------------------------------
    # Save full audit table
    # --------------------------------------------------------

    audit_path = (
        OUTPUT_DIR
        / "drugcomb_model_ready_audit.tsv"
    )


    model_master.to_csv(
        audit_path,
        sep="\t",
        index=False,
    )


    # --------------------------------------------------------
    # Save strict usable subset
    # --------------------------------------------------------

    usable = (
        model_master[
            model_master[
                "usable_v1"
            ]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    usable_path = (
        OUTPUT_DIR
        / "drugcomb_model_ready_v1.tsv"
    )


    usable.to_csv(
        usable_path,
        sep="\t",
        index=False,
    )


    # --------------------------------------------------------
    # Dataset diagnostic
    # --------------------------------------------------------

    dataset = (
        DrugCombinationDataset(

            master=
                model_master,

            features=
                features,

            pair_context_col=
                columns[
                    "pair_context"
                ],

            study_col=
                columns[
                    "study"
                ],

            observation_col=
                columns[
                    "observation"
                ],

            strict_only=
                True,
        )
    )


    print_section(
        "DATASET DIAGNOSTIC"
    )


    print(
        "Dataset length:",
        len(
            dataset
        ),
    )


    if len(
        dataset
    ) > 0:

        sample = dataset[
            0
        ]


        print(
            "\nFirst sample tensor shapes:"
        )


        tensor_keys = [

            "drug_a_morgan",
            "drug_a_targets",
            "drug_a_ppi",
            "drug_a_availability",

            "drug_b_morgan",
            "drug_b_targets",
            "drug_b_ppi",
            "drug_b_availability",

            "context_expression",
            "context_pathways",
            "context_regulons",

            "label",
        ]


        for key in tensor_keys:

            value = sample[
                key
            ]

            print(
                f"{key:<28}",
                tuple(
                    value.shape
                ),
                value.dtype,
            )


        print(
            "\nMetadata:"
        )


        for key in [

            "drug_a_id",
            "drug_b_id",
            "depmap_id",
            "pair_context_id",

        ]:

            print(
                f"{key}:",
                sample[
                    key
                ],
            )


        if "study" in sample:

            print(
                "study:",
                sample[
                    "study"
                ],
            )


        if "observation_id" in sample:

            print(
                "observation_id:",
                sample[
                    "observation_id"
                ],
            )


    # --------------------------------------------------------
    # Save run summary
    # --------------------------------------------------------

    try:
        master_path_for_summary = str(
            master_path.resolve().relative_to(ROOT)
        )
    except ValueError:
        master_path_for_summary = str(
            master_path.resolve()
        )

    summary = {

        "master_path":
            master_path_for_summary,

        "n_master":
            int(
                len(
                    model_master
                )
            ),

        "n_usable_v1":
            int(
                model_master[
                    "usable_v1"
                ].sum()
            ),

        "pct_usable_v1":
            float(
                model_master[
                    "usable_v1"
                ].mean()
                * 100
            ),

        "n_unique_drugs_v1":
            int(
                len(
                    set(
                        usable[
                            "drug_a_id_std"
                        ]
                    )
                    |
                    set(
                        usable[
                            "drug_b_id_std"
                        ]
                    )
                )
            ),

        "n_contexts_v1":
            int(
                usable[
                    "depmap_id_std"
                ].nunique()
            ),

        "n_pair_contexts_v1":
            int(
                usable[
                    "pair_context_id"
                ].nunique()
            )
            if (
                "pair_context_id"
                in usable.columns
            )
            else None,

        "class_counts_v1":
            {
                INT_TO_LABEL[
                    int(
                        k
                    )
                ]:
                    int(
                        v
                    )

                for k, v
                in usable[
                    "label_int"
                ]
                .value_counts()
                .sort_index()
                .items()
            },
    }


    with open(
        OUTPUT_DIR
        / "run_summary.json",
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )


    print_section(
        "DONE"
    )


    print(
        "Audit table:"
    )

    print(
        audit_path
    )


    print(
        "\nStrict V1 master:"
    )

    print(
        usable_path
    )


    print(
        "\nSummary:"
    )

    print(
        OUTPUT_DIR
        / "run_summary.json"
    )


if __name__ == "__main__":

    main()
