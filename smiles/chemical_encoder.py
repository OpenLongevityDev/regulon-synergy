#!/usr/bin/env python3

"""
chemical_encoder.py

Standalone chemical-structure encoder for DrugComb drugs.

Input
-----
Morgan fingerprints generated previously:

smiles/smiles_results_morgan/Morgan_ECFP4_2048.npz

Representation
--------------
2048-bit Morgan / ECFP4-style fingerprint
        |
        v
Linear 2048 -> 512
        |
        v
GELU
        |
        v
Dropout
        |
        v
Linear 512 -> 128
        |
        v
LayerNorm
        |
        v
GELU
        |
        v
128-dimensional chemical embedding


Important
---------
The encoder is NOT trained in this script.

The final chemical embedding should be learned end-to-end
during synergy-model training.

This script only:
    - loads fingerprints
    - performs QC
    - defines dataset / encoder classes
    - runs a diagnostic forward pass
    - saves metadata / feature order
"""

from pathlib import Path
import argparse
import json
import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Default paths
# ============================================================

# Repository root. Override when the checkout/data root lives elsewhere.
ROOT = Path(
    os.environ.get("DRUG_SYNERGY_ROOT", Path(__file__).resolve().parents[1])
).resolve()

DEFAULT_MORGAN = (
    ROOT
    / "smiles"
    / "smiles_results_morgan"
    / "Morgan_ECFP4_2048.npz"
)

DEFAULT_OUTDIR = ROOT / "smiles" / "chemical_encoder"


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# Load Morgan fingerprints
# ============================================================

def load_morgan_npz(path):
    """
    Load Morgan fingerprints saved by smiles_drugcomb.py.

    Expected arrays
    ---------------
    X
        shape:
            n_drugs x 2048

    drug_id
    dname
    canonical_smiles
    feature_names
    """

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Morgan fingerprint file not found:\n{path}"
        )

    print("\n========================================")
    print("Loading Morgan fingerprints")
    print("========================================")

    data = np.load(
        path,
        allow_pickle=True,
    )

    required = [
        "X",
        "drug_id",
        "dname",
        "canonical_smiles",
        "feature_names",
    ]

    missing = [
        x
        for x in required
        if x not in data.files
    ]

    if missing:

        raise ValueError(
            "Missing arrays in NPZ: "
            + ", ".join(missing)
        )

    X = data[
        "X"
    ].astype(
        np.float32
    )

    drug_ids = data[
        "drug_id"
    ].astype(str)

    drug_names = data[
        "dname"
    ].astype(str)

    canonical_smiles = data[
        "canonical_smiles"
    ].astype(str)

    feature_names = data[
        "feature_names"
    ].astype(str)

    # --------------------------------------------------------
    # QC
    # --------------------------------------------------------

    if X.ndim != 2:

        raise ValueError(
            f"X must be 2D; got shape {X.shape}"
        )

    if len(
        drug_ids
    ) != X.shape[0]:

        raise ValueError(
            "drug_id length does not match X rows."
        )

    if len(
        drug_names
    ) != X.shape[0]:

        raise ValueError(
            "dname length does not match X rows."
        )

    if len(
        canonical_smiles
    ) != X.shape[0]:

        raise ValueError(
            "canonical_smiles length does not match X rows."
        )

    if len(
        feature_names
    ) != X.shape[1]:

        raise ValueError(
            "feature_names length does not match X columns."
        )

    if len(
        np.unique(
            drug_ids
        )
    ) != len(
        drug_ids
    ):

        raise ValueError(
            "Duplicate drug IDs found in Morgan matrix."
        )

    if not np.isfinite(
        X
    ).all():

        raise ValueError(
            "Morgan matrix contains non-finite values."
        )

    unique_values = np.unique(
        X
    )

    if not np.all(
        np.isin(
            unique_values,
            [0.0, 1.0],
        )
    ):

        raise ValueError(
            "Morgan matrix is expected to be binary."
        )

    print(
        f"Fingerprint matrix: "
        f"{X.shape}"
    )

    print(
        f"Unique drug IDs: "
        f"{len(np.unique(drug_ids)):,}"
    )

    print(
        f"Fingerprint dimensions: "
        f"{X.shape[1]:,}"
    )

    print(
        f"Mean active bits/drug: "
        f"{X.sum(axis=1).mean():.2f}"
    )

    print(
        f"Median active bits/drug: "
        f"{np.median(X.sum(axis=1)):.1f}"
    )

    return {
        "X":
            X,

        "drug_id":
            drug_ids,

        "dname":
            drug_names,

        "canonical_smiles":
            canonical_smiles,

        "feature_names":
            feature_names,
    }


# ============================================================
# Drug lookup
# ============================================================

def build_drug_lookup(
    drug_ids,
    drug_names,
    canonical_smiles,
):
    """
    Build metadata table indexed by DrugComb ID.
    """

    lookup = pd.DataFrame(
        {
            "drug_id":
                drug_ids,

            "dname":
                drug_names,

            "canonical_smiles":
                canonical_smiles,
        }
    )

    lookup[
        "matrix_index"
    ] = np.arange(
        len(
            lookup
        )
    )

    lookup = lookup[
        [
            "matrix_index",
            "drug_id",
            "dname",
            "canonical_smiles",
        ]
    ]

    return lookup


# ============================================================
# Dataset
# ============================================================

class MorganDrugDataset(
    Dataset
):
    """
    Dataset for fixed Morgan fingerprints.

    Returns
    -------
    dictionary with:
        drug_id
        fingerprint
    """

    def __init__(
        self,
        X,
        drug_ids,
    ):

        self.X = torch.as_tensor(
            X,
            dtype=torch.float32,
        )

        self.drug_ids = np.asarray(
            drug_ids,
            dtype=str,
        )

        if len(
            self.drug_ids
        ) != self.X.shape[0]:

            raise ValueError(
                "drug_ids length does not match fingerprint rows."
            )

    def __len__(
        self
    ):

        return self.X.shape[0]

    def __getitem__(
        self,
        idx,
    ):

        return {
            "drug_id":
                self.drug_ids[
                    idx
                ],

            "fingerprint":
                self.X[
                    idx
                ],
        }


# ============================================================
# Chemical encoder
# ============================================================

class ChemicalEncoder(
    nn.Module
):
    """
    Morgan fingerprint -> learned chemical embedding.

    Default architecture
    --------------------
    2048
      ->
    512
      ->
    GELU
      ->
    Dropout
      ->
    128
      ->
    LayerNorm
      ->
    GELU
    """

    def __init__(
        self,
        input_dim=2048,
        hidden_dim=512,
        embedding_dim=128,
        dropout=0.20,
    ):

        super().__init__()

        self.input_dim = (
            input_dim
        )

        self.hidden_dim = (
            hidden_dim
        )

        self.embedding_dim = (
            embedding_dim
        )

        self.dropout = (
            dropout
        )

        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                embedding_dim,
            ),

            nn.LayerNorm(
                embedding_dim
            ),

            nn.GELU(),
        )

    def forward(
        self,
        fingerprint,
    ):

        return self.encoder(
            fingerprint
        )


# ============================================================
# Optional convenience wrapper for ID-based access
# ============================================================

class DrugFingerprintStore:
    """
    Convenience lookup from DrugComb ID -> Morgan fingerprint.

    Useful later when constructing synergy pair batches.
    """

    def __init__(
        self,
        X,
        drug_ids,
    ):

        self.X = torch.as_tensor(
            X,
            dtype=torch.float32,
        )

        self.drug_ids = [
            str(
                x
            )
            for x in drug_ids
        ]

        self.id_to_index = {
            drug_id:
                i
            for i, drug_id
            in enumerate(
                self.drug_ids
            )
        }

    def __len__(
        self
    ):

        return len(
            self.drug_ids
        )

    def has_drug(
        self,
        drug_id,
    ):

        return (
            str(
                drug_id
            )
            in self.id_to_index
        )

    def get_index(
        self,
        drug_id,
    ):

        drug_id = str(
            drug_id
        )

        if drug_id not in self.id_to_index:

            raise KeyError(
                f"Drug ID not found in Morgan fingerprints: "
                f"{drug_id}"
            )

        return self.id_to_index[
            drug_id
        ]

    def get_fingerprint(
        self,
        drug_id,
    ):

        idx = self.get_index(
            drug_id
        )

        return self.X[
            idx
        ]

    def get_batch(
        self,
        drug_ids,
    ):
        """
        Retrieve fingerprint tensor for multiple drug IDs.

        Output:
            batch_size x 2048
        """

        indices = [
            self.get_index(
                x
            )
            for x in drug_ids
        ]

        return self.X[
            indices
        ]


# ============================================================
# Save metadata
# ============================================================

def save_metadata(
    outdir,
    data,
    lookup,
    hidden_dim,
    embedding_dim,
    dropout,
):

    outdir = Path(
        outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Drug lookup
    # --------------------------------------------------------

    lookup.to_csv(
        outdir
        / "chemical_encoder_drug_lookup.tsv",
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # Fingerprint feature order
    # --------------------------------------------------------

    feature_order = pd.DataFrame(
        {
            "feature_index":
                np.arange(
                    len(
                        data[
                            "feature_names"
                        ]
                    )
                ),

            "feature":
                data[
                    "feature_names"
                ],
        }
    )

    feature_order.to_csv(
        outdir
        / "chemical_encoder_feature_order.tsv",
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # Architecture config
    # --------------------------------------------------------

    config = {

        "input_representation":
            "Morgan / ECFP4-style binary fingerprint",

        "input_dim":
            int(
                data[
                    "X"
                ].shape[1]
            ),

        "hidden_dim":
            int(
                hidden_dim
            ),

        "embedding_dim":
            int(
                embedding_dim
            ),

        "dropout":
            float(
                dropout
            ),

        "activation":
            "GELU",

        "normalization":
            "LayerNorm on final embedding",

        "fingerprint_normalization":
            "none",

        "encoder_training":
            "end-to-end with synergy objective",

        "n_drugs":
            int(
                data[
                    "X"
                ].shape[0]
            ),
    }

    with open(
        outdir
        / "chemical_encoder_config.json",
        "w",
    ) as handle:

        json.dump(
            config,
            handle,
            indent=2,
        )


# ============================================================
# Diagnostic forward pass
# ============================================================

def diagnostic_forward_pass(
    data,
    hidden_dim,
    embedding_dim,
    dropout,
    batch_size=8,
    device=None,
):
    """
    Instantiate the untrained encoder and verify tensor shapes.
    """

    if device is None:

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print("\n========================================")
    print("Chemical encoder diagnostic")
    print("========================================")

    print(
        f"Device: {device}"
    )

    dataset = MorganDrugDataset(
        X=data[
            "X"
        ],
        drug_ids=data[
            "drug_id"
        ],
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    encoder = ChemicalEncoder(
        input_dim=data[
            "X"
        ].shape[1],
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        dropout=dropout,
    ).to(
        device
    )

    batch = next(
        iter(
            loader
        )
    )

    fingerprints = batch[
        "fingerprint"
    ].to(
        device
    )

    with torch.no_grad():

        embeddings = encoder(
            fingerprints
        )

    print(
        f"Input fingerprint batch: "
        f"{tuple(fingerprints.shape)}"
    )

    print(
        f"Chemical embedding batch: "
        f"{tuple(embeddings.shape)}"
    )

    n_params = sum(
        p.numel()
        for p in encoder.parameters()
    )

    n_trainable = sum(
        p.numel()
        for p in encoder.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters: "
        f"{n_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{n_trainable:,}"
    )

    print(
        "\nExample DrugComb IDs:"
    )

    for x in batch[
        "drug_id"
    ]:

        print(
            f"  {x}"
        )

    return encoder


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--morgan",
        default=str(
            DEFAULT_MORGAN
        ),
    )

    parser.add_argument(
        "--outdir",
        default=str(
            DEFAULT_OUTDIR
        ),
    )

    parser.add_argument(
        "--hidden-dim",
        "--hidden_dim",
        dest="hidden_dim",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--embedding-dim",
        "--embedding_dim",
        dest="embedding_dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    set_seed(
        args.seed
    )

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. Load fixed Morgan representation
    # ========================================================

    data = load_morgan_npz(
        args.morgan
    )

    # ========================================================
    # 2. Drug lookup
    # ========================================================

    lookup = build_drug_lookup(
        drug_ids=data[
            "drug_id"
        ],
        drug_names=data[
            "dname"
        ],
        canonical_smiles=data[
            "canonical_smiles"
        ],
    )

    # ========================================================
    # 3. Save metadata
    # ========================================================

    save_metadata(
        outdir=outdir,
        data=data,
        lookup=lookup,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    )

    # ========================================================
    # 4. Diagnostic forward pass
    # ========================================================

    encoder = diagnostic_forward_pass(
        data=data,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
    )

    # ========================================================
    # Done
    # ========================================================

    print("\n========================================")
    print("DONE")
    print("========================================")

    print(
        f"\nDrug fingerprints: "
        f"{data['X'].shape}"
    )

    print(
        f"Encoder architecture: "
        f"{data['X'].shape[1]} "
        f"-> {args.hidden_dim} "
        f"-> {args.embedding_dim}"
    )

    print(
        f"\nOutput directory:\n"
        f"{outdir.resolve()}"
    )

    print(
        "\nSaved files:"
    )

    print(
        "  chemical_encoder_drug_lookup.tsv"
    )

    print(
        "  chemical_encoder_feature_order.tsv"
    )

    print(
        "  chemical_encoder_config.json"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The encoder above is UNTRAINED."
    )

    print(
        "Do not export these diagnostic embeddings "
        "as final drug embeddings."
    )

    print(
        "The ChemicalEncoder should be inserted into "
        "the synergy model and trained end-to-end."
    )


if __name__ == "__main__":
    main()
