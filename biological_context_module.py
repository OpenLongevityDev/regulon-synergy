#!/usr/bin/env python3

"""
biological_context_module.py

Combine three biological representations:

    1. L1000 landmark expression
    2. Hallmark pathway activity
    3. CollecTRI TF/regulon activity

into one learned biological-context embedding.

IMPORTANT
---------
This script defines the representation and neural encoder.

The neural network is NOT meant to be trained independently here.
It should later be inserted into the drug-synergy prediction model
and optimized end-to-end using the synergy classification loss.

Pipeline
--------
Expression (942 genes)
        |
        v
expression encoder
        |
       64

Pathways (50 Hallmarks)
        |
        v
pathway encoder
        |
       64

Regulons (~771 fixed / ~580 contextualized)
        |
        v
regulon encoder
        |
       64

        concatenate
            |
           192
            |
        fusion encoder
            |
           128
            |
 biological context embedding
"""

from pathlib import Path
import argparse
import json
import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ============================================================
# Default paths
# ============================================================

BASE_DIR = Path(
    os.environ.get("DRUG_SYNERGY_ROOT", Path(__file__).resolve().parent)
).resolve()

DEFAULT_EXPRESSION_TPM = (
    BASE_DIR
    / "expression"
    / "expression_results_l1000"
    / "L1000_expression_TPMLogp1.csv"
)

DEFAULT_EXPRESSION_RANK = (
    BASE_DIR
    / "expression"
    / "expression_results_l1000"
    / "L1000_expression_within_sample_percentile.csv"
)

DEFAULT_PATHWAY = (
    BASE_DIR
    / "pathways"
    / "pathway_results_hallmark_ulm"
    / "Hallmark_activity_ULM.csv"
)

DEFAULT_REGULON_FIXED = (
    BASE_DIR
    / "regulons"
    / "regulon_results_ulm"
    / "TF_activity_fixed_collectri_ULM.csv"
)

DEFAULT_REGULON_CONTEXTUALIZED = (
    BASE_DIR
    / "regulons"
    / "regulon_results_ulm"
    / "TF_activity_contextualized_collectri_ULM.csv"
)

DEFAULT_OUTDIR = (
    BASE_DIR
    / "biological_context"
)


# ============================================================
# Data loading
# ============================================================

def load_matrix(path, name):
    """
    Load ModelID x features CSV.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{name} matrix not found:\n{path}"
        )

    df = pd.read_csv(
        path,
        index_col=0,
    )

    df.index = df.index.astype(str)
    df.index.name = "ModelID"

    # ensure numeric
    df = df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if df.index.duplicated().any():
        dup = df.index[
            df.index.duplicated()
        ].unique().tolist()

        raise ValueError(
            f"{name}: duplicated ModelIDs detected: "
            f"{dup[:10]}"
        )

    n_missing = int(
        df.isna().sum().sum()
    )

    if n_missing > 0:
        raise ValueError(
            f"{name}: {n_missing:,} missing values detected."
        )

    print(
        f"{name:<12}: {df.shape}"
    )

    return df


# ============================================================
# Align ModelIDs
# ============================================================

def align_representations(
    expression,
    pathway,
    regulon,
):
    """
    Keep the exact ModelID intersection in all three branches.
    """

    common = (
        set(expression.index)
        & set(pathway.index)
        & set(regulon.index)
    )

    if len(common) == 0:
        raise ValueError(
            "No ModelIDs are shared among all three representations."
        )

    # Preserve expression matrix order.
    common_ordered = [
        model_id
        for model_id in expression.index
        if model_id in common
    ]

    expression = expression.loc[
        common_ordered
    ].copy()

    pathway = pathway.loc[
        common_ordered
    ].copy()

    regulon = regulon.loc[
        common_ordered
    ].copy()

    assert expression.index.equals(
        pathway.index
    )

    assert expression.index.equals(
        regulon.index
    )

    print(
        "\nShared biological contexts:",
        len(common_ordered),
    )

    print(
        "Expression features:",
        expression.shape[1],
    )

    print(
        "Pathway features:",
        pathway.shape[1],
    )

    print(
        "Regulon features:",
        regulon.shape[1],
    )

    return (
        expression,
        pathway,
        regulon,
    )


# ============================================================
# Train-only feature standardization
# ============================================================

class FeatureStandardizer:
    """
    Feature-wise standardization:

        x_standardized = (x - mean_train) / std_train

    Critically:
    mean/std MUST be fitted on training biological contexts only.

    Validation/test/external samples are transformed using
    frozen training statistics.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.columns_ = None

    def fit(self, df):
        self.columns_ = list(
            df.columns
        )

        self.mean_ = (
            df.mean(axis=0)
            .to_numpy(dtype=np.float32)
        )

        self.std_ = (
            df.std(
                axis=0,
                ddof=0,
            )
            .to_numpy(dtype=np.float32)
        )

        # Constant / nearly constant feature:
        # scaling factor = 1 so it becomes zero after centering.
        self.std_[
            self.std_ < 1e-8
        ] = 1.0

        return self

    def transform(self, df):

        if self.mean_ is None:
            raise RuntimeError(
                "Standardizer has not been fitted."
            )

        missing = (
            set(self.columns_)
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                "Input is missing required features: "
                f"{list(missing)[:20]}"
            )

        # Exact feature order used at training.
        x = (
            df[self.columns_]
            .to_numpy(dtype=np.float32)
        )

        x = (
            x - self.mean_[None, :]
        ) / self.std_[None, :]

        return x

    def fit_transform(self, df):
        return (
            self.fit(df)
            .transform(df)
        )

    def state_dict(self):
        return {
            "columns":
                self.columns_,
            "mean":
                self.mean_.tolist(),
            "std":
                self.std_.tolist(),
        }

    def load_state_dict(self, state):
        self.columns_ = state[
            "columns"
        ]

        self.mean_ = np.asarray(
            state["mean"],
            dtype=np.float32,
        )

        self.std_ = np.asarray(
            state["std"],
            dtype=np.float32,
        )


# ============================================================
# Three-branch preprocessing object
# ============================================================

class BiologicalContextPreprocessor:
    """
    Holds independent train-fitted scalers for:

        expression
        pathway
        regulon
    """

    def __init__(self):

        self.expression_scaler = (
            FeatureStandardizer()
        )

        self.pathway_scaler = (
            FeatureStandardizer()
        )

        self.regulon_scaler = (
            FeatureStandardizer()
        )

    def fit(
        self,
        expression,
        pathway,
        regulon,
    ):

        self.expression_scaler.fit(
            expression
        )

        self.pathway_scaler.fit(
            pathway
        )

        self.regulon_scaler.fit(
            regulon
        )

        return self

    def transform(
        self,
        expression,
        pathway,
        regulon,
    ):

        expression_x = (
            self.expression_scaler
            .transform(expression)
        )

        pathway_x = (
            self.pathway_scaler
            .transform(pathway)
        )

        regulon_x = (
            self.regulon_scaler
            .transform(regulon)
        )

        return {
            "expression":
                expression_x,

            "pathway":
                pathway_x,

            "regulon":
                regulon_x,
        }

    def state_dict(self):
        return {
            "expression":
                self.expression_scaler
                .state_dict(),

            "pathway":
                self.pathway_scaler
                .state_dict(),

            "regulon":
                self.regulon_scaler
                .state_dict(),
        }

    def save(self, path):

        path = Path(path)

        with open(
            path,
            "w",
        ) as handle:

            json.dump(
                self.state_dict(),
                handle,
                indent=2,
            )

    def load(self, path):

        with open(path) as handle:
            state = json.load(handle)

        self.expression_scaler.load_state_dict(
            state["expression"]
        )

        self.pathway_scaler.load_state_dict(
            state["pathway"]
        )

        self.regulon_scaler.load_state_dict(
            state["regulon"]
        )

        return self


# ============================================================
# Dataset
# ============================================================

class BiologicalContextDataset(Dataset):
    """
    Dataset returning the three biological branches
    for each ModelID.
    """

    def __init__(
        self,
        model_ids,
        expression_x,
        pathway_x,
        regulon_x,
    ):

        self.model_ids = list(
            model_ids
        )

        self.expression = torch.tensor(
            expression_x,
            dtype=torch.float32,
        )

        self.pathway = torch.tensor(
            pathway_x,
            dtype=torch.float32,
        )

        self.regulon = torch.tensor(
            regulon_x,
            dtype=torch.float32,
        )

        n = len(
            self.model_ids
        )

        if not (
            len(self.expression)
            == len(self.pathway)
            == len(self.regulon)
            == n
        ):
            raise ValueError(
                "Biological-context branches "
                "have inconsistent sample counts."
            )

    def __len__(self):
        return len(
            self.model_ids
        )

    def __getitem__(self, idx):

        return {
            "ModelID":
                self.model_ids[idx],

            "expression":
                self.expression[idx],

            "pathway":
                self.pathway[idx],

            "regulon":
                self.regulon[idx],
        }


# ============================================================
# Branch encoder
# ============================================================

class BranchEncoder(nn.Module):
    """
    Small encoder used independently for each biological view.

    Each branch is compressed to the SAME output dimension.
    This prevents the 942-gene expression branch from
    automatically dominating simply because it has more inputs.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout=0.15,
    ):

        super().__init__()

        self.network = nn.Sequential(

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
                output_dim,
            ),

            nn.LayerNorm(
                output_dim
            ),

            nn.GELU(),
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# Biological-context encoder
# ============================================================

class BiologicalContextEncoder(nn.Module):
    """
    Expression + pathways + regulons
             |
         three encoders
             |
        concatenate
             |
           fusion
             |
    biological-context embedding
    """

    def __init__(
        self,
        expression_dim,
        pathway_dim,
        regulon_dim,

        branch_dim=64,

        expression_hidden=256,
        pathway_hidden=64,
        regulon_hidden=256,

        fusion_hidden=256,

        embedding_dim=128,

        dropout=0.15,
    ):

        super().__init__()

        # ----------------------------------------------------
        # Independent biological-view encoders
        # ----------------------------------------------------

        self.expression_encoder = (
            BranchEncoder(
                input_dim=expression_dim,
                hidden_dim=expression_hidden,
                output_dim=branch_dim,
                dropout=dropout,
            )
        )

        self.pathway_encoder = (
            BranchEncoder(
                input_dim=pathway_dim,
                hidden_dim=pathway_hidden,
                output_dim=branch_dim,
                dropout=dropout,
            )
        )

        self.regulon_encoder = (
            BranchEncoder(
                input_dim=regulon_dim,
                hidden_dim=regulon_hidden,
                output_dim=branch_dim,
                dropout=dropout,
            )
        )

        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        fusion_input_dim = (
            branch_dim * 3
        )

        self.fusion = nn.Sequential(

            nn.Linear(
                fusion_input_dim,
                fusion_hidden,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                fusion_hidden,
                embedding_dim,
            ),

            nn.LayerNorm(
                embedding_dim
            ),
        )

        self.embedding_dim = (
            embedding_dim
        )

    def forward(
        self,
        expression,
        pathway,
        regulon,
        return_branches=False,
    ):

        # ----------------------------------------------------
        # Encode the three complementary biological views
        # ----------------------------------------------------

        expression_embedding = (
            self.expression_encoder(
                expression
            )
        )

        pathway_embedding = (
            self.pathway_encoder(
                pathway
            )
        )

        regulon_embedding = (
            self.regulon_encoder(
                regulon
            )
        )

        # ----------------------------------------------------
        # Concatenate
        # ----------------------------------------------------

        combined = torch.cat(
            [
                expression_embedding,
                pathway_embedding,
                regulon_embedding,
            ],
            dim=-1,
        )

        # ----------------------------------------------------
        # Fused biological-context representation
        # ----------------------------------------------------

        context_embedding = (
            self.fusion(
                combined
            )
        )

        if return_branches:

            return {
                "context":
                    context_embedding,

                "expression":
                    expression_embedding,

                "pathway":
                    pathway_embedding,

                "regulon":
                    regulon_embedding,
            }

        return context_embedding


# ============================================================
# Convenience loading function
# ============================================================

def load_biological_context(
    expression_path,
    pathway_path,
    regulon_path,
):

    print(
        "\n========================================"
    )
    print(
        "Loading biological-context representations"
    )
    print(
        "========================================"
    )

    expression = load_matrix(
        expression_path,
        "Expression",
    )

    pathway = load_matrix(
        pathway_path,
        "Pathways",
    )

    regulon = load_matrix(
        regulon_path,
        "Regulons",
    )

    return align_representations(
        expression,
        pathway,
        regulon,
    )


# ============================================================
# Save aligned feature definitions
# ============================================================

def save_feature_manifest(
    expression,
    pathway,
    regulon,
    outdir,
):

    outdir = Path(outdir)

    rows = []

    for i, feature in enumerate(
        expression.columns
    ):
        rows.append(
            {
                "branch":
                    "expression",
                "feature_index":
                    i,
                "feature":
                    feature,
            }
        )

    for i, feature in enumerate(
        pathway.columns
    ):
        rows.append(
            {
                "branch":
                    "pathway",
                "feature_index":
                    i,
                "feature":
                    feature,
            }
        )

    for i, feature in enumerate(
        regulon.columns
    ):
        rows.append(
            {
                "branch":
                    "regulon",
                "feature_index":
                    i,
                "feature":
                    feature,
            }
        )

    manifest = pd.DataFrame(
        rows
    )

    manifest.to_csv(
        outdir
        / "biological_context_feature_manifest.tsv",
        sep="\t",
        index=False,
    )


# ============================================================
# Diagnostic main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--expression-mode",
        choices=[
            "tpm",
            "rank",
        ],
        default="tpm",
        help=(
            "Primary expression input. "
            "'rank' is the robustness ablation."
        ),
    )

    parser.add_argument(
        "--regulon-mode",
        choices=[
            "fixed",
            "contextualized",
        ],
        default="fixed",
        help=(
            "Fixed is leakage-safe for the current "
            "precomputed matrices. Contextualized "
            "requires train-only network fitting "
            "during strict CV."
        ),
    )

    parser.add_argument(
        "--expression-tpm",
        default=str(
            DEFAULT_EXPRESSION_TPM
        ),
    )

    parser.add_argument(
        "--expression-rank",
        default=str(
            DEFAULT_EXPRESSION_RANK
        ),
    )

    parser.add_argument(
        "--pathway",
        default=str(
            DEFAULT_PATHWAY
        ),
    )

    parser.add_argument(
        "--regulon-fixed",
        default=str(
            DEFAULT_REGULON_FIXED
        ),
    )

    parser.add_argument(
        "--regulon-contextualized",
        default=str(
            DEFAULT_REGULON_CONTEXTUALIZED
        ),
    )

    parser.add_argument(
        "--outdir",
        default=str(
            DEFAULT_OUTDIR
        ),
    )

    parser.add_argument(
        "--branch-dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
    )

    args = parser.parse_args()

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Choose expression representation
    # ========================================================

    if args.expression_mode == "tpm":

        expression_path = (
            args.expression_tpm
        )

    else:

        expression_path = (
            args.expression_rank
        )

    # ========================================================
    # Choose regulon representation
    # ========================================================

    if args.regulon_mode == "fixed":

        regulon_path = (
            args.regulon_fixed
        )

    else:

        regulon_path = (
            args.regulon_contextualized
        )

        print(
            "\nWARNING:"
        )
        print(
            "The current contextualized regulon matrix "
            "was fitted using the full DepMap panel."
        )
        print(
            "Do NOT use it directly for strict "
            "leave-cell-line validation."
        )

    # ========================================================
    # Load / align
    # ========================================================

    (
        expression,
        pathway,
        regulon,
    ) = load_biological_context(

        expression_path=
            expression_path,

        pathway_path=
            args.pathway,

        regulon_path=
            regulon_path,
    )

    # ========================================================
    # Save aligned ModelIDs
    # ========================================================

    pd.DataFrame(
        {
            "ModelID":
                expression.index
        }
    ).to_csv(
        outdir
        / "biological_context_ModelIDs.tsv",
        sep="\t",
        index=False,
    )

    save_feature_manifest(
        expression,
        pathway,
        regulon,
        outdir,
    )

    # ========================================================
    # IMPORTANT:
    #
    # Below we fit preprocessing on all models ONLY as a
    # diagnostic test of the module.
    #
    # During actual synergy-model CV:
    #
    #     preprocessor.fit(TRAIN MODELS ONLY)
    #
    # Never fit it globally.
    # ========================================================

    preprocessor = (
        BiologicalContextPreprocessor()
    )

    preprocessor.fit(
        expression,
        pathway,
        regulon,
    )

    transformed = (
        preprocessor.transform(
            expression,
            pathway,
            regulon,
        )
    )

    # ========================================================
    # Instantiate network
    # ========================================================

    model = BiologicalContextEncoder(

        expression_dim=
            expression.shape[1],

        pathway_dim=
            pathway.shape[1],

        regulon_dim=
            regulon.shape[1],

        branch_dim=
            args.branch_dim,

        embedding_dim=
            args.embedding_dim,
    )

    # ========================================================
    # Architecture summary
    # ========================================================

    n_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "\n========================================"
    )
    print(
        "Biological-context encoder"
    )
    print(
        "========================================"
    )

    print(model)

    print(
        f"\nTrainable parameters: "
        f"{n_parameters:,}"
    )

    # ========================================================
    # Test forward pass
    #
    # NOTE:
    # output is NOT biologically meaningful yet because
    # network weights have not been trained.
    # ========================================================

    test_n = min(
        8,
        len(expression),
    )

    expression_tensor = torch.tensor(
        transformed[
            "expression"
        ][:test_n],
        dtype=torch.float32,
    )

    pathway_tensor = torch.tensor(
        transformed[
            "pathway"
        ][:test_n],
        dtype=torch.float32,
    )

    regulon_tensor = torch.tensor(
        transformed[
            "regulon"
        ][:test_n],
        dtype=torch.float32,
    )

    model.eval()

    with torch.no_grad():

        output = model(
            expression_tensor,
            pathway_tensor,
            regulon_tensor,
            return_branches=True,
        )

    print(
        "\nTest batch:"
    )

    print(
        "Expression branch:",
        tuple(
            output[
                "expression"
            ].shape
        ),
    )

    print(
        "Pathway branch:",
        tuple(
            output[
                "pathway"
            ].shape
        ),
    )

    print(
        "Regulon branch:",
        tuple(
            output[
                "regulon"
            ].shape
        ),
    )

    print(
        "Final context embedding:",
        tuple(
            output[
                "context"
            ].shape
        ),
    )

    # ========================================================
    # Save configuration
    # ========================================================

    config = {

        "expression_mode":
            args.expression_mode,

        "regulon_mode":
            args.regulon_mode,

        "n_contexts":
            int(
                expression.shape[0]
            ),

        "expression_dim":
            int(
                expression.shape[1]
            ),

        "pathway_dim":
            int(
                pathway.shape[1]
            ),

        "regulon_dim":
            int(
                regulon.shape[1]
            ),

        "branch_embedding_dim":
            int(
                args.branch_dim
            ),

        "final_context_embedding_dim":
            int(
                args.embedding_dim
            ),

        "fusion_strategy":
            "independent encoders -> concatenate -> MLP fusion",

        "normalization":
            "feature-wise train-only standardization",

        "training_strategy":
            "end-to-end with downstream drug-synergy objective",
    }

    with open(
        outdir
        / "biological_context_config.json",
        "w",
    ) as handle:

        json.dump(
            config,
            handle,
            indent=2,
        )

    print(
        "\n========================================"
    )
    print(
        "MODULE CHECK COMPLETE"
    )
    print(
        "========================================"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "No embedding CSV has been saved."
    )

    print(
        "The neural encoder is currently untrained."
    )

    print(
        "Its 128-dimensional output becomes meaningful "
        "only after training with the synergy model."
    )

    print(
        f"\nConfiguration written to:\n"
        f"{outdir.resolve()}"
    )


if __name__ == "__main__":
    main()
