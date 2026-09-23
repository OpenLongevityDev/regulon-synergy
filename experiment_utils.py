#!/usr/bin/env python3

"""
experiment_utils.py

Shared utilities used by the DrugComb publication experiment runners.

This module is derived from the final audited regulon-ablation implementation
and contains the common model construction, data loading, scaffold-aware split
generation and verification, train-only context normalization, batching,
training/evaluation, symmetry checks, prediction export, and plotting helpers.

The experiment-specific regulon ON/OFF driver lives in
``train_regulon_ablation.py``.

Compatibility note
------------------
Historical result dictionaries use keys ending in ``_auprc``. These values are
computed with ``sklearn.metrics.average_precision_score`` and therefore
correspond to average precision (AP). The legacy keys are retained so that the
publication code remains compatible with the archived result files.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


# ============================================================
# Paths
# ============================================================

# Repository root. By default this is the directory containing this module.
# Set DRUG_SYNERGY_ROOT to point to another checkout/data root if needed.
ROOT = Path(
    os.environ.get("DRUG_SYNERGY_ROOT", Path(__file__).resolve().parent)
).resolve()

MODEL_READY_PATH = (
    ROOT / "model_ready" / "drugcomb_model_ready_v1.tsv"
)

SYNERGY_DATASET_PATH = ROOT / "synergy_dataset.py"
SYNERGY_MODEL_PATH = ROOT / "synergy_model.py"
BIOLOGICAL_CONTEXT_PATH = ROOT / "biological_context_module.py"
INTRINSIC_ENCODER_PATH = ROOT / "drugs_x_context" / "intrinsic_drug_encoder.py"
CONTEXT_CONDITIONER_PATH = ROOT / "drugs_x_context" / "context_conditioning.py"

# Canonical SMILES created by the earlier Morgan preprocessing.
# Used only to construct scaffold-aware drug holdouts.
DRUG_CANONICAL_PATH = (
    ROOT / "smiles" / "smiles_results_morgan" / "DrugComb_SMILES_canonical.tsv"
)

# Default split location. Experiment runners may override this module variable
# before calling save_split_assignment().
SPLIT_DIR = ROOT / "training" / "splits"


# ============================================================
# Constants
# ============================================================

CLASS_NAMES = ["antagonism", "no_interaction", "synergy"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
N_CLASSES = 3

HOLDOUTS = [
    "unseen_cell_line",
    "unseen_drug",
    "unseen_both",
]


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    seed: int = 42

    # Desired observation-level proportions.
    # Group constraints mean actual proportions can differ.
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15

    split_search_trials: int = 50

    # Drug holdout:
    # "scaffold" = Bemis-Murcko scaffold groups
    # "structure" = canonical SMILES groups
    # "drug_id" = DrugComb ID (least strict; not recommended for paper)
    drug_grouping: str = "scaffold"

    # Test observation eligibility for unseen-drug experiments:
    # "any"  = at least one drug is held out
    # "both" = both drugs are held out
    drug_test_mode: str = "any"

    # Validation follows the same semantics as test:
    # at least one validation drug group in the pair by default.
    # Train excludes ANY pair containing val/test drug groups.

    batch_size: int = 128
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    dropout: float = 0.20
    gradient_clip: float = 5.0
    early_stopping_patience: int = 8

    scheduler_factor: float = 0.5
    scheduler_patience: int = 2

    num_workers: int = 0
    pin_memory: bool = True
    use_amp: bool = True
    deterministic: bool = False


# ============================================================
# General utilities
# ============================================================

def print_section(title: str):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def import_module_from_path(module_name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(module_name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")

    module = importlib.util.module_from_spec(spec)

    # Required for @dataclass and some typing machinery during execution.
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return module


def find_class(module, candidates: Sequence[str]):
    for name in candidates:
        if hasattr(module, name):
            obj = getattr(module, name)
            if inspect.isclass(obj):
                return obj

    raise AttributeError(
        f"Could not find class among {list(candidates)} in {module.__name__}"
    )


def instantiate_with_supported_kwargs(cls, kwargs: Dict):
    signature = inspect.signature(cls.__init__)
    params = signature.parameters

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        supported = kwargs
    else:
        supported = {k: v for k, v in kwargs.items() if k in params}

    try:
        return cls(**supported)
    except Exception:
        print(f"\nCould not instantiate {cls.__name__}")
        print("Signature:", signature)
        print("Supplied:", supported)
        raise


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def safe_float(x):
    if x is None:
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


# ============================================================
# Context encoders
# ============================================================

class BranchEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.network(x)


class BiologicalContextEncoderNoRegulons(nn.Module):
    """
    Same design philosophy as the existing BiologicalContextEncoder,
    but only expression + pathway branches.

    Expression: 942 -> 256 -> 64
    Pathway:     50  -> 64  -> 64
    Concat:      128
    Fusion:      128 -> 256 -> 128

    The final context embedding remains 128-d, so the rest of the
    architecture is identical to the +regulon condition.
    """

    def __init__(
        self,
        expression_dim: int = 942,
        pathway_dim: int = 50,
        expression_hidden: int = 256,
        pathway_hidden: int = 64,
        branch_dim: int = 64,
        fusion_hidden: int = 256,
        output_dim: int = 128,
        dropout: float = 0.20,
        **kwargs,
    ):
        super().__init__()

        self.expression_encoder = BranchEncoder(
            input_dim=expression_dim,
            hidden_dim=expression_hidden,
            output_dim=branch_dim,
            dropout=dropout,
        )

        self.pathway_encoder = BranchEncoder(
            input_dim=pathway_dim,
            hidden_dim=pathway_hidden,
            output_dim=branch_dim,
            dropout=dropout,
        )

        self.fusion = nn.Sequential(
            nn.Linear(branch_dim * 2, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        expression,
        pathways,
        regulons=None,
        return_aux=False,
        **kwargs,
    ):
        expression_embedding = self.expression_encoder(expression)
        pathway_embedding = self.pathway_encoder(pathways)

        fused = torch.cat(
            [expression_embedding, pathway_embedding],
            dim=-1,
        )

        embedding = self.fusion(fused)

        if return_aux:
            return {
                "embedding": embedding,
                "expression_embedding": expression_embedding,
                "pathway_embedding": pathway_embedding,
            }

        return embedding


# ============================================================
# Full model construction
# ============================================================

def build_core_model(cfg: Config, use_regulons: bool):
    print_section(
        f"BUILD MODEL | regulons={'ON' if use_regulons else 'OFF'}"
    )

    bio_module = import_module_from_path(
        f"biological_context_module_{id(cfg)}",
        BIOLOGICAL_CONTEXT_PATH,
    )
    intrinsic_module = import_module_from_path(
        f"intrinsic_drug_encoder_{id(cfg)}",
        INTRINSIC_ENCODER_PATH,
    )
    conditioner_module = import_module_from_path(
        f"context_conditioning_{id(cfg)}",
        CONTEXT_CONDITIONER_PATH,
    )
    synergy_module = import_module_from_path(
        f"synergy_model_{id(cfg)}",
        SYNERGY_MODEL_PATH,
    )

    IntrinsicClass = find_class(
        intrinsic_module,
        ["IntrinsicDrugEncoder"],
    )
    ConditionerClass = find_class(
        conditioner_module,
        ["ContextConditioner", "DrugContextConditioner"],
    )
    ModelClass = find_class(
        synergy_module,
        ["DrugCombinationModel"],
    )

    if use_regulons:
        BioClass = find_class(
            bio_module,
            ["BiologicalContextEncoder", "BiologicalContextModule"],
        )

        biological_context_encoder = instantiate_with_supported_kwargs(
            BioClass,
            {
                "expression_dim": 942,
                "pathway_dim": 50,
                "regulon_dim": 771,
                "expression_input_dim": 942,
                "pathway_input_dim": 50,
                "regulon_input_dim": 771,
                "output_dim": 128,
                "embedding_dim": 128,
                "context_dim": 128,
                "dropout": cfg.dropout,
            },
        )
    else:
        biological_context_encoder = BiologicalContextEncoderNoRegulons(
            expression_dim=942,
            pathway_dim=50,
            output_dim=128,
            dropout=cfg.dropout,
        )

    intrinsic_drug_encoder = instantiate_with_supported_kwargs(
        IntrinsicClass,
        {
            "morgan_dim": 2048,
            "target_dim": 8245,
            "direct_target_dim": 8245,
            "ppi_dim": 8245,
            "availability_dim": 3,
            "output_dim": 128,
            "embedding_dim": 128,
            "drug_dim": 128,
            "dropout": cfg.dropout,
        },
    )

    context_conditioner = instantiate_with_supported_kwargs(
        ConditionerClass,
        {
            "drug_dim": 128,
            "context_dim": 128,
            "output_dim": 128,
            "embedding_dim": 128,
            "dropout": cfg.dropout,
        },
    )

    sig = inspect.signature(ModelClass.__init__)
    params = sig.parameters
    model_kwargs = {}

    for name in [
        "intrinsic_drug_encoder",
        "drug_encoder",
        "intrinsic_encoder",
    ]:
        if name in params:
            model_kwargs[name] = intrinsic_drug_encoder
            break

    for name in [
        "biological_context_encoder",
        "context_encoder",
        "biological_context_module",
    ]:
        if name in params:
            model_kwargs[name] = biological_context_encoder
            break

    for name in [
        "context_conditioner",
        "drug_context_conditioner",
        "conditioner",
    ]:
        if name in params:
            model_kwargs[name] = context_conditioner
            break

    optional = {
        "embedding_dim": 128,
        "n_classes": 3,
        "num_classes": 3,
        "dropout": cfg.dropout,
        "use_l1000_aux": False,
        "use_l1000_auxiliary": False,
        "use_l1000": False,
    }

    for k, v in optional.items():
        if k in params:
            model_kwargs[k] = v

    model = ModelClass(**model_kwargs)

    total, trainable = count_parameters(model)
    print("Total parameters:", f"{total:,}")
    print("Trainable parameters:", f"{trainable:,}")

    return model


# ============================================================
# Feature/data loading
# ============================================================

def load_dataset_module():
    return import_module_from_path(
        "synergy_dataset_regulon_ablation",
        SYNERGY_DATASET_PATH,
    )


def load_master(dataset_module):
    master = pd.read_csv(
        MODEL_READY_PATH,
        sep="\t",
        low_memory=False,
    )

    for col in ["drug_a_id_std", "drug_b_id_std", "depmap_id_std"]:
        master[col] = master[col].map(dataset_module.normalize_id)

    required = [
        "sample_id",
        "pair_context_id",
        "drug_a_id_std",
        "drug_b_id_std",
        "depmap_id_std",
        "label_int",
        "morgan_a_idx",
        "morgan_b_idx",
        "direct_a_idx",
        "direct_b_idx",
        "ppi_a_idx",
        "ppi_b_idx",
        "expression_idx",
        "pathways_idx",
        "regulons_idx",
    ]

    missing = [c for c in required if c not in master.columns]

    if missing:
        raise ValueError(
            "Model-ready file is missing required columns:\n"
            + "\n".join(missing)
        )

    return master


# ============================================================
# Drug scaffold groups
# ============================================================

def autodetect_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    return None


def load_drug_groups(
    master: pd.DataFrame,
    dataset_module,
    grouping: str,
) -> Dict[str, str]:
    """
    Returns:
        normalized DrugComb ID -> holdout group
    """

    grouping = grouping.lower()

    all_drug_ids = sorted(
        set(master["drug_a_id_std"].dropna().astype(str))
        | set(master["drug_b_id_std"].dropna().astype(str))
    )

    if grouping == "drug_id":
        return {x: f"drug::{x}" for x in all_drug_ids}

    if not DRUG_CANONICAL_PATH.exists():
        raise FileNotFoundError(
            f"Canonical SMILES table not found:\n{DRUG_CANONICAL_PATH}\n"
            "Use --drug-grouping drug_id only as a temporary fallback."
        )

    smiles_df = pd.read_csv(
        DRUG_CANONICAL_PATH,
        sep="\t",
        low_memory=False,
    )

    id_col = autodetect_column(
        smiles_df,
        ["drugcomb_id", "drug_id", "id"],
    )

    smiles_col = autodetect_column(
        smiles_df,
        ["canonical_smiles", "canonical_smile", "smiles_canonical"],
    )

    if id_col is None or smiles_col is None:
        raise ValueError(
            "Could not detect DrugComb ID / canonical SMILES columns in:\n"
            f"{DRUG_CANONICAL_PATH}\n"
            f"Columns: {smiles_df.columns.tolist()}"
        )

    smiles_df = smiles_df[[id_col, smiles_col]].copy()

    smiles_df["drug_id_std"] = smiles_df[id_col].map(
        dataset_module.normalize_id
    )

    smiles_df = smiles_df.dropna(
        subset=["drug_id_std", smiles_col]
    )

    smiles_df["drug_id_std"] = smiles_df["drug_id_std"].astype(str)
    smiles_df[smiles_col] = smiles_df[smiles_col].astype(str)

    # If duplicates exist, require one canonical SMILES per DrugComb ID.
    n_smiles = smiles_df.groupby("drug_id_std")[smiles_col].nunique()

    ambiguous_ids = set(
        n_smiles[n_smiles > 1].index.astype(str)
    )

    if ambiguous_ids:
        print(
            f"WARNING: {len(ambiguous_ids)} DrugComb IDs have >1 canonical SMILES; "
            "they will be grouped by DrugComb ID to avoid accidental leakage."
        )

    first_smiles = (
        smiles_df
        .drop_duplicates("drug_id_std")
        .set_index("drug_id_std")[smiles_col]
        .to_dict()
    )

    if grouping == "structure":
        result = {}

        for drug_id in all_drug_ids:
            if drug_id in ambiguous_ids or drug_id not in first_smiles:
                result[drug_id] = f"unresolved::{drug_id}"
            else:
                result[drug_id] = f"structure::{first_smiles[drug_id]}"

        return result

    if grouping != "scaffold":
        raise ValueError(
            f"Unsupported drug grouping: {grouping}"
        )

    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as e:
        raise ImportError(
            "RDKit is required for scaffold-aware holdout.\n"
            "Install RDKit or use --drug-grouping structure."
        ) from e

    result = {}
    failed = 0
    empty_scaffold = 0

    for drug_id in all_drug_ids:
        smi = first_smiles.get(drug_id)

        if drug_id in ambiguous_ids or smi is None:
            result[drug_id] = f"unresolved::{drug_id}"
            continue

        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            failed += 1
            result[drug_id] = f"unresolved::{drug_id}"
            continue

        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=False,
        )

        # Acyclic compounds can have an empty Murcko scaffold.
        # Do NOT put all of them into one giant holdout group.
        if scaffold is None or scaffold == "":
            empty_scaffold += 1
            result[drug_id] = f"acyclic_structure::{smi}"
        else:
            result[drug_id] = f"scaffold::{scaffold}"

    print(
        f"Drug groups: {len(set(result.values())):,} groups for "
        f"{len(result):,} drugs | failed SMILES={failed:,} | "
        f"acyclic structure groups={empty_scaffold:,}"
    )

    return result


# ============================================================
# Split helpers
# ============================================================

def class_proportions(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=N_CLASSES).astype(float)
    return counts / counts.sum() if counts.sum() else np.zeros(N_CLASSES)


def all_classes_present(df: pd.DataFrame) -> bool:
    return set(df["label_int"].astype(int).unique()) == {0, 1, 2}


def split_balance_score(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    master: pd.DataFrame,
    cfg: Config,
) -> float:
    global_prop = class_proportions(master["label_int"].to_numpy(dtype=int))

    target_sizes = np.array(
        [cfg.train_fraction, cfg.val_fraction, cfg.test_fraction]
    )

    actual_sizes = np.array(
        [
            len(train_df) / len(master),
            len(val_df) / len(master),
            len(test_df) / len(master),
        ]
    )

    size_error = np.abs(actual_sizes - target_sizes).sum()

    class_error = 0.0

    for df in [train_df, val_df, test_df]:
        p = class_proportions(df["label_int"].to_numpy(dtype=int))
        class_error += np.abs(p - global_prop).mean()

    # Heavier penalty if test becomes tiny.
    tiny_penalty = 0.0

    if len(test_df) < 0.05 * len(master):
        tiny_penalty += 5.0

    if len(val_df) < 0.05 * len(master):
        tiny_penalty += 5.0

    return float(size_error + class_error + tiny_penalty)


def split_context_groups(
    contexts: np.ndarray,
    seed: int,
    cfg: Config,
) -> Tuple[set, set, set]:
    """
    Splits unique contexts themselves, not observations.
    """

    unique_contexts = np.array(sorted(set(contexts.astype(str))))

    rng = np.random.default_rng(seed)
    shuffled = unique_contexts.copy()
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_test = max(1, int(round(cfg.test_fraction * n_total)))
    n_val = max(1, int(round(cfg.val_fraction * n_total)))

    test = set(shuffled[:n_test])
    val = set(shuffled[n_test:n_test + n_val])
    train = set(shuffled[n_test + n_val:])

    return train, val, test


def assign_drug_group_sets(
    unique_groups: Sequence[str],
    seed: int,
    cfg: Config,
) -> Tuple[set, set, set]:
    groups = np.array(sorted(set(unique_groups)), dtype=object)

    rng = np.random.default_rng(seed)
    shuffled = groups.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)

    n_test = max(1, int(round(cfg.test_fraction * n)))
    n_val = max(1, int(round(cfg.val_fraction * n)))

    test_groups = set(shuffled[:n_test])
    val_groups = set(shuffled[n_test:n_test + n_val])
    train_groups = set(shuffled[n_test + n_val:])

    return train_groups, val_groups, test_groups


def pair_has_group(
    ga: pd.Series,
    gb: pd.Series,
    groups: set,
    mode: str,
) -> pd.Series:
    a = ga.isin(groups)
    b = gb.isin(groups)

    if mode == "any":
        return a | b

    if mode == "both":
        return a & b

    raise ValueError(f"Unknown drug_test_mode: {mode}")


def add_drug_group_columns(
    master: pd.DataFrame,
    drug_group_map: Dict[str, str],
) -> pd.DataFrame:
    out = master.copy()

    out["drug_a_group"] = (
        out["drug_a_id_std"].astype(str).map(drug_group_map)
    )
    out["drug_b_group"] = (
        out["drug_b_id_std"].astype(str).map(drug_group_map)
    )

    if out["drug_a_group"].isna().any() or out["drug_b_group"].isna().any():
        missing = sorted(
            set(out.loc[out["drug_a_group"].isna(), "drug_a_id_std"].astype(str))
            | set(out.loc[out["drug_b_group"].isna(), "drug_b_id_std"].astype(str))
        )
        raise ValueError(
            "Missing drug holdout groups for IDs:\n"
            + "\n".join(missing[:50])
        )

    return out


def construct_split(
    master: pd.DataFrame,
    holdout: str,
    seed: int,
    cfg: Config,
    drug_group_map: Optional[Dict[str, str]] = None,
):
    """
    Returns train_idx, val_idx, test_idx, metadata.

    Important semantics
    -------------------
    unseen_cell_line:
        test rows have held-out contexts.
        train/val/test contexts are disjoint.

    unseen_drug:
        train excludes any row containing validation/test drug groups.
        val rows contain validation drug group(s), but no test drug group.
        test rows contain test drug group(s), but no validation drug group.

    unseen_both:
        train rows use train contexts and train drugs only.
        validation rows require validation context AND validation drug(s).
        test rows require test context AND test drug(s).
        Cross-partition rows are discarded.
    """

    if holdout not in HOLDOUTS:
        raise ValueError(holdout)

    best = None
    best_score = np.inf

    contexts = master["depmap_id_std"].astype(str).to_numpy()

    if holdout in {"unseen_drug", "unseen_both"}:
        if drug_group_map is None:
            raise ValueError("drug_group_map required")

        working = add_drug_group_columns(master, drug_group_map)

        all_groups = sorted(
            set(working["drug_a_group"])
            | set(working["drug_b_group"])
        )
    else:
        working = master.copy()
        all_groups = None

    for trial in range(cfg.split_search_trials):
        trial_seed = seed + trial

        if holdout == "unseen_cell_line":
            train_ctx, val_ctx, test_ctx = split_context_groups(
                contexts,
                trial_seed,
                cfg,
            )

            train_mask = working["depmap_id_std"].astype(str).isin(train_ctx)
            val_mask = working["depmap_id_std"].astype(str).isin(val_ctx)
            test_mask = working["depmap_id_std"].astype(str).isin(test_ctx)

            split_meta = {
                "train_contexts": sorted(train_ctx),
                "validation_contexts": sorted(val_ctx),
                "test_contexts": sorted(test_ctx),
                "train_drug_groups": None,
                "validation_drug_groups": None,
                "test_drug_groups": None,
            }

        else:
            train_dg, val_dg, test_dg = assign_drug_group_sets(
                all_groups,
                trial_seed + 100_000,
                cfg,
            )

            has_val_any = (
                working["drug_a_group"].isin(val_dg)
                | working["drug_b_group"].isin(val_dg)
            )
            has_test_any = (
                working["drug_a_group"].isin(test_dg)
                | working["drug_b_group"].isin(test_dg)
            )

            val_eligible = pair_has_group(
                working["drug_a_group"],
                working["drug_b_group"],
                val_dg,
                cfg.drug_test_mode,
            )
            test_eligible = pair_has_group(
                working["drug_a_group"],
                working["drug_b_group"],
                test_dg,
                cfg.drug_test_mode,
            )

            train_drug_mask = ~(has_val_any | has_test_any)
            val_drug_mask = val_eligible & ~has_test_any
            test_drug_mask = test_eligible & ~has_val_any

            if holdout == "unseen_drug":
                # Contexts are allowed across splits.
                train_mask = train_drug_mask
                val_mask = val_drug_mask
                test_mask = test_drug_mask

                split_meta = {
                    "train_contexts": None,
                    "validation_contexts": None,
                    "test_contexts": None,
                    "train_drug_groups": sorted(train_dg),
                    "validation_drug_groups": sorted(val_dg),
                    "test_drug_groups": sorted(test_dg),
                }

            else:
                train_ctx, val_ctx, test_ctx = split_context_groups(
                    contexts,
                    trial_seed,
                    cfg,
                )

                context_train = working["depmap_id_std"].astype(str).isin(train_ctx)
                context_val = working["depmap_id_std"].astype(str).isin(val_ctx)
                context_test = working["depmap_id_std"].astype(str).isin(test_ctx)

                train_mask = train_drug_mask & context_train
                val_mask = val_drug_mask & context_val
                test_mask = test_drug_mask & context_test

                split_meta = {
                    "train_contexts": sorted(train_ctx),
                    "validation_contexts": sorted(val_ctx),
                    "test_contexts": sorted(test_ctx),
                    "train_drug_groups": sorted(train_dg),
                    "validation_drug_groups": sorted(val_dg),
                    "test_drug_groups": sorted(test_dg),
                }

        train_df = working.loc[train_mask]
        val_df = working.loc[val_mask]
        test_df = working.loc[test_mask]

        if min(len(train_df), len(val_df), len(test_df)) == 0:
            continue

        if not all_classes_present(train_df):
            continue
        if not all_classes_present(val_df):
            continue
        if not all_classes_present(test_df):
            continue

        score = split_balance_score(
            train_df,
            val_df,
            test_df,
            working,
            cfg,
        )

        if score < best_score:
            best_score = score
            best = (
                train_df.index.to_numpy(dtype=np.int64),
                val_df.index.to_numpy(dtype=np.int64),
                test_df.index.to_numpy(dtype=np.int64),
                trial_seed,
                split_meta,
                working,
            )

    if best is None:
        raise RuntimeError(
            f"Could not create valid split for {holdout}. "
            "Try increasing split_search_trials or using drug_test_mode='any'."
        )

    train_idx, val_idx, test_idx, chosen_seed, split_meta, working = best

    verify_split(
        working,
        train_idx,
        val_idx,
        test_idx,
        holdout,
        split_meta,
    )

    metadata = {
        "holdout": holdout,
        "requested_seed": seed,
        "chosen_seed": int(chosen_seed),
        "drug_grouping": cfg.drug_grouping,
        "drug_test_mode": cfg.drug_test_mode,
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_train_contexts": int(
            working.loc[train_idx, "depmap_id_std"].nunique()
        ),
        "n_validation_contexts": int(
            working.loc[val_idx, "depmap_id_std"].nunique()
        ),
        "n_test_contexts": int(
            working.loc[test_idx, "depmap_id_std"].nunique()
        ),
        "split_balance_score": float(best_score),
        **split_meta,
    }

    return train_idx, val_idx, test_idx, metadata, working


def verify_split(
    master: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    holdout: str,
    metadata: Dict,
):
    train = master.loc[train_idx]
    val = master.loc[val_idx]
    test = master.loc[test_idx]

    # pair_context_id must never cross partitions.
    train_pc = set(train["pair_context_id"].astype(str))
    val_pc = set(val["pair_context_id"].astype(str))
    test_pc = set(test["pair_context_id"].astype(str))

    assert not (train_pc & val_pc)
    assert not (train_pc & test_pc)
    assert not (val_pc & test_pc)

    if holdout in {"unseen_cell_line", "unseen_both"}:
        train_c = set(train["depmap_id_std"].astype(str))
        val_c = set(val["depmap_id_std"].astype(str))
        test_c = set(test["depmap_id_std"].astype(str))

        assert not (train_c & val_c)
        assert not (train_c & test_c)
        assert not (val_c & test_c)

    if holdout in {"unseen_drug", "unseen_both"}:
        # With drug_test_mode="any", a validation/test pair may contain
        # one held-out drug plus one ordinary training drug. Therefore the
        # complete sets of drugs observed in the partitions are NOT expected
        # to be disjoint. What must be disjoint are the held-out GROUPS:
        # validation/test scaffold groups may never appear in training, and
        # test scaffold groups may never appear in validation (vice versa).
        train_groups_actual = (
            set(train["drug_a_group"])
            | set(train["drug_b_group"])
        )
        val_groups_actual = (
            set(val["drug_a_group"])
            | set(val["drug_b_group"])
        )
        test_groups_actual = (
            set(test["drug_a_group"])
            | set(test["drug_b_group"])
        )

        val_heldout = set(metadata["validation_drug_groups"])
        test_heldout = set(metadata["test_drug_groups"])

        assert not (train_groups_actual & val_heldout)
        assert not (train_groups_actual & test_heldout)
        assert not (val_groups_actual & test_heldout)
        assert not (test_groups_actual & val_heldout)

        # Every held-out partition must actually contain the intended group.
        if not (val_groups_actual & val_heldout):
            raise AssertionError("Validation contains no validation-held-out drug group.")
        if not (test_groups_actual & test_heldout):
            raise AssertionError("Test contains no test-held-out drug group.")


# ============================================================
# Split saving/loading
# ============================================================

def split_file_base(holdout: str, seed: int, cfg: Config) -> Path:
    suffix = (
        f"{holdout}"
        f"__seed{seed}"
        f"__druggroup-{cfg.drug_grouping}"
        f"__drugtest-{cfg.drug_test_mode}"
    )
    return SPLIT_DIR / suffix


def save_split_assignment(
    master: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    metadata: Dict,
    holdout: str,
    seed: int,
    cfg: Config,
):
    base = split_file_base(holdout, seed, cfg)

    assignment = master[
        [
            "sample_id",
            "pair_context_id",
            "drug_a_id_std",
            "drug_b_id_std",
            "depmap_id_std",
            "label_int",
        ]
    ].copy()

    assignment["split"] = "discarded"
    assignment.loc[train_idx, "split"] = "train"
    assignment.loc[val_idx, "split"] = "validation"
    assignment.loc[test_idx, "split"] = "test"

    assignment.to_csv(
        str(base) + ".tsv",
        sep="\t",
        index=False,
    )

    with open(str(base) + ".json", "w") as f:
        json.dump(metadata, f, indent=2)

    return assignment


# ============================================================
# Train-only normalization
# ============================================================

def fit_standardizer(X: np.ndarray, rows: np.ndarray):
    train_X = X[rows]

    mean = train_X.mean(axis=0, dtype=np.float64)
    std = train_X.std(axis=0, dtype=np.float64)

    std[std < 1e-8] = 1.0

    return mean.astype(np.float32), std.astype(np.float32)


def standardize_matrix(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    chunk_size: int = 256,
):
    for start in range(0, X.shape[0], chunk_size):
        end = min(start + chunk_size, X.shape[0])
        X[start:end] = (X[start:end] - mean) / std


def normalize_context_train_only(
    features,
    master: pd.DataFrame,
    train_idx: np.ndarray,
    use_regulons: bool,
    output_dir: Path,
):
    train = master.loc[train_idx]

    expr_rows = train["expression_idx"].astype(int).unique()
    path_rows = train["pathways_idx"].astype(int).unique()

    expr_mean, expr_std = fit_standardizer(features.expression, expr_rows)
    path_mean, path_std = fit_standardizer(features.pathways, path_rows)

    standardize_matrix(features.expression, expr_mean, expr_std)
    standardize_matrix(features.pathways, path_mean, path_std)

    save_dict = {
        "expression_mean": expr_mean,
        "expression_std": expr_std,
        "pathway_mean": path_mean,
        "pathway_std": path_std,
    }

    if use_regulons:
        reg_rows = train["regulons_idx"].astype(int).unique()

        reg_mean, reg_std = fit_standardizer(
            features.regulons,
            reg_rows,
        )

        standardize_matrix(
            features.regulons,
            reg_mean,
            reg_std,
        )

        save_dict["regulon_mean"] = reg_mean
        save_dict["regulon_std"] = reg_std

    np.savez_compressed(
        output_dir / "context_standardizers.npz",
        **save_dict,
    )


# ============================================================
# Efficient batching
# ============================================================

class RowIndexDataset(Dataset):
    def __init__(self, row_indices: np.ndarray):
        self.row_indices = np.asarray(row_indices, dtype=np.int64)

    def __len__(self):
        return len(self.row_indices)

    def __getitem__(self, i):
        return int(self.row_indices[i])


class FastBatchCollator:
    def __init__(self, master: pd.DataFrame, features):
        self.master = master
        self.features = features

    def __call__(self, row_indices: List[int]):
        rows = self.master.loc[row_indices]

        ma = rows["morgan_a_idx"].to_numpy(dtype=np.int64)
        mb = rows["morgan_b_idx"].to_numpy(dtype=np.int64)

        ta = rows["direct_a_idx"].to_numpy(dtype=np.int64)
        tb = rows["direct_b_idx"].to_numpy(dtype=np.int64)

        pa = rows["ppi_a_idx"].to_numpy(dtype=np.int64)
        pb = rows["ppi_b_idx"].to_numpy(dtype=np.int64)

        ex = rows["expression_idx"].to_numpy(dtype=np.int64)
        pw = rows["pathways_idx"].to_numpy(dtype=np.int64)
        rg = rows["regulons_idx"].to_numpy(dtype=np.int64)

        a_morgan = self.features.morgan[ma].astype(np.float32, copy=False)
        b_morgan = self.features.morgan[mb].astype(np.float32, copy=False)

        a_targets = (
            self.features.direct_targets[ta]
            .toarray()
            .astype(np.float32, copy=False)
        )
        b_targets = (
            self.features.direct_targets[tb]
            .toarray()
            .astype(np.float32, copy=False)
        )

        a_ppi = self.features.ppi[pa].astype(np.float32, copy=False)
        b_ppi = self.features.ppi[pb].astype(np.float32, copy=False)

        a_availability = np.stack(
            [
                self.features.availability[x]
                for x in rows["drug_a_id_std"]
            ]
        ).astype(np.float32, copy=False)

        b_availability = np.stack(
            [
                self.features.availability[x]
                for x in rows["drug_b_id_std"]
            ]
        ).astype(np.float32, copy=False)

        expression = self.features.expression[ex].astype(
            np.float32,
            copy=False,
        )
        pathways = self.features.pathways[pw].astype(
            np.float32,
            copy=False,
        )
        regulons = self.features.regulons[rg].astype(
            np.float32,
            copy=False,
        )

        labels = rows["label_int"].to_numpy(dtype=np.int64)

        return {
            "drug_a_morgan": torch.from_numpy(a_morgan),
            "drug_a_targets": torch.from_numpy(a_targets),
            "drug_a_ppi": torch.from_numpy(a_ppi),
            "drug_a_availability": torch.from_numpy(a_availability),

            "drug_b_morgan": torch.from_numpy(b_morgan),
            "drug_b_targets": torch.from_numpy(b_targets),
            "drug_b_ppi": torch.from_numpy(b_ppi),
            "drug_b_availability": torch.from_numpy(b_availability),

            "context_expression": torch.from_numpy(expression),
            "context_pathways": torch.from_numpy(pathways),
            "context_regulons": torch.from_numpy(regulons),

            "label": torch.from_numpy(labels),

            "row_index": torch.tensor(
                row_indices,
                dtype=torch.long,
            ),
        }


def make_loader(
    indices: np.ndarray,
    master: pd.DataFrame,
    features,
    cfg: Config,
    shuffle: bool,
):
    return DataLoader(
        RowIndexDataset(indices),
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
        collate_fn=FastBatchCollator(master, features),
        drop_last=False,
    )


# ============================================================
# Model forwarding
# ============================================================

MODEL_INPUT_KEYS = [
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
]


def move_batch_to_device(batch: Dict, device: torch.device):
    out = {}

    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v

    return out


def forward_model(model: nn.Module, batch: Dict):
    kwargs = {k: batch[k] for k in MODEL_INPUT_KEYS}

    output = model(**kwargs)

    if torch.is_tensor(output):
        return output

    if isinstance(output, dict):
        if "logits" in output:
            return output["logits"]
        if "synergy_logits" in output:
            return output["synergy_logits"]

    if isinstance(output, (tuple, list)) and len(output):
        if torch.is_tensor(output[0]):
            return output[0]

    raise TypeError(
        f"Could not extract logits from model output: {type(output)}"
    )


# ============================================================
# Metrics
# ============================================================

def calculate_class_weights(master: pd.DataFrame, train_idx: np.ndarray):
    y = master.loc[train_idx, "label_int"].to_numpy(dtype=int)

    counts = np.bincount(
        y,
        minlength=N_CLASSES,
    ).astype(np.float64)

    weights = counts.sum() / (N_CLASSES * counts)

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
):
    y_pred = np.argmax(probabilities, axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0,
    )

    metrics["per_class"] = {}

    for i, name in enumerate(CLASS_NAMES):
        metrics["per_class"][name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    y_bin = label_binarize(
        y_true,
        classes=[0, 1, 2],
    )

    per_class_auroc = {}
    per_class_auprc = {}

    for i, name in enumerate(CLASS_NAMES):
        try:
            per_class_auroc[name] = float(
                roc_auc_score(
                    y_bin[:, i],
                    probabilities[:, i],
                )
            )
        except ValueError:
            per_class_auroc[name] = None

        try:
            per_class_auprc[name] = float(
                average_precision_score(
                    y_bin[:, i],
                    probabilities[:, i],
                )
            )
        except ValueError:
            per_class_auprc[name] = None

    valid_roc = [
        x for x in per_class_auroc.values()
        if x is not None
    ]
    valid_pr = [
        x for x in per_class_auprc.values()
        if x is not None
    ]

    metrics["per_class_auroc"] = per_class_auroc
    metrics["per_class_auprc"] = per_class_auprc

    metrics["macro_ovr_auroc"] = (
        float(np.mean(valid_roc))
        if valid_roc
        else None
    )
    metrics["macro_auprc"] = (
        float(np.mean(valid_pr))
        if valid_pr
        else None
    )

    metrics["confusion_matrix"] = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2],
    ).tolist()

    return metrics


# ============================================================
# Training/evaluation
# ============================================================

def make_grad_scaler(enabled: bool):
    """
    Compatible with both newer and older PyTorch APIs.
    """
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
        )
    except Exception:
        return torch.cuda.amp.GradScaler(
            enabled=enabled,
        )


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    cfg,
):
    model.train()

    total_loss = 0.0
    n_total = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        labels = batch["label"]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=cfg.use_amp and device.type == "cuda",
        ):
            logits = forward_model(model, batch)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg.gradient_clip,
        )

        scaler.step(optimizer)
        scaler.update()

        bs = labels.shape[0]
        total_loss += loss.item() * bs
        n_total += bs

    return total_loss / n_total


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    cfg,
):
    model.eval()

    total_loss = 0.0
    n_total = 0

    labels_all = []
    probs_all = []
    rows_all = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        labels = batch["label"]

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=cfg.use_amp and device.type == "cuda",
        ):
            logits = forward_model(model, batch)
            loss = criterion(logits, labels)

        probs = torch.softmax(
            logits.float(),
            dim=1,
        )

        bs = labels.shape[0]

        total_loss += loss.item() * bs
        n_total += bs

        labels_all.append(
            labels.detach().cpu().numpy()
        )
        probs_all.append(
            probs.detach().cpu().numpy()
        )
        rows_all.append(
            batch["row_index"].detach().cpu().numpy()
        )

    y_true = np.concatenate(labels_all)
    probabilities = np.concatenate(probs_all)
    row_indices = np.concatenate(rows_all)

    metrics = calculate_metrics(
        y_true,
        probabilities,
    )
    metrics["loss"] = float(total_loss / n_total)

    return metrics, y_true, probabilities, row_indices


@torch.no_grad()
def check_pair_symmetry(model, loader, device):
    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)

    model.eval()

    logits_ab = forward_model(model, batch)

    swapped = dict(batch)

    for suffix in [
        "morgan",
        "targets",
        "ppi",
        "availability",
    ]:
        a = f"drug_a_{suffix}"
        b = f"drug_b_{suffix}"

        swapped[a] = batch[b]
        swapped[b] = batch[a]

    logits_ba = forward_model(model, swapped)

    diff = torch.abs(logits_ab - logits_ba)

    return {
        "mean_abs_difference": float(diff.mean().item()),
        "max_abs_difference": float(diff.max().item()),
    }


# ============================================================
# Prediction saving
# ============================================================

def save_predictions(
    master: pd.DataFrame,
    row_indices: np.ndarray,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    path: Path,
):
    pred = master.loc[
        row_indices,
        [
            "sample_id",
            "pair_context_id",
            "drug_a_id_std",
            "drug_b_id_std",
            "depmap_id_std",
            "label_int",
        ],
    ].copy()

    y_pred = np.argmax(
        probabilities,
        axis=1,
    )

    pred["true_label"] = [
        CLASS_NAMES[int(x)]
        for x in y_true
    ]
    pred["predicted_class"] = y_pred
    pred["predicted_label"] = [
        CLASS_NAMES[int(x)]
        for x in y_pred
    ]

    pred["p_antagonism"] = probabilities[:, 0]
    pred["p_no_interaction"] = probabilities[:, 1]
    pred["p_synergy"] = probabilities[:, 2]

    pred.to_csv(
        path,
        sep="\t",
        index=False,
    )


# ============================================================
# Paper plots
# ============================================================

def save_figure(fig, base_path: Path):
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_path = Path(str(base_path) + ".pdf")
    png_path = Path(str(base_path) + ".png")

    try:
        fig.tight_layout()
    except Exception:
        pass

    fig.canvas.draw()

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    for out_path in [pdf_path, png_path]:
        if (not out_path.exists()) or out_path.stat().st_size == 0:
            raise IOError(f"Figure was not written correctly: {out_path}")

    return pdf_path, png_path


def save_curve_tables(y_true, probabilities, output_dir: Path, prefix="test"):
    """Save all ROC/PR coordinates so figures can be regenerated later."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    y_bin = label_binarize(y_true, classes=[0, 1, 2])

    roc_rows = []
    pr_rows = []
    summary_rows = []

    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, thresholds = roc_curve(y_bin[:, i], probabilities[:, i])
        auc_value = roc_auc_score(y_bin[:, i], probabilities[:, i])

        for j, (fpr_j, tpr_j, threshold_j) in enumerate(zip(fpr, tpr, thresholds)):
            roc_rows.append({
                "split": prefix,
                "class": name,
                "point_index": j,
                "fpr": float(fpr_j),
                "tpr": float(tpr_j),
                "threshold": float(threshold_j),
                "auroc": float(auc_value),
            })

        precision, recall, pr_thresholds = precision_recall_curve(
            y_bin[:, i], probabilities[:, i]
        )
        ap = average_precision_score(y_bin[:, i], probabilities[:, i])
        prevalence = float(y_bin[:, i].mean())

        for j in range(len(precision)):
            threshold_j = (
                float(pr_thresholds[j]) if j < len(pr_thresholds) else np.nan
            )
            pr_rows.append({
                "split": prefix,
                "class": name,
                "point_index": j,
                "recall": float(recall[j]),
                "precision": float(precision[j]),
                "threshold": threshold_j,
                "auprc": float(ap),
                "prevalence": prevalence,
            })

        summary_rows.append({
            "split": prefix,
            "class": name,
            "support": int(y_bin[:, i].sum()),
            "prevalence": prevalence,
            "auroc": float(auc_value),
            "auprc": float(ap),
        })

    pd.DataFrame(roc_rows).to_csv(
        output_dir / f"{prefix}_roc_curve_points.tsv", sep="\t", index=False
    )
    pd.DataFrame(pr_rows).to_csv(
        output_dir / f"{prefix}_pr_curve_points.tsv", sep="\t", index=False
    )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / f"{prefix}_curve_summary.tsv", sep="\t", index=False
    )


def save_confusion_tables(y_true, probabilities, output_dir: Path, prefix="test"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y_pred = np.argmax(probabilities, axis=1)

    raw = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    norm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2], normalize="true")

    pd.DataFrame(raw, index=CLASS_NAMES, columns=CLASS_NAMES).rename_axis(
        "true_class"
    ).to_csv(output_dir / f"{prefix}_confusion_matrix_raw.tsv", sep="\t")

    pd.DataFrame(norm, index=CLASS_NAMES, columns=CLASS_NAMES).rename_axis(
        "true_class"
    ).to_csv(output_dir / f"{prefix}_confusion_matrix_normalized.tsv", sep="\t")


def plot_training_history(history_df: pd.DataFrame, base_path: Path):
    if history_df is None or history_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train loss")
    ax.plot(history_df["epoch"], history_df["val_loss"], marker="o", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training history")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, Path(str(base_path) + "_loss"))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history_df["epoch"], history_df["val_macro_f1"], marker="o", label="Macro-F1")
    ax.plot(history_df["epoch"], history_df["val_balanced_accuracy"], marker="o", label="Balanced accuracy")
    ax.plot(history_df["epoch"], history_df["val_macro_auroc"], marker="o", label="Macro AUROC")
    ax.plot(history_df["epoch"], history_df["val_macro_auprc"], marker="o", label="Macro AUPRC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1)
    ax.set_title("Validation metrics during training")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, Path(str(base_path) + "_metrics"))


def plot_roc_curves(
    y_true,
    probabilities,
    title,
    base_path,
):
    y_bin = label_binarize(
        y_true,
        classes=[0, 1, 2],
    )

    fig, ax = plt.subplots(
        figsize=(6.5, 5.5)
    )

    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(
            y_bin[:, i],
            probabilities[:, i],
        )

        auc_value = roc_auc_score(
            y_bin[:, i],
            probabilities[:, i],
        )

        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUROC={auc_value:.3f})",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, base_path)


def plot_pr_curves(
    y_true,
    probabilities,
    title,
    base_path,
):
    y_bin = label_binarize(
        y_true,
        classes=[0, 1, 2],
    )

    fig, ax = plt.subplots(
        figsize=(6.5, 5.5)
    )

    for i, name in enumerate(CLASS_NAMES):
        precision, recall, _ = precision_recall_curve(
            y_bin[:, i],
            probabilities[:, i],
        )

        ap = average_precision_score(
            y_bin[:, i],
            probabilities[:, i],
        )

        prevalence = y_bin[:, i].mean()

        ax.plot(
            recall,
            precision,
            label=(
                f"{name} "
                f"(AP={ap:.3f}, prevalence={prevalence:.3f})"
            ),
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, base_path)


def plot_confusion(
    y_true,
    probabilities,
    title,
    base_path,
):
    y_pred = np.argmax(
        probabilities,
        axis=1,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        normalize="true",
    )

    fig, ax = plt.subplots(
        figsize=(5.7, 5.2)
    )

    im = ax.imshow(cm)

    ax.set_xticks(
        np.arange(N_CLASSES),
        labels=CLASS_NAMES,
        rotation=30,
        ha="right",
    )
    ax.set_yticks(
        np.arange(N_CLASSES),
        labels=CLASS_NAMES,
    )

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(
                j,
                i,
                f"{cm[i, j]:.2f}",
                ha="center",
                va="center",
            )

    fig.colorbar(
        im,
        ax=ax,
        label="Fraction of true class",
    )

    save_figure(fig, base_path)
