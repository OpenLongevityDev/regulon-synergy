#!/usr/bin/env python3

"""
train_synergy.py

End-to-end training of the multimodal DrugComb synergy model.

Primary evaluation:
    held-out-cell-line generalization

Classes
-------
0 = antagonism
1 = no_interaction
2 = synergy

Core model
----------
Drug A fixed features
    -> shared IntrinsicDrugEncoder -> dA

Drug B fixed features
    -> same IntrinsicDrugEncoder -> dB

Biological context
    -> BiologicalContextEncoder -> c

(dA, c) -> shared ContextConditioner -> dA|c
(dB, c) -> same ContextConditioner -> dB|c

Symmetric pair representation:
    A+B
    |A-B|
    A*B
    context

-> classifier
-> 3 classes

Important leakage rules
-----------------------
- Cell lines are completely disjoint between train/validation/test.
- Context normalization is fit ONLY on training cell lines.
- Fixed CollecTRI regulons are used.
- Same pair_context_id cannot cross splits because contexts are disjoint.
- Drug fingerprints / HuRI / fixed pathway/regulon priors are external fixed features.

This is the CORE V1 training run.
LINCS auxiliary supervision is intentionally NOT included yet.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from sklearn.model_selection import GroupShuffleSplit

from torch.utils.data import DataLoader, Dataset


# ============================================================
# Paths
# ============================================================

ROOT = Path(
    "/data/analysis/combinations/models/in-house"
)

MODEL_READY_PATH = (
    ROOT
    / "model_ready"
    / "drugcomb_model_ready_v1.tsv"
)

SYNERGY_DATASET_PATH = (
    ROOT
    / "synergy_dataset.py"
)

BIOLOGICAL_CONTEXT_PATH = (
    ROOT
    / "biological_context_module.py"
)

INTRINSIC_ENCODER_PATH = (
    ROOT
    / "drugs_x_context"
    / "intrinsic_drug_encoder.py"
)

CONTEXT_CONDITIONER_PATH = (
    ROOT
    / "drugs_x_context"
    / "context_conditioning.py"
)

SYNERGY_MODEL_PATH = (
    ROOT
    / "synergy_model.py"
)

OUTPUT_DIR = (
    ROOT
    / "training"
    / "core_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Class convention
# ============================================================

CLASS_NAMES = [
    "antagonism",
    "no_interaction",
    "synergy",
]

N_CLASSES = 3


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:

    seed: int = 42

    # --------------------------------------------------------
    # Held-out context split
    # --------------------------------------------------------

    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15

    split_search_trials: int = 30

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    batch_size: int = 128
    epochs: int = 50

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    dropout: float = 0.20

    gradient_clip: float = 5.0

    early_stopping_patience: int = 8

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    num_workers: int = 0
    pin_memory: bool = True

    use_amp: bool = True

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler_factor: float = 0.5
    scheduler_patience: int = 2

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    deterministic: bool = False


CFG = Config()


# ============================================================
# Utilities
# ============================================================

def print_section(
    title: str,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        title
    )

    print(
        "=" * 80
    )


def set_seed(
    seed: int,
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


def import_module_from_path(
    module_name: str,
    path: Path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"Could not find:\n{path}"
        )

    spec = (
        importlib.util
        .spec_from_file_location(
            module_name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise ImportError(
            f"Could not import module from {path}"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    # IMPORTANT:
    # dataclasses and some typing machinery expect the
    # module to already be registered during execution.
    import sys

    sys.modules[
        module_name
    ] = module

    try:

        spec.loader.exec_module(
            module
        )

    except Exception:

        # Do not leave a partially imported module behind.
        sys.modules.pop(
            module_name,
            None,
        )

        raise

    return module

def count_parameters(
    model: nn.Module,
):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return (
        total,
        trainable,
    )


# ============================================================
# Dynamic model component loading
# ============================================================

def find_class(
    module,
    candidates: Sequence[str],
):

    for name in candidates:

        if hasattr(
            module,
            name,
        ):

            obj = getattr(
                module,
                name,
            )

            if inspect.isclass(
                obj
            ):

                return obj

    raise AttributeError(
        f"Could not find any class from:\n"
        f"{list(candidates)}\n"
        f"in module {module.__name__}"
    )


def instantiate_with_supported_kwargs(
    cls,
    kwargs: Dict,
):

    """
    Pass only constructor arguments actually supported by the class.

    This makes train_synergy.py slightly robust to harmless naming
    differences between versions of the component files.
    """

    signature = inspect.signature(
        cls.__init__
    )

    params = signature.parameters

    has_var_kwargs = any(
        p.kind
        == inspect.Parameter.VAR_KEYWORD
        for p in params.values()
    )

    if has_var_kwargs:

        supported = kwargs

    else:

        supported = {
            k: v
            for k, v in kwargs.items()
            if k in params
        }

    try:

        return cls(
            **supported
        )

    except TypeError as e:

        print(
            f"\nCould not instantiate {cls.__name__} "
            f"with filtered arguments:"
        )

        print(
            supported
        )

        print(
            "\nConstructor signature:"
        )

        print(
            signature
        )

        raise e


def build_core_model():

    print_section(
        "LOAD MODEL COMPONENTS"
    )

    bio_module = (
        import_module_from_path(
            "biological_context_module",
            BIOLOGICAL_CONTEXT_PATH,
        )
    )

    intrinsic_module = (
        import_module_from_path(
            "intrinsic_drug_encoder",
            INTRINSIC_ENCODER_PATH,
        )
    )

    conditioner_module = (
        import_module_from_path(
            "context_conditioning",
            CONTEXT_CONDITIONER_PATH,
        )
    )

    synergy_module = (
        import_module_from_path(
            "synergy_model",
            SYNERGY_MODEL_PATH,
        )
    )


    # --------------------------------------------------------
    # Find classes
    # --------------------------------------------------------

    BioClass = find_class(
        bio_module,
        [
            "BiologicalContextEncoder",
            "BiologicalContextModule",
        ],
    )

    IntrinsicClass = find_class(
        intrinsic_module,
        [
            "IntrinsicDrugEncoder",
        ],
    )

    ConditionerClass = find_class(
        conditioner_module,
        [
            "ContextConditioner",
            "DrugContextConditioner",
        ],
    )

    ModelClass = find_class(
        synergy_module,
        [
            "DrugCombinationModel",
        ],
    )


    print(
        "Biological context:",
        BioClass.__name__,
    )

    print(
        "Intrinsic drug:",
        IntrinsicClass.__name__,
    )

    print(
        "Context conditioner:",
        ConditionerClass.__name__,
    )

    print(
        "Full model:",
        ModelClass.__name__,
    )


    # --------------------------------------------------------
    # Instantiate biological context encoder
    # --------------------------------------------------------

    biological_context_encoder = (
        instantiate_with_supported_kwargs(
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

                "dropout": CFG.dropout,
            },
        )
    )


    # --------------------------------------------------------
    # Intrinsic drug encoder
    # --------------------------------------------------------

    intrinsic_drug_encoder = (
        instantiate_with_supported_kwargs(
            IntrinsicClass,
            {
                "morgan_dim": 2048,
                "fingerprint_dim": 2048,

                "target_dim": 8245,
                "direct_target_dim": 8245,

                "ppi_dim": 8245,

                "availability_dim": 3,

                "output_dim": 128,
                "embedding_dim": 128,
                "drug_dim": 128,

                "dropout": CFG.dropout,
            },
        )
    )


    # --------------------------------------------------------
    # Drug × context conditioner
    # --------------------------------------------------------

    context_conditioner = (
        instantiate_with_supported_kwargs(
            ConditionerClass,
            {
                "drug_dim": 128,
                "context_dim": 128,
                "output_dim": 128,
                "embedding_dim": 128,

                "dropout": CFG.dropout,
            },
        )
    )


    # --------------------------------------------------------
    # Full combination model
    #
    # Constructor names may differ slightly, so inspect and map.
    # --------------------------------------------------------

    model_signature = inspect.signature(
        ModelClass.__init__
    )

    model_params = (
        model_signature.parameters
    )

    model_kwargs = {}


    intrinsic_names = [
        "intrinsic_drug_encoder",
        "drug_encoder",
        "intrinsic_encoder",
    ]

    context_names = [
        "biological_context_encoder",
        "context_encoder",
        "biological_context_module",
    ]

    conditioner_names = [
        "context_conditioner",
        "drug_context_conditioner",
        "conditioner",
    ]


    for name in intrinsic_names:

        if name in model_params:

            model_kwargs[
                name
            ] = intrinsic_drug_encoder

            break


    for name in context_names:

        if name in model_params:

            model_kwargs[
                name
            ] = biological_context_encoder

            break


    for name in conditioner_names:

        if name in model_params:

            model_kwargs[
                name
            ] = context_conditioner

            break


    optional_model_kwargs = {

        "drug_dim": 128,
        "context_dim": 128,

        "embedding_dim": 128,

        "n_classes": 3,
        "num_classes": 3,

        "dropout": CFG.dropout,

        "use_l1000_auxiliary": False,
        "use_l1000": False,
    }


    for key, value in optional_model_kwargs.items():

        if key in model_params:

            model_kwargs[
                key
            ] = value


    print(
        "\nDrugCombinationModel constructor:"
    )

    print(
        model_signature
    )

    print(
        "\nArguments supplied:"
    )

    print(
        list(
            model_kwargs.keys()
        )
    )


    model = ModelClass(
        **model_kwargs
    )


    return model


# ============================================================
# Load synergy_dataset helpers
# ============================================================

def load_dataset_module():

    return import_module_from_path(
        "synergy_dataset",
        SYNERGY_DATASET_PATH,
    )


# ============================================================
# Grouped held-out-cell-line split
# ============================================================

def class_proportions(
    y: np.ndarray,
):

    counts = np.bincount(
        y,
        minlength=N_CLASSES,
    ).astype(
        float
    )

    if counts.sum() == 0:

        return np.zeros(
            N_CLASSES,
            dtype=float,
        )

    return (
        counts
        / counts.sum()
    )


def split_score(
    master: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
):

    n = len(
        master
    )

    target = np.array(
        [
            CFG.train_fraction,
            CFG.val_fraction,
            CFG.test_fraction,
        ]
    )

    actual = np.array(
        [
            len(train_idx) / n,
            len(val_idx) / n,
            len(test_idx) / n,
        ]
    )


    size_error = np.abs(
        actual
        - target
    ).sum()


    global_prop = class_proportions(
        master[
            "label_int"
        ]
        .to_numpy(
            dtype=int
        )
    )


    class_error = 0.0

    for idx in [
        train_idx,
        val_idx,
        test_idx,
    ]:

        p = class_proportions(
            master.iloc[
                idx
            ][
                "label_int"
            ]
            .to_numpy(
                dtype=int
            )
        )

        class_error += np.abs(
            p
            - global_prop
        ).mean()


    return (
        size_error
        + class_error
    )


def make_context_split(
    master: pd.DataFrame,
):

    print_section(
        "HELD-OUT-CELL-LINE SPLIT"
    )


    groups = (
        master[
            "depmap_id_std"
        ]
        .astype(str)
        .to_numpy()
    )

    y = (
        master[
            "label_int"
        ]
        .to_numpy(
            dtype=int
        )
    )


    relative_val = (
        CFG.val_fraction
        / (
            CFG.train_fraction
            + CFG.val_fraction
        )
    )


    best = None
    best_score = np.inf


    for trial in range(
        CFG.split_search_trials
    ):

        seed = (
            CFG.seed
            + trial
        )


        outer = GroupShuffleSplit(
            n_splits=1,
            test_size=CFG.test_fraction,
            random_state=seed,
        )


        trainval_idx, test_idx = next(
            outer.split(
                np.zeros(
                    len(master)
                ),
                y,
                groups,
            )
        )


        trainval_groups = groups[
            trainval_idx
        ]

        trainval_y = y[
            trainval_idx
        ]


        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=relative_val,
            random_state=seed + 10_000,
        )


        train_sub, val_sub = next(
            inner.split(
                np.zeros(
                    len(
                        trainval_idx
                    )
                ),
                trainval_y,
                trainval_groups,
            )
        )


        train_idx = trainval_idx[
            train_sub
        ]

        val_idx = trainval_idx[
            val_sub
        ]


        # All classes should appear in every split
        valid = True

        for idx in [
            train_idx,
            val_idx,
            test_idx,
        ]:

            classes = set(
                master.iloc[
                    idx
                ][
                    "label_int"
                ]
                .unique()
                .tolist()
            )

            if classes != {
                0,
                1,
                2,
            }:

                valid = False
                break


        if not valid:

            continue


        score = split_score(
            master,
            train_idx,
            val_idx,
            test_idx,
        )


        if score < best_score:

            best_score = score

            best = (
                train_idx,
                val_idx,
                test_idx,
                seed,
            )


    if best is None:

        raise RuntimeError(
            "Could not construct a valid "
            "held-out-cell-line split."
        )


    (
        train_idx,
        val_idx,
        test_idx,
        chosen_seed,
    ) = best


    print(
        "Chosen split seed:",
        chosen_seed,
    )


    # --------------------------------------------------------
    # Verify context disjointness
    # --------------------------------------------------------

    train_contexts = set(
        master.iloc[
            train_idx
        ][
            "depmap_id_std"
        ]
    )

    val_contexts = set(
        master.iloc[
            val_idx
        ][
            "depmap_id_std"
        ]
    )

    test_contexts = set(
        master.iloc[
            test_idx
        ][
            "depmap_id_std"
        ]
    )


    assert not (
        train_contexts
        & val_contexts
    )

    assert not (
        train_contexts
        & test_contexts
    )

    assert not (
        val_contexts
        & test_contexts
    )


    # --------------------------------------------------------
    # Pair-context leakage check
    # --------------------------------------------------------

    train_pairs = set(
        master.iloc[
            train_idx
        ][
            "pair_context_id"
        ]
    )

    val_pairs = set(
        master.iloc[
            val_idx
        ][
            "pair_context_id"
        ]
    )

    test_pairs = set(
        master.iloc[
            test_idx
        ][
            "pair_context_id"
        ]
    )


    assert not (
        train_pairs
        & val_pairs
    )

    assert not (
        train_pairs
        & test_pairs
    )

    assert not (
        val_pairs
        & test_pairs
    )


    print(
        "\nObservations:"
    )

    for name, idx in [
        (
            "Train",
            train_idx,
        ),
        (
            "Validation",
            val_idx,
        ),
        (
            "Test",
            test_idx,
        ),
    ]:

        print(
            f"{name:<12}"
            f"{len(idx):>9,} "
            f"({len(idx)/len(master):6.2%})"
        )


    print(
        "\nCell lines:"
    )

    print(
        "Train:",
        len(
            train_contexts
        ),
    )

    print(
        "Validation:",
        len(
            val_contexts
        ),
    )

    print(
        "Test:",
        len(
            test_contexts
        ),
    )


    print(
        "\nClass proportions:"
    )

    for name, idx in [
        (
            "Train",
            train_idx,
        ),
        (
            "Validation",
            val_idx,
        ),
        (
            "Test",
            test_idx,
        ),
    ]:

        p = class_proportions(
            master.iloc[
                idx
            ][
                "label_int"
            ]
            .to_numpy(
                dtype=int
            )
        )

        print(
            name,
            dict(
                zip(
                    CLASS_NAMES,
                    np.round(
                        p,
                        4,
                    ),
                )
            ),
        )


    print(
        "\nContext leakage: NONE ✓"
    )

    print(
        "pair_context_id leakage: NONE ✓"
    )


    return (
        train_idx,
        val_idx,
        test_idx,
    )


# ============================================================
# Train-only context normalization
# ============================================================

def fit_standardizer(
    X: np.ndarray,
    rows: np.ndarray,
):

    train_X = X[
        rows
    ]


    mean = train_X.mean(
        axis=0,
        dtype=np.float64,
    )


    std = train_X.std(
        axis=0,
        dtype=np.float64,
    )


    std[
        std < 1e-8
    ] = 1.0


    return (
        mean.astype(
            np.float32
        ),
        std.astype(
            np.float32
        ),
    )


def standardize_inplace(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    chunk_size: int = 256,
):

    for start in range(
        0,
        X.shape[0],
        chunk_size,
    ):

        end = min(
            start
            + chunk_size,
            X.shape[0],
        )

        X[
            start:end
        ] = (
            X[
                start:end
            ]
            - mean
        ) / std


def normalize_context_train_only(
    features,
    master: pd.DataFrame,
    train_idx: np.ndarray,
):

    print_section(
        "TRAIN-ONLY CONTEXT NORMALIZATION"
    )


    train = master.iloc[
        train_idx
    ]


    expression_rows = (
        train[
            "expression_idx"
        ]
        .astype(int)
        .unique()
    )


    pathway_rows = (
        train[
            "pathways_idx"
        ]
        .astype(int)
        .unique()
    )


    regulon_rows = (
        train[
            "regulons_idx"
        ]
        .astype(int)
        .unique()
    )


    print(
        "Unique training expression contexts:",
        len(
            expression_rows
        ),
    )

    print(
        "Unique training pathway contexts:",
        len(
            pathway_rows
        ),
    )

    print(
        "Unique training regulon contexts:",
        len(
            regulon_rows
        ),
    )


    expr_mean, expr_std = fit_standardizer(
        features.expression,
        expression_rows,
    )

    path_mean, path_std = fit_standardizer(
        features.pathways,
        pathway_rows,
    )

    reg_mean, reg_std = fit_standardizer(
        features.regulons,
        regulon_rows,
    )


    standardize_inplace(
        features.expression,
        expr_mean,
        expr_std,
    )

    standardize_inplace(
        features.pathways,
        path_mean,
        path_std,
    )

    standardize_inplace(
        features.regulons,
        reg_mean,
        reg_std,
    )


    scaler_path = (
        OUTPUT_DIR
        / "context_standardizers.npz"
    )


    np.savez_compressed(
        scaler_path,

        expression_mean=
            expr_mean,

        expression_std=
            expr_std,

        pathway_mean=
            path_mean,

        pathway_std=
            path_std,

        regulon_mean=
            reg_mean,

        regulon_std=
            reg_std,
    )


    print(
        "Saved:",
        scaler_path,
    )


# ============================================================
# Fast indexed Dataset + batch collator
# ============================================================

class RowIndexDataset(
    Dataset
):

    def __init__(
        self,
        row_indices: np.ndarray,
    ):

        self.row_indices = np.asarray(
            row_indices,
            dtype=np.int64,
        )


    def __len__(
        self,
    ):

        return len(
            self.row_indices
        )


    def __getitem__(
        self,
        i: int,
    ):

        return int(
            self.row_indices[
                i
            ]
        )


class FastBatchCollator:

    """
    Retrieve matrices once PER BATCH instead of densifying
    direct-target vectors once per observation.
    """

    def __init__(
        self,
        master: pd.DataFrame,
        features,
    ):

        self.master = master
        self.features = features


    def __call__(
        self,
        row_indices: List[int],
    ):

        rows = self.master.iloc[
            row_indices
        ]


        # ----------------------------------------------------
        # Matrix indices
        # ----------------------------------------------------

        ma = (
            rows[
                "morgan_a_idx"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        mb = (
            rows[
                "morgan_b_idx"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )


        ta = (
            rows[
                "direct_a_idx"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        tb = (
            rows[
                "direct_b_idx"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )


        pa = (
            rows[
                "ppi_a_idx"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        pb = (
            rows[
                "ppi_b_idx"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )


        ex = (
            rows[
                "expression_idx"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        pw = (
            rows[
                "pathways_idx"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        rg = (
            rows[
                "regulons_idx"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )


        # ----------------------------------------------------
        # Drug features
        # ----------------------------------------------------

        drug_a_morgan = (
            self.features
            .morgan[
                ma
            ]
            .astype(
                np.float32,
                copy=False,
            )
        )

        drug_b_morgan = (
            self.features
            .morgan[
                mb
            ]
            .astype(
                np.float32,
                copy=False,
            )
        )


        # Sparse -> dense ONCE PER BATCH
        drug_a_targets = (
            self.features
            .direct_targets[
                ta
            ]
            .toarray()
            .astype(
                np.float32,
                copy=False,
            )
        )

        drug_b_targets = (
            self.features
            .direct_targets[
                tb
            ]
            .toarray()
            .astype(
                np.float32,
                copy=False,
            )
        )


        drug_a_ppi = (
            self.features
            .ppi[
                pa
            ]
            .astype(
                np.float32,
                copy=False,
            )
        )

        drug_b_ppi = (
            self.features
            .ppi[
                pb
            ]
            .astype(
                np.float32,
                copy=False,
            )
        )


        drug_a_availability = np.stack(
            [
                self.features
                .availability[
                    drug_id
                ]

                for drug_id
                in rows[
                    "drug_a_id_std"
                ]
            ]
        ).astype(
            np.float32,
            copy=False,
        )


        drug_b_availability = np.stack(
            [
                self.features
                .availability[
                    drug_id
                ]

                for drug_id
                in rows[
                    "drug_b_id_std"
                ]
            ]
        ).astype(
            np.float32,
            copy=False,
        )


        # ----------------------------------------------------
        # Biological context
        # ----------------------------------------------------

        context_expression = (
            self.features
            .expression[
                ex
            ]
            .astype(
                np.float32,
                copy=False,
            )
        )


        context_pathways = (
            self.features
            .pathways[
                pw
            ]
            .astype(
                np.float32,
                copy=False,
            )
        )


        context_regulons = (
            self.features
            .regulons[
                rg
            ]
            .astype(
                np.float32,
                copy=False,
            )
        )


        label = (
            rows[
                "label_int"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )


        return {

            "drug_a_morgan":
                torch.from_numpy(
                    drug_a_morgan
                ),

            "drug_a_targets":
                torch.from_numpy(
                    drug_a_targets
                ),

            "drug_a_ppi":
                torch.from_numpy(
                    drug_a_ppi
                ),

            "drug_a_availability":
                torch.from_numpy(
                    drug_a_availability
                ),


            "drug_b_morgan":
                torch.from_numpy(
                    drug_b_morgan
                ),

            "drug_b_targets":
                torch.from_numpy(
                    drug_b_targets
                ),

            "drug_b_ppi":
                torch.from_numpy(
                    drug_b_ppi
                ),

            "drug_b_availability":
                torch.from_numpy(
                    drug_b_availability
                ),


            "context_expression":
                torch.from_numpy(
                    context_expression
                ),

            "context_pathways":
                torch.from_numpy(
                    context_pathways
                ),

            "context_regulons":
                torch.from_numpy(
                    context_regulons
                ),


            "label":
                torch.from_numpy(
                    label
                ),

            "row_index":
                torch.tensor(
                    row_indices,
                    dtype=torch.long,
                ),
        }


# ============================================================
# Loader
# ============================================================

def make_loader(
    indices: np.ndarray,
    master: pd.DataFrame,
    features,
    shuffle: bool,
):

    dataset = RowIndexDataset(
        indices
    )

    collator = FastBatchCollator(
        master=
            master,

        features=
            features,
    )


    return DataLoader(

        dataset,

        batch_size=
            CFG.batch_size,

        shuffle=
            shuffle,

        num_workers=
            CFG.num_workers,

        pin_memory=
            (
                CFG.pin_memory
                and torch.cuda.is_available()
            ),

        persistent_workers=
            (
                CFG.num_workers
                > 0
            ),

        collate_fn=
            collator,

        drop_last=
            False,
    )


# ============================================================
# Device transfer
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


def move_batch_to_device(
    batch: Dict,
    device: torch.device,
):

    result = {}

    for key, value in batch.items():

        if torch.is_tensor(
            value
        ):

            result[
                key
            ] = value.to(
                device,
                non_blocking=True,
            )

        else:

            result[
                key
            ] = value


    return result


# ============================================================
# Forward compatibility helper
# ============================================================

def forward_model(
    model: nn.Module,
    batch: Dict,
):

    kwargs = {
        key: batch[
            key
        ]
        for key in MODEL_INPUT_KEYS
    }


    output = model(
        **kwargs
    )


    if torch.is_tensor(
        output
    ):

        return output


    if isinstance(
        output,
        dict,
    ):

        if "logits" in output:

            return output[
                "logits"
            ]

        if "synergy_logits" in output:

            return output[
                "synergy_logits"
            ]


    if isinstance(
        output,
        (
            tuple,
            list,
        ),
    ):

        if len(
            output
        ) > 0:

            if torch.is_tensor(
                output[
                    0
                ]
            ):

                return output[
                    0
                ]


    raise TypeError(
        "Could not extract logits from model output.\n"
        f"Output type: {type(output)}"
    )


# ============================================================
# Class weights
# ============================================================

def calculate_class_weights(
    master: pd.DataFrame,
    train_idx: np.ndarray,
):

    y = (
        master.iloc[
            train_idx
        ][
            "label_int"
        ]
        .to_numpy(
            dtype=int
        )
    )


    counts = np.bincount(
        y,
        minlength=N_CLASSES,
    ).astype(
        np.float64
    )


    weights = (
        counts.sum()
        / (
            N_CLASSES
            * counts
        )
    )


    print_section(
        "TRAINING CLASS WEIGHTS"
    )


    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{class_name:<16}"
            f"n={int(counts[class_id]):>8,} "
            f"weight={weights[class_id]:.4f}"
        )


    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
):

    y_pred = np.argmax(
        probabilities,
        axis=1,
    )


    metrics = {

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),

        "weighted_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
    }


    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[
                0,
                1,
                2,
            ],
            zero_division=0,
        )
    )


    per_class = {}

    for i, name in enumerate(
        CLASS_NAMES
    ):

        per_class[
            name
        ] = {

            "precision":
                float(
                    precision[
                        i
                    ]
                ),

            "recall":
                float(
                    recall[
                        i
                    ]
                ),

            "f1":
                float(
                    f1[
                        i
                    ]
                ),

            "support":
                int(
                    support[
                        i
                    ]
                ),
        }


    metrics[
        "per_class"
    ] = per_class


    # --------------------------------------------------------
    # AUROC
    # --------------------------------------------------------

    try:

        metrics[
            "macro_ovr_auroc"
        ] = float(
            roc_auc_score(
                y_true,
                probabilities,
                multi_class="ovr",
                average="macro",
            )
        )

    except ValueError:

        metrics[
            "macro_ovr_auroc"
        ] = None


    # --------------------------------------------------------
    # AUPRC one-vs-rest
    # --------------------------------------------------------

    auprc = {}


    for class_id, name in enumerate(
        CLASS_NAMES
    ):

        binary_true = (
            y_true
            == class_id
        ).astype(
            int
        )

        try:

            value = (
                average_precision_score(
                    binary_true,
                    probabilities[
                        :,
                        class_id
                    ],
                )
            )

            auprc[
                name
            ] = float(
                value
            )

        except ValueError:

            auprc[
                name
            ] = None


    valid_auprc = [
        x
        for x in auprc.values()
        if x is not None
    ]


    metrics[
        "per_class_auprc"
    ] = auprc


    metrics[
        "macro_auprc"
    ] = (
        float(
            np.mean(
                valid_auprc
            )
        )
        if valid_auprc
        else None
    )


    metrics[
        "confusion_matrix"
    ] = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[
                0,
                1,
                2,
            ],
        )
        .tolist()
    )


    return metrics


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
):

    model.eval()


    total_loss = 0.0
    n_total = 0


    all_labels = []
    all_probs = []


    for batch in loader:

        batch = move_batch_to_device(
            batch,
            device,
        )


        labels = batch[
            "label"
        ]


        with torch.autocast(
            device_type=
                device.type,

            dtype=
                torch.float16,

            enabled=
                (
                    CFG.use_amp
                    and device.type
                    == "cuda"
                ),
        ):

            logits = forward_model(
                model,
                batch,
            )

            loss = criterion(
                logits,
                labels,
            )


        probs = torch.softmax(
            logits.float(),
            dim=1,
        )


        batch_size = labels.shape[
            0
        ]


        total_loss += (
            loss.item()
            * batch_size
        )

        n_total += batch_size


        all_labels.append(
            labels.detach()
            .cpu()
            .numpy()
        )

        all_probs.append(
            probs.detach()
            .cpu()
            .numpy()
        )


    y_true = np.concatenate(
        all_labels
    )

    probabilities = np.concatenate(
        all_probs
    )


    metrics = calculate_metrics(
        y_true,
        probabilities,
    )


    metrics[
        "loss"
    ] = (
        total_loss
        / n_total
    )


    return (
        metrics,
        y_true,
        probabilities,
    )


# ============================================================
# Training epoch
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    criterion,
    scaler,
    device: torch.device,
):

    model.train()


    running_loss = 0.0
    n_total = 0


    for batch in loader:

        batch = move_batch_to_device(
            batch,
            device,
        )


        labels = batch[
            "label"
        ]


        optimizer.zero_grad(
            set_to_none=True
        )


        with torch.autocast(
            device_type=
                device.type,

            dtype=
                torch.float16,

            enabled=
                (
                    CFG.use_amp
                    and device.type
                    == "cuda"
                ),
        ):

            logits = forward_model(
                model,
                batch,
            )

            loss = criterion(
                logits,
                labels,
            )


        scaler.scale(
            loss
        ).backward()


        scaler.unscale_(
            optimizer
        )


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            CFG.gradient_clip,
        )


        scaler.step(
            optimizer
        )

        scaler.update()


        batch_size = labels.shape[
            0
        ]


        running_loss += (
            loss.item()
            * batch_size
        )

        n_total += batch_size


    return (
        running_loss
        / n_total
    )


# ============================================================
# Symmetry check
# ============================================================

@torch.no_grad()
def check_pair_symmetry(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):

    print_section(
        "PAIR SYMMETRY CHECK"
    )


    batch = next(
        iter(
            loader
        )
    )


    batch = move_batch_to_device(
        batch,
        device,
    )


    model.eval()


    logits_ab = forward_model(
        model,
        batch,
    )


    swapped = dict(
        batch
    )


    for suffix in [
        "morgan",
        "targets",
        "ppi",
        "availability",
    ]:

        a = (
            f"drug_a_{suffix}"
        )

        b = (
            f"drug_b_{suffix}"
        )

        swapped[
            a
        ] = batch[
            b
        ]

        swapped[
            b
        ] = batch[
            a
        ]


    logits_ba = forward_model(
        model,
        swapped,
    )


    difference = torch.abs(
        logits_ab
        - logits_ba
    )


    max_diff = float(
        difference.max().item()
    )

    mean_diff = float(
        difference.mean().item()
    )


    print(
        "Mean |AB - BA|:",
        f"{mean_diff:.8g}",
    )

    print(
        "Max  |AB - BA|:",
        f"{max_diff:.8g}",
    )


    if max_diff < 1e-5:

        print(
            "Pair symmetry: PASS ✓"
        )

    else:

        print(
            "WARNING: model is not exactly symmetric."
        )


    return {
        "mean_abs_difference":
            mean_diff,

        "max_abs_difference":
            max_diff,
    }


# ============================================================
# Save prediction table
# ============================================================

def save_predictions(
    master: pd.DataFrame,
    indices: np.ndarray,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    split_name: str,
):

    pred = master.iloc[
        indices
    ][
        [
            "sample_id",
            "pair_context_id",
            "drug_a_id_std",
            "drug_b_id_std",
            "depmap_id_std",
            "study_name",
            "label_int",
        ]
    ].copy()


    pred[
        "true_label"
    ] = [
        CLASS_NAMES[
            x
        ]
        for x in y_true
    ]


    y_pred = np.argmax(
        probabilities,
        axis=1,
    )


    pred[
        "predicted_class"
    ] = y_pred


    pred[
        "predicted_label"
    ] = [
        CLASS_NAMES[
            x
        ]
        for x in y_pred
    ]


    pred[
        "p_antagonism"
    ] = probabilities[
        :,
        0
    ]


    pred[
        "p_no_interaction"
    ] = probabilities[
        :,
        1
    ]


    pred[
        "p_synergy"
    ] = probabilities[
        :,
        2
    ]


    path = (
        OUTPUT_DIR
        / f"{split_name}_predictions.tsv"
    )


    pred.to_csv(
        path,
        sep="\t",
        index=False,
    )


    print(
        f"Saved {split_name} predictions:",
        path,
    )


# ============================================================
# Main
# ============================================================

def main():

    set_seed(
        CFG.seed
    )


    if CFG.deterministic:

        torch.use_deterministic_algorithms(
            True
        )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print_section(
        "CONFIGURATION"
    )


    print(
        json.dumps(
            asdict(
                CFG
            ),
            indent=2,
        )
    )


    print(
        "\nDevice:",
        device,
    )


    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )


    # ========================================================
    # Load dataset feature store
    # ========================================================

    dataset_module = (
        load_dataset_module()
    )


    print_section(
        "LOAD MODEL-READY MASTER"
    )


    if not MODEL_READY_PATH.exists():

        raise FileNotFoundError(
            f"Run synergy_dataset.py first.\n"
            f"Missing:\n{MODEL_READY_PATH}"
        )


    master = pd.read_csv(
        MODEL_READY_PATH,
        sep="\t",
        low_memory=False,
    )


    print(
        "Model-ready observations:",
        f"{len(master):,}",
    )


    print(
        "Unique cell lines:",
        master[
            "depmap_id_std"
        ].nunique(),
    )


    print(
        "Unique drugs:",
        len(
            set(
                master[
                    "drug_a_id_std"
                ].astype(str)
            )
            |
            set(
                master[
                    "drug_b_id_std"
                ].astype(str)
            )
        ),
    )


    # Make sure IDs remain strings after reading TSV
    master[
        "drug_a_id_std"
    ] = (
        master[
            "drug_a_id_std"
        ]
        .map(
            dataset_module.normalize_id
        )
    )


    master[
        "drug_b_id_std"
    ] = (
        master[
            "drug_b_id_std"
        ]
        .map(
            dataset_module.normalize_id
        )
    )


    master[
        "depmap_id_std"
    ] = (
        master[
            "depmap_id_std"
        ]
        .map(
            dataset_module.normalize_id
        )
    )


    # ========================================================
    # Load matrices
    # ========================================================

    print_section(
        "LOAD FEATURE MATRICES"
    )


    features = (
        dataset_module
        .load_all_features()
    )


    # ========================================================
    # Split BEFORE normalization
    # ========================================================

    (
        train_idx,
        val_idx,
        test_idx,
    ) = make_context_split(
        master
    )


    # Save split assignment
    split_assignment = (
        master[
            [
                "sample_id",
                "pair_context_id",
                "depmap_id_std",
                "label_int",
            ]
        ]
        .copy()
    )


    split_assignment[
        "split"
    ] = ""


    split_assignment.loc[
        train_idx,
        "split",
    ] = "train"


    split_assignment.loc[
        val_idx,
        "split",
    ] = "validation"


    split_assignment.loc[
        test_idx,
        "split",
    ] = "test"


    split_assignment.to_csv(
        OUTPUT_DIR
        / "split_assignments.tsv",
        sep="\t",
        index=False,
    )


    # ========================================================
    # Train-only context normalization
    # ========================================================

    normalize_context_train_only(
        features=
            features,

        master=
            master,

        train_idx=
            train_idx,
    )


    # ========================================================
    # DataLoaders
    # ========================================================

    print_section(
        "BUILD DATALOADERS"
    )


    train_loader = make_loader(
        train_idx,
        master,
        features,
        shuffle=True,
    )


    val_loader = make_loader(
        val_idx,
        master,
        features,
        shuffle=False,
    )


    test_loader = make_loader(
        test_idx,
        master,
        features,
        shuffle=False,
    )


    print(
        "Train batches:",
        len(
            train_loader
        ),
    )

    print(
        "Validation batches:",
        len(
            val_loader
        ),
    )

    print(
        "Test batches:",
        len(
            test_loader
        ),
    )


    # ========================================================
    # Model
    # ========================================================

    model = (
        build_core_model()
        .to(
            device
        )
    )


    total_params, trainable_params = (
        count_parameters(
            model
        )
    )


    print_section(
        "MODEL"
    )


    print(
        model
    )


    print(
        "\nTotal parameters:",
        f"{total_params:,}",
    )


    print(
        "Trainable parameters:",
        f"{trainable_params:,}",
    )


    # ========================================================
    # Loss
    # ========================================================

    class_weights = (
        calculate_class_weights(
            master,
            train_idx,
        )
        .to(
            device
        )
    )


    criterion = nn.CrossEntropyLoss(
        weight=
            class_weights
    )


    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=
            CFG.learning_rate,

        weight_decay=
            CFG.weight_decay,
    )


    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(

            optimizer,

            mode="max",

            factor=
                CFG.scheduler_factor,

            patience=
                CFG.scheduler_patience,
        )
    )


    scaler = torch.cuda.amp.GradScaler(
        enabled=
            (
                CFG.use_amp
                and device.type
                == "cuda"
            )
    )


    # ========================================================
    # Pre-training symmetry test
    # ========================================================

    check_pair_symmetry(
        model,
        val_loader,
        device,
    )


    # ========================================================
    # Train
    # ========================================================

    print_section(
        "TRAINING"
    )


    history = []

    best_macro_f1 = -np.inf
    best_epoch = -1

    epochs_without_improvement = 0


    checkpoint_path = (
        OUTPUT_DIR
        / "best_model.pt"
    )


    for epoch in range(
        1,
        CFG.epochs + 1,
    ):

        start_time = time.time()


        train_loss = train_one_epoch(

            model=
                model,

            loader=
                train_loader,

            optimizer=
                optimizer,

            criterion=
                criterion,

            scaler=
                scaler,

            device=
                device,
        )


        (
            val_metrics,
            _,
            _,
        ) = evaluate(

            model=
                model,

            loader=
                val_loader,

            criterion=
                criterion,

            device=
                device,
        )


        scheduler.step(
            val_metrics[
                "macro_f1"
            ]
        )


        lr = optimizer.param_groups[
            0
        ][
            "lr"
        ]


        elapsed = (
            time.time()
            - start_time
        )


        row = {

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "val_loss":
                val_metrics[
                    "loss"
                ],

            "val_macro_f1":
                val_metrics[
                    "macro_f1"
                ],

            "val_balanced_accuracy":
                val_metrics[
                    "balanced_accuracy"
                ],

            "val_accuracy":
                val_metrics[
                    "accuracy"
                ],

            "val_macro_auroc":
                val_metrics[
                    "macro_ovr_auroc"
                ],

            "val_macro_auprc":
                val_metrics[
                    "macro_auprc"
                ],

            "learning_rate":
                lr,

            "seconds":
                elapsed,
        }


        history.append(
            row
        )


        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"macro-F1 {val_metrics['macro_f1']:.4f} | "
            f"bal acc {val_metrics['balanced_accuracy']:.4f} | "
            f"AUROC {val_metrics['macro_ovr_auroc']:.4f} | "
            f"AUPRC {val_metrics['macro_auprc']:.4f} | "
            f"lr {lr:.2e} | "
            f"{elapsed:.1f}s"
        )


        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        current = (
            val_metrics[
                "macro_f1"
            ]
        )


        if current > best_macro_f1:

            best_macro_f1 = current

            best_epoch = epoch

            epochs_without_improvement = 0


            torch.save(

                {
                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "val_metrics":
                        val_metrics,

                    "config":
                        asdict(
                            CFG
                        ),

                    "class_names":
                        CLASS_NAMES,

                    "class_weights":
                        class_weights
                        .detach()
                        .cpu(),

                },

                checkpoint_path,
            )


            print(
                "  -> best checkpoint saved"
            )


        else:

            epochs_without_improvement += 1


        # Save history every epoch
        pd.DataFrame(
            history
        ).to_csv(
            OUTPUT_DIR
            / "training_history.tsv",
            sep="\t",
            index=False,
        )


        if (
            epochs_without_improvement
            >= CFG.early_stopping_patience
        ):

            print(
                "\nEarly stopping."
            )

            break


    # ========================================================
    # Load best checkpoint
    # ========================================================

    print_section(
        "LOAD BEST MODEL"
    )


    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation macro-F1:",
        f"{best_macro_f1:.4f}",
    )


    # ========================================================
    # Final validation
    # ========================================================

    print_section(
        "FINAL VALIDATION"
    )


    (
        val_metrics,
        val_true,
        val_probs,
    ) = evaluate(

        model,
        val_loader,
        criterion,
        device,
    )


    print(
        json.dumps(
            val_metrics,
            indent=2,
        )
    )


    save_predictions(
        master,
        val_idx,
        val_true,
        val_probs,
        "validation",
    )


    # ========================================================
    # Final unseen-cell-line test
    # ========================================================

    print_section(
        "FINAL HELD-OUT-CELL-LINE TEST"
    )


    (
        test_metrics,
        test_true,
        test_probs,
    ) = evaluate(

        model,
        test_loader,
        criterion,
        device,
    )


    print(
        json.dumps(
            test_metrics,
            indent=2,
        )
    )


    save_predictions(
        master,
        test_idx,
        test_true,
        test_probs,
        "test",
    )


    # ========================================================
    # Final symmetry
    # ========================================================

    symmetry = check_pair_symmetry(
        model,
        test_loader,
        device,
    )


    # ========================================================
    # Save final metrics
    # ========================================================

    final_results = {

        "best_epoch":
            best_epoch,

        "best_validation_macro_f1":
            best_macro_f1,

        "validation":
            val_metrics,

        "test":
            test_metrics,

        "symmetry":
            symmetry,

        "n_train":
            int(
                len(
                    train_idx
                )
            ),

        "n_validation":
            int(
                len(
                    val_idx
                )
            ),

        "n_test":
            int(
                len(
                    test_idx
                )
            ),

        "n_train_contexts":
            int(
                master.iloc[
                    train_idx
                ][
                    "depmap_id_std"
                ]
                .nunique()
            ),

        "n_validation_contexts":
            int(
                master.iloc[
                    val_idx
                ][
                    "depmap_id_std"
                ]
                .nunique()
            ),

        "n_test_contexts":
            int(
                master.iloc[
                    test_idx
                ][
                    "depmap_id_std"
                ]
                .nunique()
            ),
    }


    with open(
        OUTPUT_DIR
        / "final_metrics.json",
        "w",
    ) as f:

        json.dump(
            final_results,
            f,
            indent=2,
        )


    # ========================================================
    # Done
    # ========================================================

    print_section(
        "TRAINING COMPLETE"
    )


    print(
        "Best checkpoint:"
    )

    print(
        checkpoint_path
    )


    print(
        "\nTraining history:"
    )

    print(
        OUTPUT_DIR
        / "training_history.tsv"
    )


    print(
        "\nFinal metrics:"
    )

    print(
        OUTPUT_DIR
        / "final_metrics.json"
    )


    print(
        "\nTest predictions:"
    )

    print(
        OUTPUT_DIR
        / "test_predictions.tsv"
    )


if __name__ == "__main__":

    main()
