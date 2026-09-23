#!/usr/bin/env python3
"""
Nested, leakage-safe abstraction experiment for DrugComb biological context.

Question
--------
Do latent biological PROGRAMS learned above Hallmark pathway activities improve
cross-domain drug-combination prediction?

Program construction
--------------------
Hallmark ULM activities are signed, so NMF cannot be applied directly. For each
holdout x seed split, this script:
  1) fits feature-wise min/max ONLY on unique training contexts;
  2) maps training Hallmark activities to [0,1] and clips transformed held-out
     contexts to [0,1];
  3) fits NMF ONLY on unique training contexts;
  4) transforms all contexts using the frozen training-fitted NMF basis;
  5) z-standardizes resulting program scores using training contexts only.

Nested model selection
----------------------
K is selected ONLY by validation macro-F1. Test predictions are produced only
for the selected K. Thus K=5/10/15/20/30 is not selected using test performance.

Program context configurations
------------------------------
  programs
  expression_programs
  regulons_programs
  expression_regulons_programs

Default full design
-------------------
  4 program representations x 3 holdouts x 5 seeds x 5 candidate K values
  = 300 candidate trainings, followed by 60 selected-model test evaluations.

This script reuses the audited DrugComb loaders, split construction, drug encoder,
context conditioning, metrics, and plotting utilities from ``experiment_utils.py``.
Program vectors are passed through the existing context_pathways slot; this is
intentional and avoids changing the downstream DrugCombinationModel.

Historical output columns containing ``auprc`` are retained for compatibility.
Their values are computed with sklearn average_precision_score and therefore
correspond to average precision (AP).
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
import os
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.decomposition import NMF

# Repository root. By default this is the directory containing this script.
# Set DRUG_SYNERGY_ROOT to point to another checkout/data root if needed.
ROOT = Path(
    os.environ.get("DRUG_SYNERGY_ROOT", Path(__file__).resolve().parent)
).resolve()

BASE_SCRIPT_CANDIDATES = [
    ROOT / "experiment_utils.py",
]

OUTPUT_ROOT = ROOT / "training" / "program_ablation"
SPLIT_DIR = OUTPUT_ROOT / "splits"
PROGRAM_DIR = OUTPUT_ROOT / "program_representations"
CANDIDATE_DIR = OUTPUT_ROOT / "candidate_runs"
SELECTED_DIR = OUTPUT_ROOT / "selected_runs"
SUMMARY_DIR = OUTPUT_ROOT / "summary"
for d in [OUTPUT_ROOT, SPLIT_DIR, PROGRAM_DIR, CANDIDATE_DIR, SELECTED_DIR, SUMMARY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

HOLDOUTS = ["unseen_cell_line", "unseen_drug", "unseen_both"]
CLASS_NAMES = ["antagonism", "no_interaction", "synergy"]
PROGRAM_CONFIGS = {
    "programs": ("programs",),
    "expression_programs": ("expression", "programs"),
    "regulons_programs": ("regulons", "programs"),
    "expression_regulons_programs": ("expression", "regulons", "programs"),
}
PROGRAM_LABELS = {
    "programs": "Programs only",
    "expression_programs": "Expression + programs",
    "regulons_programs": "Regulons + programs",
    "expression_regulons_programs": "Expression + regulons + programs",
}
PROGRAM_ORDER = list(PROGRAM_CONFIGS)
DEFAULT_KS = [5, 10, 15, 20, 30]

# Previous 105-run experiment; used only for a combined descriptive summary.
PREVIOUS_CONTEXT_METRICS = ROOT / "training" / "context_ablation" / "summary" / "all_run_metrics.tsv"


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_base_module(explicit: Optional[str] = None):
    candidates = [Path(explicit)] if explicit else BASE_SCRIPT_CANDIDATES
    for p in candidates:
        if p.exists():
            print(f"Using base utilities from: {p}")
            return import_module_from_path("program_ablation_base", p)
    raise FileNotFoundError(
        f"Could not find experiment_utils.py under repository root {ROOT}. "
        "Use --base-script PATH to override."
    )


def find_class(module, names):
    for name in names:
        obj = getattr(module, name, None)
        if inspect.isclass(obj):
            return obj
    raise AttributeError(f"Could not find any of {names} in {module.__name__}")


def instantiate_with_supported_kwargs(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        use = kwargs
    else:
        use = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**use)


class BranchEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=64, dropout=0.20):
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


class ProgramContextEncoder(nn.Module):
    """Selected expression/regulon/program branches -> 128-d context.

    Programs arrive through the `pathways` argument because the existing model
    has three context input slots: expression, pathways, regulons.
    """
    def __init__(
        self,
        modalities,
        program_dim,
        expression_dim=942,
        regulon_dim=771,
        branch_dim=64,
        fusion_hidden=256,
        output_dim=128,
        dropout=0.20,
    ):
        super().__init__()
        self.modalities = tuple(modalities)
        if "expression" in self.modalities:
            self.expression_encoder = BranchEncoder(expression_dim, 256, branch_dim, dropout)
        if "regulons" in self.modalities:
            self.regulon_encoder = BranchEncoder(regulon_dim, 256, branch_dim, dropout)
        if "programs" in self.modalities:
            self.program_encoder = BranchEncoder(program_dim, 64, branch_dim, dropout)

        self.fusion = nn.Sequential(
            nn.Linear(branch_dim * len(self.modalities), fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, expression, pathways, regulons, return_aux=False, **kwargs):
        parts, aux = [], {}
        if "expression" in self.modalities:
            z = self.expression_encoder(expression)
            parts.append(z); aux["expression_embedding"] = z
        if "regulons" in self.modalities:
            z = self.regulon_encoder(regulons)
            parts.append(z); aux["regulon_embedding"] = z
        if "programs" in self.modalities:
            z = self.program_encoder(pathways)
            parts.append(z); aux["program_embedding"] = z
        embedding = self.fusion(torch.cat(parts, dim=-1))
        return {"embedding": embedding, **aux} if return_aux else embedding


def build_model(base, cfg, context_config: str, k: int):
    modalities = PROGRAM_CONFIGS[context_config]
    intrinsic_module = base.import_module_from_path(
        f"intrinsic_program_{context_config}_{k}_{id(cfg)}", base.INTRINSIC_ENCODER_PATH
    )
    conditioner_module = base.import_module_from_path(
        f"conditioner_program_{context_config}_{k}_{id(cfg)}", base.CONTEXT_CONDITIONER_PATH
    )
    synergy_module = base.import_module_from_path(
        f"synergy_program_{context_config}_{k}_{id(cfg)}", base.SYNERGY_MODEL_PATH
    )
    IntrinsicClass = find_class(intrinsic_module, ["IntrinsicDrugEncoder"])
    ConditionerClass = find_class(conditioner_module, ["ContextConditioner", "DrugContextConditioner"])
    ModelClass = find_class(synergy_module, ["DrugCombinationModel"])

    context_encoder = ProgramContextEncoder(
        modalities=modalities, program_dim=k, dropout=cfg.dropout, output_dim=128
    )
    intrinsic_encoder = instantiate_with_supported_kwargs(IntrinsicClass, {
        "morgan_dim": 2048, "target_dim": 8245, "direct_target_dim": 8245,
        "ppi_dim": 8245, "availability_dim": 3, "output_dim": 128,
        "embedding_dim": 128, "drug_dim": 128, "dropout": cfg.dropout,
    })
    conditioner = instantiate_with_supported_kwargs(ConditionerClass, {
        "drug_dim": 128, "context_dim": 128, "output_dim": 128,
        "embedding_dim": 128, "dropout": cfg.dropout,
    })

    params = inspect.signature(ModelClass.__init__).parameters
    kwargs = {}
    for n in ["intrinsic_drug_encoder", "drug_encoder", "intrinsic_encoder"]:
        if n in params: kwargs[n] = intrinsic_encoder; break
    for n in ["biological_context_encoder", "context_encoder", "biological_context_module"]:
        if n in params: kwargs[n] = context_encoder; break
    for n in ["context_conditioner", "drug_context_conditioner", "conditioner"]:
        if n in params: kwargs[n] = conditioner; break
    for key, value in {
        "embedding_dim": 128, "n_classes": 3, "num_classes": 3,
        "dropout": cfg.dropout, "use_l1000_aux": False,
        "use_l1000_auxiliary": False, "use_l1000": False,
    }.items():
        if key in params: kwargs[key] = value
    return ModelClass(**kwargs)


def fit_program_representation(pathways, train_rows, k, seed, outdir):
    """Train-only min/max -> NMF -> all-context program scores."""
    X = np.asarray(pathways, dtype=np.float64)
    train_rows = np.asarray(sorted(set(map(int, train_rows))), dtype=int)
    Xtr = X[train_rows]

    xmin = np.nanmin(Xtr, axis=0)
    xmax = np.nanmax(Xtr, axis=0)
    scale = xmax - xmin
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0

    X01 = (X - xmin) / scale
    X01 = np.nan_to_num(X01, nan=0.0, posinf=1.0, neginf=0.0)
    # Held-out values can lie outside training range; clipping keeps the frozen
    # training-defined non-negative coordinate system valid for NMF transform.
    X01 = np.clip(X01, 0.0, 1.0)

    nmf = NMF(
        n_components=int(k), init="nndsvda", random_state=int(seed),
        max_iter=1500, tol=1e-4, solver="cd", beta_loss="frobenius",
    )
    Wtr = nmf.fit_transform(X01[train_rows])
    Wall = nmf.transform(X01)
    H = nmf.components_

    recon = Wtr @ H
    denom = np.linalg.norm(X01[train_rows])
    rel_err = float(np.linalg.norm(X01[train_rows] - recon) / denom) if denom > 0 else np.nan

    # Standardize programs using training contexts only for neural input.
    pmean = Wall[train_rows].mean(axis=0)
    pstd = Wall[train_rows].std(axis=0)
    pstd[pstd < 1e-8] = 1.0
    Wall_z = ((Wall - pmean) / pstd).astype(np.float32)

    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outdir / "program_representation.npz",
        pathway_min=xmin.astype(np.float32), pathway_scale=scale.astype(np.float32),
        nmf_components=H.astype(np.float32), program_scores_raw=Wall.astype(np.float32),
        program_mean=pmean.astype(np.float32), program_std=pstd.astype(np.float32),
        train_pathway_rows=train_rows, k=np.array(k), seed=np.array(seed),
        relative_reconstruction_error=np.array(rel_err),
    )
    pd.DataFrame(H, index=[f"Program_{i+1}" for i in range(k)]).to_csv(
        outdir / "nmf_program_loadings.tsv", sep="\t"
    )
    pd.DataFrame({
        "k": [k], "seed": [seed], "n_train_contexts": [len(train_rows)],
        "n_iter": [int(nmf.n_iter_)], "reconstruction_err_sklearn": [float(nmf.reconstruction_err_)],
        "relative_reconstruction_error": [rel_err],
    }).to_csv(outdir / "nmf_diagnostics.tsv", sep="\t", index=False)
    return Wall_z, rel_err


def standardize_optional_context(base, features, working, train_idx, context_config, outdir):
    mods = PROGRAM_CONFIGS[context_config]
    train = working.loc[train_idx]
    save = {}
    if "expression" in mods:
        rows = train["expression_idx"].astype(int).unique()
        mean, std = base.fit_standardizer(features.expression, rows)
        base.standardize_matrix(features.expression, mean, std)
        save["expression_mean"], save["expression_std"] = mean, std
    if "regulons" in mods:
        rows = train["regulons_idx"].astype(int).unique()
        mean, std = base.fit_standardizer(features.regulons, rows)
        base.standardize_matrix(features.regulons, mean, std)
        save["regulon_mean"], save["regulon_std"] = mean, std
    if save:
        np.savez_compressed(outdir / "other_context_standardizers.npz", **save)


def candidate_name(holdout, config, seed, k, cfg):
    return (f"{holdout}__context-{config}__K{k}__seed{seed}"
            f"__druggroup-{cfg.drug_grouping}__drugtest-{cfg.drug_test_mode}")


def flatten_metrics(prefix, m, row):
    for metric in ["loss", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_ovr_auroc", "macro_auprc"]:
        row[f"{prefix}_{metric}"] = m.get(metric)
    for cls in CLASS_NAMES:
        for metric in ["precision", "recall", "f1", "support"]:
            row[f"{prefix}_{cls}_{metric}"] = m["per_class"][cls].get(metric)
        row[f"{prefix}_{cls}_auroc"] = m["per_class_auroc"].get(cls)
        row[f"{prefix}_{cls}_auprc"] = m["per_class_auprc"].get(cls)
    return row


def prepare_split_and_programs(base, dataset_module, master, holdout, seed, k, cfg, drug_group_map):
    train_idx, val_idx, test_idx, split_meta, working = base.construct_split(
        master=master, holdout=holdout, seed=seed, cfg=cfg, drug_group_map=drug_group_map
    )
    pdir = PROGRAM_DIR / f"{holdout}__seed{seed}__K{k}"
    program_file = pdir / "program_scores_z.npy"
    if program_file.exists():
        programs = np.load(program_file)
        diag = pd.read_csv(pdir / "nmf_diagnostics.tsv", sep="\t").iloc[0].to_dict()
        rel_err = float(diag["relative_reconstruction_error"])
    else:
        raw_features = dataset_module.load_all_features()
        train_rows = working.loc[train_idx, "pathways_idx"].astype(int).unique()
        programs, rel_err = fit_program_representation(raw_features.pathways, train_rows, k, seed, pdir)
        np.save(program_file, programs)
    return train_idx, val_idx, test_idx, split_meta, working, programs, rel_err


def train_candidate(base, dataset_module, master, holdout, context_config, seed, k, cfg, drug_group_map, device):
    name = candidate_name(holdout, context_config, seed, k, cfg)
    outdir = CANDIDATE_DIR / name
    outdir.mkdir(parents=True, exist_ok=True)
    result_path = outdir / "candidate_metrics.json"
    if result_path.exists():
        print(f"SKIP candidate: {name}")
        return json.loads(result_path.read_text())

    base.set_seed(seed, cfg.deterministic)
    train_idx, val_idx, test_idx, split_meta, working, programs, rel_err = prepare_split_and_programs(
        base, dataset_module, master, holdout, seed, k, cfg, drug_group_map
    )

    features = dataset_module.load_all_features()
    # Repurpose pathways slot for program scores; row indexing is identical.
    features.pathways = programs.copy()
    standardize_optional_context(base, features, working, train_idx, context_config, outdir)

    train_loader = base.make_loader(train_idx, working, features, cfg, shuffle=True)
    val_loader = base.make_loader(val_idx, working, features, cfg, shuffle=False)

    model = build_model(base, cfg, context_config, k).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    class_weights = base.calculate_class_weights(working, train_idx).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience
    )
    scaler = base.make_grad_scaler(cfg.use_amp and device.type == "cuda")

    history, best_f1, best_epoch, stale = [], -np.inf, -1, 0
    ckpt_path = outdir / "best_model.pt"
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss = base.train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, cfg)
        vm, _, _, _ = base.evaluate(model, val_loader, criterion, device, cfg)
        scheduler.step(vm["macro_f1"])
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": vm["loss"],
            "val_macro_f1": vm["macro_f1"], "val_balanced_accuracy": vm["balanced_accuracy"],
            "val_macro_auroc": vm["macro_ovr_auroc"], "val_macro_auprc": vm["macro_auprc"],
            "learning_rate": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0,
        })
        pd.DataFrame(history).to_csv(outdir / "training_history.tsv", sep="\t", index=False)
        print(f"{name} | ep {epoch:02d} | val F1={vm['macro_f1']:.4f} | AP={vm['macro_auprc']:.4f}")
        if vm["macro_f1"] > best_f1:
            best_f1, best_epoch, stale = vm["macro_f1"], epoch, 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(), "val_metrics": vm,
                "class_weights": class_weights.detach().cpu(), "config": asdict(cfg),
                "holdout": holdout, "context_config": context_config, "seed": seed, "k": k,
                "total_parameters": total, "trainable_parameters": trainable,
            }, ckpt_path)
        else:
            stale += 1
        if stale >= cfg.early_stopping_patience:
            break

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    vm, vy, vp, vrows = base.evaluate(model, val_loader, criterion, device, cfg)
    base.save_predictions(working, vrows, vy, vp, outdir / "validation_predictions.tsv")
    base.save_curve_tables(vy, vp, outdir, prefix="validation")

    result = {
        "candidate_name": name, "holdout": holdout, "context_config": context_config,
        "context_label": PROGRAM_LABELS[context_config], "modalities": list(PROGRAM_CONFIGS[context_config]),
        "seed": int(seed), "k": int(k), "best_epoch": int(best_epoch),
        "selection_metric": "validation_macro_f1", "best_validation_macro_f1": float(best_f1),
        "nmf_relative_reconstruction_error": float(rel_err),
        "n_train": int(len(train_idx)), "n_validation": int(len(val_idx)), "n_test_reserved": int(len(test_idx)),
        "total_parameters": int(total), "trainable_parameters": int(trainable),
        "validation": vm, "split_metadata": split_meta,
    }
    result_path.write_text(json.dumps(result, indent=2))
    row = {k0: v for k0, v in result.items() if k0 not in {"validation", "split_metadata", "modalities"}}
    row["modalities"] = "+".join(result["modalities"])
    flatten_metrics("validation", vm, row)
    pd.DataFrame([row]).to_csv(outdir / "candidate_metrics_flat.tsv", sep="\t", index=False)
    return result


def select_k_table():
    rows = []
    for p in CANDIDATE_DIR.glob("*/candidate_metrics.json"):
        try:
            r = json.loads(p.read_text())
            row = {
                "candidate_name": r["candidate_name"], "holdout": r["holdout"],
                "context_config": r["context_config"], "context_label": r["context_label"],
                "seed": r["seed"], "k": r["k"], "best_epoch": r["best_epoch"],
                "validation_macro_f1": r["validation"]["macro_f1"],
                "validation_macro_auprc": r["validation"]["macro_auprc"],
                "validation_macro_auroc": r["validation"]["macro_ovr_auroc"],
                "nmf_relative_reconstruction_error": r["nmf_relative_reconstruction_error"],
            }
            rows.append(row)
        except Exception as e:
            print("WARNING reading", p, e)
    df = pd.DataFrame(rows)
    if df.empty:
        return df, df
    # Tie-break: macro-F1, then macro-AP, then smaller K.
    selected = (df.sort_values(
        ["holdout", "context_config", "seed", "validation_macro_f1", "validation_macro_auprc", "k"],
        ascending=[True, True, True, False, False, True]
    ).groupby(["holdout", "context_config", "seed"], as_index=False).head(1).copy())
    selected["selected"] = True
    return df.sort_values(["holdout", "context_config", "seed", "k"]), selected


def evaluate_selected(base, dataset_module, master, selected_row, cfg, drug_group_map, device):
    holdout = selected_row["holdout"]
    context_config = selected_row["context_config"]
    seed = int(selected_row["seed"]); k = int(selected_row["k"])
    name = candidate_name(holdout, context_config, seed, k, cfg)
    candidate_dir = CANDIDATE_DIR / name
    outdir = SELECTED_DIR / name
    outdir.mkdir(parents=True, exist_ok=True)
    final_path = outdir / "final_metrics.json"
    if final_path.exists():
        print(f"SKIP selected test: {name}")
        return json.loads(final_path.read_text())

    base.set_seed(seed, cfg.deterministic)
    train_idx, val_idx, test_idx, split_meta, working, programs, rel_err = prepare_split_and_programs(
        base, dataset_module, master, holdout, seed, k, cfg, drug_group_map
    )
    features = dataset_module.load_all_features()
    features.pathways = programs.copy()
    standardize_optional_context(base, features, working, train_idx, context_config, outdir)
    val_loader = base.make_loader(val_idx, working, features, cfg, shuffle=False)
    test_loader = base.make_loader(test_idx, working, features, cfg, shuffle=False)

    model = build_model(base, cfg, context_config, k).to(device)
    ckpt = torch.load(candidate_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    class_weights = ckpt["class_weights"].to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    vm, vy, vp, vrows = base.evaluate(model, val_loader, criterion, device, cfg)
    tm, ty, tp, trows = base.evaluate(model, test_loader, criterion, device, cfg)
    symmetry = base.check_pair_symmetry(model, test_loader, device)

    base.save_predictions(working, vrows, vy, vp, outdir / "validation_predictions.tsv")
    base.save_predictions(working, trows, ty, tp, outdir / "test_predictions.tsv")
    base.save_curve_tables(ty, tp, outdir, prefix="test")
    base.save_confusion_tables(ty, tp, outdir, prefix="test")
    title = f"{holdout.replace('_',' ')} — {PROGRAM_LABELS[context_config]} (K={k})"
    base.plot_roc_curves(ty, tp, title, outdir / "roc_curves")
    base.plot_pr_curves(ty, tp, title, outdir / "pr_curves")
    base.plot_confusion(ty, tp, title, outdir / "confusion_matrix")

    result = {
        "run_name": name, "holdout": holdout, "context_config": context_config,
        "context_label": PROGRAM_LABELS[context_config], "modalities": list(PROGRAM_CONFIGS[context_config]),
        "seed": seed, "selected_k": k, "k_selected_on": "validation_macro_f1",
        "best_epoch": int(ckpt["epoch"]), "nmf_relative_reconstruction_error": float(rel_err),
        "n_train": int(len(train_idx)), "n_validation": int(len(val_idx)), "n_test": int(len(test_idx)),
        "total_parameters": int(ckpt["total_parameters"]), "trainable_parameters": int(ckpt["trainable_parameters"]),
        "validation": vm, "test": tm, "final_symmetry": symmetry, "split_metadata": split_meta,
    }
    final_path.write_text(json.dumps(result, indent=2))
    row = {k0: v for k0, v in result.items() if k0 not in {"validation", "test", "split_metadata", "modalities", "final_symmetry"}}
    row["modalities"] = "+".join(result["modalities"])
    flatten_metrics("validation", vm, row); flatten_metrics("test", tm, row)
    pd.DataFrame([row]).to_csv(outdir / "final_metrics_flat.tsv", sep="\t", index=False)
    print(f"TEST {name} | F1={tm['macro_f1']:.4f} | AUROC={tm['macro_ovr_auroc']:.4f} | AP={tm['macro_auprc']:.4f}")
    return result


def collect_selected_results():
    rows = []
    for p in SELECTED_DIR.glob("*/final_metrics.json"):
        try:
            r = json.loads(p.read_text())
            row = {
                "run_name": r["run_name"], "holdout": r["holdout"], "context_config": r["context_config"],
                "context_label": r["context_label"], "seed": r["seed"], "selected_k": r["selected_k"],
                "best_epoch": r["best_epoch"], "nmf_relative_reconstruction_error": r["nmf_relative_reconstruction_error"],
                "total_parameters": r["total_parameters"], "trainable_parameters": r["trainable_parameters"],
            }
            flatten_metrics("validation", r["validation"], row)
            flatten_metrics("test", r["test"], row)
            rows.append(row)
        except Exception as e:
            print("WARNING reading", p, e)
    return pd.DataFrame(rows)


def build_summary():
    cand, selected = select_k_table()
    cand.to_csv(SUMMARY_DIR / "all_candidate_validation_metrics.tsv", sep="\t", index=False)
    selected.to_csv(SUMMARY_DIR / "selected_k_by_validation.tsv", sep="\t", index=False)

    final = collect_selected_results()
    final.to_csv(SUMMARY_DIR / "selected_program_test_metrics.tsv", sep="\t", index=False)
    if final.empty:
        return

    metrics = ["test_macro_f1", "test_balanced_accuracy", "test_macro_ovr_auroc", "test_macro_auprc",
               "test_antagonism_auprc", "test_synergy_auprc"]
    summary_rows = []
    for (h, c), g in final.groupby(["holdout", "context_config"]):
        row = {"holdout": h, "context_config": c, "context_label": PROGRAM_LABELS[c], "n_seeds": g["seed"].nunique()}
        row["selected_k_values"] = ",".join(map(str, sorted(g["selected_k"].astype(int).tolist())))
        for m in metrics:
            x = pd.to_numeric(g[m], errors="coerce")
            row[m+"_mean"] = x.mean(); row[m+"_sd"] = x.std(ddof=1); row[m+"_median"] = x.median()
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_DIR / "program_condition_summary.tsv", sep="\t", index=False)

    # K selection frequency: a useful diagnostic showing whether abstraction depth is stable.
    (final.groupby(["holdout", "context_config", "selected_k"]).size().rename("n_seeds")
     .reset_index().to_csv(SUMMARY_DIR / "selected_k_frequency.tsv", sep="\t", index=False))

    # Pairwise comparisons among program configurations, paired by holdout+seed.
    pair_rows, stat_rows = [], []
    for h in HOLDOUTS:
        sub = final[final["holdout"] == h]
        for i, a in enumerate(PROGRAM_ORDER):
            for b in PROGRAM_ORDER[i+1:]:
                ga = sub[sub.context_config == a].set_index("seed")
                gb = sub[sub.context_config == b].set_index("seed")
                common = sorted(set(ga.index) & set(gb.index))
                for m in metrics:
                    deltas = []
                    for s in common:
                        av = float(ga.loc[s, m]); bv = float(gb.loc[s, m]); d = bv-av
                        deltas.append(d)
                        pair_rows.append({"holdout":h,"metric":m,"context_a":a,"context_b":b,"seed":s,"a":av,"b":bv,"delta_b_minus_a":d})
                    if deltas:
                        try:
                            w,p = wilcoxon(deltas) if any(abs(x)>1e-15 for x in deltas) else (0.0,1.0)
                        except Exception:
                            w,p = np.nan,np.nan
                        stat_rows.append({"holdout":h,"metric":m,"context_a":a,"context_b":b,"n_pairs":len(deltas),
                                          "mean_delta_b_minus_a":np.mean(deltas),"median_delta_b_minus_a":np.median(deltas),
                                          "n_b_better":sum(x>0 for x in deltas),"n_a_better":sum(x<0 for x in deltas),
                                          "wilcoxon_statistic":w,"wilcoxon_p":p})
    pd.DataFrame(pair_rows).to_csv(SUMMARY_DIR / "pairwise_program_comparisons.tsv", sep="\t", index=False)
    pd.DataFrame(stat_rows).to_csv(SUMMARY_DIR / "pairwise_program_statistics.tsv", sep="\t", index=False)

    # Combined table with the previous 7-representation benchmark if available.
    combined_rows = []
    if PREVIOUS_CONTEXT_METRICS.exists():
        old = pd.read_csv(PREVIOUS_CONTEXT_METRICS, sep="\t")
        old["experiment_family"] = "original_context"
        old["selected_k"] = np.nan
        combined_rows.append(old)
    f2 = final.copy(); f2["experiment_family"] = "program_abstraction"
    combined_rows.append(f2)
    combined = pd.concat(combined_rows, ignore_index=True, sort=False)
    combined.to_csv(SUMMARY_DIR / "combined_with_previous_context_ablation.tsv", sep="\t", index=False)

    # Heatmaps for selected program models only.
    for metric in ["test_macro_f1", "test_macro_ovr_auroc", "test_macro_auprc"]:
        mat = final.groupby(["context_label", "holdout"])[metric].mean().unstack("holdout")
        mat = mat.reindex([PROGRAM_LABELS[c] for c in PROGRAM_ORDER])
        mat = mat.reindex(columns=HOLDOUTS)
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(mat.values, aspect="auto")
        ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels([x.replace("_"," ") for x in mat.columns], rotation=20, ha="right")
        ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(mat.index)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.iloc[i,j]
                if pd.notna(v): ax.text(j,i,f"{v:.3f}",ha="center",va="center")
        ax.set_title(metric.replace("test_","").replace("_"," ").upper()+" — validation-selected K")
        fig.colorbar(im, ax=ax, shrink=.8)
        fig.tight_layout()
        basepath = SUMMARY_DIR / f"heatmap_{metric.replace('test_','')}"
        fig.savefig(str(basepath)+".pdf", bbox_inches="tight")
        fig.savefig(str(basepath)+".png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    print("\nPROGRAM ABLATION SUMMARY")
    print(summary.to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-script", default=None)
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument("--holdout", choices=HOLDOUTS, default="unseen_cell_line")
    p.add_argument("--context-config", choices=PROGRAM_ORDER, default="programs")
    p.add_argument("--context-configs", choices=PROGRAM_ORDER, nargs="+", default=PROGRAM_ORDER)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=[42,43,44,45,46])
    p.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_KS)
    p.add_argument("--drug-grouping", choices=["scaffold","structure","drug_id"], default="scaffold")
    p.add_argument("--drug-test-mode", choices=["any","both"], default="any")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--split-search-trials", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    base = load_base_module(args.base_script)
    # Keep any generated split metadata under this experiment's output tree.
    base.SPLIT_DIR = SPLIT_DIR

    if args.plot_only:
        build_summary(); return

    cfg = base.Config(seed=args.seed, batch_size=args.batch_size, epochs=args.epochs,
                      split_search_trials=args.split_search_trials,
                      drug_grouping=args.drug_grouping, drug_test_mode=args.drug_test_mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda": print("GPU:", torch.cuda.get_device_name(0))
    dataset_module = base.load_dataset_module()
    master = base.load_master(dataset_module)
    need_drugs = args.run_all or args.holdout in {"unseen_drug","unseen_both"}
    drug_group_map = base.load_drug_groups(master, dataset_module, cfg.drug_grouping) if need_drugs else None

    if args.run_all:
        groups = [(h,c,s) for s in args.seeds for h in HOLDOUTS for c in args.context_configs]
    else:
        groups = [(args.holdout,args.context_config,args.seed)]
    plan = [{"holdout":h,"context_config":c,"seed":s,"k":k} for h,c,s in groups for k in args.k_values]
    pd.DataFrame(plan).to_csv(OUTPUT_ROOT / "experiment_plan.tsv", sep="\t", index=False)
    print(f"Candidate trainings planned: {len(plan)}")
    print(f"Final selected test evaluations planned: {len(groups)}")

    # Stage 1: validation-only candidate training for every K.
    for i,(h,c,s) in enumerate(groups,1):
        for k in args.k_values:
            run_cfg = copy.deepcopy(cfg); run_cfg.seed = s
            outdir = CANDIDATE_DIR / candidate_name(h,c,s,k,run_cfg)
            if args.force_rerun and outdir.exists(): shutil.rmtree(outdir)
            print(f"\nCANDIDATE group {i}/{len(groups)} | {h} | {c} | seed={s} | K={k}")
            train_candidate(base,dataset_module,master,h,c,s,k,run_cfg,drug_group_map,device)

    # Stage 2: select K strictly on validation, then touch test once.
    cand, selected = select_k_table()
    cand.to_csv(SUMMARY_DIR / "all_candidate_validation_metrics.tsv", sep="\t", index=False)
    selected.to_csv(SUMMARY_DIR / "selected_k_by_validation.tsv", sep="\t", index=False)
    expected = len(groups)
    if len(selected) != expected:
        raise RuntimeError(f"Expected {expected} selected configurations but found {len(selected)}")

    for _, row in selected.iterrows():
        run_cfg = copy.deepcopy(cfg); run_cfg.seed = int(row.seed)
        evaluate_selected(base,dataset_module,master,row,run_cfg,drug_group_map,device)

    build_summary()
    print("ALL REQUESTED PROGRAM-ABSTRACTION EXPERIMENTS COMPLETE")


if __name__ == "__main__":
    main()
