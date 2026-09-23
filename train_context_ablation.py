#!/usr/bin/env python3
"""
Biological-context representation ablation used for the DrugComb publication experiments.

Context configurations:
  expression
  pathways
  regulons
  expression_pathways
  expression_regulons
  pathways_regulons
  expression_pathways_regulons

Default full design:
  7 representations x 3 holdouts x 5 seeds = 105 runs.

This script runs the seven biological-context representations across the three
generalization settings and five optimization seeds used in the publication.
Shared data loading, split construction, batching, training, evaluation, and
plotting utilities are imported from train_regulon_ablation_final.py.

For compatibility with archived publication outputs, metric dictionary keys
ending in ``_auprc`` are retained. In the underlying implementation these
values are computed with sklearn.metrics.average_precision_score and should be
interpreted/reported as average precision (AP), not trapezoidal PR-AUC.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# Repository root. By default this is the directory containing this script.
# Set DRUG_SYNERGY_ROOT to override it without editing the source.
ROOT = Path(
    os.environ.get("DRUG_SYNERGY_ROOT", Path(__file__).resolve().parent)
).resolve()

# Shared utilities used by the publication experiments.
BASE_SCRIPT_CANDIDATES = [
    ROOT / "train_regulon_ablation_final.py",
]

OUTPUT_ROOT = ROOT / "training" / "context_ablation"
SPLIT_DIR = OUTPUT_ROOT / "splits"
RUN_DIR = OUTPUT_ROOT / "runs"
SUMMARY_DIR = OUTPUT_ROOT / "summary"
for d in [OUTPUT_ROOT, SPLIT_DIR, RUN_DIR, SUMMARY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CONTEXT_CONFIGS = {
    "expression": ("expression",),
    "pathways": ("pathways",),
    "regulons": ("regulons",),
    "expression_pathways": ("expression", "pathways"),
    "expression_regulons": ("expression", "regulons"),
    "pathways_regulons": ("pathways", "regulons"),
    "expression_pathways_regulons": ("expression", "pathways", "regulons"),
}
CONTEXT_ORDER = list(CONTEXT_CONFIGS)
CONTEXT_LABELS = {
    "expression": "Expression only",
    "pathways": "Pathways only",
    "regulons": "Regulons only",
    "expression_pathways": "Expression + pathways",
    "expression_regulons": "Expression + regulons",
    "pathways_regulons": "Pathways + regulons",
    "expression_pathways_regulons": "Expression + pathways + regulons",
}
BASELINE_CONTEXT = "expression_pathways"
CLASS_NAMES = ["antagonism", "no_interaction", "synergy"]
HOLDOUTS = ["unseen_cell_line", "unseen_drug", "unseen_both"]


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
            return import_module_from_path("regulon_ablation_base", p)
    raise FileNotFoundError(
        "Could not find train_regulon_ablation_final.py "
        f"under repository root {ROOT}. Use --base-script PATH to override."
    )


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


class ConfigurableBiologicalContextEncoder(nn.Module):
    """Any non-empty subset of expression/pathways/regulons -> 128-d context."""

    def __init__(
        self,
        modalities,
        expression_dim=942,
        pathway_dim=50,
        regulon_dim=771,
        branch_dim=64,
        fusion_hidden=256,
        output_dim=128,
        dropout=0.20,
    ):
        super().__init__()
        self.modalities = tuple(modalities)
        if not self.modalities:
            raise ValueError("At least one modality is required")

        if "expression" in self.modalities:
            self.expression_encoder = BranchEncoder(expression_dim, 256, branch_dim, dropout)
        if "pathways" in self.modalities:
            self.pathway_encoder = BranchEncoder(pathway_dim, 64, branch_dim, dropout)
        if "regulons" in self.modalities:
            self.regulon_encoder = BranchEncoder(regulon_dim, 256, branch_dim, dropout)

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
            parts.append(z)
            aux["expression_embedding"] = z
        if "pathways" in self.modalities:
            z = self.pathway_encoder(pathways)
            parts.append(z)
            aux["pathway_embedding"] = z
        if "regulons" in self.modalities:
            z = self.regulon_encoder(regulons)
            parts.append(z)
            aux["regulon_embedding"] = z
        embedding = self.fusion(torch.cat(parts, dim=-1))
        return {"embedding": embedding, **aux} if return_aux else embedding


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


def build_model(base, cfg, context_config: str):
    modalities = CONTEXT_CONFIGS[context_config]
    print(f"\nBUILD MODEL: {CONTEXT_LABELS[context_config]} | {modalities}")

    intrinsic_module = base.import_module_from_path(
        f"intrinsic_context_{context_config}_{id(cfg)}", base.INTRINSIC_ENCODER_PATH
    )
    conditioner_module = base.import_module_from_path(
        f"conditioner_context_{context_config}_{id(cfg)}", base.CONTEXT_CONDITIONER_PATH
    )
    synergy_module = base.import_module_from_path(
        f"synergy_context_{context_config}_{id(cfg)}", base.SYNERGY_MODEL_PATH
    )

    IntrinsicClass = find_class(intrinsic_module, ["IntrinsicDrugEncoder"])
    ConditionerClass = find_class(conditioner_module, ["ContextConditioner", "DrugContextConditioner"])
    ModelClass = find_class(synergy_module, ["DrugCombinationModel"])

    context_encoder = ConfigurableBiologicalContextEncoder(
        modalities=modalities, dropout=cfg.dropout, output_dim=128
    )
    intrinsic_encoder = instantiate_with_supported_kwargs(IntrinsicClass, {
        "morgan_dim": 2048,
        "target_dim": 8245,
        "direct_target_dim": 8245,
        "ppi_dim": 8245,
        "availability_dim": 3,
        "output_dim": 128,
        "embedding_dim": 128,
        "drug_dim": 128,
        "dropout": cfg.dropout,
    })
    conditioner = instantiate_with_supported_kwargs(ConditionerClass, {
        "drug_dim": 128,
        "context_dim": 128,
        "output_dim": 128,
        "embedding_dim": 128,
        "dropout": cfg.dropout,
    })

    sig = inspect.signature(ModelClass.__init__)
    params = sig.parameters
    kwargs = {}
    for n in ["intrinsic_drug_encoder", "drug_encoder", "intrinsic_encoder"]:
        if n in params:
            kwargs[n] = intrinsic_encoder
            break
    for n in ["biological_context_encoder", "context_encoder", "biological_context_module"]:
        if n in params:
            kwargs[n] = context_encoder
            break
    for n in ["context_conditioner", "drug_context_conditioner", "conditioner"]:
        if n in params:
            kwargs[n] = conditioner
            break
    for k, v in {
        "embedding_dim": 128,
        "n_classes": 3,
        "num_classes": 3,
        "dropout": cfg.dropout,
        "use_l1000_aux": False,
        "use_l1000_auxiliary": False,
        "use_l1000": False,
    }.items():
        if k in params:
            kwargs[k] = v

    model = ModelClass(**kwargs)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total / {trainable:,} trainable")
    return model


def normalize_context_train_only(base, features, master, train_idx, context_config, output_dir):
    modalities = CONTEXT_CONFIGS[context_config]
    train = master.loc[train_idx]
    save = {}

    if "expression" in modalities:
        rows = train["expression_idx"].astype(int).unique()
        mean, std = base.fit_standardizer(features.expression, rows)
        base.standardize_matrix(features.expression, mean, std)
        save["expression_mean"], save["expression_std"] = mean, std
    if "pathways" in modalities:
        rows = train["pathways_idx"].astype(int).unique()
        mean, std = base.fit_standardizer(features.pathways, rows)
        base.standardize_matrix(features.pathways, mean, std)
        save["pathway_mean"], save["pathway_std"] = mean, std
    if "regulons" in modalities:
        rows = train["regulons_idx"].astype(int).unique()
        mean, std = base.fit_standardizer(features.regulons, rows)
        base.standardize_matrix(features.regulons, mean, std)
        save["regulon_mean"], save["regulon_std"] = mean, std

    save["context_config"] = np.array(context_config)
    np.savez_compressed(output_dir / "context_standardizers.npz", **save)


def run_name(holdout, context_config, seed, cfg):
    return (
        f"{holdout}__context-{context_config}__seed{seed}"
        f"__druggroup-{cfg.drug_grouping}__drugtest-{cfg.drug_test_mode}"
    )


def flatten_result(result):
    row = {
        "run_name": result["run_name"],
        "holdout": result["holdout"],
        "context_config": result["context_config"],
        "context_label": result["context_label"],
        "modalities": "+".join(result["modalities"]),
        "seed": result["seed"],
        "best_epoch": result["best_epoch"],
        "best_validation_macro_f1": result["best_validation_macro_f1"],
        "n_train": result["n_train"],
        "n_validation": result["n_validation"],
        "n_test": result["n_test"],
        "n_train_contexts": result["n_train_contexts"],
        "n_validation_contexts": result["n_validation_contexts"],
        "n_test_contexts": result["n_test_contexts"],
        "total_parameters": result["total_parameters"],
        "trainable_parameters": result["trainable_parameters"],
    }
    for split in ["validation", "test"]:
        m = result[split]
        for metric in ["loss", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_ovr_auroc", "macro_auprc"]:
            row[f"{split}_{metric}"] = m.get(metric)
        for cls in CLASS_NAMES:
            for metric in ["precision", "recall", "f1", "support"]:
                row[f"{split}_{cls}_{metric}"] = m["per_class"][cls].get(metric)
            row[f"{split}_{cls}_auroc"] = m["per_class_auroc"].get(cls)
            row[f"{split}_{cls}_auprc"] = m["per_class_auprc"].get(cls)
    return row


def run_experiment(base, holdout, context_config, seed, cfg, master, dataset_module, drug_group_map, device):
    base.set_seed(seed, cfg.deterministic)
    name = run_name(holdout, context_config, seed, cfg)
    outdir = RUN_DIR / name
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / "final_metrics.json"
    if metrics_path.exists():
        print(f"SKIP completed: {name}")
        return json.loads(metrics_path.read_text())

    train_idx, val_idx, test_idx, split_meta, working = base.construct_split(
        master=master, holdout=holdout, seed=seed, cfg=cfg, drug_group_map=drug_group_map
    )
    # Save in this experiment's own split folder.
    old_split_dir = base.SPLIT_DIR
    base.SPLIT_DIR = SPLIT_DIR
    try:
        base.save_split_assignment(working, train_idx, val_idx, test_idx, split_meta, holdout, seed, cfg)
    finally:
        base.SPLIT_DIR = old_split_dir

    print(f"Train/val/test: {len(train_idx):,}/{len(val_idx):,}/{len(test_idx):,}")
    features = dataset_module.load_all_features()
    normalize_context_train_only(base, features, working, train_idx, context_config, outdir)

    train_loader = base.make_loader(train_idx, working, features, cfg, shuffle=True)
    val_loader = base.make_loader(val_idx, working, features, cfg, shuffle=False)
    test_loader = base.make_loader(test_idx, working, features, cfg, shuffle=False)

    model = build_model(base, cfg, context_config).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    class_weights = base.calculate_class_weights(working, train_idx).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience
    )
    scaler = base.make_grad_scaler(cfg.use_amp and device.type == "cuda")
    initial_symmetry = base.check_pair_symmetry(model, val_loader, device)

    history, best_f1, best_epoch, stale = [], -np.inf, -1, 0
    checkpoint_path = outdir / "best_model.pt"
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss = base.train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, cfg)
        vm, _, _, _ = base.evaluate(model, val_loader, criterion, device, cfg)
        scheduler.step(vm["macro_f1"])
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": vm["loss"],
            "val_macro_f1": vm["macro_f1"],
            "val_balanced_accuracy": vm["balanced_accuracy"],
            "val_macro_auroc": vm["macro_ovr_auroc"],
            "val_macro_auprc": vm["macro_auprc"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - t0,
        })
        pd.DataFrame(history).to_csv(outdir / "training_history.tsv", sep="\t", index=False)
        print(
            f"Epoch {epoch:02d} | train={train_loss:.4f} | val={vm['loss']:.4f} | "
            f"F1={vm['macro_f1']:.4f} | bal.acc={vm['balanced_accuracy']:.4f} | "
            f"AUROC={vm['macro_ovr_auroc']:.4f} | AUPRC={vm['macro_auprc']:.4f}"
        )
        if vm["macro_f1"] > best_f1:
            best_f1, best_epoch, stale = vm["macro_f1"], epoch, 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": vm,
                "class_weights": class_weights.detach().cpu(),
                "config": asdict(cfg),
                "holdout": holdout,
                "context_config": context_config,
                "modalities": list(CONTEXT_CONFIGS[context_config]),
                "seed": seed,
                "total_parameters": total,
                "trainable_parameters": trainable,
            }, checkpoint_path)
        else:
            stale += 1
        if stale >= cfg.early_stopping_patience:
            print("Early stopping.")
            break

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    val_m, val_y, val_p, val_rows = base.evaluate(model, val_loader, criterion, device, cfg)
    test_m, test_y, test_p, test_rows = base.evaluate(model, test_loader, criterion, device, cfg)
    final_symmetry = base.check_pair_symmetry(model, test_loader, device)

    base.save_predictions(working, val_rows, val_y, val_p, outdir / "validation_predictions.tsv")
    base.save_predictions(working, test_rows, test_y, test_p, outdir / "test_predictions.tsv")
    base.save_curve_tables(val_y, val_p, outdir, prefix="validation")
    base.save_curve_tables(test_y, test_p, outdir, prefix="test")
    base.save_confusion_tables(val_y, val_p, outdir, prefix="validation")
    base.save_confusion_tables(test_y, test_p, outdir, prefix="test")
    base.plot_training_history(pd.DataFrame(history), outdir / "training_history")

    title = f"{holdout.replace('_', ' ')} — {CONTEXT_LABELS[context_config]}"
    base.plot_roc_curves(test_y, test_p, title, outdir / "roc_curves")
    base.plot_pr_curves(test_y, test_p, title, outdir / "pr_curves")
    base.plot_confusion(test_y, test_p, title, outdir / "confusion_matrix")

    result = {
        "run_name": name,
        "holdout": holdout,
        "context_config": context_config,
        "context_label": CONTEXT_LABELS[context_config],
        "modalities": list(CONTEXT_CONFIGS[context_config]),
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_validation_macro_f1": float(best_f1),
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_train_contexts": int(working.loc[train_idx, "depmap_id_std"].nunique()),
        "n_validation_contexts": int(working.loc[val_idx, "depmap_id_std"].nunique()),
        "n_test_contexts": int(working.loc[test_idx, "depmap_id_std"].nunique()),
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "split_metadata": split_meta,
        "initial_symmetry": initial_symmetry,
        "final_symmetry": final_symmetry,
        "validation": val_m,
        "test": test_m,
        "config": asdict(cfg),
    }
    metrics_path.write_text(json.dumps(result, indent=2))
    pd.DataFrame([flatten_result(result)]).to_csv(outdir / "final_metrics_flat.tsv", sep="\t", index=False)
    pd.DataFrame({
        "class_id": [0, 1, 2],
        "class_name": CLASS_NAMES,
        "weight": class_weights.detach().cpu().numpy(),
    }).to_csv(outdir / "class_weights.tsv", sep="\t", index=False)

    print(
        f"DONE {name}: macro-F1={test_m['macro_f1']:.4f}, "
        f"macro AUROC={test_m['macro_ovr_auroc']:.4f}, macro AUPRC={test_m['macro_auprc']:.4f}"
    )
    return result


def collect_results():
    rows = []
    for p in RUN_DIR.glob("*/final_metrics.json"):
        try:
            r = json.loads(p.read_text())
            if "context_config" in r:
                rows.append(flatten_result(r))
        except Exception as e:
            print(f"WARNING reading {p}: {e}")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    order = {c: i for i, c in enumerate(CONTEXT_ORDER)}
    df["_o"] = df["context_config"].map(order)
    return df.sort_values(["holdout", "seed", "_o"]).drop(columns="_o")


SUMMARY_METRICS = [
    "test_macro_f1", "test_balanced_accuracy", "test_macro_ovr_auroc", "test_macro_auprc",
    "test_antagonism_f1", "test_antagonism_auroc", "test_antagonism_auprc",
    "test_no_interaction_f1", "test_no_interaction_auroc", "test_no_interaction_auprc",
    "test_synergy_f1", "test_synergy_auroc", "test_synergy_auprc",
]


def condition_summary(df):
    rows = []
    for (h, c), g in df.groupby(["holdout", "context_config"]):
        row = {
            "holdout": h,
            "context_config": c,
            "context_label": CONTEXT_LABELS[c],
            "n_seeds": int(g["seed"].nunique()),
            "mean_total_parameters": float(g["total_parameters"].mean()),
        }
        for m in SUMMARY_METRICS:
            v = pd.to_numeric(g[m], errors="coerce")
            row[f"{m}_mean"] = v.mean()
            row[f"{m}_sd"] = v.std(ddof=1) if v.notna().sum() > 1 else 0.0
            row[f"{m}_median"] = v.median()
            row[f"{m}_min"] = v.min()
            row[f"{m}_max"] = v.max()
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_comparison(df):
    rows = []
    for (h, seed), g in df.groupby(["holdout", "seed"]):
        b = g[g["context_config"] == BASELINE_CONTEXT]
        if b.empty:
            continue
        b = b.iloc[0]
        for _, r in g.iterrows():
            if r["context_config"] == BASELINE_CONTEXT:
                continue
            row = {"holdout": h, "seed": int(seed), "context_config": r["context_config"], "context_label": r["context_label"]}
            for m in SUMMARY_METRICS:
                row[f"{m}_baseline"] = b[m]
                row[f"{m}_context"] = r[m]
                row[f"{m}_delta_vs_baseline"] = r[m] - b[m]
            rows.append(row)
    return pd.DataFrame(rows)


def baseline_statistics(comp):
    rows = []
    if comp.empty:
        return pd.DataFrame()
    for (h, c), g in comp.groupby(["holdout", "context_config"]):
        for m in SUMMARY_METRICS:
            col = f"{m}_delta_vs_baseline"
            vals = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy(float)
            if not len(vals):
                continue
            try:
                stat, p = wilcoxon(vals) if np.any(vals != 0) else (0.0, 1.0)
            except Exception:
                stat, p = np.nan, np.nan
            rows.append({
                "holdout": h, "context_config": c, "context_label": CONTEXT_LABELS[c], "metric": m,
                "n_pairs": len(vals), "mean_delta": vals.mean(),
                "sd_delta": vals.std(ddof=1) if len(vals) > 1 else 0.0,
                "median_delta": np.median(vals), "min_delta": vals.min(), "max_delta": vals.max(),
                "n_improved": int((vals > 0).sum()), "n_worsened": int((vals < 0).sum()),
                "wilcoxon_statistic": stat, "wilcoxon_p": p,
            })
    return pd.DataFrame(rows)


def pairwise_tables(df):
    raw = []
    stats = []
    for (h, seed), g in df.groupby(["holdout", "seed"]):
        available = [c for c in CONTEXT_ORDER if c in set(g["context_config"])]
        for i, a in enumerate(available):
            ra = g[g["context_config"] == a].iloc[0]
            for b in available[i+1:]:
                rb = g[g["context_config"] == b].iloc[0]
                row = {"holdout": h, "seed": int(seed), "context_a": a, "context_b": b}
                for m in SUMMARY_METRICS:
                    row[f"{m}_a"] = ra[m]
                    row[f"{m}_b"] = rb[m]
                    row[f"{m}_delta_b_minus_a"] = rb[m] - ra[m]
                raw.append(row)
    raw_df = pd.DataFrame(raw)
    if not raw_df.empty:
        for (h, a, b), g in raw_df.groupby(["holdout", "context_a", "context_b"]):
            for m in SUMMARY_METRICS:
                vals = pd.to_numeric(g[f"{m}_delta_b_minus_a"], errors="coerce").dropna().to_numpy(float)
                if not len(vals):
                    continue
                try:
                    stat, p = wilcoxon(vals) if np.any(vals != 0) else (0.0, 1.0)
                except Exception:
                    stat, p = np.nan, np.nan
                stats.append({
                    "holdout": h, "context_a": a, "context_b": b, "metric": m,
                    "n_pairs": len(vals), "mean_delta_b_minus_a": vals.mean(),
                    "median_delta_b_minus_a": np.median(vals),
                    "n_b_better": int((vals > 0).sum()), "n_a_better": int((vals < 0).sum()),
                    "wilcoxon_statistic": stat, "wilcoxon_p": p,
                })
    return raw_df, pd.DataFrame(stats)


def plot_context_metric(df, metric, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), squeeze=False)
    x = np.arange(len(CONTEXT_ORDER))
    for ax, h in zip(axes[0], HOLDOUTS):
        sub = df[df["holdout"] == h]
        means, sds = [], []
        for i, c in enumerate(CONTEXT_ORDER):
            vals = pd.to_numeric(sub[sub["context_config"] == c][metric], errors="coerce").dropna()
            means.append(vals.mean() if len(vals) else np.nan)
            sds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
            if len(vals):
                ax.scatter(np.repeat(i, len(vals)), vals, alpha=0.45, s=25)
        ax.errorbar(x, means, yerr=sds, fmt="o-", capsize=3)
        ax.set_xticks(x, [CONTEXT_LABELS[c] for c in CONTEXT_ORDER], rotation=45, ha="right")
        ax.set_title(h.replace("_", " "))
        ax.set_ylabel(metric.replace("test_", "").replace("_", " "))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"Context representation ablation: {metric.replace('test_', '').replace('_', ' ')}")
    base = str(path)
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(cond, metric, path):
    col = f"{metric}_mean"
    mat = np.full((len(CONTEXT_ORDER), len(HOLDOUTS)), np.nan)
    for i, c in enumerate(CONTEXT_ORDER):
        for j, h in enumerate(HOLDOUTS):
            s = cond[(cond["context_config"] == c) & (cond["holdout"] == h)][col]
            if len(s):
                mat[i, j] = float(s.iloc[0])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(np.arange(3), [h.replace("_", " ") for h in HOLDOUTS], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(CONTEXT_ORDER)), [CONTEXT_LABELS[c] for c in CONTEXT_ORDER])
    ax.set_title(f"Mean {metric.replace('test_', '').replace('_', ' ')}")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.savefig(str(path) + ".pdf", bbox_inches="tight")
    fig.savefig(str(path) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_summary():
    df = collect_results()
    if df.empty:
        print("No completed runs.")
        return
    df.to_csv(SUMMARY_DIR / "all_run_metrics.tsv", sep="\t", index=False)
    cond = condition_summary(df)
    cond.to_csv(SUMMARY_DIR / "condition_summary.tsv", sep="\t", index=False)

    comp = baseline_comparison(df)
    comp.to_csv(SUMMARY_DIR / "paired_vs_expression_pathways.tsv", sep="\t", index=False)
    baseline_statistics(comp).to_csv(
        SUMMARY_DIR / "paired_vs_expression_pathways_statistics.tsv", sep="\t", index=False
    )
    raw_pair, stats_pair = pairwise_tables(df)
    raw_pair.to_csv(SUMMARY_DIR / "all_pairwise_context_comparisons.tsv", sep="\t", index=False)
    stats_pair.to_csv(SUMMARY_DIR / "all_pairwise_context_statistics.tsv", sep="\t", index=False)

    # Ranking table.
    ranking = []
    for h in HOLDOUTS:
        sub = cond[cond["holdout"] == h]
        for m in ["test_macro_f1", "test_macro_auprc", "test_macro_ovr_auroc", "test_synergy_auprc", "test_antagonism_auprc"]:
            col = f"{m}_mean"
            for rank, (_, r) in enumerate(sub.sort_values(col, ascending=False).iterrows(), start=1):
                ranking.append({"holdout": h, "metric": m, "rank": rank, "context_config": r["context_config"], "context_label": r["context_label"], "mean": r[col], "sd": r[f"{m}_sd"]})
    pd.DataFrame(ranking).to_csv(SUMMARY_DIR / "context_ranking.tsv", sep="\t", index=False)

    for metric, short in [
        ("test_macro_f1", "macro_f1"),
        ("test_balanced_accuracy", "balanced_accuracy"),
        ("test_macro_ovr_auroc", "macro_auroc"),
        ("test_macro_auprc", "macro_auprc"),
        ("test_synergy_auprc", "synergy_auprc"),
        ("test_antagonism_auprc", "antagonism_auprc"),
    ]:
        plot_context_metric(df, metric, SUMMARY_DIR / f"context_ablation_{short}")
        plot_heatmap(cond, metric, SUMMARY_DIR / f"heatmap_{short}")

    # Completion matrix for overnight monitoring/restart.
    seeds = sorted(df["seed"].unique())
    completion = []
    for h in HOLDOUTS:
        for s in seeds:
            present = set(df[(df["holdout"] == h) & (df["seed"] == s)]["context_config"])
            for c in CONTEXT_ORDER:
                completion.append({"holdout": h, "seed": int(s), "context_config": c, "completed": c in present})
    pd.DataFrame(completion).to_csv(SUMMARY_DIR / "completion_status.tsv", sep="\t", index=False)

    print(f"Summary written to {SUMMARY_DIR}; completed runs={len(df)}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-script", default=None)
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument("--summary-at-end-only", action="store_true")
    p.add_argument("--holdout", choices=HOLDOUTS, default="unseen_cell_line")
    p.add_argument("--context-config", choices=CONTEXT_ORDER, default="expression_pathways_regulons")
    p.add_argument("--context-configs", choices=CONTEXT_ORDER, nargs="+", default=CONTEXT_ORDER)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--drug-grouping", choices=["scaffold", "structure", "drug_id"], default="scaffold")
    p.add_argument("--drug-test-mode", choices=["any", "both"], default="any")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--split-search-trials", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    base = load_base_module(args.base_script)
    if args.plot_only:
        build_summary()
        return

    cfg = base.Config(
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        split_search_trials=args.split_search_trials,
        drug_grouping=args.drug_grouping,
        drug_test_mode=args.drug_test_mode,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps(asdict(cfg), indent=2))
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    dataset_module = base.load_dataset_module()
    master = base.load_master(dataset_module)
    need_drug_groups = args.run_all or args.holdout in {"unseen_drug", "unseen_both"}
    drug_group_map = base.load_drug_groups(master, dataset_module, cfg.drug_grouping) if need_drug_groups else None

    if args.run_all:
        experiments = [
            (h, c, s)
            for s in args.seeds
            for h in HOLDOUTS
            for c in args.context_configs
        ]
    else:
        experiments = [(args.holdout, args.context_config, args.seed)]

    print(f"Planned runs: {len(experiments)}")
    pd.DataFrame([
        {"holdout": h, "context_config": c, "context_label": CONTEXT_LABELS[c], "seed": s}
        for h, c, s in experiments
    ]).to_csv(OUTPUT_ROOT / "experiment_plan.tsv", sep="\t", index=False)

    for i, (h, c, s) in enumerate(experiments, start=1):
        print(f"\n[{i}/{len(experiments)}] {h} | {c} | seed={s}")
        run_cfg = copy.deepcopy(cfg)
        run_cfg.seed = s
        outdir = RUN_DIR / run_name(h, c, s, run_cfg)
        if args.force_rerun and outdir.exists():
            shutil.rmtree(outdir)
        if h in {"unseen_drug", "unseen_both"} and drug_group_map is None:
            drug_group_map = base.load_drug_groups(master, dataset_module, run_cfg.drug_grouping)
        run_experiment(base, h, c, s, run_cfg, master, dataset_module, drug_group_map, device)
        if not args.summary_at_end_only:
            build_summary()

    build_summary()
    print("ALL REQUESTED RUNS COMPLETE")


if __name__ == "__main__":
    main()
