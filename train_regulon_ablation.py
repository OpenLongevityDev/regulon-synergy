#!/usr/bin/env python3
"""
train_regulon_ablation.py

Regulon-ablation experiment used in development of the DrugComb biological
context model.

Comparison
----------
WITHOUT regulons:
    expression + Hallmark pathway activity

WITH regulons:
    expression + Hallmark pathway activity + CollecTRI TF activity

The experiment is evaluated under unseen-cell-line, unseen-drug, and
unseen-both generalization settings. Paired ON/OFF runs use the same training
seed and the same split construction procedure.

The reusable audited implementation for splitting, train-only normalization,
training, evaluation, prediction export, and plotting is in
``experiment_utils.py``.

Historical output columns containing ``auprc`` are retained for compatibility;
their values are sklearn average precision (AP).
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon

import experiment_utils as utils
from experiment_utils import *  # noqa: F401,F403

ROOT = utils.ROOT
OUTPUT_ROOT = ROOT / "training" / "regulon_ablation"
SPLIT_DIR = OUTPUT_ROOT / "splits"
RUN_DIR = OUTPUT_ROOT / "runs"
SUMMARY_DIR = OUTPUT_ROOT / "summary"

for directory in (OUTPUT_ROOT, SPLIT_DIR, RUN_DIR, SUMMARY_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# save_split_assignment() is defined in experiment_utils.py and resolves
# SPLIT_DIR in that module's namespace.
utils.SPLIT_DIR = SPLIT_DIR

def run_name(
    holdout: str,
    use_regulons: bool,
    seed: int,
    cfg: Config,
):
    return (
        f"{holdout}"
        f"__regulons-{'on' if use_regulons else 'off'}"
        f"__seed{seed}"
        f"__druggroup-{cfg.drug_grouping}"
        f"__drugtest-{cfg.drug_test_mode}"
    )


def run_experiment(
    holdout: str,
    use_regulons: bool,
    seed: int,
    cfg: Config,
    master: pd.DataFrame,
    dataset_module,
    drug_group_map: Optional[Dict[str, str]],
    device: torch.device,
):
    set_seed(
        seed,
        cfg.deterministic,
    )

    name = run_name(
        holdout,
        use_regulons,
        seed,
        cfg,
    )

    outdir = RUN_DIR / name
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_metrics_path = outdir / "final_metrics.json"

    if final_metrics_path.exists():
        print(
            f"\nSKIP existing completed run: {name}"
        )
        with open(final_metrics_path) as f:
            return json.load(f)

    print_section(
        f"EXPERIMENT | {name}"
    )

    train_idx, val_idx, test_idx, split_meta, working_master = construct_split(
        master=master,
        holdout=holdout,
        seed=seed,
        cfg=cfg,
        drug_group_map=drug_group_map,
    )

    # Persist exactly the split used in the comparison.
    save_split_assignment(
        working_master,
        train_idx,
        val_idx,
        test_idx,
        split_meta,
        holdout,
        seed,
        cfg,
    )

    print(
        f"Train / val / test observations: "
        f"{len(train_idx):,} / {len(val_idx):,} / {len(test_idx):,}"
    )

    print(
        "Train / val / test cell lines:",
        working_master.loc[train_idx, "depmap_id_std"].nunique(),
        working_master.loc[val_idx, "depmap_id_std"].nunique(),
        working_master.loc[test_idx, "depmap_id_std"].nunique(),
    )

    for split_label, idx in [
        ("train", train_idx),
        ("validation", val_idx),
        ("test", test_idx),
    ]:
        p = class_proportions(
            working_master.loc[idx, "label_int"].to_numpy(dtype=int)
        )
        print(
            split_label,
            {
                name: round(float(v), 4)
                for name, v in zip(CLASS_NAMES, p)
            },
        )

    # Reload a fresh feature store for every run because standardization
    # mutates the context arrays in memory.
    features = dataset_module.load_all_features()

    normalize_context_train_only(
        features=features,
        master=working_master,
        train_idx=train_idx,
        use_regulons=use_regulons,
        output_dir=outdir,
    )

    train_loader = make_loader(
        train_idx,
        working_master,
        features,
        cfg,
        shuffle=True,
    )
    val_loader = make_loader(
        val_idx,
        working_master,
        features,
        cfg,
        shuffle=False,
    )
    test_loader = make_loader(
        test_idx,
        working_master,
        features,
        cfg,
        shuffle=False,
    )

    model = build_core_model(
        cfg,
        use_regulons,
    ).to(device)

    total_params, trainable_params = count_parameters(
        model
    )

    class_weights = calculate_class_weights(
        working_master,
        train_idx,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
    )

    scaler = make_grad_scaler(
        cfg.use_amp and device.type == "cuda"
    )

    initial_symmetry = check_pair_symmetry(
        model,
        val_loader,
        device,
    )

    history = []
    best_macro_f1 = -np.inf
    best_epoch = -1
    no_improvement = 0

    checkpoint_path = outdir / "best_model.pt"

    for epoch in range(
        1,
        cfg.epochs + 1,
    ):
        t0 = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            cfg,
        )

        val_metrics, _, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
            cfg,
        )

        scheduler.step(
            val_metrics["macro_f1"]
        )

        lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_macro_auroc": val_metrics["macro_ovr_auroc"],
                "val_macro_auprc": val_metrics["macro_auprc"],
                "learning_rate": lr,
                "seconds": time.time() - t0,
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train={train_loss:.4f} | "
            f"val={val_metrics['loss']:.4f} | "
            f"F1={val_metrics['macro_f1']:.4f} | "
            f"bal.acc={val_metrics['balanced_accuracy']:.4f} | "
            f"AUROC={val_metrics['macro_ovr_auroc']:.4f} | "
            f"AP={val_metrics['macro_auprc']:.4f}"
        )

        current = val_metrics["macro_f1"]

        if current > best_macro_f1:
            best_macro_f1 = current
            best_epoch = epoch
            no_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "class_weights": class_weights.detach().cpu(),
                    "config": asdict(cfg),
                    "holdout": holdout,
                    "use_regulons": use_regulons,
                    "seed": seed,
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                },
                checkpoint_path,
            )
        else:
            no_improvement += 1

        pd.DataFrame(history).to_csv(
            outdir / "training_history.tsv",
            sep="\t",
            index=False,
        )

        if no_improvement >= cfg.early_stopping_patience:
            print("Early stopping.")
            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    val_metrics, val_true, val_probs, val_rows = evaluate(
        model,
        val_loader,
        criterion,
        device,
        cfg,
    )

    test_metrics, test_true, test_probs, test_rows = evaluate(
        model,
        test_loader,
        criterion,
        device,
        cfg,
    )

    final_symmetry = check_pair_symmetry(
        model,
        test_loader,
        device,
    )

    save_predictions(
        working_master,
        val_rows,
        val_true,
        val_probs,
        outdir / "validation_predictions.tsv",
    )

    save_predictions(
        working_master,
        test_rows,
        test_true,
        test_probs,
        outdir / "test_predictions.tsv",
    )

    # Save every numeric object required to recreate figures later.
    save_curve_tables(
        val_true,
        val_probs,
        output_dir=outdir,
        prefix="validation",
    )
    save_curve_tables(
        test_true,
        test_probs,
        output_dir=outdir,
        prefix="test",
    )
    save_confusion_tables(
        val_true,
        val_probs,
        output_dir=outdir,
        prefix="validation",
    )
    save_confusion_tables(
        test_true,
        test_probs,
        output_dir=outdir,
        prefix="test",
    )

    plot_training_history(
        pd.DataFrame(history),
        outdir / "training_history",
    )

    regulon_label = (
        "+ regulons"
        if use_regulons
        else "no regulons"
    )

    plot_roc_curves(
        test_true,
        test_probs,
        title=f"{holdout.replace('_', ' ')} — {regulon_label}",
        base_path=outdir / "roc_curves",
    )

    plot_pr_curves(
        test_true,
        test_probs,
        title=f"{holdout.replace('_', ' ')} — {regulon_label}",
        base_path=outdir / "pr_curves",
    )

    plot_confusion(
        test_true,
        test_probs,
        title=f"{holdout.replace('_', ' ')} — {regulon_label}",
        base_path=outdir / "confusion_matrix",
    )

    result = {
        "run_name": name,
        "holdout": holdout,
        "use_regulons": bool(use_regulons),
        "seed": int(seed),

        "best_epoch": int(best_epoch),
        "best_validation_macro_f1": float(best_macro_f1),

        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "n_test": int(len(test_idx)),

        "n_train_contexts": int(
            working_master.loc[train_idx, "depmap_id_std"].nunique()
        ),
        "n_validation_contexts": int(
            working_master.loc[val_idx, "depmap_id_std"].nunique()
        ),
        "n_test_contexts": int(
            working_master.loc[test_idx, "depmap_id_std"].nunique()
        ),

        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),

        "split_metadata": split_meta,

        "initial_symmetry": initial_symmetry,
        "final_symmetry": final_symmetry,

        "validation": val_metrics,
        "test": test_metrics,

        "config": asdict(cfg),
    }

    with open(final_metrics_path, "w") as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    # Flat one-row metrics file for quick inspection of this run.
    pd.DataFrame([flatten_result(result)]).to_csv(
        outdir / "final_metrics_flat.tsv",
        sep="\t",
        index=False,
    )

    # Save training class weights explicitly outside the checkpoint.
    pd.DataFrame({
        "class_id": [0, 1, 2],
        "class_name": CLASS_NAMES,
        "weight": class_weights.detach().cpu().numpy(),
    }).to_csv(
        outdir / "class_weights.tsv",
        sep="\t",
        index=False,
    )

    print(
        f"\nDONE: {name}\n"
        f"test macro-F1={test_metrics['macro_f1']:.4f} | "
        f"balanced acc={test_metrics['balanced_accuracy']:.4f} | "
        f"macro AUROC={test_metrics['macro_ovr_auroc']:.4f} | "
        f"macro AP={test_metrics['macro_auprc']:.4f}"
    )

    return result


def regenerate_run_outputs_from_predictions(run_dir: Path):
    """Regenerate plots/tables from saved test predictions without retraining."""
    run_dir = Path(run_dir)
    pred_path = run_dir / "test_predictions.tsv"

    if not pred_path.exists():
        return False

    pred = pd.read_csv(pred_path, sep="\t", low_memory=False)
    required = [
        "label_int",
        "p_antagonism",
        "p_no_interaction",
        "p_synergy",
    ]

    if any(c not in pred.columns for c in required):
        print(f"WARNING: missing prediction columns in {pred_path}")
        return False

    y_true = pred["label_int"].to_numpy(dtype=int)
    probs = pred[[
        "p_antagonism",
        "p_no_interaction",
        "p_synergy",
    ]].to_numpy(dtype=float)

    title = run_dir.name
    metrics_path = run_dir / "final_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            result = json.load(f)
        holdout = result.get("holdout", "test")
        regulon_label = "+ regulons" if result.get("use_regulons") else "no regulons"
        title = f"{holdout.replace('_', ' ')} — {regulon_label}"

    plot_roc_curves(y_true, probs, title, run_dir / "roc_curves")
    plot_pr_curves(y_true, probs, title, run_dir / "pr_curves")
    plot_confusion(y_true, probs, title, run_dir / "confusion_matrix")
    save_curve_tables(y_true, probs, run_dir, prefix="test")
    save_confusion_tables(y_true, probs, run_dir, prefix="test")

    history_path = run_dir / "training_history.tsv"
    if history_path.exists():
        history = pd.read_csv(history_path, sep="\t")
        plot_training_history(history, run_dir / "training_history")

    return True


def build_pair_status_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "holdout",
        "seed",
        "has_without_regulons",
        "has_with_regulons",
        "pair_complete",
    ]

    if df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (holdout, seed), group in df.groupby(["holdout", "seed"]):
        values = set(group["use_regulons"].astype(bool))
        rows.append({
            "holdout": holdout,
            "seed": int(seed),
            "has_without_regulons": False in values,
            "has_with_regulons": True in values,
            "pair_complete": values == {False, True},
        })

    return pd.DataFrame(rows, columns=columns).sort_values(["holdout", "seed"])


def load_run_predictions(run_name_value: str) -> pd.DataFrame:
    path = RUN_DIR / run_name_value / "test_predictions.tsv"
    return pd.read_csv(path, sep="\t", low_memory=False)


def plot_paired_roc_pr(df: pd.DataFrame):
    """Create direct ON-vs-OFF ROC and PR comparisons for every complete pair."""
    for (holdout, seed), group in df.groupby(["holdout", "seed"]):
        values = set(group["use_regulons"].astype(bool))
        if values != {False, True}:
            continue

        off_run = group.loc[group["use_regulons"] == False, "run_name"].iloc[0]
        on_run = group.loc[group["use_regulons"] == True, "run_name"].iloc[0]

        off_path = RUN_DIR / off_run / "test_predictions.tsv"
        on_path = RUN_DIR / on_run / "test_predictions.tsv"
        if not off_path.exists() or not on_path.exists():
            continue

        off = load_run_predictions(off_run)
        on = load_run_predictions(on_run)

        # ROC: one class per figure, both models overlaid.
        for class_name, class_id, prob_col in [
            ("antagonism", 0, "p_antagonism"),
            ("no_interaction", 1, "p_no_interaction"),
            ("synergy", 2, "p_synergy"),
        ]:
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            for label, pred in [("Without regulons", off), ("With regulons", on)]:
                binary = (pred["label_int"].to_numpy(dtype=int) == class_id).astype(int)
                score = pred[prob_col].to_numpy(dtype=float)
                fpr, tpr, _ = roc_curve(binary, score)
                auc_value = roc_auc_score(binary, score)
                ax.plot(fpr, tpr, label=f"{label} (AUROC={auc_value:.3f})")
            ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title(f"{class_name}: {holdout.replace('_', ' ')}, seed {seed}")
            ax.legend(frameon=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            save_figure(
                fig,
                SUMMARY_DIR / f"paired_roc__{holdout}__{class_name}__seed{seed}",
            )

            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            for label, pred in [("Without regulons", off), ("With regulons", on)]:
                binary = (pred["label_int"].to_numpy(dtype=int) == class_id).astype(int)
                score = pred[prob_col].to_numpy(dtype=float)
                precision, recall, _ = precision_recall_curve(binary, score)
                ap = average_precision_score(binary, score)
                ax.plot(recall, precision, label=f"{label} (AP={ap:.3f})")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"{class_name}: {holdout.replace('_', ' ')}, seed {seed}")
            ax.legend(frameon=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            save_figure(
                fig,
                SUMMARY_DIR / f"paired_pr__{holdout}__{class_name}__seed{seed}",
            )


# ============================================================
# Summary table / paired plots
# ============================================================

def flatten_result(result: Dict) -> Dict:
    row = {
        "run_name": result["run_name"],
        "holdout": result["holdout"],
        "use_regulons": result["use_regulons"],
        "seed": result["seed"],
        "best_epoch": result["best_epoch"],
        "n_train": result["n_train"],
        "n_validation": result["n_validation"],
        "n_test": result["n_test"],
        "n_train_contexts": result["n_train_contexts"],
        "n_validation_contexts": result["n_validation_contexts"],
        "n_test_contexts": result["n_test_contexts"],
        "total_parameters": result["total_parameters"],
        "trainable_parameters": result["trainable_parameters"],
    }

    for split_name in ["validation", "test"]:
        m = result[split_name]

        for metric in [
            "loss",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "macro_ovr_auroc",
            "macro_auprc",
        ]:
            row[f"{split_name}_{metric}"] = m.get(metric)

        for class_name in CLASS_NAMES:
            class_metrics = m["per_class"][class_name]

            for metric in ["precision", "recall", "f1", "support"]:
                row[
                    f"{split_name}_{class_name}_{metric}"
                ] = class_metrics.get(metric)

            row[
                f"{split_name}_{class_name}_auroc"
            ] = m["per_class_auroc"].get(class_name)

            row[
                f"{split_name}_{class_name}_auprc"
            ] = m["per_class_auprc"].get(class_name)

    return row


def collect_completed_results() -> pd.DataFrame:
    rows = []

    for metrics_path in RUN_DIR.glob(
        "*/final_metrics.json"
    ):
        with open(metrics_path) as f:
            result = json.load(f)

        rows.append(
            flatten_result(result)
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["holdout", "seed", "use_regulons"]
    )


def build_paired_table(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "test_macro_f1",
        "test_balanced_accuracy",
        "test_macro_ovr_auroc",
        "test_macro_auprc",
        "test_antagonism_auprc",
        "test_no_interaction_auprc",
        "test_synergy_auprc",
    ]

    rows = []

    for (holdout, seed), group in df.groupby(
        ["holdout", "seed"]
    ):
        if set(group["use_regulons"].astype(bool)) != {False, True}:
            continue

        off = group.loc[
            group["use_regulons"] == False
        ].iloc[0]

        on = group.loc[
            group["use_regulons"] == True
        ].iloc[0]

        row = {
            "holdout": holdout,
            "seed": seed,
        }

        for metric in metrics:
            row[f"{metric}_without_regulons"] = off[metric]
            row[f"{metric}_with_regulons"] = on[metric]
            row[f"{metric}_delta_regulons"] = (
                on[metric] - off[metric]
            )

        rows.append(row)

    return pd.DataFrame(rows)


def metric_label(metric: str) -> str:
    labels = {
        "test_macro_f1": "Macro-F1",
        "test_balanced_accuracy": "Balanced accuracy",
        "test_macro_ovr_auroc": "Macro AUROC",
        "test_macro_auprc": "Macro AUPRC",
        "test_antagonism_auprc": "Antagonism AUPRC",
        "test_no_interaction_auprc": "No-interaction AUPRC",
        "test_synergy_auprc": "Synergy AUPRC",
    }
    return labels.get(metric, metric)


def plot_paired_metric(
    df: pd.DataFrame,
    metric: str,
    base_path: Path,
):
    """
    Paired seed plot: same split/seed connected between no-regulon and +regulon.
    This is the most direct paper visualization of the regulon effect.
    """

    holdouts = [
        h for h in HOLDOUTS
        if h in set(df["holdout"])
    ]

    if not holdouts:
        return

    fig, axes = plt.subplots(
        1,
        len(holdouts),
        figsize=(4.4 * len(holdouts), 5),
        squeeze=False,
    )

    for ax, holdout in zip(
        axes[0],
        holdouts,
    ):
        sub = df.loc[
            df["holdout"] == holdout
        ]

        seeds = sorted(
            set(sub["seed"])
        )

        for seed in seeds:
            pair = sub.loc[
                sub["seed"] == seed
            ]

            if set(pair["use_regulons"].astype(bool)) != {False, True}:
                continue

            y0 = float(
                pair.loc[
                    pair["use_regulons"] == False,
                    metric,
                ].iloc[0]
            )

            y1 = float(
                pair.loc[
                    pair["use_regulons"] == True,
                    metric,
                ].iloc[0]
            )

            ax.plot(
                [0, 1],
                [y0, y1],
                marker="o",
                alpha=0.65,
            )

        means = (
            sub.groupby("use_regulons")[metric]
            .mean()
            .reindex([False, True])
        )

        if means.notna().all():
            ax.plot(
                [0, 1],
                means.values,
                marker="s",
                linewidth=3,
                label="Mean",
            )

        ax.set_xticks(
            [0, 1],
            labels=["Without\nregulons", "With\nregulons"],
        )
        ax.set_ylabel(
            metric_label(metric)
        )
        ax.set_title(
            holdout.replace("_", " ")
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Effect of regulon input on {metric_label(metric)}"
    )

    save_figure(
        fig,
        base_path,
    )


def plot_per_class_auprc_summary(
    df: pd.DataFrame,
    base_path: Path,
):
    """
    Mean ± SD over seeds for per-class AUPRC.
    """

    rows = []

    for _, r in df.iterrows():
        for class_name in CLASS_NAMES:
            rows.append(
                {
                    "holdout": r["holdout"],
                    "use_regulons": r["use_regulons"],
                    "seed": r["seed"],
                    "class": class_name,
                    "auprc": r[
                        f"test_{class_name}_auprc"
                    ],
                }
            )

    long = pd.DataFrame(rows)

    holdouts = [
        h for h in HOLDOUTS
        if h in set(long["holdout"])
    ]

    if not holdouts:
        return

    fig, axes = plt.subplots(
        1,
        len(holdouts),
        figsize=(5.0 * len(holdouts), 5),
        squeeze=False,
    )

    x = np.arange(
        len(CLASS_NAMES)
    )

    width = 0.34

    for ax, holdout in zip(
        axes[0],
        holdouts,
    ):
        sub = long.loc[
            long["holdout"] == holdout
        ]

        for j, use_regulons in enumerate([False, True]):
            ss = sub.loc[
                sub["use_regulons"] == use_regulons
            ]

            means = (
                ss.groupby("class")["auprc"]
                .mean()
                .reindex(CLASS_NAMES)
            )

            stds = (
                ss.groupby("class")["auprc"]
                .std()
                .reindex(CLASS_NAMES)
                .fillna(0.0)
            )

            offset = (
                -width / 2
                if not use_regulons
                else width / 2
            )

            ax.bar(
                x + offset,
                means.values,
                width=width,
                yerr=stds.values,
                capsize=3,
                label=(
                    "+ regulons"
                    if use_regulons
                    else "without regulons"
                ),
            )

        ax.set_xticks(
            x,
            labels=CLASS_NAMES,
            rotation=25,
            ha="right",
        )
        ax.set_ylabel("AUPRC")
        ax.set_title(
            holdout.replace("_", " ")
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0][0].legend(
        frameon=False
    )

    fig.suptitle(
        "Per-class precision–recall performance"
    )

    save_figure(
        fig,
        base_path,
    )



def run_slug(row) -> str:
    return (
        f"{row['holdout']}"
        f"__regulons-{'on' if bool(row['use_regulons']) else 'off'}"
        f"__seed{int(row['seed'])}"
    )


def export_run_outputs_to_summary(df: pd.DataFrame):
    """
    Copy/regenerate the most useful per-run figures into SUMMARY_DIR with
    unique names. This guarantees that even ONE completed run produces plots
    directly in summary/, instead of plots existing only under runs/<run>/.
    """
    exported_rows = []

    for _, row in df.iterrows():
        run_dir = RUN_DIR / row['run_name']
        slug = run_slug(row)

        # Regenerate first, so stale/missing figures are repaired.
        try:
            regenerate_run_outputs_from_predictions(run_dir)
        except Exception as e:
            print(f"WARNING: could not regenerate {row['run_name']}: {e}")

        for stem in [
            'roc_curves',
            'pr_curves',
            'confusion_matrix',
            'training_history_loss',
            'training_history_metrics',
        ]:
            for ext in ['png', 'pdf']:
                src = run_dir / f'{stem}.{ext}'
                dst = SUMMARY_DIR / f'{stem}__{slug}.{ext}'
                if src.exists() and src.stat().st_size > 0:
                    shutil.copy2(src, dst)
                    exported_rows.append({
                        'run_name': row['run_name'],
                        'artifact_type': stem,
                        'format': ext,
                        'path': str(dst),
                        'size_bytes': int(dst.stat().st_size),
                    })

        # Also copy the numeric paper-source tables.
        for fname in [
            'test_roc_curve_points.tsv',
            'test_pr_curve_points.tsv',
            'test_curve_summary.tsv',
            'test_confusion_matrix_raw.tsv',
            'test_confusion_matrix_normalized.tsv',
            'validation_roc_curve_points.tsv',
            'validation_pr_curve_points.tsv',
            'validation_curve_summary.tsv',
            'validation_confusion_matrix_raw.tsv',
            'validation_confusion_matrix_normalized.tsv',
            'final_metrics_flat.tsv',
            'class_weights.tsv',
            'training_history.tsv',
        ]:
            src = run_dir / fname
            if src.exists() and src.stat().st_size > 0:
                dst = SUMMARY_DIR / f'{src.stem}__{slug}{src.suffix}'
                shutil.copy2(src, dst)
                exported_rows.append({
                    'run_name': row['run_name'],
                    'artifact_type': src.stem,
                    'format': src.suffix.lstrip('.'),
                    'path': str(dst),
                    'size_bytes': int(dst.stat().st_size),
                })

    artifact_df = pd.DataFrame(exported_rows)
    artifact_df.to_csv(
        SUMMARY_DIR / 'artifact_manifest.tsv',
        sep='\t',
        index=False,
    )
    return artifact_df


def make_condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/SD/median across seeds for every holdout × regulon condition."""
    metrics = [
        'test_macro_f1',
        'test_balanced_accuracy',
        'test_macro_ovr_auroc',
        'test_macro_auprc',
        'test_antagonism_f1',
        'test_antagonism_auroc',
        'test_antagonism_auprc',
        'test_no_interaction_f1',
        'test_no_interaction_auroc',
        'test_no_interaction_auprc',
        'test_synergy_f1',
        'test_synergy_auroc',
        'test_synergy_auprc',
    ]
    metrics = [m for m in metrics if m in df.columns]
    rows = []
    for (holdout, use_regulons), g in df.groupby(['holdout', 'use_regulons']):
        row = {
            'holdout': holdout,
            'use_regulons': bool(use_regulons),
            'n_seeds': int(g['seed'].nunique()),
            'seeds': ','.join(str(int(x)) for x in sorted(g['seed'].unique())),
        }
        for m in metrics:
            values = pd.to_numeric(g[m], errors='coerce').dropna()
            row[f'{m}_mean'] = float(values.mean()) if len(values) else np.nan
            row[f'{m}_sd'] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[f'{m}_median'] = float(values.median()) if len(values) else np.nan
            row[f'{m}_min'] = float(values.min()) if len(values) else np.nan
            row[f'{m}_max'] = float(values.max()) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_metric_overview(df: pd.DataFrame, metric: str, base_path: Path):
    """
    Plot every completed run, so summary/ always contains usable figures even
    before ON/OFF pairs or all five seeds are complete.
    """
    if df.empty or metric not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    holdouts = [h for h in HOLDOUTS if h in set(df['holdout'])]
    x_positions = {h: i for i, h in enumerate(holdouts)}

    for _, row in df.iterrows():
        base_x = x_positions[row['holdout']]
        x = base_x + (0.14 if bool(row['use_regulons']) else -0.14)
        ax.scatter(x, float(row[metric]), s=55)
        ax.text(
            x,
            float(row[metric]),
            f" {int(row['seed'])}",
            fontsize=8,
            va='center',
        )

    # Means for every currently available condition.
    means = df.groupby(['holdout', 'use_regulons'])[metric].mean()
    for (holdout, use_regulons), value in means.items():
        x = x_positions[holdout] + (0.14 if bool(use_regulons) else -0.14)
        ax.plot([x - 0.08, x + 0.08], [value, value], linewidth=3)

    ax.set_xticks(range(len(holdouts)), labels=[h.replace('_', '\n') for h in holdouts])
    ax.set_ylabel(metric_label(metric))
    ax.set_title(f"Completed runs: {metric_label(metric)}")
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(
        0.01,
        0.01,
        'Left within each holdout: without regulons; right: with regulons. Numbers are seeds.',
        transform=ax.transAxes,
        fontsize=8,
        va='bottom',
    )
    save_figure(fig, base_path)


def build_paired_statistics(paired: pd.DataFrame) -> pd.DataFrame:
    """
    Paired across-seed statistics for the regulon effect. Wilcoxon is reported
    only when at least two non-zero paired differences exist; with five seeds,
    interpret p-values cautiously and emphasize effect sizes/direction.
    """
    if paired.empty:
        return pd.DataFrame(columns=[
            'holdout', 'metric', 'n_pairs', 'mean_delta', 'sd_delta',
            'median_delta', 'n_positive', 'n_negative', 'n_zero',
            'wilcoxon_statistic', 'wilcoxon_pvalue',
        ])

    metric_prefixes = [
        'test_macro_f1',
        'test_balanced_accuracy',
        'test_macro_ovr_auroc',
        'test_macro_auprc',
        'test_antagonism_auprc',
        'test_no_interaction_auprc',
        'test_synergy_auprc',
    ]

    rows = []
    for holdout, g in paired.groupby('holdout'):
        for metric in metric_prefixes:
            col = f'{metric}_delta_regulons'
            if col not in g.columns:
                continue
            d = pd.to_numeric(g[col], errors='coerce').dropna().to_numpy(dtype=float)
            if not len(d):
                continue
            stat = np.nan
            pval = np.nan
            nonzero = d[np.abs(d) > 1e-15]
            if len(nonzero) >= 2:
                try:
                    w = wilcoxon(nonzero, alternative='two-sided', zero_method='wilcox')
                    stat = float(w.statistic)
                    pval = float(w.pvalue)
                except Exception:
                    pass
            rows.append({
                'holdout': holdout,
                'metric': metric,
                'n_pairs': int(len(d)),
                'mean_delta': float(np.mean(d)),
                'sd_delta': float(np.std(d, ddof=1)) if len(d) > 1 else np.nan,
                'median_delta': float(np.median(d)),
                'n_positive': int(np.sum(d > 0)),
                'n_negative': int(np.sum(d < 0)),
                'n_zero': int(np.sum(d == 0)),
                'wilcoxon_statistic': stat,
                'wilcoxon_pvalue': pval,
            })
    return pd.DataFrame(rows)


def write_summary_readme(df, pair_status, paired, condition_summary, artifact_df):
    complete_pairs = int(pair_status['pair_complete'].sum()) if not pair_status.empty else 0
    lines = [
        'REGULON ABLATION OUTPUT SUMMARY',
        '================================',
        '',
        f'Completed training runs: {len(df)}',
        f'Complete ON/OFF pairs: {complete_pairs}',
        f'Conditions represented: {len(condition_summary)}',
        f'Exported/rebuilt artifacts: {len(artifact_df)}',
        '',
        'Important files:',
        '  all_run_metrics.tsv              - every metric from every completed run',
        '  completed_run_manifest.tsv        - compact run-level summary',
        '  condition_summary.tsv             - mean/SD/median across seeds',
        '  pair_status.tsv                   - which ON/OFF pairs are complete',
        '  paired_regulon_comparison.tsv     - same-seed ON minus OFF differences',
        '  paired_statistics.tsv             - across-seed paired delta statistics',
        '  artifact_manifest.tsv             - exact paths/sizes of plots and numeric plot data',
        '',
        'Per-run plots are copied into this summary directory with unique run tags.',
        'Paired ON-vs-OFF ROC/PR plots appear only when both arms for the same holdout/seed exist.',
        'All ROC/PR coordinates are also stored as TSV so figures can be regenerated without retraining.',
        '',
    ]
    (SUMMARY_DIR / 'README.txt').write_text('\n'.join(lines))

def build_summary_outputs():
    """Rebuild a complete paper-oriented summary from every finished run."""
    df = collect_completed_results()

    if df.empty:
        print('No completed runs found.')
        return

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Rebuild and COPY per-run outputs to summary/ so plots are visible there.
    artifact_df = export_run_outputs_to_summary(df)

    # 2) Full and compact run tables.
    df.to_csv(SUMMARY_DIR / 'all_run_metrics.tsv', sep='\t', index=False)

    manifest_columns = [
        'run_name', 'holdout', 'use_regulons', 'seed', 'best_epoch',
        'n_train', 'n_validation', 'n_test',
        'n_train_contexts', 'n_validation_contexts', 'n_test_contexts',
        'total_parameters', 'trainable_parameters',
        'test_macro_f1', 'test_balanced_accuracy',
        'test_macro_ovr_auroc', 'test_macro_auprc',
        'test_antagonism_f1', 'test_antagonism_auroc', 'test_antagonism_auprc',
        'test_no_interaction_f1', 'test_no_interaction_auroc', 'test_no_interaction_auprc',
        'test_synergy_f1', 'test_synergy_auroc', 'test_synergy_auprc',
    ]
    df[[c for c in manifest_columns if c in df.columns]].to_csv(
        SUMMARY_DIR / 'completed_run_manifest.tsv', sep='\t', index=False
    )

    # 3) Condition-level aggregation across seeds.
    condition_summary = make_condition_summary(df)
    condition_summary.to_csv(
        SUMMARY_DIR / 'condition_summary.tsv', sep='\t', index=False
    )

    # These overview plots are ALWAYS made, even with one completed run.
    for metric, filename in [
        ('test_macro_f1', 'overview_macro_f1'),
        ('test_balanced_accuracy', 'overview_balanced_accuracy'),
        ('test_macro_ovr_auroc', 'overview_macro_auroc'),
        ('test_macro_auprc', 'overview_macro_auprc'),
        ('test_synergy_auprc', 'overview_synergy_auprc'),
        ('test_antagonism_auprc', 'overview_antagonism_auprc'),
    ]:
        plot_metric_overview(df, metric, SUMMARY_DIR / filename)

    # 4) Pair status and ON-vs-OFF deltas.
    pair_status = build_pair_status_table(df)
    pair_status.to_csv(SUMMARY_DIR / 'pair_status.tsv', sep='\t', index=False)

    paired = build_paired_table(df)
    paired_columns = [
        'holdout', 'seed',
        'test_macro_f1_without_regulons', 'test_macro_f1_with_regulons', 'test_macro_f1_delta_regulons',
        'test_balanced_accuracy_without_regulons', 'test_balanced_accuracy_with_regulons', 'test_balanced_accuracy_delta_regulons',
        'test_macro_ovr_auroc_without_regulons', 'test_macro_ovr_auroc_with_regulons', 'test_macro_ovr_auroc_delta_regulons',
        'test_macro_auprc_without_regulons', 'test_macro_auprc_with_regulons', 'test_macro_auprc_delta_regulons',
        'test_antagonism_auprc_without_regulons', 'test_antagonism_auprc_with_regulons', 'test_antagonism_auprc_delta_regulons',
        'test_no_interaction_auprc_without_regulons', 'test_no_interaction_auprc_with_regulons', 'test_no_interaction_auprc_delta_regulons',
        'test_synergy_auprc_without_regulons', 'test_synergy_auprc_with_regulons', 'test_synergy_auprc_delta_regulons',
    ]
    if paired.empty:
        paired = pd.DataFrame(columns=paired_columns)
    else:
        paired = paired.reindex(columns=paired_columns)
    paired.to_csv(
        SUMMARY_DIR / 'paired_regulon_comparison.tsv', sep='\t', index=False
    )

    paired_stats = build_paired_statistics(paired)
    paired_stats.to_csv(
        SUMMARY_DIR / 'paired_statistics.tsv', sep='\t', index=False
    )

    # 5) True paired paper plots only when both arms exist.
    complete_pairs = pair_status.loc[pair_status['pair_complete'] == True]
    if not complete_pairs.empty:
        complete_keys = set(zip(complete_pairs['holdout'], complete_pairs['seed']))
        paired_df = df.loc[
            [(h, s) in complete_keys for h, s in zip(df['holdout'], df['seed'])]
        ].copy()

        for metric, filename in [
            ('test_macro_f1', 'paired_macro_f1'),
            ('test_balanced_accuracy', 'paired_balanced_accuracy'),
            ('test_macro_ovr_auroc', 'paired_macro_auroc'),
            ('test_macro_auprc', 'paired_macro_auprc'),
            ('test_synergy_auprc', 'paired_synergy_auprc'),
            ('test_antagonism_auprc', 'paired_antagonism_auprc'),
        ]:
            plot_paired_metric(paired_df, metric, SUMMARY_DIR / filename)

        plot_per_class_auprc_summary(
            paired_df, SUMMARY_DIR / 'per_class_auprc_summary'
        )
        plot_paired_roc_pr(paired_df)
    else:
        print('\nNo complete regulon ON/OFF pair yet: paired plots are correctly deferred.')

    # 6) Human-readable output index.
    write_summary_readme(df, pair_status, paired, condition_summary, artifact_df)

    print_section('SUMMARY')
    print(df[[
        'holdout', 'use_regulons', 'seed', 'test_macro_f1',
        'test_balanced_accuracy', 'test_macro_ovr_auroc',
        'test_macro_auprc', 'test_synergy_auprc', 'test_antagonism_auprc'
    ]].to_string(index=False))
    print('\nPair completeness:')
    print(pair_status.to_string(index=False))
    print(f'\nSummary plots/tables are in: {SUMMARY_DIR}')
    print(f'Artifact manifest: {SUMMARY_DIR / "artifact_manifest.tsv"}')


# ============================================================
# CLI
# ============================================================

def parse_bool(x: str) -> bool:
    x = x.lower().strip()

    if x in {"true", "1", "yes", "y", "on"}:
        return True

    if x in {"false", "0", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(
        f"Expected true/false, got {x}"
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run both regulon conditions for all three holdout schemes.",
    )

    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Rebuild every per-run and summary plot/table from saved predictions; no retraining.",
    )

    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Delete the selected run directory before training it again.",
    )

    parser.add_argument(
        "--summary-at-end-only",
        action="store_true",
        help="During --run-all, rebuild summary only after all requested runs finish.",
    )

    parser.add_argument(
        "--holdout",
        choices=HOLDOUTS,
        default="unseen_cell_line",
    )

    parser.add_argument(
        "--use-regulons",
        type=parse_bool,
        default=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
    )

    parser.add_argument(
        "--drug-grouping",
        choices=["scaffold", "structure", "drug_id"],
        default="scaffold",
    )

    parser.add_argument(
        "--drug-test-mode",
        choices=["any", "both"],
        default="any",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--split-search-trials",
        type=int,
        default=50,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.plot_only:
        build_summary_outputs()
        return

    cfg = Config(
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        split_search_trials=args.split_search_trials,
        drug_grouping=args.drug_grouping,
        drug_test_mode=args.drug_test_mode,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print_section("CONFIGURATION")
    print(
        json.dumps(
            asdict(cfg),
            indent=2,
        )
    )
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    dataset_module = load_dataset_module()

    master = load_master(
        dataset_module
    )

    print(
        "Model-ready observations:",
        f"{len(master):,}",
    )
    print(
        "Unique contexts:",
        master["depmap_id_std"].nunique(),
    )
    print(
        "Unique drugs:",
        len(
            set(master["drug_a_id_std"].astype(str))
            | set(master["drug_b_id_std"].astype(str))
        ),
    )

    need_drug_groups = (
        args.run_all
        or args.holdout in {"unseen_drug", "unseen_both"}
    )

    drug_group_map = None

    if need_drug_groups:
        drug_group_map = load_drug_groups(
            master,
            dataset_module,
            cfg.drug_grouping,
        )

    if args.run_all:
        experiments = []

        for seed in args.seeds:
            for holdout in HOLDOUTS:
                for use_regulons in [False, True]:
                    experiments.append(
                        (holdout, use_regulons, seed)
                    )
    else:
        experiments = [
            (
                args.holdout,
                args.use_regulons,
                args.seed,
            )
        ]

    print_section(
        f"EXPERIMENT PLAN | {len(experiments)} run(s)"
    )

    for holdout, use_regulons, seed in experiments:
        print(
            f"{holdout:<20} "
            f"regulons={'ON' if use_regulons else 'OFF':<3} "
            f"seed={seed}"
        )

    # Persist the requested experiment plan before training starts.
    pd.DataFrame([
        {
            "holdout": h,
            "use_regulons": bool(r),
            "seed": int(s),
            "drug_grouping": cfg.drug_grouping,
            "drug_test_mode": cfg.drug_test_mode,
        }
        for h, r, s in experiments
    ]).to_csv(
        OUTPUT_ROOT / "experiment_plan.tsv",
        sep="\t",
        index=False,
    )

    for holdout, use_regulons, seed in experiments:
        run_cfg = copy.deepcopy(cfg)
        run_cfg.seed = seed

        if args.force_rerun:
            selected_name = run_name(holdout, use_regulons, seed, run_cfg)
            selected_dir = RUN_DIR / selected_name
            if selected_dir.exists():
                print(f"FORCE RERUN: deleting {selected_dir}")
                shutil.rmtree(selected_dir)

        if holdout in {"unseen_drug", "unseen_both"} and drug_group_map is None:
            drug_group_map = load_drug_groups(
                master,
                dataset_module,
                run_cfg.drug_grouping,
            )

        run_experiment(
            holdout=holdout,
            use_regulons=use_regulons,
            seed=seed,
            cfg=run_cfg,
            master=master,
            dataset_module=dataset_module,
            drug_group_map=drug_group_map,
            device=device,
        )

        # Update paper summaries after each run unless explicitly deferred.
        if not args.summary_at_end_only:
            build_summary_outputs()

    # Always build a final complete summary.
    build_summary_outputs()

    print_section("ALL REQUESTED RUNS COMPLETE")
    print("Results:", OUTPUT_ROOT)
    print("Summary:", SUMMARY_DIR)


if __name__ == "__main__":
    main()
