#!/usr/bin/env python3
"""
DepMap regulon activity inference using CollecTRI + decoupler ULM.

Two regulon representations are produced:

1. Fixed CollecTRI
   - curated TF -> target network
   - signed CollecTRI weights
   - TF activity inferred with ULM

2. Contextualized CollecTRI
   - CollecTRI network
   - TF-target support estimated across DepMap using Spearman rho
   - self-edges excluded from correlation contextualization
   - weak edges removed
   - final edge weight = CollecTRI sign * abs(rho)
   - TF activity inferred with the SAME ULM method

Important:
- DepMap expression is already TPMLogp1 and is NOT log-transformed again.
- ULM uses TPMLogp1 directly.
- Spearman contextualization uses ranked expression.
- During final cross-validation, correlation contextualization must be fitted
  on training cell lines only and then frozen for validation/test.
"""

import argparse
import json
import warnings
from pathlib import Path

import anndata as ad
import decoupler as dc
import numpy as np
import pandas as pd


# ===============================================================
# CollecTRI
# ===============================================================

def load_collectri():
    """
    Load human CollecTRI and normalize network columns to:

        source
        target
        prior_weight
    """

    print("\n========================================")
    print("Loading human CollecTRI")
    print("========================================")

    net = dc.op.collectri(
        organism="human"
    ).copy()

    print("\nCollecTRI columns:")
    print(net.columns.tolist())
    print(net.head())

    if "source" not in net.columns:
        raise ValueError(
            "CollecTRI does not contain 'source'."
        )

    if "target" not in net.columns:
        raise ValueError(
            "CollecTRI does not contain 'target'."
        )

    weight_candidates = [
        "weight",
        "mor",
        "likelihood",
        "score",
        "interaction_weight",
    ]

    weight_col = None

    for col in weight_candidates:
        if col in net.columns:
            weight_col = col
            break

    if weight_col is None:

        warnings.warn(
            "No CollecTRI weight column detected. "
            "Using +1 for all interactions."
        )

        net["prior_weight"] = 1.0

    else:

        print(
            f"Using CollecTRI weight column: "
            f"{weight_col}"
        )

        net["prior_weight"] = pd.to_numeric(
            net[weight_col],
            errors="coerce",
        )

    net = (
        net[
            [
                "source",
                "target",
                "prior_weight",
            ]
        ]
        .dropna()
        .drop_duplicates(
            ["source", "target"]
        )
        .reset_index(drop=True)
    )

    net = net[
        net["prior_weight"] != 0
    ].copy()

    print(
        f"\nCollecTRI edges: "
        f"{len(net):,}"
    )

    print(
        f"CollecTRI TFs: "
        f"{net['source'].nunique():,}"
    )

    print(
        f"CollecTRI targets: "
        f"{net['target'].nunique():,}"
    )

    return net


# ===============================================================
# DepMap loading
# ===============================================================

METADATA_COLUMNS = {
    "Unnamed: 0",
    "SequencingID",
    "ModelConditionID",
    "ModelID",
    "IsDefaultEntryForMC",
    "IsDefaultEntryForModel",
}


def detect_gene_mapping(columns):
    """
    Convert DepMap column names such as:

        TP53 (7157)

    into:

        TP53 -> original column name
    """

    mapping = {}

    for col in columns:

        col = str(col)

        if " (" in col and col.endswith(")"):
            symbol = col.split(" (")[0]

        else:
            symbol = col

        if symbol not in mapping:
            mapping[symbol] = col

    return mapping


def load_depmap_expression(
    expr_path,
    needed_genes,
):
    """
    Read only genes needed for CollecTRI.

    Keeps exactly one canonical profile per ModelID:

        IsDefaultEntryForModel == Yes
    """

    print("\n========================================")
    print("Loading DepMap expression")
    print("========================================")

    print("\nReading CSV header...")

    header = pd.read_csv(
        expr_path,
        nrows=0,
    ).columns.tolist()

    gene_columns = [
        c
        for c in header
        if c not in METADATA_COLUMNS
    ]

    mapping = detect_gene_mapping(
        gene_columns
    )

    genes_present = [
        gene
        for gene in needed_genes
        if gene in mapping
    ]

    original_gene_columns = [
        mapping[gene]
        for gene in genes_present
    ]

    print(
        f"CollecTRI genes requested: "
        f"{len(needed_genes)}"
    )

    print(
        f"CollecTRI genes present:   "
        f"{len(genes_present)}"
    )

    print("\nReading selected expression columns...")

    expr = pd.read_csv(
        expr_path,
        usecols=[
            "ModelID",
            "IsDefaultEntryForModel",
        ]
        + original_gene_columns,
    )

    # -----------------------------------------------------------
    # One canonical expression profile per DepMap model
    # -----------------------------------------------------------

    expr = expr[
        expr["IsDefaultEntryForModel"] == "Yes"
    ].copy()

    expr = expr.drop(
        columns="IsDefaultEntryForModel"
    )

    expr = expr.set_index(
        "ModelID"
    )

    reverse_mapping = {
        mapping[gene]: gene
        for gene in genes_present
    }

    expr = expr.rename(
        columns=reverse_mapping
    )

    # Defensive duplicate removal
    expr = expr.loc[
        :,
        ~expr.columns.duplicated()
    ].copy()

    # Ensure numeric matrix
    expr = expr.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Missing expression values should be rare.
    # ULM cannot use NaN values.
    n_missing = int(
        expr.isna().sum().sum()
    )

    if n_missing > 0:

        warnings.warn(
            f"{n_missing:,} missing expression values "
            "found. Filling with gene medians."
        )

        expr = expr.fillna(
            expr.median(axis=0)
        )

    print(
        f"Loaded expression matrix: "
        f"{expr.shape}"
    )

    print(
        "Unique ModelIDs:",
        expr.index.nunique(),
    )

    print(
        "Duplicated ModelIDs:",
        expr.index.duplicated().sum(),
    )

    print(
        "Example IDs:",
        expr.index[:5].tolist(),
    )

    return expr


# ===============================================================
# Expression matrix for correlation calculation
# ===============================================================

def prepare_correlation_matrix(
    expr,
    method="spearman",
):
    """
    Prepare standardized expression only for TF-target
    correlation estimation.

    For Spearman:
      expression -> ranks across cell lines -> z-score ranks.

    The resulting matrix is NOT used as ULM input.
    """

    if method == "spearman":

        ranked = expr.rank(
            axis=0,
            method="average",
        )

        mean = ranked.mean(
            axis=0
        )

        std = (
            ranked
            .std(
                axis=0,
                ddof=0,
            )
            .replace(
                0,
                np.nan,
            )
        )

        z_corr = (
            (ranked - mean)
            / std
        )

    elif method == "pearson":

        mean = expr.mean(
            axis=0
        )

        std = (
            expr
            .std(
                axis=0,
                ddof=0,
            )
            .replace(
                0,
                np.nan,
            )
        )

        z_corr = (
            (expr - mean)
            / std
        )

    else:

        raise ValueError(
            "Correlation method must be "
            "'spearman' or 'pearson'."
        )

    z_corr = (
        z_corr
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    return z_corr


# ===============================================================
# ULM scoring
# ===============================================================

def score_tf_activity_ulm(
    expr,
    network,
    weight_col,
    min_targets=5,
    verbose=True,
):
    """
    Infer TF activity using decoupler ULM.

    Parameters
    ----------
    expr
        samples x genes DataFrame.
        Here this is DepMap TPMLogp1.

    network
        Long-format regulatory network.

    weight_col
        Network column containing signed edge weights.

    Returns
    -------
    scores
        ModelID x TF ULM t-values.

    padj
        ModelID x TF BH-adjusted p-values.
    """

    print(
        f"\nRunning ULM using network weight: "
        f"{weight_col}"
    )

    # -----------------------------------------------------------
    # ULM expects:
    #
    # source | target | weight
    # -----------------------------------------------------------

    ulm_net = (
        network[
            [
                "source",
                "target",
                weight_col,
            ]
        ]
        .rename(
            columns={
                weight_col: "weight"
            }
        )
        .copy()
    )

    # Only target genes need to be measured for ULM.
    ulm_net = ulm_net[
        ulm_net["target"].isin(
            expr.columns
        )
    ].copy()

    # Remove TFs with too few measurable targets before ULM.
    target_counts = (
        ulm_net
        .groupby("source")["target"]
        .nunique()
    )

    keep_tfs = target_counts[
        target_counts >= min_targets
    ].index

    ulm_net = ulm_net[
        ulm_net["source"].isin(
            keep_tfs
        )
    ].copy()

    print(
        f"ULM network edges: "
        f"{len(ulm_net):,}"
    )

    print(
        f"ULM network TFs: "
        f"{ulm_net['source'].nunique():,}"
    )

    # -----------------------------------------------------------
    # AnnData gives us a robust interface to current decoupler
    # and keeps score names / TF names attached.
    # -----------------------------------------------------------

    adata = ad.AnnData(
        X=expr.to_numpy(
            dtype=np.float32
        )
    )

    adata.obs_names = (
        expr.index
        .astype(str)
        .tolist()
    )

    adata.var_names = (
        expr.columns
        .astype(str)
        .tolist()
    )

    # -----------------------------------------------------------
    # ULM
    #
    # Default tval=True:
    # activity score = t statistic of the ULM slope.
    # -----------------------------------------------------------

    dc.mt.ulm(
        data=adata,
        net=ulm_net,
        tmin=min_targets,
        verbose=verbose,
        tval=True,
    )

    # -----------------------------------------------------------
    # Extract score_ulm
    # -----------------------------------------------------------

    score_adata = dc.pp.get_obsm(
        adata=adata,
        key="score_ulm",
    )

    scores = pd.DataFrame(
        score_adata.X,
        index=score_adata.obs_names,
        columns=score_adata.var_names,
    )

    scores.index.name = "ModelID"

    # -----------------------------------------------------------
    # Extract adjusted p-values
    # -----------------------------------------------------------

    padj_adata = dc.pp.get_obsm(
        adata=adata,
        key="padj_ulm",
    )

    padj = pd.DataFrame(
        padj_adata.X,
        index=padj_adata.obs_names,
        columns=padj_adata.var_names,
    )

    padj.index.name = "ModelID"

    print(
        "ULM activity matrix:",
        scores.shape,
    )

    return scores, padj, ulm_net


# ===============================================================
# Correlation contextualization
# ===============================================================

def contextualize_collectri(
    expr,
    prior,
    method="spearman",
    min_abs_rho=0.10,
    min_targets=5,
):
    """
    Contextualize CollecTRI using TF-target co-expression
    across the DepMap panel.

    CollecTRI determines regulatory direction.

    abs(rho) determines data-derived support:

        final_weight =
            prior_weight * abs(rho)

    NOTE
    ----
    Correlation direction itself is intentionally NOT used as
    activation/repression direction.

    TF -> same TF self-edges are excluded because
    rho(TF, TF) == 1 trivially.
    """

    print("\n========================================")
    print("Contextualizing CollecTRI")
    print("========================================")

    print(
        f"Correlation method: {method}"
    )

    print(
        f"Minimum |rho|: {min_abs_rho}"
    )

    z_corr = prepare_correlation_matrix(
        expr,
        method=method,
    )

    genes = set(
        z_corr.columns
    )

    # Correlation requires expression for both source and target.
    net = prior[
        prior["source"].isin(genes)
        & prior["target"].isin(genes)
    ].copy()

    n_before_self = len(net)

    # -----------------------------------------------------------
    # Remove trivial self-correlation edges
    # -----------------------------------------------------------

    net = net[
        net["source"]
        != net["target"]
    ].copy()

    n_self_edges = (
        n_before_self
        - len(net)
    )

    print(
        f"Removed TF self-edges: "
        f"{n_self_edges:,}"
    )

    support_rows = []

    print(
        "\nCalculating TF-target "
        f"{method} support..."
    )

    for tf, reg in net.groupby(
        "source"
    ):

        tf_vec = (
            z_corr[tf]
            .to_numpy(
                dtype=np.float64
            )
        )

        targets = (
            reg["target"]
            .tolist()
        )

        target_matrix = (
            z_corr[targets]
            .to_numpy(
                dtype=np.float64
            )
        )

        # Because columns are standardized,
        # mean product = Pearson correlation.
        # For ranked data, this is Spearman rho.
        rho = np.mean(
            target_matrix
            * tf_vec[:, None],
            axis=0,
        )

        tmp = reg[
            [
                "source",
                "target",
                "prior_weight",
            ]
        ].copy()

        tmp["rho"] = rho
        tmp["abs_rho"] = np.abs(
            rho
        )

        support_rows.append(
            tmp
        )

    all_edges = pd.concat(
        support_rows,
        ignore_index=True,
    )

    # -----------------------------------------------------------
    # Edge-level threshold
    # -----------------------------------------------------------

    supported = all_edges[
        all_edges["abs_rho"]
        >= min_abs_rho
    ].copy()

    n_edges_threshold = len(
        supported
    )

    # -----------------------------------------------------------
    # Regulon-level minimum target requirement
    # -----------------------------------------------------------

    target_counts = (
        supported
        .groupby("source")["target"]
        .nunique()
    )

    keep_tfs = target_counts[
        target_counts >= min_targets
    ].index

    supported = supported[
        supported["source"].isin(
            keep_tfs
        )
    ].copy()

    # -----------------------------------------------------------
    # Final signed contextualized weight
    # -----------------------------------------------------------

    supported["final_weight"] = (
        supported["prior_weight"]
        * supported["abs_rho"]
    )

    supported = (
        supported
        .sort_values(
            [
                "source",
                "abs_rho",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"\nEvaluable non-self prior edges: "
        f"{len(all_edges):,}"
    )

    print(
        f"Edges with |rho| >= "
        f"{min_abs_rho}: "
        f"{n_edges_threshold:,}"
    )

    print(
        f"Edges after TF minimum-target filter: "
        f"{len(supported):,}"
    )

    print(
        f"Contextualized TFs: "
        f"{supported['source'].nunique():,}"
    )

    return (
        supported,
        all_edges,
        n_self_edges,
    )


# ===============================================================
# QC
# ===============================================================

def make_fixed_qc(
    network,
):
    """
    Per-TF QC for fixed CollecTRI.
    """

    qc = (
        network
        .groupby("source")
        .agg(
            n_targets=(
                "target",
                "nunique",
            ),
            mean_abs_prior_weight=(
                "prior_weight",
                lambda x:
                np.mean(
                    np.abs(x)
                ),
            ),
        )
        .sort_values(
            "n_targets",
            ascending=False,
        )
    )

    return qc


def make_correlation_qc(
    fixed_network,
    all_edges,
    supported,
):
    """
    Per-TF QC for contextualized CollecTRI.
    """

    prior_counts = (
        fixed_network
        .groupby("source")[
            "target"
        ]
        .nunique()
        .rename(
            "n_prior_targets"
        )
    )

    evaluated_counts = (
        all_edges
        .groupby("source")[
            "target"
        ]
        .nunique()
        .rename(
            "n_evaluated_nonself_targets"
        )
    )

    supported_counts = (
        supported
        .groupby("source")[
            "target"
        ]
        .nunique()
        .rename(
            "n_supported_targets"
        )
    )

    rho_summary = (
        all_edges
        .groupby("source")
        .agg(
            mean_abs_rho=(
                "abs_rho",
                "mean",
            ),
            median_abs_rho=(
                "abs_rho",
                "median",
            ),
            max_abs_rho=(
                "abs_rho",
                "max",
            ),
        )
    )

    qc = pd.concat(
        [
            prior_counts,
            evaluated_counts,
            supported_counts,
            rho_summary,
        ],
        axis=1,
    )

    qc[
        "n_evaluated_nonself_targets"
    ] = (
        qc[
            "n_evaluated_nonself_targets"
        ]
        .fillna(0)
        .astype(int)
    )

    qc[
        "n_supported_targets"
    ] = (
        qc[
            "n_supported_targets"
        ]
        .fillna(0)
        .astype(int)
    )

    qc[
        "fraction_supported"
    ] = (
        qc[
            "n_supported_targets"
        ]
        / qc[
            "n_evaluated_nonself_targets"
        ].replace(
            0,
            np.nan,
        )
    )

    qc = qc.sort_values(
        [
            "n_supported_targets",
            "median_abs_rho",
        ],
        ascending=[
            False,
            False,
        ],
    )

    return qc


# ===============================================================
# Main
# ===============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--expr",
        default=(
            "/data/analysis/combinations/input/"
            "cell_lines/depmap/"
            "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
        ),
        help=(
            "DepMap protein-coding TPMLogp1 CSV"
        ),
    )

    parser.add_argument(
        "--outdir",
        default="./regulon_results_ulm",
    )

    parser.add_argument(
        "--min-abs-rho",
        "--min_abs_rho",
        dest="min_abs_rho",
        type=float,
        default=0.10,
        help=(
            "Minimum absolute TF-target correlation "
            "for contextualized network."
        ),
    )

    parser.add_argument(
        "--min-targets",
        "--min_targets",
        dest="min_targets",
        type=int,
        default=5,
        help=(
            "Minimum number of measurable/supported "
            "targets per TF."
        ),
    )

    parser.add_argument(
        "--correlation-method",
        choices=[
            "spearman",
            "pearson",
        ],
        default="spearman",
    )

    parser.add_argument(
        "--quiet-ulm",
        action="store_true",
        help=(
            "Disable decoupler ULM progress output."
        ),
    )

    args = parser.parse_args()

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ===========================================================
    # 1. CollecTRI
    # ===========================================================

    prior = load_collectri()

    prior.to_csv(
        outdir
        / "collectri_prior.tsv",
        sep="\t",
        index=False,
    )

    # ===========================================================
    # 2. DepMap expression
    # ===========================================================

    needed_genes = sorted(
        set(prior["source"])
        | set(prior["target"])
    )

    expr = load_depmap_expression(
        args.expr,
        needed_genes,
    )

    genes = set(
        expr.columns
    )

    print("\n========================================")
    print("Expression / CollecTRI coverage")
    print("========================================")

    available_tfs = (
        set(prior["source"])
        & genes
    )

    available_targets = (
        set(prior["target"])
        & genes
    )

    print(
        f"Expression matrix: "
        f"{expr.shape[0]} cell lines x "
        f"{expr.shape[1]} genes"
    )

    print(
        f"TF genes measured: "
        f"{len(available_tfs)} / "
        f"{prior['source'].nunique()}"
    )

    print(
        f"Target genes measured: "
        f"{len(available_targets)} / "
        f"{prior['target'].nunique()}"
    )

    # ===========================================================
    # 3. Fixed CollecTRI + ULM
    # ===========================================================

    print("\n========================================")
    print("A. Fixed CollecTRI + ULM")
    print("========================================")

    # Important:
    # ULM only needs target genes to be measured.
    # TF/source expression is NOT required to infer its activity.
    fixed_network = prior[
        prior["target"].isin(
            expr.columns
        )
    ].copy()

    fixed_activity, fixed_padj, fixed_ulm_network = (
        score_tf_activity_ulm(
            expr=expr,
            network=fixed_network,
            weight_col="prior_weight",
            min_targets=args.min_targets,
            verbose=not args.quiet_ulm,
        )
    )

    fixed_activity.to_csv(
        outdir
        / "TF_activity_fixed_collectri_ULM.csv"
    )

    fixed_padj.to_csv(
        outdir
        / "TF_activity_fixed_collectri_ULM_padj.csv"
    )

    fixed_ulm_network.to_csv(
        outdir
        / "network_fixed_collectri_ULM.tsv",
        sep="\t",
        index=False,
    )

    make_fixed_qc(
        fixed_network
    ).to_csv(
        outdir
        / "QC_fixed_collectri.tsv",
        sep="\t",
    )

    # ===========================================================
    # 4. Contextualize CollecTRI
    # ===========================================================

    print("\n========================================")
    print("B. Correlation-contextualized CollecTRI")
    print("========================================")

    (
        contextual_network,
        all_correlation_edges,
        n_self_edges,
    ) = contextualize_collectri(
        expr=expr,
        prior=prior,
        method=args.correlation_method,
        min_abs_rho=args.min_abs_rho,
        min_targets=args.min_targets,
    )

    all_correlation_edges.to_csv(
        outdir
        / "network_collectri_correlation_all_edges.tsv",
        sep="\t",
        index=False,
    )

    contextual_network.to_csv(
        outdir
        / "network_collectri_correlation_supported.tsv",
        sep="\t",
        index=False,
    )

    # ===========================================================
    # 5. Contextualized CollecTRI + same ULM
    # ===========================================================

    print("\n========================================")
    print("C. Contextualized CollecTRI + ULM")
    print("========================================")

    (
        contextual_activity,
        contextual_padj,
        contextual_ulm_network,
    ) = score_tf_activity_ulm(
        expr=expr,
        network=contextual_network,
        weight_col="final_weight",
        min_targets=args.min_targets,
        verbose=not args.quiet_ulm,
    )

    contextual_activity.to_csv(
        outdir
        / "TF_activity_contextualized_collectri_ULM.csv"
    )

    contextual_padj.to_csv(
        outdir
        / "TF_activity_contextualized_collectri_ULM_padj.csv"
    )

    contextual_ulm_network.to_csv(
        outdir
        / "network_contextualized_collectri_ULM.tsv",
        sep="\t",
        index=False,
    )

    # ===========================================================
    # 6. QC
    # ===========================================================

    corr_qc = make_correlation_qc(
        fixed_network=fixed_network,
        all_edges=all_correlation_edges,
        supported=contextual_network,
    )

    corr_qc.to_csv(
        outdir
        / "QC_collectri_correlation.tsv",
        sep="\t",
    )

    rho_summary = (
        all_correlation_edges[
            "abs_rho"
        ]
        .describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
    )

    rho_summary.to_csv(
        outdir
        / "rho_distribution.tsv",
        sep="\t",
    )

    # ===========================================================
    # 7. Compare TF feature spaces
    # ===========================================================

    fixed_tfs = set(
        fixed_activity.columns
    )

    contextual_tfs = set(
        contextual_activity.columns
    )

    shared_tfs = (
        fixed_tfs
        & contextual_tfs
    )

    feature_comparison = pd.DataFrame(
        {
            "TF": sorted(
                fixed_tfs
                | contextual_tfs
            )
        }
    )

    feature_comparison[
        "in_fixed_ULM"
    ] = (
        feature_comparison[
            "TF"
        ].isin(
            fixed_tfs
        )
    )

    feature_comparison[
        "in_contextualized_ULM"
    ] = (
        feature_comparison[
            "TF"
        ].isin(
            contextual_tfs
        )
    )

    feature_comparison.to_csv(
        outdir
        / "TF_feature_space_comparison.tsv",
        sep="\t",
        index=False,
    )

    # ===========================================================
    # 8. Summary
    # ===========================================================

    supported_fraction = (
        len(contextual_network)
        / len(all_correlation_edges)
        if len(all_correlation_edges) > 0
        else np.nan
    )

    summary = {
        "n_cell_lines": int(
            expr.shape[0]
        ),
        "n_expression_genes": int(
            expr.shape[1]
        ),
        "collectri_edges_total": int(
            len(prior)
        ),
        "collectri_tfs_total": int(
            prior["source"].nunique()
        ),
        "collectri_targets_total": int(
            prior["target"].nunique()
        ),
        "fixed_ulm_edges": int(
            len(fixed_ulm_network)
        ),
        "fixed_ulm_tfs": int(
            fixed_activity.shape[1]
        ),
        "correlation_method":
            args.correlation_method,
        "min_abs_rho": float(
            args.min_abs_rho
        ),
        "min_targets": int(
            args.min_targets
        ),
        "self_edges_removed_from_contextualization":
            int(n_self_edges),
        "correlation_edges_evaluated": int(
            len(all_correlation_edges)
        ),
        "contextualized_edges": int(
            len(contextual_network)
        ),
        "contextualized_edge_fraction": float(
            supported_fraction
        ),
        "contextualized_ulm_tfs": int(
            contextual_activity.shape[1]
        ),
        "shared_fixed_contextualized_tfs": int(
            len(shared_tfs)
        ),
        "activity_method": "decoupler ULM",
        "ulm_statistic": "t-value",
        "ulm_input":
            "DepMap TPMLogp1 expression",
    }

    with open(
        outdir / "run_summary.json",
        "w",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
        )

    with open(
        outdir / "summary.txt",
        "w",
    ) as handle:

        handle.write(
            "DepMap CollecTRI + ULM summary\n"
        )

        handle.write(
            "=" * 55
            + "\n\n"
        )

        for key, value in summary.items():

            handle.write(
                f"{key}: {value}\n"
            )

    # ===========================================================
    # Done
    # ===========================================================

    print("\n========================================")
    print("DONE")
    print("========================================")

    print(
        f"\nFixed ULM activity: "
        f"{fixed_activity.shape}"
    )

    print(
        f"Contextualized ULM activity: "
        f"{contextual_activity.shape}"
    )

    print(
        f"Shared TFs: "
        f"{len(shared_tfs)}"
    )

    print(
        f"\nResults written to:\n"
        f"{outdir.resolve()}"
    )


if __name__ == "__main__":
    main()
