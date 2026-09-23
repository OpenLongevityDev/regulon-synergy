#!/usr/bin/env python3

"""
DepMap Hallmark pathway activity inference using decoupler ULM.

Biological-context pathway branch:

    DepMap TPMLogp1 expression
              |
              v
    fixed human MSigDB Hallmark gene sets
              |
              v
        decoupler ULM
              |
              v
    ModelID x Hallmark activity matrix


Important
---------
1. The SAME Hallmark definitions are used for all cell lines and,
   later, all primary cell types / pseudobulk samples.

2. Hallmark pathways are NOT contextualized using DepMap.
   This intentionally keeps the pathway coordinate system fixed
   across biological domains.

3. DepMap expression is already TPMLogp1.
   No additional log transformation is performed.

4. ULM uses TPMLogp1 directly.

5. Hallmark gene sets are unsigned gene sets.
   Their pathway memberships are therefore represented with
   weight = +1.

6. The exact Hallmark reference retrieved during this run is saved
   locally so that the final model can reuse the identical reference.
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
# DepMap metadata columns
# ===============================================================

METADATA_COLUMNS = {
    "Unnamed: 0",
    "SequencingID",
    "ModelConditionID",
    "ModelID",
    "IsDefaultEntryForMC",
    "IsDefaultEntryForModel",
}


# ===============================================================
# Hallmark reference
# ===============================================================

def load_hallmark():
    """
    Load fixed human MSigDB Hallmark gene sets through decoupler.

    Expected columns:

        source = Hallmark pathway
        target = gene

    We explicitly add:

        weight = 1.0

    because Hallmark gene sets are unsigned memberships.
    """

    print("\n========================================")
    print("Loading human Hallmark gene sets")
    print("========================================")

    hallmark = dc.op.hallmark(
        organism="human"
    ).copy()

    print("\nHallmark columns:")
    print(
        hallmark.columns.tolist()
    )

    print("\nFirst rows:")
    print(
        hallmark.head()
    )

    if "source" not in hallmark.columns:
        raise ValueError(
            "Hallmark resource does not contain 'source'."
        )

    if "target" not in hallmark.columns:
        raise ValueError(
            "Hallmark resource does not contain 'target'."
        )

    # -----------------------------------------------------------
    # Keep only the shared coordinate definition:
    #
    # pathway -> gene
    #
    # Hallmarks are treated as unsigned sets.
    # -----------------------------------------------------------

    hallmark = (
        hallmark[
            [
                "source",
                "target",
            ]
        ]
        .dropna()
        .drop_duplicates(
            [
                "source",
                "target",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    hallmark["weight"] = 1.0

    print(
        f"\nHallmark memberships: "
        f"{len(hallmark):,}"
    )

    print(
        f"Hallmark pathways: "
        f"{hallmark['source'].nunique():,}"
    )

    print(
        f"Unique Hallmark genes: "
        f"{hallmark['target'].nunique():,}"
    )

    pathways = (
        hallmark["source"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    print("\nPathways:")
    for pathway in pathways:
        print(
            "  ",
            pathway,
        )

    return hallmark


# ===============================================================
# DepMap gene name handling
# ===============================================================

def detect_gene_mapping(columns):
    """
    Convert DepMap columns such as:

        TP53 (7157)

    into:

        TP53 -> original column name
    """

    mapping = {}

    for col in columns:

        col = str(
            col
        )

        if (
            " (" in col
            and col.endswith(")")
        ):

            symbol = col.split(
                " ("
            )[0]

        else:

            symbol = col

        # Defensive handling if repeated symbols occur.
        if symbol not in mapping:
            mapping[
                symbol
            ] = col

    return mapping


# ===============================================================
# DepMap expression loading
# ===============================================================

def load_depmap_expression(
    expr_path,
    needed_genes,
):
    """
    Load only genes required by the Hallmark reference.

    Keeps one canonical DepMap profile per model:

        IsDefaultEntryForModel == "Yes"

    Returns
    -------
    expr
        ModelID x gene TPMLogp1 matrix.
    """

    print("\n========================================")
    print("Loading DepMap expression")
    print("========================================")

    print(
        "\nReading CSV header..."
    )

    header = pd.read_csv(
        expr_path,
        nrows=0,
    ).columns.tolist()

    gene_columns = [
        col
        for col in header
        if col not in METADATA_COLUMNS
    ]

    mapping = detect_gene_mapping(
        gene_columns
    )

    genes_present = [
        gene
        for gene in needed_genes
        if gene in mapping
    ]

    genes_missing = [
        gene
        for gene in needed_genes
        if gene not in mapping
    ]

    original_gene_columns = [
        mapping[
            gene
        ]
        for gene in genes_present
    ]

    print(
        f"Hallmark genes requested: "
        f"{len(needed_genes):,}"
    )

    print(
        f"Hallmark genes present:   "
        f"{len(genes_present):,}"
    )

    print(
        f"Hallmark genes missing:   "
        f"{len(genes_missing):,}"
    )

    print(
        "\nReading selected expression columns..."
    )

    expr = pd.read_csv(
        expr_path,
        usecols=[
            "ModelID",
            "IsDefaultEntryForModel",
        ]
        + original_gene_columns,
    )

    # -----------------------------------------------------------
    # Same canonical DepMap profile filtering as regulon pipeline
    # -----------------------------------------------------------

    expr = expr[
        expr[
            "IsDefaultEntryForModel"
        ]
        == "Yes"
    ].copy()

    expr = expr.drop(
        columns=[
            "IsDefaultEntryForModel"
        ]
    )

    expr = expr.set_index(
        "ModelID"
    )

    # Convert original "GENE (id)" columns back to symbols.
    reverse_mapping = {
        mapping[
            gene
        ]: gene
        for gene in genes_present
    }

    expr = expr.rename(
        columns=reverse_mapping
    )

    # Defensive duplicate removal.
    expr = expr.loc[
        :,
        ~expr.columns.duplicated()
    ].copy()

    expr = expr.apply(
        pd.to_numeric,
        errors="coerce",
    )

    n_missing = int(
        expr
        .isna()
        .sum()
        .sum()
    )

    if n_missing > 0:

        warnings.warn(
            f"{n_missing:,} missing expression values found. "
            "Filling with gene medians."
        )

        expr = expr.fillna(
            expr.median(
                axis=0
            )
        )

    print(
        f"\nLoaded expression matrix: "
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

    return (
        expr,
        genes_present,
        genes_missing,
    )


# ===============================================================
# Hallmark coverage QC
# ===============================================================

def make_hallmark_coverage_qc(
    hallmark,
    expr,
):
    """
    Calculate pathway-wise Hallmark gene coverage in DepMap.
    """

    measured_genes = set(
        expr.columns
    )

    qc = (
        hallmark
        .groupby(
            "source"
        )["target"]
        .agg(
            [
                "nunique",
            ]
        )
        .rename(
            columns={
                "nunique":
                    "n_reference_genes"
            }
        )
    )

    measured_counts = (
        hallmark[
            hallmark[
                "target"
            ].isin(
                measured_genes
            )
        ]
        .groupby(
            "source"
        )["target"]
        .nunique()
        .rename(
            "n_measured_genes"
        )
    )

    qc = qc.join(
        measured_counts,
        how="left",
    )

    qc[
        "n_measured_genes"
    ] = (
        qc[
            "n_measured_genes"
        ]
        .fillna(0)
        .astype(int)
    )

    qc[
        "fraction_measured"
    ] = (
        qc[
            "n_measured_genes"
        ]
        /
        qc[
            "n_reference_genes"
        ]
    )

    qc = qc.sort_values(
        [
            "fraction_measured",
            "n_measured_genes",
        ],
        ascending=[
            True,
            True,
        ],
    )

    return qc


# ===============================================================
# ULM pathway scoring
# ===============================================================

def score_hallmark_ulm(
    expr,
    hallmark,
    min_targets=5,
    verbose=True,
):
    """
    Infer Hallmark program activity with decoupler ULM.

    Parameters
    ----------
    expr
        ModelID x gene TPMLogp1 matrix.

    hallmark
        Long-format Hallmark network:

            source | target | weight

    min_targets
        Minimum number of measured genes required for a pathway.

    Returns
    -------
    scores
        ModelID x Hallmark ULM t-values.

    padj
        ModelID x Hallmark BH-adjusted p-values.

    ulm_net
        Hallmark network actually passed to ULM.
    """

    print("\n========================================")
    print("Running Hallmark + ULM")
    print("========================================")

    # -----------------------------------------------------------
    # Keep only measurable Hallmark genes.
    # -----------------------------------------------------------

    ulm_net = hallmark[
        hallmark[
            "target"
        ].isin(
            expr.columns
        )
    ].copy()

    # -----------------------------------------------------------
    # Remove pathways with too few measured genes.
    # -----------------------------------------------------------

    target_counts = (
        ulm_net
        .groupby(
            "source"
        )["target"]
        .nunique()
    )

    keep_pathways = (
        target_counts[
            target_counts
            >= min_targets
        ]
        .index
    )

    ulm_net = ulm_net[
        ulm_net[
            "source"
        ].isin(
            keep_pathways
        )
    ].copy()

    print(
        f"ULM pathway-gene memberships: "
        f"{len(ulm_net):,}"
    )

    print(
        f"ULM pathways: "
        f"{ulm_net['source'].nunique():,}"
    )

    print(
        f"Unique genes used by ULM: "
        f"{ulm_net['target'].nunique():,}"
    )

    # -----------------------------------------------------------
    # AnnData
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
    # Hallmark weights are all +1.
    #
    # tval=True:
    # score = t-statistic of the ULM slope.
    # -----------------------------------------------------------

    dc.mt.ulm(
        data=adata,
        net=ulm_net[
            [
                "source",
                "target",
                "weight",
            ]
        ],
        tmin=min_targets,
        tval=True,
        verbose=verbose,
    )

    # -----------------------------------------------------------
    # Extract ULM scores
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

    scores.index.name = (
        "ModelID"
    )

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

    padj.index.name = (
        "ModelID"
    )

    print(
        "\nHallmark activity matrix:",
        scores.shape,
    )

    return (
        scores,
        padj,
        ulm_net,
    )


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
            "DepMap protein-coding TPMLogp1 expression CSV."
        ),
    )

    parser.add_argument(
        "--outdir",
        default=(
            "/data/analysis/combinations/models/"
            "in-house/pathways/"
            "pathway_results_hallmark_ulm"
        ),
    )

    parser.add_argument(
        "--min-targets",
        "--min_targets",
        dest="min_targets",
        type=int,
        default=5,
        help=(
            "Minimum number of measurable genes required "
            "for each Hallmark pathway."
        ),
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
    # 1. Load fixed Hallmark reference
    # ===========================================================

    hallmark = load_hallmark()

    # -----------------------------------------------------------
    # Save the EXACT reference used in this run.
    #
    # This should later become the frozen reference used for
    # training, validation and external primary-cell scoring.
    # -----------------------------------------------------------

    hallmark.to_csv(
        outdir
        / "hallmark_reference.tsv",
        sep="\t",
        index=False,
    )

    # ===========================================================
    # 2. Load DepMap expression
    # ===========================================================

    needed_genes = sorted(
        hallmark[
            "target"
        ]
        .unique()
    )

    (
        expr,
        genes_present,
        genes_missing,
    ) = load_depmap_expression(
        expr_path=args.expr,
        needed_genes=needed_genes,
    )

    # ===========================================================
    # 3. Coverage QC
    # ===========================================================

    print("\n========================================")
    print("Hallmark / DepMap coverage")
    print("========================================")

    coverage_qc = make_hallmark_coverage_qc(
        hallmark=hallmark,
        expr=expr,
    )

    coverage_qc.to_csv(
        outdir
        / "QC_hallmark_gene_coverage.tsv",
        sep="\t",
    )

    print(
        coverage_qc
    )

    print(
        "\nMinimum pathway coverage:"
    )

    print(
        coverage_qc[
            "fraction_measured"
        ].min()
    )

    print(
        "\nMedian pathway coverage:"
    )

    print(
        coverage_qc[
            "fraction_measured"
        ].median()
    )

    # Save missing Hallmark genes.
    pd.DataFrame(
        {
            "missing_gene":
                genes_missing
        }
    ).to_csv(
        outdir
        / "missing_hallmark_genes.tsv",
        sep="\t",
        index=False,
    )

    # ===========================================================
    # 4. Hallmark + ULM
    # ===========================================================

    (
        pathway_activity,
        pathway_padj,
        ulm_network,
    ) = score_hallmark_ulm(
        expr=expr,
        hallmark=hallmark,
        min_targets=args.min_targets,
        verbose=not args.quiet_ulm,
    )

    # ===========================================================
    # 5. Save activity matrix
    # ===========================================================

    pathway_activity.to_csv(
        outdir
        / "Hallmark_activity_ULM.csv"
    )

    pathway_padj.to_csv(
        outdir
        / "Hallmark_activity_ULM_padj.csv"
    )

    ulm_network.to_csv(
        outdir
        / "hallmark_network_ULM.tsv",
        sep="\t",
        index=False,
    )

    # ===========================================================
    # 6. Save pathway feature order
    # ===========================================================
    #
    # This is useful because the neural-network pathway encoder
    # should receive pathways in a fixed deterministic order.
    # ===========================================================

    pathway_order = (
        pathway_activity
        .columns
        .tolist()
    )

    pd.DataFrame(
        {
            "feature_index":
                np.arange(
                    len(
                        pathway_order
                    )
                ),

            "pathway":
                pathway_order,
        }
    ).to_csv(
        outdir
        / "Hallmark_feature_order.tsv",
        sep="\t",
        index=False,
    )

    # ===========================================================
    # 7. Basic activity QC
    # ===========================================================

    activity_qc = pd.DataFrame(
        {
            "mean":
                pathway_activity.mean(
                    axis=0
                ),

            "std":
                pathway_activity.std(
                    axis=0,
                    ddof=0,
                ),

            "median":
                pathway_activity.median(
                    axis=0
                ),

            "min":
                pathway_activity.min(
                    axis=0
                ),

            "max":
                pathway_activity.max(
                    axis=0
                ),
        }
    )

    activity_qc[
        "fraction_significant_padj_0.05"
    ] = (
        pathway_padj
        < 0.05
    ).mean(
        axis=0
    )

    activity_qc.to_csv(
        outdir
        / "QC_Hallmark_activity.tsv",
        sep="\t",
    )

    # ===========================================================
    # 8. Summary
    # ===========================================================

    summary = {

        "n_cell_lines":
            int(
                expr.shape[0]
            ),

        "n_expression_genes_loaded":
            int(
                expr.shape[1]
            ),

        "n_hallmark_pathways_reference":
            int(
                hallmark[
                    "source"
                ].nunique()
            ),

        "n_hallmark_memberships_reference":
            int(
                len(
                    hallmark
                )
            ),

        "n_unique_hallmark_genes_reference":
            int(
                hallmark[
                    "target"
                ].nunique()
            ),

        "n_hallmark_genes_present_depmap":
            int(
                len(
                    genes_present
                )
            ),

        "n_hallmark_genes_missing_depmap":
            int(
                len(
                    genes_missing
                )
            ),

        "n_pathways_scored":
            int(
                pathway_activity.shape[1]
            ),

        "n_ulm_memberships":
            int(
                len(
                    ulm_network
                )
            ),

        "min_targets":
            int(
                args.min_targets
            ),

        "pathway_reference":
            "MSigDB Hallmark via decoupler.op.hallmark",

        "pathway_reference_strategy":
            "fixed across all biological contexts",

        "pathway_contextualization":
            "none",

        "activity_method":
            "decoupler ULM",

        "ulm_statistic":
            "t-value",

        "ulm_input":
            "DepMap TPMLogp1 expression",

        "hallmark_edge_weight":
            1.0,
    }

    with open(
        outdir
        / "run_summary.json",
        "w",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
        )

    with open(
        outdir
        / "summary.txt",
        "w",
    ) as handle:

        handle.write(
            "DepMap Hallmark + ULM summary\n"
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
        f"\nHallmark activity matrix: "
        f"{pathway_activity.shape}"
    )

    print(
        "\nFirst pathways:"
    )

    print(
        pathway_activity
        .columns[:10]
        .tolist()
    )

    print(
        f"\nResults written to:\n"
        f"{outdir.resolve()}"
    )

    print(
        "\nMain output:"
    )

    print(
        "  Hallmark_activity_ULM.csv"
    )


if __name__ == "__main__":
    main()
