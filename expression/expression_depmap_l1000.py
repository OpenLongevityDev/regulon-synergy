#!/usr/bin/env python3

"""
Build the DepMap expression branch using a fixed L1000 landmark gene list.

Input
-----
1. expression/L1000.txt
   One gene symbol per line.

2. DepMap:
   OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv

Output
------
ModelID x L1000 landmark-gene expression matrix.

Important
---------
- DepMap values are already TPMLogp1.
- No additional log transformation is performed.
- No z-scoring is performed here.
- The fixed L1000 gene vocabulary is preserved.
- Only canonical DepMap profiles are retained:
      IsDefaultEntryForModel == "Yes"
"""

from pathlib import Path
import argparse
import json
import os

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = Path(
    os.environ.get("DRUG_SYNERGY_ROOT", Path(__file__).resolve().parents[1])
).resolve()

INPUT_ROOT = Path(
    os.environ.get("DRUG_SYNERGY_INPUT_ROOT", ROOT / "input")
).resolve()

DEFAULT_L1000 = ROOT / "expression" / "L1000.txt"
DEFAULT_EXPR = INPUT_ROOT / "cell_lines" / "depmap" / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
DEFAULT_OUTDIR = ROOT / "expression" / "expression_results_l1000"


# ============================================================
# DepMap metadata
# ============================================================

METADATA_COLUMNS = {
    "Unnamed: 0",
    "SequencingID",
    "ModelConditionID",
    "ModelID",
    "IsDefaultEntryForMC",
    "IsDefaultEntryForModel",
}


# ============================================================
# Load L1000 gene list
# ============================================================

def load_l1000_genes(path):
    """
    Read newline-separated gene symbols.

    Returns a unique gene list preserving input order.
    """

    with open(path) as handle:
        genes = [
            line.strip()
            for line in handle
            if line.strip()
        ]

    # preserve order, remove duplicates
    genes = list(dict.fromkeys(genes))

    print("\n========================================")
    print("Loading L1000 landmark genes")
    print("========================================")

    print(
        f"L1000 genes in input file: "
        f"{len(genes):,}"
    )

    print(
        "\nFirst genes:"
    )

    print(
        genes[:20]
    )

    return genes


# ============================================================
# DepMap column parsing
# ============================================================

def detect_gene_mapping(columns):
    """
    Convert DepMap columns such as:

        TP53 (7157)

    to mapping:

        TP53 -> "TP53 (7157)"
    """

    mapping = {}

    for col in columns:

        col = str(col)

        if (
            " (" in col
            and col.endswith(")")
        ):
            symbol = col.split(" (")[0]
        else:
            symbol = col

        # defensive handling of duplicated gene symbols
        if symbol not in mapping:
            mapping[symbol] = col

    return mapping


# ============================================================
# Load DepMap expression
# ============================================================

def load_depmap_l1000(
    expr_path,
    l1000_genes,
):
    """
    Load only L1000 genes from DepMap.

    Keeps canonical model profiles only.
    """

    print("\n========================================")
    print("Loading DepMap L1000 expression")
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
        for gene in l1000_genes
        if gene in mapping
    ]

    genes_missing = [
        gene
        for gene in l1000_genes
        if gene not in mapping
    ]

    original_columns = [
        mapping[gene]
        for gene in genes_present
    ]

    print(
        f"L1000 genes requested: "
        f"{len(l1000_genes):,}"
    )

    print(
        f"L1000 genes present:   "
        f"{len(genes_present):,}"
    )

    print(
        f"L1000 genes missing:   "
        f"{len(genes_missing):,}"
    )

    if genes_missing:
        print(
            "\nMissing genes:"
        )
        print(
            genes_missing
        )

    print(
        "\nReading selected DepMap columns..."
    )

    expr = pd.read_csv(
        expr_path,
        usecols=[
            "ModelID",
            "IsDefaultEntryForModel",
        ] + original_columns,
    )

    # --------------------------------------------------------
    # Keep one canonical profile per DepMap model
    # --------------------------------------------------------

    expr = expr[
        expr["IsDefaultEntryForModel"] == "Yes"
    ].copy()

    expr = expr.drop(
        columns=[
            "IsDefaultEntryForModel"
        ]
    )

    expr = expr.set_index(
        "ModelID"
    )

    # --------------------------------------------------------
    # Rename GENE (EntrezID) -> GENE
    # --------------------------------------------------------

    reverse_mapping = {
        mapping[gene]: gene
        for gene in genes_present
    }

    expr = expr.rename(
        columns=reverse_mapping
    )

    # remove duplicated columns defensively
    expr = expr.loc[
        :,
        ~expr.columns.duplicated()
    ].copy()

    # --------------------------------------------------------
    # Preserve L1000 input order
    # --------------------------------------------------------

    final_gene_order = [
        gene
        for gene in l1000_genes
        if gene in expr.columns
    ]

    expr = expr[
        final_gene_order
    ]

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    expr = expr.apply(
        pd.to_numeric,
        errors="coerce",
    )

    n_missing_values = int(
        expr.isna().sum().sum()
    )

    print(
        f"\nMissing expression values: "
        f"{n_missing_values:,}"
    )

    if n_missing_values > 0:

        print(
            "Filling missing values with "
            "gene-wise median."
        )

        expr = expr.fillna(
            expr.median(axis=0)
        )

    print(
        f"\nFinal expression matrix: "
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
        "Example ModelIDs:",
        expr.index[:5].tolist(),
    )

    return (
        expr,
        genes_present,
        genes_missing,
    )


# ============================================================
# QC
# ============================================================

def make_gene_qc(expr):
    """
    Basic per-gene expression QC.
    """

    qc = pd.DataFrame(
        {
            "mean":
                expr.mean(axis=0),

            "std":
                expr.std(
                    axis=0,
                    ddof=0,
                ),

            "median":
                expr.median(axis=0),

            "min":
                expr.min(axis=0),

            "max":
                expr.max(axis=0),

            "fraction_zero":
                (expr == 0).mean(axis=0),

            "fraction_positive":
                (expr > 0).mean(axis=0),

            "n_unique_values":
                expr.nunique(axis=0),
        }
    )

    return qc


# ============================================================
# Optional rank representation
# ============================================================

def make_within_sample_percentile_ranks(expr):
    """
    Create within-sample percentile ranks across the L1000 genes.

    This is NOT the primary output.
    It is saved as an optional cross-domain robustness
    representation for later ablation.

    Range approximately:
        0 ... 1
    """

    ranks = expr.rank(
        axis=1,
        method="average",
        pct=True,
    )

    return ranks


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--l1000",
        default=str(DEFAULT_L1000),
        help=(
            "Newline-separated L1000 landmark gene list."
        ),
    )

    parser.add_argument(
        "--expr",
        default=str(DEFAULT_EXPR),
        help=(
            "DepMap protein-coding TPMLogp1 expression CSV."
        ),
    )

    parser.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTDIR),
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
    # 1. Load fixed L1000 vocabulary
    # ========================================================

    l1000_genes = load_l1000_genes(
        args.l1000
    )

    # save frozen input list
    pd.DataFrame(
        {
            "gene":
                l1000_genes
        }
    ).to_csv(
        outdir
        / "L1000_reference_input.tsv",
        sep="\t",
        index=False,
    )

    # ========================================================
    # 2. Load DepMap expression
    # ========================================================

    (
        expr,
        genes_present,
        genes_missing,
    ) = load_depmap_l1000(
        expr_path=args.expr,
        l1000_genes=l1000_genes,
    )

    # ========================================================
    # 3. Save primary expression matrix
    # ========================================================

    expr.index.name = (
        "ModelID"
    )

    expr.to_csv(
        outdir
        / "L1000_expression_TPMLogp1.csv"
    )

    # ========================================================
    # 4. Save exact feature order
    # ========================================================

    feature_order = pd.DataFrame(
        {
            "feature_index":
                np.arange(
                    expr.shape[1]
                ),

            "gene":
                expr.columns,
        }
    )

    feature_order.to_csv(
        outdir
        / "L1000_feature_order.tsv",
        sep="\t",
        index=False,
    )

    # ========================================================
    # 5. Missing genes
    # ========================================================

    pd.DataFrame(
        {
            "gene":
                genes_missing
        }
    ).to_csv(
        outdir
        / "L1000_missing_genes.tsv",
        sep="\t",
        index=False,
    )

    # ========================================================
    # 6. Gene-level QC
    # ========================================================

    qc = make_gene_qc(
        expr
    )

    qc.to_csv(
        outdir
        / "QC_L1000_expression.tsv",
        sep="\t",
    )

    # ========================================================
    # 7. Optional within-sample rank representation
    # ========================================================
    #
    # Useful later for:
    #
    #   bulk DepMap
    #       vs
    #   pseudobulk primary human cells
    #
    # This is NOT a replacement for TPMLogp1 yet.
    # It is simply prepared as an ablation.
    # ========================================================

    ranks = make_within_sample_percentile_ranks(
        expr
    )

    ranks.index.name = (
        "ModelID"
    )

    ranks.to_csv(
        outdir
        / "L1000_expression_within_sample_percentile.csv"
    )

    # ========================================================
    # 8. Model-level QC
    # ========================================================

    model_qc = pd.DataFrame(
        {
            "mean_expression":
                expr.mean(axis=1),

            "median_expression":
                expr.median(axis=1),

            "fraction_zero":
                (expr == 0).mean(axis=1),

            "std_expression":
                expr.std(
                    axis=1,
                    ddof=0,
                ),
        }
    )

    model_qc.to_csv(
        outdir
        / "QC_L1000_models.tsv",
        sep="\t",
    )

    # ========================================================
    # 9. Summary
    # ========================================================

    summary = {
        "n_depmap_cell_lines":
            int(expr.shape[0]),

        "n_l1000_genes_requested":
            int(len(l1000_genes)),

        "n_l1000_genes_present":
            int(len(genes_present)),

        "n_l1000_genes_missing":
            int(len(genes_missing)),

        "fraction_l1000_genes_present":
            float(
                len(genes_present)
                /
                len(l1000_genes)
            ),

        "expression_input":
            "DepMap TPMLogp1",

        "extra_log_transform":
            False,

        "global_zscore_applied":
            False,

        "primary_representation":
            "L1000 landmark gene TPMLogp1 expression",

        "optional_representation":
            "within-sample percentile rank",

        "feature_definition":
            "fixed external L1000 landmark gene set",
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

    # ========================================================
    # Done
    # ========================================================

    print("\n========================================")
    print("DONE")
    print("========================================")

    print(
        f"\nPrimary expression matrix: "
        f"{expr.shape}"
    )

    print(
        f"L1000 coverage: "
        f"{len(genes_present)}/"
        f"{len(l1000_genes)} "
        f"({100 * len(genes_present) / len(l1000_genes):.2f}%)"
    )

    print(
        f"\nResults written to:\n"
        f"{outdir.resolve()}"
    )

    print(
        "\nMain outputs:"
    )

    print(
        "  L1000_expression_TPMLogp1.csv"
    )

    print(
        "  L1000_expression_within_sample_percentile.csv"
    )

    print(
        "  L1000_feature_order.tsv"
    )

    print(
        "  QC_L1000_expression.tsv"
    )


if __name__ == "__main__":
    main()
