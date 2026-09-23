#!/usr/bin/env python3

"""
build_drug_ppi.py

Build direct-target and HuRI-propagated PPI representations for DrugComb drugs.

INPUTS
------
1. Drug seed dictionary:
   input/drugs/drug_seed_dictionary.tsv

   Expected columns:
       drugcomb_id
       dname
       inchikey
       seed_genes
       n_seed_genes
       has_any_seed_gene

2. HuRI PPI:
   input/targets/HuRI_PPI/HuRI.tsv

   Two-column Ensembl gene interaction edge list:
       ENSG...
       ENSG...

OUTPUT
------
Default:
    targets/target_ppi_results/

Files:
    huri_edges_clean.tsv
    huri_node_order.tsv

    target_symbol_to_ensembl.tsv
    target_mapping_qc.tsv
    drug_target_mapping_qc.tsv
    rwr_convergence_qc.tsv

    direct_target_binary.npz
    direct_target_binary.csv.gz

    ppi_propagated_rwr.npz
    ppi_propagated_rwr.csv.gz

    drug_target_ppi_lookup.tsv
    run_summary.json


REPRESENTATIONS
---------------

For each drug:

1. Direct target vector

       S_binary[i] = 1
           if protein i is a direct drug target in HuRI

2. RWR seed vector

       S_norm[i] = 1 / n_targets_in_HuRI

3. PPI-propagated profile

       p_(t+1) = (1-r) * W * p_t + r * S_norm

   where W is a column-normalized HuRI adjacency matrix.

Direct target and PPI propagation use the SAME fixed HuRI node order.


IMPORTANT
---------
- HuRI interactions are treated as undirected.
- Self-edges are removed.
- Duplicate edges are removed.
- PPI is unweighted.
- No context-specific expression is used here.
- Drugs with known targets but zero HuRI targets are distinguished
  from drugs with no known targets.
- Propagation is deterministic; it is not fitted on synergy labels.
"""

from pathlib import Path
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

from scipy import sparse


# ============================================================
# Paths
# ============================================================

# Repository root. Override when the checkout/data root lives elsewhere.
ROOT = Path(
    os.environ.get("DRUG_SYNERGY_ROOT", Path(__file__).resolve().parents[1])
).resolve()

# External source data are expected under input/ and are not distributed here.
INPUT_ROOT = Path(
    os.environ.get("DRUG_SYNERGY_INPUT_ROOT", ROOT / "input")
).resolve()

DEFAULT_SEEDS = INPUT_ROOT / "drugs" / "drug_seed_dictionary.tsv"
DEFAULT_HURI = INPUT_ROOT / "targets" / "HuRI_PPI" / "HuRI.tsv"
DEFAULT_OUTDIR = ROOT / "targets" / "target_ppi_results"


# ============================================================
# Utilities
# ============================================================

def print_header(text):

    print()
    print("=" * 60)
    print(text)
    print("=" * 60)


def decode_str_array(x):

    return np.asarray(
        x,
        dtype=str,
    )


def portable_path(path):
    """Return a repository-relative path when possible."""
    resolved = Path(path).resolve()

    for base in (ROOT, INPUT_ROOT):
        try:
            return str(resolved.relative_to(base))
        except ValueError:
            pass

    return str(resolved)


# ============================================================
# Load seed dictionary
# ============================================================

def load_seed_dictionary(path):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Seed dictionary not found:\n{path}"
        )

    print_header(
        "Loading DrugComb seed dictionary"
    )

    df = pd.read_csv(
        path,
        sep="\t",
        dtype={
            "drugcomb_id": str,
            "dname": str,
            "inchikey": str,
            "seed_genes": str,
        },
    )

    required = [
        "drugcomb_id",
        "dname",
        "inchikey",
        "seed_genes",
        "n_seed_genes",
        "has_any_seed_gene",
    ]

    missing = [
        x
        for x in required
        if x not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    if df[
        "drugcomb_id"
    ].duplicated().any():

        duplicated = df.loc[
            df["drugcomb_id"].duplicated(
                keep=False
            ),
            "drugcomb_id",
        ].tolist()

        raise ValueError(
            "Duplicate drugcomb_id values found, e.g. "
            + ", ".join(
                duplicated[:10]
            )
        )

    # ----------------------------------------
    # Normalize seed strings
    # ----------------------------------------

    def parse_seed_genes(value):

        if pd.isna(value):

            return []

        value = str(
            value
        ).strip()

        if (
            value == ""
            or value.lower() == "nan"
        ):

            return []

        genes = [
            x.strip()
            for x
            in value.split(";")
            if x.strip()
        ]

        # preserve order while deduplicating
        genes = list(
            dict.fromkeys(
                genes
            )
        )

        return genes

    df[
        "seed_gene_list"
    ] = df[
        "seed_genes"
    ].apply(
        parse_seed_genes
    )

    df[
        "n_seed_genes_reparsed"
    ] = df[
        "seed_gene_list"
    ].apply(
        len
    )

    n_mismatch = (
        pd.to_numeric(
            df[
                "n_seed_genes"
            ],
            errors="coerce",
        ).fillna(0).astype(int)
        !=
        df[
            "n_seed_genes_reparsed"
        ]
    ).sum()

    print(
        f"Drug rows: "
        f"{len(df):,}"
    )

    print(
        f"Unique DrugComb IDs: "
        f"{df['drugcomb_id'].nunique():,}"
    )

    print(
        f"Drugs with >=1 target symbol: "
        f"{(df['n_seed_genes_reparsed'] > 0).sum():,}"
    )

    print(
        f"Drugs with no target symbols: "
        f"{(df['n_seed_genes_reparsed'] == 0).sum():,}"
    )

    print(
        f"Rows where parsed seed count differs "
        f"from n_seed_genes: {n_mismatch:,}"
    )

    unique_symbols = sorted(
        {
            gene
            for genes
            in df[
                "seed_gene_list"
            ]
            for gene
            in genes
        }
    )

    print(
        f"Unique target symbols: "
        f"{len(unique_symbols):,}"
    )

    return (
        df,
        unique_symbols,
    )


# ============================================================
# HuRI
# ============================================================

def load_and_clean_huri(path):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"HuRI network not found:\n{path}"
        )

    print_header(
        "Loading and cleaning HuRI"
    )

    edges = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=[
            "protein_a",
            "protein_b",
        ],
        dtype=str,
    )

    edges = edges.dropna(
        subset=[
            "protein_a",
            "protein_b",
        ]
    ).copy()

    edges[
        "protein_a"
    ] = edges[
        "protein_a"
    ].str.strip()

    edges[
        "protein_b"
    ] = edges[
        "protein_b"
    ].str.strip()

    n_input_edges = len(
        edges
    )

    # ----------------------------------------
    # Remove self interactions
    # ----------------------------------------

    self_mask = (
        edges[
            "protein_a"
        ]
        ==
        edges[
            "protein_b"
        ]
    )

    n_self_edges = int(
        self_mask.sum()
    )

    edges = edges.loc[
        ~self_mask
    ].copy()

    # ----------------------------------------
    # Canonical ordering for undirected edges
    # ----------------------------------------

    pair_array = np.sort(
        edges[
            [
                "protein_a",
                "protein_b",
            ]
        ].to_numpy(
            dtype=str
        ),
        axis=1,
    )

    edges[
        "protein_a"
    ] = pair_array[
        :,
        0
    ]

    edges[
        "protein_b"
    ] = pair_array[
        :,
        1
    ]

    before_dedup = len(
        edges
    )

    edges = (
        edges
        .drop_duplicates(
            subset=[
                "protein_a",
                "protein_b",
            ]
        )
        .sort_values(
            [
                "protein_a",
                "protein_b",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    n_duplicate_edges = (
        before_dedup
        -
        len(
            edges
        )
    )

    nodes = sorted(
        set(
            edges[
                "protein_a"
            ]
        )
        |
        set(
            edges[
                "protein_b"
            ]
        )
    )

    print(
        f"Input edges: "
        f"{n_input_edges:,}"
    )

    print(
        f"Self-edges removed: "
        f"{n_self_edges:,}"
    )

    print(
        f"Duplicate undirected edges removed: "
        f"{n_duplicate_edges:,}"
    )

    print(
        f"Final HuRI edges: "
        f"{len(edges):,}"
    )

    print(
        f"HuRI proteins: "
        f"{len(nodes):,}"
    )

    return (
        edges,
        nodes,
        {
            "n_huri_input_edges":
                int(
                    n_input_edges
                ),

            "n_huri_self_edges_removed":
                int(
                    n_self_edges
                ),

            "n_huri_duplicate_edges_removed":
                int(
                    n_duplicate_edges
                ),

            "n_huri_edges":
                int(
                    len(
                        edges
                    )
                ),

            "n_huri_nodes":
                int(
                    len(
                        nodes
                    )
                ),
        },
    )


# ============================================================
# Symbol -> Ensembl mapping
# ============================================================

def query_mygene_mapping(
    symbols,
):

    try:

        import mygene

    except ImportError as exc:

        raise ImportError(
            "\nThe target mapping file does not exist, "
            "so symbol -> Ensembl mapping must be generated.\n"
            "Install mygene first:\n\n"
            "    pip install mygene\n\n"
            "Then rerun this script."
        ) from exc

    print_header(
        "Mapping target symbols to Ensembl using mygene"
    )

    mg = mygene.MyGeneInfo()

    result = mg.querymany(
        symbols,
        scopes="symbol",
        fields=(
            "symbol,"
            "ensembl.gene,"
            "entrezgene"
        ),
        species="human",
        as_dataframe=False,
        returnall=False,
        verbose=False,
    )

    rows = []

    for item in result:

        query = str(
            item.get(
                "query",
                ""
            )
        )

        notfound = bool(
            item.get(
                "notfound",
                False
            )
        )

        score = item.get(
            "_score",
            np.nan,
        )

        symbol_returned = item.get(
            "symbol",
            None,
        )

        entrez = item.get(
            "entrezgene",
            None,
        )

        ensembl_obj = item.get(
            "ensembl",
            None,
        )

        ensembl_ids = []

        if isinstance(
            ensembl_obj,
            dict,
        ):

            gene_id = ensembl_obj.get(
                "gene",
                None,
            )

            if gene_id:

                ensembl_ids = [
                    str(
                        gene_id
                    )
                ]

        elif isinstance(
            ensembl_obj,
            list,
        ):

            for x in ensembl_obj:

                if not isinstance(
                    x,
                    dict,
                ):

                    continue

                gene_id = x.get(
                    "gene",
                    None,
                )

                if gene_id:

                    ensembl_ids.append(
                        str(
                            gene_id
                        )
                    )

        ensembl_ids = sorted(
            set(
                ensembl_ids
            )
        )

        if len(
            ensembl_ids
        ) == 0:

            rows.append(
                {
                    "query_symbol":
                        query,

                    "returned_symbol":
                        symbol_returned,

                    "ensembl_gene":
                        np.nan,

                    "entrezgene":
                        entrez,

                    "mygene_score":
                        score,

                    "mygene_notfound":
                        notfound,
                }
            )

        else:

            for ensembl_gene in ensembl_ids:

                rows.append(
                    {
                        "query_symbol":
                            query,

                        "returned_symbol":
                            symbol_returned,

                        "ensembl_gene":
                            ensembl_gene,

                        "entrezgene":
                            entrez,

                        "mygene_score":
                            score,

                        "mygene_notfound":
                            notfound,
                    }
                )

    mapping = pd.DataFrame(
        rows
    )

    return mapping


def resolve_symbol_mapping(
    symbols,
    huri_nodes,
    mapping_file,
):

    mapping_file = Path(
        mapping_file
    )

    if mapping_file.exists():

        print_header(
            "Loading frozen symbol -> Ensembl mapping"
        )

        mapping = pd.read_csv(
            mapping_file,
            sep="\t",
            dtype=str,
        )

        print(
            f"Loaded mapping: "
            f"{len(mapping):,} rows"
        )

    else:

        mapping = query_mygene_mapping(
            symbols
        )

        mapping_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        mapping.to_csv(
            mapping_file,
            sep="\t",
            index=False,
        )

        print(
            "\nFrozen mapping saved to:"
        )

        print(
            mapping_file
        )

    required = [
        "query_symbol",
        "ensembl_gene",
    ]

    missing = [
        x
        for x in required
        if x not in mapping.columns
    ]

    if missing:

        raise ValueError(
            "Mapping file is missing columns: "
            + ", ".join(
                missing
            )
        )

    mapping[
        "query_symbol"
    ] = mapping[
        "query_symbol"
    ].astype(
        str
    )

    # ----------------------------------------
    # Restrict to symbols actually needed
    # ----------------------------------------

    mapping = mapping.loc[
        mapping[
            "query_symbol"
        ].isin(
            symbols
        )
    ].copy()

    # ----------------------------------------
    # Clean Ensembl IDs
    # ----------------------------------------

    mapping[
        "ensembl_gene"
    ] = mapping[
        "ensembl_gene"
    ].replace(
        {
            "nan": np.nan,
            "None": np.nan,
            "": np.nan,
        }
    )

    # In case version suffixes appear
    mapping[
        "ensembl_gene"
    ] = mapping[
        "ensembl_gene"
    ].str.replace(
        r"\.\d+$",
        "",
        regex=True,
    )

    huri_nodes = set(
        huri_nodes
    )

    mapping[
        "in_huri"
    ] = mapping[
        "ensembl_gene"
    ].isin(
        huri_nodes
    )

    # ----------------------------------------
    # We want one preferred HuRI Ensembl ID
    # per symbol.
    #
    # If multiple Ensembl genes are returned,
    # prefer one present in HuRI.
    # ----------------------------------------

    selected_rows = []

    for symbol in symbols:

        group = mapping.loc[
            mapping[
                "query_symbol"
            ]
            ==
            symbol
        ].copy()

        if len(
            group
        ) == 0:

            selected_rows.append(
                {
                    "query_symbol":
                        symbol,

                    "selected_ensembl":
                        np.nan,

                    "mapping_status":
                        "unmapped_symbol",

                    "n_ensembl_candidates":
                        0,

                    "n_huri_candidates":
                        0,

                    "all_ensembl_candidates":
                        "",
                }
            )

            continue

        candidates = sorted(
            set(
                group[
                    "ensembl_gene"
                ].dropna().astype(
                    str
                )
            )
        )

        huri_candidates = sorted(
            set(
                group.loc[
                    group[
                        "in_huri"
                    ],
                    "ensembl_gene",
                ].dropna().astype(
                    str
                )
            )
        )

        if len(
            huri_candidates
        ) == 1:

            selected = (
                huri_candidates[
                    0
                ]
            )

            status = (
                "mapped_to_huri"
            )

        elif len(
            huri_candidates
        ) > 1:

            # Rare ambiguous case:
            # choose lexicographically first,
            # but explicitly flag it.
            selected = (
                huri_candidates[
                    0
                ]
            )

            status = (
                "multiple_huri_candidates_selected_first"
            )

        elif len(
            candidates
        ) > 0:

            selected = (
                candidates[
                    0
                ]
            )

            status = (
                "mapped_not_in_huri"
            )

        else:

            selected = np.nan

            status = (
                "unmapped_symbol"
            )

        selected_rows.append(
            {
                "query_symbol":
                    symbol,

                "selected_ensembl":
                    selected,

                "mapping_status":
                    status,

                "n_ensembl_candidates":
                    len(
                        candidates
                    ),

                "n_huri_candidates":
                    len(
                        huri_candidates
                    ),

                "all_ensembl_candidates":
                    ";".join(
                        candidates
                    ),
            }
        )

    selected = pd.DataFrame(
        selected_rows
    )

    print_header(
        "Target mapping coverage"
    )

    print(
        selected[
            "mapping_status"
        ].value_counts(
            dropna=False
        )
    )

    return (
        mapping,
        selected,
    )


# ============================================================
# Per-drug mapping
# ============================================================

def map_drugs_to_huri(
    seed_df,
    selected_mapping,
):

    print_header(
        "Mapping DrugComb targets into HuRI"
    )

    symbol_to_ensembl = (
        selected_mapping
        .set_index(
            "query_symbol"
        )[
            "selected_ensembl"
        ]
        .to_dict()
    )

    symbol_to_status = (
        selected_mapping
        .set_index(
            "query_symbol"
        )[
            "mapping_status"
        ]
        .to_dict()
    )

    rows = []

    huri_target_lists = []

    for _, row in seed_df.iterrows():

        symbols = row[
            "seed_gene_list"
        ]

        huri_targets = []

        n_symbol_mapped = 0

        n_symbol_in_huri = 0

        for symbol in symbols:

            status = symbol_to_status.get(
                symbol,
                "unmapped_symbol",
            )

            ensembl = symbol_to_ensembl.get(
                symbol,
                np.nan,
            )

            if status != "unmapped_symbol":

                n_symbol_mapped += 1

            if (
                status
                in {
                    "mapped_to_huri",
                    "multiple_huri_candidates_selected_first",
                }
                and pd.notna(
                    ensembl
                )
            ):

                huri_targets.append(
                    str(
                        ensembl
                    )
                )

                n_symbol_in_huri += 1

        huri_targets = list(
            dict.fromkeys(
                huri_targets
            )
        )

        n_known = len(
            symbols
        )

        n_huri = len(
            huri_targets
        )

        if n_known == 0:

            drug_status = (
                "no_known_targets"
            )

        elif n_huri == 0:

            drug_status = (
                "targets_known_but_none_in_huri"
            )

        else:

            drug_status = (
                "targets_mapped_to_huri"
            )

        huri_target_lists.append(
            huri_targets
        )

        rows.append(
            {
                "drugcomb_id":
                    row[
                        "drugcomb_id"
                    ],

                "dname":
                    row[
                        "dname"
                    ],

                "inchikey":
                    row[
                        "inchikey"
                    ],

                "n_seed_symbols":
                    n_known,

                "n_symbols_mapped_to_ensembl":
                    n_symbol_mapped,

                "n_targets_in_huri":
                    n_huri,

                "fraction_targets_in_huri":
                    (
                        n_huri
                        /
                        n_known
                        if n_known > 0
                        else np.nan
                    ),

                "drug_target_status":
                    drug_status,

                "seed_symbols":
                    ";".join(
                        symbols
                    ),

                "huri_target_ensembl":
                    ";".join(
                        huri_targets
                    ),
            }
        )

    qc = pd.DataFrame(
        rows
    )

    seed_df = seed_df.copy()

    seed_df[
        "huri_target_list"
    ] = huri_target_lists

    print(
        qc[
            "drug_target_status"
        ].value_counts()
    )

    with_targets = qc[
        "n_targets_in_huri"
    ] > 0

    if with_targets.any():

        print(
            "\nHuRI targets/drug among drugs with coverage:"
        )

        print(
            qc.loc[
                with_targets,
                "n_targets_in_huri",
            ].describe()
        )

    return (
        seed_df,
        qc,
    )


# ============================================================
# HuRI adjacency
# ============================================================

def build_huri_adjacency(
    edges,
    nodes,
):

    print_header(
        "Building sparse HuRI adjacency matrix"
    )

    node_to_idx = {
        node:
            i
        for i, node
        in enumerate(
            nodes
        )
    }

    row_idx = edges[
        "protein_a"
    ].map(
        node_to_idx
    ).to_numpy()

    col_idx = edges[
        "protein_b"
    ].map(
        node_to_idx
    ).to_numpy()

    n_nodes = len(
        nodes
    )

    # Undirected graph:
    # add both A->B and B->A

    rows = np.concatenate(
        [
            row_idx,
            col_idx,
        ]
    )

    cols = np.concatenate(
        [
            col_idx,
            row_idx,
        ]
    )

    values = np.ones(
        len(
            rows
        ),
        dtype=np.float32,
    )

    adjacency = sparse.csr_matrix(
        (
            values,
            (
                rows,
                cols,
            ),
        ),
        shape=(
            n_nodes,
            n_nodes,
        ),
        dtype=np.float32,
    )

    adjacency.sum_duplicates()

    # Force binary just in case
    adjacency.data[
        :
    ] = 1.0

    degree = np.asarray(
        adjacency.sum(
            axis=1
        )
    ).ravel()

    print(
        f"Adjacency shape: "
        f"{adjacency.shape}"
    )

    print(
        f"Undirected nonzero entries: "
        f"{adjacency.nnz:,}"
    )

    print(
        f"Mean degree: "
        f"{degree.mean():.2f}"
    )

    print(
        f"Median degree: "
        f"{np.median(degree):.1f}"
    )

    return (
        adjacency,
        node_to_idx,
        degree,
    )


# ============================================================
# Direct target matrices
# ============================================================

def build_direct_target_matrices(
    seed_df,
    nodes,
    node_to_idx,
):

    print_header(
        "Building direct target matrices"
    )

    n_drugs = len(
        seed_df
    )

    n_nodes = len(
        nodes
    )

    rows = []

    cols = []

    for drug_idx, targets in enumerate(
        seed_df[
            "huri_target_list"
        ]
    ):

        for target in targets:

            idx = node_to_idx[
                target
            ]

            rows.append(
                drug_idx
            )

            cols.append(
                idx
            )

    values = np.ones(
        len(
            rows
        ),
        dtype=np.float32,
    )

    binary = sparse.csr_matrix(
        (
            values,
            (
                rows,
                cols,
            ),
        ),
        shape=(
            n_drugs,
            n_nodes,
        ),
        dtype=np.float32,
    )

    binary.sum_duplicates()

    binary.data[
        :
    ] = 1.0

    # ----------------------------------------
    # Normalize seed weights per drug
    # ----------------------------------------

    row_sums = np.asarray(
        binary.sum(
            axis=1
        )
    ).ravel()

    inv = np.zeros_like(
        row_sums,
        dtype=np.float32,
    )

    nonzero = (
        row_sums > 0
    )

    inv[
        nonzero
    ] = (
        1.0
        /
        row_sums[
            nonzero
        ]
    )

    normalized = (
        sparse.diags(
            inv
        )
        @ binary
    ).tocsr()

    print(
        f"Direct target matrix: "
        f"{binary.shape}"
    )

    print(
        f"Direct target nonzeros: "
        f"{binary.nnz:,}"
    )

    print(
        f"Drugs with >=1 HuRI target: "
        f"{nonzero.sum():,}"
    )

    return (
        binary,
        normalized,
    )


# ============================================================
# Transition matrix
# ============================================================

def build_transition_matrix(
    adjacency,
):
    """
    Build row-normalized transition matrix W.

    We represent each drug profile as a row vector and use:

        P_next = (1-r) * P @ W + r * S

    Therefore each row of W contains the outgoing transition
    probabilities from the corresponding HuRI node.
    """

    degree = np.asarray(
        adjacency.sum(
            axis=1
        )
    ).ravel()

    inv_degree = np.zeros_like(
        degree,
        dtype=np.float32,
    )

    mask = degree > 0

    inv_degree[
        mask
    ] = (
        1.0
        /
        degree[
            mask
        ]
    )

    # Row-normalized first
    row_transition = (
        sparse.diags(
            inv_degree
        )
        @ adjacency
    )

    # We propagate row vectors below:
    #
    # P_next = (1-r) * P @ row_transition + r*S
    #
    # This is equivalent and more convenient for
    # a drug × proteins matrix.

    return row_transition.tocsr()


# ============================================================
# Random Walk with Restart
# ============================================================

def run_rwr(
    seed_matrix,
    transition,
    restart=0.5,
    tol=1e-6,
    max_iter=500,
    batch_size=256,
):
    """
    Random Walk with Restart over HuRI.

    Parameters
    ----------
    seed_matrix
        Drug x protein matrix. Each covered drug row should sum to 1.
    transition
        Row-normalized HuRI transition matrix.
    restart
        Probability of restarting at the drug's direct targets.
    tol
        Convergence threshold on the maximum absolute change between
        successive iterations. Default 1e-6 is suitable for float32.
    max_iter
        Maximum iterations per batch.
    batch_size
        Number of drugs propagated simultaneously.

    Returns
    -------
    propagated : np.ndarray, float32
        Drug x protein propagated PPI profiles.
    convergence_qc : pd.DataFrame
        One row per batch with iteration count, final delta, and
        whether the convergence criterion was reached.
    """

    print_header(
        "Running Random Walk with Restart"
    )

    if not (
        0.0
        <
        restart
        <=
        1.0
    ):

        raise ValueError(
            "restart must be in (0, 1]."
        )

    if tol <= 0:

        raise ValueError(
            "tol must be > 0."
        )

    if max_iter < 1:

        raise ValueError(
            "max_iter must be >= 1."
        )

    if batch_size < 1:

        raise ValueError(
            "batch_size must be >= 1."
        )

    n_drugs = (
        seed_matrix.shape[
            0
        ]
    )

    n_nodes = (
        seed_matrix.shape[
            1
        ]
    )

    propagated = np.zeros(
        (
            n_drugs,
            n_nodes,
        ),
        dtype=np.float32,
    )

    covered = np.asarray(
        seed_matrix.sum(
            axis=1
        )
    ).ravel() > 0

    covered_indices = np.where(
        covered
    )[0]

    print(
        f"Drugs to propagate: "
        f"{len(covered_indices):,}"
    )

    print(
        f"Restart probability: "
        f"{restart}"
    )

    print(
        f"Tolerance: {tol}"
    )

    print(
        f"Max iterations: "
        f"{max_iter}"
    )

    print(
        f"Batch size: "
        f"{batch_size}"
    )

    if len(
        covered_indices
    ) == 0:

        empty_qc = pd.DataFrame(
            columns=[
                "batch",
                "n_drugs",
                "iterations",
                "final_delta",
                "converged",
            ]
        )

        return (
            propagated,
            empty_qc,
        )

    n_batches = int(
        np.ceil(
            len(
                covered_indices
            )
            /
            batch_size
        )
    )

    convergence_rows = []

    for batch_number, start in enumerate(
        range(
            0,
            len(
                covered_indices
            ),
            batch_size,
        ),
        start=1,
    ):

        idx = covered_indices[
            start:
            start
            +
            batch_size
        ]

        S_sparse = seed_matrix[
            idx
        ]

        S = S_sparse.toarray().astype(
            np.float32
        )

        P = S.copy()

        converged = False
        converged_at = max_iter
        delta = np.inf

        for iteration in range(
            1,
            max_iter + 1,
        ):

            P_next = (
                (1.0 - restart)
                *
                (
                    P
                    @
                    transition
                )
                +
                restart
                *
                S
            )

            P_next = np.asarray(
                P_next,
                dtype=np.float32,
            )

            # Maximum absolute change across this whole batch.
            delta = float(
                np.max(
                    np.abs(
                        P_next
                        -
                        P
                    )
                )
            )

            P = P_next

            if delta < tol:

                converged = True

                converged_at = (
                    iteration
                )

                break

        propagated[
            idx,
            :
        ] = P

        convergence_rows.append(
            {
                "batch":
                    batch_number,

                "n_drugs":
                    len(
                        idx
                    ),

                "iterations":
                    converged_at,

                "final_delta":
                    delta,

                "converged":
                    converged,
            }
        )

        print(
            f"Batch "
            f"{batch_number:>3}/{n_batches} "
            f"| drugs {len(idx):>4} "
            f"| iterations {converged_at:>3} "
            f"| final delta {delta:.3e} "
            f"| converged {converged}"
        )

    convergence_qc = pd.DataFrame(
        convergence_rows
    )

    print(
        "\nRWR convergence:"
    )

    print(
        f"Converged batches: "
        f"{int(convergence_qc['converged'].sum())}"
        f"/{len(convergence_qc)}"
    )

    print(
        f"Median iterations: "
        f"{convergence_qc['iterations'].median():.1f}"
    )

    print(
        f"Max iterations used: "
        f"{int(convergence_qc['iterations'].max())}"
    )

    print(
        f"Median final delta: "
        f"{convergence_qc['final_delta'].median():.3e}"
    )

    print(
        f"Max final delta: "
        f"{convergence_qc['final_delta'].max():.3e}"
    )

    if not convergence_qc[
        "converged"
    ].all():

        n_not_converged = int(
            (
                ~convergence_qc[
                    "converged"
                ]
            ).sum()
        )

        warnings.warn(
            f"{n_not_converged} RWR batch(es) reached "
            f"max_iter={max_iter} before delta < {tol}. "
            "Inspect rwr_convergence_qc.tsv before using "
            "the propagated profiles."
        )

    return (
        propagated,
        convergence_qc,
    )


# ============================================================
# Save matrices
# ============================================================

def save_sparse_npz_with_metadata(
    path,
    matrix,
    drug_ids,
    nodes,
):

    path = Path(
        path
    )

    matrix = matrix.tocsr()

    np.savez_compressed(
        path,

        data=matrix.data.astype(
            np.float32
        ),

        indices=matrix.indices.astype(
            np.int32
        ),

        indptr=matrix.indptr.astype(
            np.int32
        ),

        shape=np.asarray(
            matrix.shape,
            dtype=np.int64,
        ),

        drug_id=np.asarray(
            drug_ids,
            dtype=str,
        ),

        protein_id=np.asarray(
            nodes,
            dtype=str,
        ),
    )


def save_dense_npz(
    path,
    matrix,
    drug_ids,
    nodes,
):

    np.savez_compressed(
        path,

        X=np.asarray(
            matrix,
            dtype=np.float32,
        ),

        drug_id=np.asarray(
            drug_ids,
            dtype=str,
        ),

        protein_id=np.asarray(
            nodes,
            dtype=str,
        ),
    )


def save_matrix_csv_gz(
    path,
    matrix,
    drug_ids,
    nodes,
):
    """
    Saves a potentially large dense table.

    For direct targets this converts the sparse matrix
    to dense for output. The .npz should be preferred
    for modeling.
    """

    if sparse.issparse(
        matrix
    ):

        X = matrix.toarray()

    else:

        X = matrix

    df = pd.DataFrame(
        X,
        index=np.asarray(
            drug_ids,
            dtype=str,
        ),
        columns=nodes,
    )

    df.index.name = (
        "drugcomb_id"
    )

    df.to_csv(
        path,
        compression="gzip",
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        default=str(
            DEFAULT_SEEDS
        ),
    )

    parser.add_argument(
        "--huri",
        default=str(
            DEFAULT_HURI
        ),
    )

    parser.add_argument(
        "--outdir",
        default=str(
            DEFAULT_OUTDIR
        ),
    )

    parser.add_argument(
        "--restart",
        type=float,
        default=0.5,
        help=(
            "RWR restart probability. "
            "Default: 0.5"
        ),
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help=(
            "RWR convergence tolerance. "
            "Default: 1e-6 (appropriate for float32 propagation)."
        ),
    )

    parser.add_argument(
        "--max-iter",
        "--max_iter",
        dest="max_iter",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--write-csv",
        action="store_true",
        help=(
            "Also write full drug x HuRI matrices "
            "as compressed CSV. These can be large; "
            "NPZ files are always saved."
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

    # ========================================================
    # 1. Drug targets
    # ========================================================

    seed_df, unique_symbols = (
        load_seed_dictionary(
            args.seeds
        )
    )

    # ========================================================
    # 2. HuRI
    # ========================================================

    (
        huri_edges,
        huri_nodes,
        huri_summary,
    ) = load_and_clean_huri(
        args.huri
    )

    huri_edges.to_csv(
        outdir
        / "huri_edges_clean.tsv",
        sep="\t",
        index=False,
    )

    node_order = pd.DataFrame(
        {
            "protein_index":
                np.arange(
                    len(
                        huri_nodes
                    )
                ),

            "ensembl_gene":
                huri_nodes,
        }
    )

    node_order.to_csv(
        outdir
        / "huri_node_order.tsv",
        sep="\t",
        index=False,
    )

    # ========================================================
    # 3. Symbol -> Ensembl mapping
    # ========================================================

    # This mapping is frozen after the first successful query so that
    # later runs do not depend on changes in the external MyGene service.
    mapping_file = (
        outdir
        / "target_symbol_to_ensembl.tsv"
    )

    (
        full_mapping,
        selected_mapping,
    ) = resolve_symbol_mapping(
        symbols=unique_symbols,
        huri_nodes=huri_nodes,
        mapping_file=mapping_file,
    )

    selected_mapping.to_csv(
        outdir
        / "target_mapping_qc.tsv",
        sep="\t",
        index=False,
    )

    # ========================================================
    # 4. Map drugs into HuRI
    # ========================================================

    (
        seed_df,
        drug_qc,
    ) = map_drugs_to_huri(
        seed_df=seed_df,
        selected_mapping=selected_mapping,
    )

    drug_qc.to_csv(
        outdir
        / "drug_target_mapping_qc.tsv",
        sep="\t",
        index=False,
    )

    # ========================================================
    # 5. HuRI adjacency
    # ========================================================

    (
        adjacency,
        node_to_idx,
        degree,
    ) = build_huri_adjacency(
        edges=huri_edges,
        nodes=huri_nodes,
    )

    # Save sparse adjacency
    sparse.save_npz(
        outdir
        / "huri_adjacency.npz",
        adjacency,
    )

    # ========================================================
    # 6. Direct target matrices
    # ========================================================

    (
        direct_binary,
        direct_normalized,
    ) = build_direct_target_matrices(
        seed_df=seed_df,
        nodes=huri_nodes,
        node_to_idx=node_to_idx,
    )

    drug_ids = seed_df[
        "drugcomb_id"
    ].astype(
        str
    ).to_numpy()

    save_sparse_npz_with_metadata(
        outdir
        / "direct_target_binary.npz",
        matrix=direct_binary,
        drug_ids=drug_ids,
        nodes=huri_nodes,
    )

    save_sparse_npz_with_metadata(
        outdir
        / "direct_target_normalized_for_rwr.npz",
        matrix=direct_normalized,
        drug_ids=drug_ids,
        nodes=huri_nodes,
    )

    # ========================================================
    # 7. Random Walk with Restart
    # ========================================================

    transition = (
        build_transition_matrix(
            adjacency
        )
    )

    (
        propagated,
        rwr_convergence_qc,
    ) = run_rwr(
        seed_matrix=direct_normalized,
        transition=transition,
        restart=args.restart,
        tol=args.tol,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
    )

    rwr_convergence_qc.to_csv(
        outdir
        / "rwr_convergence_qc.tsv",
        sep="\t",
        index=False,
    )

    # ========================================================
    # 8. Propagation QC
    # ========================================================

    row_sums = propagated.sum(
        axis=1
    )

    covered_mask = (
        drug_qc[
            "n_targets_in_huri"
        ].to_numpy()
        >
        0
    )

    if covered_mask.any():

        covered_sums = (
            row_sums[
                covered_mask
            ]
        )

        print_header(
            "PPI propagation QC"
        )

        print(
            f"Propagated matrix: "
            f"{propagated.shape}"
        )

        print(
            f"Mean profile sum "
            f"(covered drugs): "
            f"{covered_sums.mean():.6f}"
        )

        print(
            f"Min profile sum "
            f"(covered drugs): "
            f"{covered_sums.min():.6f}"
        )

        print(
            f"Max profile sum "
            f"(covered drugs): "
            f"{covered_sums.max():.6f}"
        )

    # ========================================================
    # 9. Save PPI matrix
    # ========================================================

    save_dense_npz(
        outdir
        / "ppi_propagated_rwr.npz",
        matrix=propagated,
        drug_ids=drug_ids,
        nodes=huri_nodes,
    )

    # ========================================================
    # 10. Drug lookup
    # ========================================================

    lookup = drug_qc.copy()

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
            "drugcomb_id",
            "dname",
            "inchikey",
            "drug_target_status",
            "n_seed_symbols",
            "n_symbols_mapped_to_ensembl",
            "n_targets_in_huri",
            "fraction_targets_in_huri",
            "seed_symbols",
            "huri_target_ensembl",
        ]
    ]

    lookup.to_csv(
        outdir
        / "drug_target_ppi_lookup.tsv",
        sep="\t",
        index=False,
    )

    # ========================================================
    # Optional human-readable matrices
    # ========================================================

    if args.write_csv:

        print_header(
            "Writing compressed CSV matrices"
        )

        print(
            "Writing direct target matrix..."
        )

        save_matrix_csv_gz(
            outdir
            / "direct_target_binary.csv.gz",
            matrix=direct_binary,
            drug_ids=drug_ids,
            nodes=huri_nodes,
        )

        print(
            "Writing propagated PPI matrix..."
        )

        save_matrix_csv_gz(
            outdir
            / "ppi_propagated_rwr.csv.gz",
            matrix=propagated,
            drug_ids=drug_ids,
            nodes=huri_nodes,
        )

    # ========================================================
    # Summary
    # ========================================================

    n_drugs = len(
        seed_df
    )

    n_known = int(
        (
            drug_qc[
                "n_seed_symbols"
            ]
            >
            0
        ).sum()
    )

    n_huri_covered = int(
        (
            drug_qc[
                "n_targets_in_huri"
            ]
            >
            0
        ).sum()
    )

    summary = {
        "seed_dictionary":
            portable_path(
                args.seeds
            ),

        "huri_file":
            portable_path(
                args.huri
            ),

        "n_drugs":
            int(
                n_drugs
            ),

        "n_drugs_with_known_target_symbols":
            n_known,

        "n_drugs_with_at_least_one_huri_target":
            n_huri_covered,

        "fraction_drugs_with_huri_target":
            (
                n_huri_covered
                /
                n_drugs
                if n_drugs > 0
                else np.nan
            ),

        "n_unique_target_symbols":
            int(
                len(
                    unique_symbols
                )
            ),

        "restart_probability":
            float(
                args.restart
            ),

        "rwr_tolerance":
            float(
                args.tol
            ),

        "rwr_max_iter":
            int(
                args.max_iter
            ),

        "rwr_n_batches":
            int(
                len(
                    rwr_convergence_qc
                )
            ),

        "rwr_n_converged_batches":
            int(
                rwr_convergence_qc[
                    "converged"
                ].sum()
            )
            if len(
                rwr_convergence_qc
            ) > 0
            else 0,

        "rwr_all_batches_converged":
            bool(
                rwr_convergence_qc[
                    "converged"
                ].all()
            )
            if len(
                rwr_convergence_qc
            ) > 0
            else True,

        "rwr_median_iterations":
            float(
                rwr_convergence_qc[
                    "iterations"
                ].median()
            )
            if len(
                rwr_convergence_qc
            ) > 0
            else None,

        "rwr_max_iterations_used":
            int(
                rwr_convergence_qc[
                    "iterations"
                ].max()
            )
            if len(
                rwr_convergence_qc
            ) > 0
            else 0,

        "rwr_median_final_delta":
            float(
                rwr_convergence_qc[
                    "final_delta"
                ].median()
            )
            if len(
                rwr_convergence_qc
            ) > 0
            else None,

        "rwr_max_final_delta":
            float(
                rwr_convergence_qc[
                    "final_delta"
                ].max()
            )
            if len(
                rwr_convergence_qc
            ) > 0
            else None,

        "ppi_type":
            "HuRI",

        "ppi_directed":
            False,

        "ppi_weighted":
            False,

        "self_edges_removed":
            True,

        "duplicate_edges_removed":
            True,

        "rwr_seed_normalization":
            "equal weight 1/n_targets_in_huri",

        "direct_target_representation":
            "binary",

        "ppi_representation":
            "random walk with restart",

        **huri_summary,
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
    # Final report
    # ========================================================

    print_header(
        "DONE"
    )

    print(
        f"Drug rows: "
        f"{n_drugs:,}"
    )

    print(
        f"Drugs with known targets: "
        f"{n_known:,}"
    )

    print(
        f"Drugs with >=1 HuRI target: "
        f"{n_huri_covered:,}"
    )

    print(
        f"HuRI proteins: "
        f"{len(huri_nodes):,}"
    )

    print(
        f"HuRI edges: "
        f"{len(huri_edges):,}"
    )

    print(
        f"\nDirect target matrix: "
        f"{direct_binary.shape}"
    )

    print(
        f"PPI propagated matrix: "
        f"{propagated.shape}"
    )

    print(
        "\nResults written to:"
    )

    print(
        outdir.resolve()
    )

    print(
        "\nMain modeling files:"
    )

    print(
        "  direct_target_binary.npz"
    )

    print(
        "  ppi_propagated_rwr.npz"
    )

    print(
        "  huri_node_order.tsv"
    )

    print(
        "  drug_target_ppi_lookup.tsv"
    )

    print(
        "\nQC files:"
    )

    print(
        "  target_mapping_qc.tsv"
    )

    print(
        "  drug_target_mapping_qc.tsv"
    )

    print(
        "  rwr_convergence_qc.tsv"
    )

    print(
        "  target_symbol_to_ensembl.tsv"
    )

    print(
        "  run_summary.json"
    )


if __name__ == "__main__":

    main()
