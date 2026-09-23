#!/usr/bin/env python3

"""
smiles_drugcomb.py

Build the chemical-structure branch of the drug encoder from DrugComb SMILES.

Input
-----
input/drugs/DrugComb_drug_identifiers.tsv

Expected columns include:
    id
    dname
    chembl_id
    inchikey
    smiles

Output
------
1. Canonicalized / validated SMILES table
2. Morgan fingerprint matrix:
       drug x 2048 fingerprint bits
3. Compressed NumPy representation
4. Invalid-SMILES report
5. Duplicate-structure report
6. Per-drug chemistry QC
7. SMILES cleanup QC
8. Repaired-SMILES report
9. Run summary

Important
---------
- We DO NOT remove salts / fragments.
- We DO NOT keep only the largest fragment.
- We preserve stereochemistry in canonical SMILES.
- We DO NOT neutralize molecules.
- We DO NOT alter protonation state.
- We conservatively handle semicolon-separated alternative
  SMILES representations:
      * first try the full original string
      * then try removing trailing semicolon(s)
      * then split internal semicolons
      * if valid candidate SMILES exist, use the first valid one
        and retain remaining valid alternatives for QC
- Morgan fingerprints use:
      radius = 2
      fpSize = 2048
  i.e. an ECFP4-style representation.

The fingerprint itself is FIXED and not learned.

Later, the synergy model will learn:

    2048 fingerprint bits
             |
             v
          MLP encoder
             |
             v
      chemical embedding
"""

from pathlib import Path
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


# ============================================================
# Default paths
# ============================================================

# Repository root. Override when the checkout/data root lives elsewhere.
ROOT = Path(
    os.environ.get("DRUG_SYNERGY_ROOT", Path(__file__).resolve().parents[1])
).resolve()

# External source data are expected under input/ and are not distributed here.
INPUT_ROOT = Path(
    os.environ.get("DRUG_SYNERGY_INPUT_ROOT", ROOT / "input")
).resolve()

DEFAULT_INPUT = INPUT_ROOT / "drugs" / "DrugComb_drug_identifiers.tsv"
DEFAULT_OUTDIR = ROOT / "smiles" / "smiles_results_morgan"


# ============================================================
# Utility: SMILES parser
# ============================================================

def try_parse_smiles(smiles):
    """
    Try to parse a SMILES string with RDKit.

    Returns
    -------
    RDKit Mol or None
    """

    try:
        mol = Chem.MolFromSmiles(
            smiles,
            sanitize=True,
        )
    except Exception:
        mol = None

    return mol


# ============================================================
# Load drug table
# ============================================================

def load_drugs(path):

    print("\n========================================")
    print("Loading DrugComb drug identifiers")
    print("========================================")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype={
            "id": str,
        },
    )

    required = [
        "id",
        "dname",
        "smiles",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Required columns missing from input: "
            + ", ".join(missing)
        )

    print(
        f"Rows in input: {len(df):,}"
    )

    print(
        f"Unique drug IDs: "
        f"{df['id'].nunique():,}"
    )

    print(
        f"Unique drug names: "
        f"{df['dname'].nunique():,}"
    )

    print(
        f"Missing SMILES: "
        f"{df['smiles'].isna().sum():,}"
    )

    # Drug ID should ideally define one DrugComb drug.
    if df["id"].duplicated().any():

        duplicated = (
            df.loc[
                df["id"].duplicated(
                    keep=False
                ),
                ["id", "dname", "smiles"],
            ]
            .sort_values("id")
        )

        warnings.warn(
            f"{duplicated['id'].nunique()} duplicated "
            "DrugComb IDs found."
        )

    return df


# ============================================================
# Conservative SMILES cleanup
# ============================================================

def clean_smiles_string(smiles):
    """
    Conservative cleanup of DrugComb SMILES.

    Strategy
    --------
    1. Missing / empty -> unresolved as missing.
    2. Try original SMILES exactly as provided.
    3. If original fails, remove TRAILING semicolon(s) only.
    4. If still invalid and internal semicolon(s) are present:
         - split into candidate SMILES
         - parse candidates independently
         - if one or more candidates are valid:
             use the first valid candidate
             retain other valid candidate(s) as alternatives
    5. Otherwise mark unresolved.

    We intentionally DO NOT:
        - strip salts
        - keep largest fragment only
        - neutralize molecules
        - alter protonation state

    Returns
    -------
    cleaned_smiles : str or None

    cleanup_status : str
        one of:
            "original"
            "removed_trailing_semicolon"
            "selected_first_valid_semicolon_candidate"
            "missing"
            "empty"
            "unresolved"

    alternative_smiles : str or None
        Remaining valid semicolon-separated alternatives,
        joined by " ; ".
    """

    if pd.isna(smiles):
        return None, "missing", None

    raw = str(
        smiles
    ).strip()

    if raw == "":
        return None, "empty", None

    # --------------------------------------------------------
    # 1. Original string
    # --------------------------------------------------------

    mol = try_parse_smiles(
        raw
    )

    if mol is not None:
        return (
            raw,
            "original",
            None,
        )

    # --------------------------------------------------------
    # 2. Remove trailing semicolon(s) only
    # --------------------------------------------------------

    trailing_fixed = (
        raw
        .rstrip(";")
        .strip()
    )

    if trailing_fixed != raw:

        mol = try_parse_smiles(
            trailing_fixed
        )

        if mol is not None:

            return (
                trailing_fixed,
                "removed_trailing_semicolon",
                None,
            )

    # --------------------------------------------------------
    # 3. Semicolon-separated alternative representations
    # --------------------------------------------------------

    if ";" in raw:

        candidates = [
            x.strip()
            for x in raw.split(";")
            if x.strip()
        ]

        valid_candidates = []

        for candidate in candidates:

            mol = try_parse_smiles(
                candidate
            )

            if mol is not None:

                valid_candidates.append(
                    candidate
                )

        if len(
            valid_candidates
        ) > 0:

            chosen = (
                valid_candidates[0]
            )

            remaining = (
                valid_candidates[1:]
            )

            if len(
                remaining
            ) > 0:

                alternative_smiles = (
                    " ; ".join(
                        remaining
                    )
                )

            else:

                alternative_smiles = None

            return (
                chosen,
                "selected_first_valid_semicolon_candidate",
                alternative_smiles,
            )

    # --------------------------------------------------------
    # 4. Still unresolved
    # --------------------------------------------------------

    return (
        None,
        "unresolved",
        None,
    )


# ============================================================
# SMILES processing
# ============================================================

def process_smiles(smiles):
    """
    Parse and canonicalize one DrugComb SMILES string.

    Returns
    -------
    dictionary containing:
        validity
        cleanup status
        cleaned SMILES
        alternative SMILES
        canonical SMILES
        basic molecular QC
    """

    (
        cleaned_smiles,
        cleanup_status,
        alternative_smiles,
    ) = clean_smiles_string(
        smiles
    )

    if cleaned_smiles is None:

        return {
            "valid_smiles": False,
            "cleaned_smiles": None,
            "alternative_smiles": alternative_smiles,
            "smiles_cleanup_status": cleanup_status,
            "canonical_smiles": None,
            "n_fragments": np.nan,
            "n_atoms": np.nan,
            "n_heavy_atoms": np.nan,
            "molecular_weight_rdkit": np.nan,
            "formal_charge": np.nan,
        }

    mol = try_parse_smiles(
        cleaned_smiles
    )

    if mol is None:

        # Should normally never happen because
        # clean_smiles_string() already validated it.

        return {
            "valid_smiles": False,
            "cleaned_smiles": cleaned_smiles,
            "alternative_smiles": alternative_smiles,
            "smiles_cleanup_status":
                "parse_failed_after_cleanup",
            "canonical_smiles": None,
            "n_fragments": np.nan,
            "n_atoms": np.nan,
            "n_heavy_atoms": np.nan,
            "molecular_weight_rdkit": np.nan,
            "formal_charge": np.nan,
        }

    # Preserve stereochemistry.
    canonical = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )

    fragments = Chem.GetMolFrags(
        mol,
        asMols=False,
        sanitizeFrags=False,
    )

    result = {

        "valid_smiles":
            True,

        "cleaned_smiles":
            cleaned_smiles,

        "alternative_smiles":
            alternative_smiles,

        "smiles_cleanup_status":
            cleanup_status,

        "canonical_smiles":
            canonical,

        "n_fragments":
            len(fragments),

        "n_atoms":
            mol.GetNumAtoms(),

        "n_heavy_atoms":
            mol.GetNumHeavyAtoms(),

        "molecular_weight_rdkit":
            Descriptors.MolWt(
                mol
            ),

        "formal_charge":
            Chem.GetFormalCharge(
                mol
            ),
    }

    return result


# ============================================================
# Canonicalize all structures
# ============================================================

def canonicalize_table(df):

    print("\n========================================")
    print("Validating and canonicalizing SMILES")
    print("========================================")

    records = []

    for _, row in df.iterrows():

        result = process_smiles(
            row["smiles"]
        )

        record = row.to_dict()

        record.update(
            result
        )

        records.append(
            record
        )

    out = pd.DataFrame(
        records
    )

    n_valid = int(
        out[
            "valid_smiles"
        ].sum()
    )

    n_invalid = (
        len(out)
        - n_valid
    )

    print(
        f"Valid SMILES:   {n_valid:,}"
    )

    print(
        f"Invalid SMILES: {n_invalid:,}"
    )

    print(
        "\nSMILES cleanup status:"
    )

    print(
        out[
            "smiles_cleanup_status"
        ]
        .value_counts(
            dropna=False
        )
    )

    if n_valid > 0:

        print(
            "\nFragment counts among valid drugs:"
        )

        print(
            out.loc[
                out[
                    "valid_smiles"
                ],
                "n_fragments",
            ]
            .value_counts()
            .sort_index()
        )

    return out


# ============================================================
# Morgan fingerprints
# ============================================================

def generate_morgan_fingerprints(
    df,
    radius=2,
    fp_size=2048,
):
    """
    Generate binary Morgan fingerprints.

    radius=2 corresponds to ECFP4-style
    neighborhoods.

    Only valid SMILES are fingerprinted.
    """

    print("\n========================================")
    print("Generating Morgan fingerprints")
    print("========================================")

    print(
        f"Radius: {radius}"
    )

    print(
        f"Fingerprint size: {fp_size}"
    )

    valid = df[
        df[
            "valid_smiles"
        ]
    ].copy()

    generator = GetMorganGenerator(
        radius=radius,
        fpSize=fp_size,
        includeChirality=True,
    )

    X = np.zeros(
        (
            len(valid),
            fp_size,
        ),
        dtype=np.uint8,
    )

    successful_rows = []

    failed_indices = []

    for df_i, row in valid.iterrows():

        try:

            mol = Chem.MolFromSmiles(
                row[
                    "canonical_smiles"
                ],
                sanitize=True,
            )

            if mol is None:

                raise ValueError(
                    "Canonical SMILES could not be parsed."
                )

            fp = (
                generator
                .GetFingerprint(
                    mol
                )
            )

            arr = np.zeros(
                (
                    fp_size,
                ),
                dtype=np.uint8,
            )

            DataStructs.ConvertToNumpyArray(
                fp,
                arr,
            )

            X[
                len(
                    successful_rows
                ),
                :
            ] = arr

            successful_rows.append(
                df_i
            )

        except Exception as e:

            print(
                f"Fingerprint failed for "
                f"{row['dname']} "
                f"({row['id']}): {e}"
            )

            failed_indices.append(
                df_i
            )

    # Trim unused preallocated rows.
    X = X[
        :len(
            successful_rows
        ),
        :
    ]

    valid = df.loc[
        successful_rows
    ].copy()

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    feature_names = [
        f"morgan_{i:04d}"
        for i in range(
            fp_size
        )
    ]

    fingerprints = pd.DataFrame(
        X,
        index=valid[
            "id"
        ].astype(str),
        columns=feature_names,
    )

    fingerprints.index.name = (
        "drug_id"
    )

    print(
        f"\nFingerprint matrix: "
        f"{fingerprints.shape}"
    )

    print(
        "Mean active bits per drug:",
        round(
            float(
                X.sum(
                    axis=1
                ).mean()
            ),
            2,
        ),
    )

    print(
        "Median active bits per drug:",
        round(
            float(
                np.median(
                    X.sum(
                        axis=1
                    )
                )
            ),
            2,
        ),
    )

    return (
        fingerprints,
        valid,
        failed_indices,
    )


# ============================================================
# Fingerprint QC
# ============================================================

def make_fingerprint_qc(
    fingerprints,
    valid_drugs,
):

    X = fingerprints.to_numpy(
        dtype=np.uint8
    )

    active_bits = X.sum(
        axis=1
    )

    fraction_active = (
        active_bits
        / X.shape[1]
    )

    qc = pd.DataFrame(
        {

            "drug_id":
                valid_drugs[
                    "id"
                ].astype(
                    str
                ).values,

            "dname":
                valid_drugs[
                    "dname"
                ].values,

            "smiles_cleanup_status":
                valid_drugs[
                    "smiles_cleanup_status"
                ].values,

            "n_active_bits":
                active_bits,

            "fraction_active_bits":
                fraction_active,

            "n_atoms":
                valid_drugs[
                    "n_atoms"
                ].values,

            "n_heavy_atoms":
                valid_drugs[
                    "n_heavy_atoms"
                ].values,

            "n_fragments":
                valid_drugs[
                    "n_fragments"
                ].values,

            "molecular_weight_rdkit":
                valid_drugs[
                    "molecular_weight_rdkit"
                ].values,

            "formal_charge":
                valid_drugs[
                    "formal_charge"
                ].values,
        }
    )

    return qc


# ============================================================
# Duplicate structure QC
# ============================================================

def find_duplicate_structures(df):

    valid = df[
        df[
            "valid_smiles"
        ]
    ].copy()

    counts = (
        valid[
            "canonical_smiles"
        ]
        .value_counts()
    )

    duplicate_smiles = set(
        counts[
            counts > 1
        ].index
    )

    duplicates = valid[
        valid[
            "canonical_smiles"
        ].isin(
            duplicate_smiles
        )
    ].copy()

    if len(
        duplicates
    ) > 0:

        duplicates[
            "n_drugcomb_ids_same_structure"
        ] = (
            duplicates[
                "canonical_smiles"
            ]
            .map(
                counts
            )
        )

        duplicates = (
            duplicates
            .sort_values(
                [
                    "canonical_smiles",
                    "id",
                ]
            )
        )

    return duplicates


# ============================================================
# Bit-level QC
# ============================================================

def make_bit_qc(
    fingerprints
):

    X = fingerprints.to_numpy(
        dtype=np.uint8
    )

    frequency = X.mean(
        axis=0
    )

    count = X.sum(
        axis=0
    )

    qc = pd.DataFrame(
        {

            "feature":
                fingerprints.columns,

            "n_drugs_active":
                count,

            "fraction_drugs_active":
                frequency,
        }
    )

    return qc


# ============================================================
# SMILES cleanup QC
# ============================================================

def make_cleanup_qc(
    processed
):

    qc = (
        processed[
            "smiles_cleanup_status"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "smiles_cleanup_status"
        )
        .reset_index(
            name="n_drugs"
        )
    )

    qc[
        "fraction"
    ] = (
        qc[
            "n_drugs"
        ]
        / len(
            processed
        )
    )

    return qc


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
    )

    parser.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTDIR),
    )

    parser.add_argument(
        "--radius",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--fp-size",
        "--fp_size",
        dest="fp_size",
        type=int,
        default=2048,
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
    # 1. Load DrugComb identifiers
    # ========================================================

    drugs = load_drugs(
        args.input
    )

    # ========================================================
    # 2. SMILES validation + cleanup + canonicalization
    # ========================================================

    processed = canonicalize_table(
        drugs
    )

    processed.to_csv(
        outdir
        / "DrugComb_SMILES_canonical.tsv",
        sep="\t",
        index=False,
    )

    # ========================================================
    # 3. SMILES cleanup QC
    # ========================================================

    cleanup_qc = make_cleanup_qc(
        processed
    )

    cleanup_qc.to_csv(
        outdir
        / "QC_SMILES_cleanup.tsv",
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # Save repaired rows
    # --------------------------------------------------------

    repaired = processed[
        processed[
            "smiles_cleanup_status"
        ].isin(
            [
                "removed_trailing_semicolon",
                "selected_first_valid_semicolon_candidate",
            ]
        )
    ].copy()

    repaired.to_csv(
        outdir
        / "repaired_smiles.tsv",
        sep="\t",
        index=False,
    )

    # ========================================================
    # 4. Invalid / unresolved structures
    # ========================================================

    invalid = processed[
        ~processed[
            "valid_smiles"
        ]
    ].copy()

    invalid.to_csv(
        outdir
        / "invalid_smiles.tsv",
        sep="\t",
        index=False,
    )

    # ========================================================
    # 5. Duplicate canonical structures
    # ========================================================

    duplicates = (
        find_duplicate_structures(
            processed
        )
    )

    duplicates.to_csv(
        outdir
        / "duplicate_structures.tsv",
        sep="\t",
        index=False,
    )

    print("\n========================================")
    print("Duplicate structure QC")
    print("========================================")

    if len(
        duplicates
    ) == 0:

        print(
            "No duplicate canonical structures."
        )

    else:

        print(
            f"Rows involved in duplicate structures: "
            f"{len(duplicates):,}"
        )

        print(
            f"Distinct duplicated structures: "
            f"{duplicates['canonical_smiles'].nunique():,}"
        )

    # ========================================================
    # 6. Morgan fingerprints
    # ========================================================

    (
        fingerprints,
        valid_drugs,
        fp_failures,
    ) = generate_morgan_fingerprints(
        processed,
        radius=args.radius,
        fp_size=args.fp_size,
    )

    # ========================================================
    # 7. Save fingerprint matrix
    # ========================================================

    fingerprints.to_csv(
        outdir
        / "Morgan_ECFP4_2048.csv.gz",
        compression="gzip",
    )

    # --------------------------------------------------------
    # NumPy compressed representation
    # --------------------------------------------------------

    np.savez_compressed(
        outdir
        / "Morgan_ECFP4_2048.npz",

        X=fingerprints.to_numpy(
            dtype=np.uint8
        ),

        drug_id=valid_drugs[
            "id"
        ].astype(
            str
        ).to_numpy(),

        dname=valid_drugs[
            "dname"
        ].astype(
            str
        ).to_numpy(),

        canonical_smiles=valid_drugs[
            "canonical_smiles"
        ].astype(
            str
        ).to_numpy(),

        feature_names=np.asarray(
            fingerprints.columns,
            dtype=str,
        ),
    )

    # ========================================================
    # 8. Feature order
    # ========================================================

    feature_order = pd.DataFrame(
        {

            "feature_index":
                np.arange(
                    args.fp_size
                ),

            "feature":
                fingerprints.columns,
        }
    )

    feature_order.to_csv(
        outdir
        / "Morgan_feature_order.tsv",
        sep="\t",
        index=False,
    )

    # ========================================================
    # 9. Per-drug QC
    # ========================================================

    fingerprint_qc = (
        make_fingerprint_qc(
            fingerprints,
            valid_drugs,
        )
    )

    fingerprint_qc.to_csv(
        outdir
        / "QC_Morgan_per_drug.tsv",
        sep="\t",
        index=False,
    )

    # ========================================================
    # 10. Bit-level QC
    # ========================================================

    bit_qc = make_bit_qc(
        fingerprints
    )

    bit_qc.to_csv(
        outdir
        / "QC_Morgan_bits.tsv",
        sep="\t",
        index=False,
    )

    # ========================================================
    # 11. Valid drug lookup
    # ========================================================

    lookup_cols = [
        col
        for col in [
            "id",
            "dname",
            "chembl_id",
            "inchikey",
            "smiles",
            "cleaned_smiles",
            "alternative_smiles",
            "smiles_cleanup_status",
            "canonical_smiles",
        ]
        if col in valid_drugs.columns
    ]

    valid_drugs[
        lookup_cols
    ].to_csv(
        outdir
        / "drug_structure_lookup.tsv",
        sep="\t",
        index=False,
    )

    # ========================================================
    # 12. Summary
    # ========================================================

    n_valid = int(
        processed[
            "valid_smiles"
        ].sum()
    )

    n_invalid = int(
        (
            ~processed[
                "valid_smiles"
            ]
        ).sum()
    )

    n_trailing_repaired = int(
        (
            processed[
                "smiles_cleanup_status"
            ]
            == "removed_trailing_semicolon"
        ).sum()
    )

    n_semicolon_recovered = int(
        (
            processed[
                "smiles_cleanup_status"
            ]
            == "selected_first_valid_semicolon_candidate"
        ).sum()
    )

    n_missing = int(
        processed[
            "smiles_cleanup_status"
        ]
        .isin(
            [
                "missing",
                "empty",
            ]
        )
        .sum()
    )

    n_unresolved = int(
        (
            processed[
                "smiles_cleanup_status"
            ]
            == "unresolved"
        ).sum()
    )

    n_unique_structures = int(
        processed.loc[
            processed[
                "valid_smiles"
            ],
            "canonical_smiles",
        ]
        .nunique()
    )

    n_multifragment = int(
        (
            processed.loc[
                processed[
                    "valid_smiles"
                ],
                "n_fragments",
            ]
            > 1
        ).sum()
    )

    summary = {

        "input_file":
            (
                str(Path(args.input).resolve().relative_to(ROOT))
                if Path(args.input).resolve().is_relative_to(ROOT)
                else str(Path(args.input).resolve())
            ),

        "n_input_rows":
            int(
                len(
                    processed
                )
            ),

        "n_unique_drug_ids":
            int(
                processed[
                    "id"
                ].nunique()
            ),

        "n_valid_smiles":
            n_valid,

        "n_invalid_smiles":
            n_invalid,

        "n_repaired_trailing_semicolon":
            n_trailing_repaired,

        "n_recovered_semicolon_alternatives":
            n_semicolon_recovered,

        "n_missing_or_empty_smiles":
            n_missing,

        "n_unresolved_smiles":
            n_unresolved,

        "n_unique_canonical_structures":
            n_unique_structures,

        "n_multifragment_structures":
            n_multifragment,

        "n_duplicate_structure_rows":
            int(
                len(
                    duplicates
                )
            ),

        "n_fingerprint_failures":
            int(
                len(
                    fp_failures
                )
            ),

        "fingerprint_type":
            "Morgan / ECFP4-style",

        "morgan_radius":
            int(
                args.radius
            ),

        "fingerprint_size":
            int(
                args.fp_size
            ),

        "include_chirality":
            True,

        "salt_or_fragment_removal":
            False,

        "largest_fragment_selection":
            False,

        "neutralization":
            False,

        "canonical_smiles_stereochemistry":
            True,

        "semicolon_alternative_resolution":
            True,

        "semicolon_resolution_rule":
            "first valid candidate",

        "mean_active_bits":
            float(
                fingerprints
                .sum(
                    axis=1
                )
                .mean()
            ),

        "median_active_bits":
            float(
                fingerprints
                .sum(
                    axis=1
                )
                .median()
            ),
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
        f"\nInput drug rows: "
        f"{len(processed):,}"
    )

    print(
        f"Valid SMILES: "
        f"{n_valid:,}"
    )

    print(
        f"Invalid SMILES: "
        f"{n_invalid:,}"
    )

    print(
        f"Recovered by removing trailing semicolon: "
        f"{n_trailing_repaired:,}"
    )

    print(
        f"Recovered from semicolon-separated alternatives: "
        f"{n_semicolon_recovered:,}"
    )

    print(
        f"Missing / empty SMILES: "
        f"{n_missing:,}"
    )

    print(
        f"Still unresolved SMILES: "
        f"{n_unresolved:,}"
    )

    print(
        f"Unique canonical structures: "
        f"{n_unique_structures:,}"
    )

    print(
        f"Multi-fragment structures retained: "
        f"{n_multifragment:,}"
    )

    print(
        f"\nFingerprint matrix: "
        f"{fingerprints.shape}"
    )

    print(
        f"\nResults written to:\n"
        f"{outdir.resolve()}"
    )

    print(
        "\nMain files:"
    )

    print(
        "  DrugComb_SMILES_canonical.tsv"
    )

    print(
        "  Morgan_ECFP4_2048.csv.gz"
    )

    print(
        "  Morgan_ECFP4_2048.npz"
    )

    print(
        "  drug_structure_lookup.tsv"
    )

    print(
        "  repaired_smiles.tsv"
    )

    print(
        "  invalid_smiles.tsv"
    )

    print(
        "  duplicate_structures.tsv"
    )

    print(
        "  QC_SMILES_cleanup.tsv"
    )

    print(
        "  QC_Morgan_per_drug.tsv"
    )

    print(
        "  QC_Morgan_bits.tsv"
    )


if __name__ == "__main__":
    main()
