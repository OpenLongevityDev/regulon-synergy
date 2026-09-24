import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn


# ============================================================
# Paths
# ============================================================

ROOT = Path(
    os.environ.get("DRUG_SYNERGY_ROOT", Path(__file__).resolve().parents[1])
).resolve()

SMILES_DIR = ROOT / "smiles" / "smiles_results_morgan"
PPI_DIR = ROOT / "targets" / "target_ppi_results"

MORGAN_NPZ = (
    SMILES_DIR
    / "Morgan_ECFP4_2048.npz"
)

DIRECT_TARGET_NPZ = (
    PPI_DIR
    / "direct_target_binary.npz"
)

PPI_RWR_NPZ = (
    PPI_DIR
    / "ppi_propagated_rwr.npz"
)

PPI_LOOKUP = (
    PPI_DIR
    / "drug_target_ppi_lookup.tsv"
)


# ============================================================
# Robust NPZ loader
# ============================================================

def load_npz_matrix(
    path,
    preferred_keys=None,
):
    """
    Robust loader for three formats used in this project:

    1. scipy.sparse.save_npz()
    2. manually serialized CSR matrix using:
         data, indices, indptr, shape
    3. ordinary NumPy archive containing a dense 2D matrix
         e.g. X
    """

    path = Path(path)

    # ========================================================
    # 1. Standard scipy sparse NPZ
    # ========================================================

    try:

        matrix = sp.load_npz(
            path
        )

        print(
            f"Detected scipy sparse NPZ: {path.name}"
        )

        print(
            "Matrix:",
            matrix.shape,
            matrix.dtype,
            "nnz =",
            matrix.nnz
        )

        return matrix.tocsr(), None

    except (ValueError, KeyError):
        pass


    # ========================================================
    # 2/3. Ordinary NumPy archive
    # ========================================================

    archive = np.load(
        path,
        allow_pickle=True
    )

    print(
        f"Detected NumPy NPZ: {path.name}"
    )

    print(
        "Keys:",
        archive.files
    )


    # ========================================================
    # 2. Manually serialized CSR matrix
    # ========================================================

    csr_keys = {
        "data",
        "indices",
        "indptr",
        "shape",
    }


    if csr_keys.issubset(
        set(
            archive.files
        )
    ):

        print(
            "Detected manually serialized CSR matrix."
        )


        data = archive[
            "data"
        ]

        indices = archive[
            "indices"
        ]

        indptr = archive[
            "indptr"
        ]

        shape_raw = archive[
            "shape"
        ]


        shape = tuple(
            int(x)
            for x in np.asarray(
                shape_raw
            ).ravel()
        )


        if len(
            shape
        ) != 2:

            raise ValueError(
                f"Invalid sparse matrix shape in {path}: "
                f"{shape}"
            )


        matrix = sp.csr_matrix(
            (
                data,
                indices,
                indptr,
            ),
            shape=shape,
        )


        print(
            "Reconstructed CSR matrix:",
            matrix.shape,
            matrix.dtype
        )

        print(
            "Non-zero entries:",
            f"{matrix.nnz:,}"
        )


        return matrix, archive


    # ========================================================
    # 3. Dense NumPy matrix
    # ========================================================

    if preferred_keys is None:

        preferred_keys = [
            "X",
            "matrix",
        ]


    for key in preferred_keys:

        if key not in archive.files:
            continue


        arr = archive[
            key
        ]


        if (
            isinstance(
                arr,
                np.ndarray
            )
            and arr.ndim == 2
        ):

            print(
                f"Using matrix key '{key}':",
                arr.shape,
                arr.dtype
            )


            return arr, archive


    # ========================================================
    # Last resort: search for a unique 2D array
    # ========================================================

    candidates = []


    for key in archive.files:

        try:

            arr = archive[
                key
            ]

        except Exception:

            continue


        if (
            isinstance(
                arr,
                np.ndarray
            )
            and arr.ndim == 2
        ):

            candidates.append(
                key
            )


    if len(
        candidates
    ) == 1:

        key = candidates[
            0
        ]

        arr = archive[
            key
        ]


        print(
            f"Using sole 2D array '{key}':",
            arr.shape,
            arr.dtype
        )


        return arr, archive


    raise ValueError(
        f"\nCould not determine matrix representation in:\n"
        f"{path}\n\n"
        f"Available keys:\n"
        f"{archive.files}\n\n"
        f"2D candidates:\n"
        f"{candidates}"
    )

# ============================================================
# Generic matrix → tensor helper
# ============================================================

def matrix_rows_to_tensor(
    matrix,
    rows,
    device,
):

    x = matrix[
        rows
    ]

    if sp.issparse(
        x
    ):

        x = x.toarray()


    x = np.asarray(
        x,
        dtype=np.float32
    )


    return torch.from_numpy(
        x
    ).to(
        device
    )


# ============================================================
# Chemical branch
# ============================================================

class ChemicalEncoder(
    nn.Module
):

    def __init__(
        self,
        input_dim=2048,
        hidden_dim=512,
        output_dim=128,
        dropout=0.20,
    ):

        super().__init__()


        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                output_dim
            ),

            nn.LayerNorm(
                output_dim
            ),

            nn.GELU(),
        )


    def forward(
        self,
        x
    ):

        return self.net(
            x
        )


# ============================================================
# Direct-target branch
# ============================================================

class TargetEncoder(
    nn.Module
):

    def __init__(
        self,
        input_dim=8245,
        hidden_dim=256,
        output_dim=32,
        dropout=0.20,
    ):

        super().__init__()


        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                output_dim
            ),

            nn.LayerNorm(
                output_dim
            ),

            nn.GELU(),
        )


    def forward(
        self,
        x
    ):

        return self.net(
            x
        )


# ============================================================
# PPI branch
# ============================================================

class PPIEncoder(
    nn.Module
):

    def __init__(
        self,
        input_dim=8245,
        hidden_dim=512,
        output_dim=128,
        dropout=0.20,
    ):

        super().__init__()


        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                output_dim
            ),

            nn.LayerNorm(
                output_dim
            ),

            nn.GELU(),
        )


    def forward(
        self,
        x
    ):

        return self.net(
            x
        )


# ============================================================
# Intrinsic drug encoder
# ============================================================

class IntrinsicDrugEncoder(
    nn.Module
):

    def __init__(
        self,

        morgan_dim=2048,
        target_dim=8245,
        ppi_dim=8245,

        chemical_hidden=512,
        target_hidden=256,
        ppi_hidden=512,

        chemical_out=128,
        target_out=32,
        ppi_out=128,

        availability_dim=3,

        fusion_hidden=256,
        output_dim=128,

        dropout=0.20,
    ):

        super().__init__()


        self.chemical_encoder = ChemicalEncoder(

            input_dim=morgan_dim,
            hidden_dim=chemical_hidden,
            output_dim=chemical_out,
            dropout=dropout,

        )


        self.target_encoder = TargetEncoder(

            input_dim=target_dim,
            hidden_dim=target_hidden,
            output_dim=target_out,
            dropout=dropout,

        )


        self.ppi_encoder = PPIEncoder(

            input_dim=ppi_dim,
            hidden_dim=ppi_hidden,
            output_dim=ppi_out,
            dropout=dropout,

        )


        fusion_input_dim = (

            chemical_out
            + target_out
            + ppi_out
            + availability_dim

        )


        self.fusion = nn.Sequential(

            nn.Linear(
                fusion_input_dim,
                fusion_hidden
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                fusion_hidden,
                output_dim
            ),

            nn.LayerNorm(
                output_dim
            ),

            nn.GELU(),
        )


    def forward(
        self,
        morgan,
        direct_target,
        ppi_profile,
        availability,
        return_aux=False,
    ):

        chemical = self.chemical_encoder(
            morgan
        )

        target = self.target_encoder(
            direct_target
        )

        ppi = self.ppi_encoder(
            ppi_profile
        )


        fused = torch.cat(

            [
                chemical,
                target,
                ppi,
                availability,
            ],

            dim=-1

        )


        intrinsic = self.fusion(
            fused
        )


        if return_aux:

            return {

                "embedding":
                    intrinsic,

                "chemical_embedding":
                    chemical,

                "target_embedding":
                    target,

                "ppi_embedding":
                    ppi,

                "availability":
                    availability,
            }


        return intrinsic


# ============================================================
# Availability encoding
# ============================================================

def availability_one_hot(status_series):
    """
    Encode target/PPI availability:

        no_known_targets
            -> [1, 0, 0]

        targets_known_but_none_in_huri
            -> [0, 1, 0]

        targets_mapped_to_huri
            -> [0, 0, 1]
    """

    mapping = {
        "no_known_targets": 0,
        "targets_known_but_none_in_huri": 1,
        "targets_mapped_to_huri": 2,
    }

    status_series = (
        status_series
        .astype(str)
        .str.strip()
    )

    result = np.zeros(
        (
            len(status_series),
            3
        ),
        dtype=np.float32
    )

    unknown = []

    for i, status in enumerate(status_series):

        if status not in mapping:
            unknown.append(status)
            continue

        result[
            i,
            mapping[status]
        ] = 1.0

    if unknown:

        raise ValueError(
            "Unrecognized target/PPI status values:\n"
            + "\n".join(
                sorted(set(unknown))
            )
        )

    return result

# ============================================================
# Main diagnostic
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(

        "cuda"
        if torch.cuda.is_available()
        else "cpu"

    )


    print(
        "Device:",
        device
    )


    # ========================================================
    # Morgan
    # ========================================================

    print(
        "\nLoading Morgan fingerprints..."
    )


    morgan, morgan_archive = load_npz_matrix(

        MORGAN_NPZ,

        preferred_keys=[
            "X",
            "fingerprints",
            "morgan",
            "matrix",
        ],

    )


    print(
        "Morgan:",
        morgan.shape,
        morgan.dtype
    )


    if morgan_archive is None:

        raise ValueError(
            "Morgan file loaded as scipy sparse format, "
            "but embedded drug_id is required for alignment."
        )


    if "drug_id" not in morgan_archive.files:

        raise ValueError(

            "'drug_id' not found in Morgan archive.\n"
            f"Available keys: {morgan_archive.files}"

        )


    morgan_ids = (

        morgan_archive[
            "drug_id"
        ]
        .astype(str)

    )


    print(
        "Morgan drug IDs:",
        len(
            morgan_ids
        )
    )


    print(
        "Unique Morgan drug IDs:",
        len(
            set(
                morgan_ids
            )
        )
    )


    if len(
        morgan_ids
    ) != morgan.shape[
        0
    ]:

        raise ValueError(
            "Morgan drug_id length does not match matrix rows."
        )


    # ========================================================
    # Direct targets
    # ========================================================

    print(
        "\nLoading direct-target matrix..."
    )


    direct, direct_archive = load_npz_matrix(

        DIRECT_TARGET_NPZ,

        preferred_keys=[
            "X",
            "direct_target_binary",
            "direct_targets",
            "target_binary",
            "matrix",
        ],

    )


    print(
        "Direct targets:",
        direct.shape,
        direct.dtype
    )


    # ========================================================
    # PPI
    # ========================================================

    print(
        "\nLoading PPI RWR matrix..."
    )


    ppi, ppi_archive = load_npz_matrix(

        PPI_RWR_NPZ,

        preferred_keys=[
            "X",
            "ppi_propagated_rwr",
            "rwr",
            "ppi",
            "matrix",
        ],

    )


    print(
        "PPI RWR:",
        ppi.shape,
        ppi.dtype
    )


    # ========================================================
    # Structural matrix checks
    # ========================================================

    if direct.ndim != 2:

        raise ValueError(
            f"Direct-target matrix must be 2D. "
            f"Shape: {direct.shape}"
        )


    if ppi.ndim != 2:

        raise ValueError(
            f"PPI matrix must be 2D. "
            f"Shape: {ppi.shape}"
        )


    if direct.shape[
        0
    ] != ppi.shape[
        0
    ]:

        raise ValueError(

            "Direct-target and PPI matrices have "
            "different numbers of drugs:\n"
            f"direct = {direct.shape}\n"
            f"ppi    = {ppi.shape}"

        )


    if direct.shape[
        1
    ] != ppi.shape[
        1
    ]:

        raise ValueError(

            "Direct-target and PPI matrices use "
            "different protein dimensions:\n"
            f"direct = {direct.shape}\n"
            f"ppi    = {ppi.shape}"

        )


    # ========================================================
    # PPI lookup
    # ========================================================

    print(
        "\nLoading target/PPI lookup..."
    )


    ppi_lookup = pd.read_csv(

        PPI_LOOKUP,
        sep="\t",
        low_memory=False

    )


    print(
        "PPI lookup:",
        ppi_lookup.shape
    )


    print(
        "\nPPI lookup columns:"
    )


    print(
        ppi_lookup.columns.tolist()
    )


    # ========================================================
    # Find DrugComb ID column
    # ========================================================

    ppi_id_col = None


    for candidate in [

        "drugcomb_id",
        "id",
        "drug_id",

    ]:

        if candidate in ppi_lookup.columns:

            ppi_id_col = candidate

            break


    if ppi_id_col is None:

        raise ValueError(

            "Could not identify DrugComb ID column "
            "in drug_target_ppi_lookup.tsv."

        )


    print(
        "\nPPI DrugComb ID column:",
        ppi_id_col
    )


    # ========================================================
    # Lookup row-count checks
    # ========================================================

    if len(
        ppi_lookup
    ) != direct.shape[
        0
    ]:

        raise ValueError(

            "PPI lookup row count does not match "
            "direct-target matrix:\n"
            f"lookup = {len(ppi_lookup)}\n"
            f"direct = {direct.shape[0]}"

        )


    if len(
        ppi_lookup
    ) != ppi.shape[
        0
    ]:

        raise ValueError(

            "PPI lookup row count does not match "
            "RWR matrix:\n"
            f"lookup = {len(ppi_lookup)}\n"
            f"ppi    = {ppi.shape[0]}"

        )


    lookup_ids = (

        ppi_lookup[
            ppi_id_col
        ]
        .astype(str)
        .to_numpy()

    )


    print(
        "PPI/direct drug rows:",
        len(
            lookup_ids
        )
    )


    print(
        "Unique PPI/direct drug IDs:",
        len(
            set(
                lookup_ids
            )
        )
    )


    # ========================================================
    # Check embedded IDs in direct/PPI archives
    # ========================================================

    for matrix_name, archive in [

        (
            "direct targets",
            direct_archive
        ),

        (
            "PPI RWR",
            ppi_archive
        ),

    ]:

        if archive is None:

            print(
                f"\n{matrix_name}: scipy sparse archive; "
                f"no embedded ID check available."
            )

            continue


        possible_id_keys = [

            "drug_id",
            "drugcomb_id",
            "id",

        ]


        embedded_id_key = None


        for key in possible_id_keys:

            if key in archive.files:

                embedded_id_key = key

                break


        if embedded_id_key is None:

            print(
                f"\n{matrix_name}: no embedded drug ID array; "
                f"using lookup row order."
            )

            continue


        embedded_ids = (

            archive[
                embedded_id_key
            ]
            .astype(str)

        )


        print(
            f"\n{matrix_name} embedded ID key:",
            embedded_id_key
        )


        print(
            f"{matrix_name} embedded IDs:",
            len(
                embedded_ids
            )
        )


        if len(
            embedded_ids
        ) != len(
            lookup_ids
        ):

            raise ValueError(

                f"{matrix_name}: embedded ID count "
                f"does not match lookup.\n"
                f"embedded = {len(embedded_ids)}\n"
                f"lookup   = {len(lookup_ids)}"

            )


        if not np.array_equal(

            embedded_ids,
            lookup_ids

        ):

            mismatch = np.where(

                embedded_ids
                != lookup_ids

            )[
                0
            ]


            first = int(
                mismatch[
                    0
                ]
            )


            raise ValueError(

                f"{matrix_name}: embedded drug row order "
                f"does NOT match lookup.\n\n"
                f"First mismatch at row {first}:\n"
                f"embedded = {embedded_ids[first]}\n"
                f"lookup   = {lookup_ids[first]}"

            )


        print(
            f"{matrix_name}: embedded drug ID order matches lookup ✓"
        )


    # ========================================================
    # Build row maps
    # ========================================================

    morgan_id_to_row = {

        drug_id: i

        for i, drug_id
        in enumerate(
            morgan_ids
        )

    }


    ppi_id_to_row = {

        drug_id: i

        for i, drug_id
        in enumerate(
            lookup_ids
        )

    }


    # ========================================================
    # Shared drugs
    # ========================================================

    common_ids = [

        drug_id

        for drug_id
        in lookup_ids

        if drug_id
        in morgan_id_to_row

    ]


    print(
        "\nDrugs present in both Morgan and target/PPI:",
        f"{len(common_ids):,}"
    )


    if len(
        common_ids
    ) == 0:

        print(
            "\nFirst Morgan IDs:"
        )

        print(
            morgan_ids[
                :10
            ]
        )


        print(
            "\nFirst target/PPI IDs:"
        )

        print(
            lookup_ids[
                :10
            ]
        )


        raise ValueError(

            "No shared DrugComb IDs between "
            "Morgan and target/PPI data."

        )


    # ========================================================
    # Availability status column
    # ========================================================

    status_col = None


    for candidate in [

        "drug_target_status",
        "target_status",
        "status",

    ]:

        if candidate in ppi_lookup.columns:

            status_col = candidate

            break


    if status_col is None:

        raise ValueError(

            "Could not identify target/PPI "
            "availability-status column."

        )


    print(
        "\nAvailability status column:",
        status_col
    )


    print(
        "\nAvailability status counts:"
    )


    print(

        ppi_lookup[
            status_col
        ]
        .value_counts(
            dropna=False
        )

    )


    # ========================================================
    # Diagnostic batch
    # ========================================================

    batch_size = min(

        32,
        len(
            common_ids
        )

    )


    batch_ids = common_ids[
        :batch_size
    ]


    morgan_rows = [

        morgan_id_to_row[
            drug_id
        ]

        for drug_id
        in batch_ids

    ]


    ppi_rows = [

        ppi_id_to_row[
            drug_id
        ]

        for drug_id
        in batch_ids

    ]


    # ========================================================
    # Convert real data to tensors
    # ========================================================

    morgan_x = matrix_rows_to_tensor(

        morgan,
        morgan_rows,
        device

    )


    direct_x = matrix_rows_to_tensor(

        direct,
        ppi_rows,
        device

    )


    ppi_x = matrix_rows_to_tensor(

        ppi,
        ppi_rows,
        device

    )


    # ========================================================
    # Availability vectors
    # ========================================================

    batch_status = (

        ppi_lookup
        .iloc[
            ppi_rows
        ][
            status_col
        ]

    )


    availability_np = availability_one_hot(
        batch_status
    )


    availability_x = torch.from_numpy(

        availability_np

    ).to(
        device
    )


    # ========================================================
    # Initialize model
    # ========================================================

    model = IntrinsicDrugEncoder(

        morgan_dim=morgan.shape[
            1
        ],

        target_dim=direct.shape[
            1
        ],

        ppi_dim=ppi.shape[
            1
        ],

        chemical_hidden=512,

        target_hidden=256,

        ppi_hidden=512,

        chemical_out=128,

        target_out=32,

        ppi_out=128,

        availability_dim=3,

        fusion_hidden=256,

        output_dim=128,

        dropout=0.20,

    ).to(
        device
    )


    # ========================================================
    # Forward pass
    # ========================================================

    model.eval()


    with torch.no_grad():

        out = model(

            morgan=morgan_x,

            direct_target=direct_x,

            ppi_profile=ppi_x,

            availability=availability_x,

            return_aux=True,

        )


    # ========================================================
    # Diagnostics
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "INTRINSIC DRUG ENCODER DIAGNOSTIC"
    )

    print(
        "========================================"
    )


    print(
        "\nInput shapes:"
    )


    print(
        "Morgan:",
        tuple(
            morgan_x.shape
        )
    )


    print(
        "Direct targets:",
        tuple(
            direct_x.shape
        )
    )


    print(
        "PPI:",
        tuple(
            ppi_x.shape
        )
    )


    print(
        "Availability:",
        tuple(
            availability_x.shape
        )
    )


    print(
        "\nBranch output shapes:"
    )


    print(
        "Chemical:",
        tuple(
            out[
                "chemical_embedding"
            ].shape
        )
    )


    print(
        "Targets:",
        tuple(
            out[
                "target_embedding"
            ].shape
        )
    )


    print(
        "PPI:",
        tuple(
            out[
                "ppi_embedding"
            ].shape
        )
    )


    print(
        "\nFinal intrinsic embedding:"
    )


    print(
        tuple(
            out[
                "embedding"
            ].shape
        )
    )


    # ========================================================
    # Numeric checks
    # ========================================================

    embedding = out[
        "embedding"
    ]


    print(
        "\nEmbedding diagnostics:"
    )


    print(
        "mean:",
        embedding.mean().item()
    )


    print(
        "std:",
        embedding.std().item()
    )


    print(
        "min:",
        embedding.min().item()
    )


    print(
        "max:",
        embedding.max().item()
    )


    if not torch.isfinite(
        embedding
    ).all():

        raise ValueError(
            "Embedding contains NaN or Inf."
        )


    # ========================================================
    # Availability sanity check
    # ========================================================

    row_sums = availability_np.sum(
        axis=1
    )


    if not np.allclose(

        row_sums,
        1.0

    ):

        raise ValueError(

            "Availability vectors are not "
            "valid one-hot vectors."

        )


    # ========================================================
    # Parameter count
    # ========================================================

    total_params = sum(

        p.numel()

        for p
        in model.parameters()

    )


    trainable_params = sum(

        p.numel()

        for p
        in model.parameters()

        if p.requires_grad

    )


    print(
        "\nTotal parameters:",
        f"{total_params:,}"
    )


    print(
        "Trainable parameters:",
        f"{trainable_params:,}"
    )


    # ========================================================
    # Diagnostic IDs/status
    # ========================================================

    diagnostic_df = pd.DataFrame(

        {

            "drugcomb_id":
                batch_ids,

            "status":
                batch_status
                .astype(str)
                .to_numpy(),

            "availability_no_known_targets":
                availability_np[
                    :,
                    0
                ],

            "availability_not_in_huri":
                availability_np[
                    :,
                    1
                ],

            "availability_in_huri":
                availability_np[
                    :,
                    2
                ],

        }

    )


    print(
        "\nDiagnostic drugs:"
    )


    print(

        diagnostic_df.to_string(
            index=False
        )

    )


    print(
        "\nAll diagnostic checks passed."
    )


    # ========================================================
    # Verify protein-column alignment
    # ========================================================

    if (
        direct_archive is not None
        and ppi_archive is not None
        and "protein_id" in direct_archive.files
        and "protein_id" in ppi_archive.files
    ):

        direct_proteins = (
            direct_archive["protein_id"]
            .astype(str)
        )

        ppi_proteins = (
            ppi_archive["protein_id"]
            .astype(str)
        )

        if len(direct_proteins) != direct.shape[1]:
            raise ValueError(
                "direct_target_binary protein_id length "
                "does not match matrix columns."
            )

        if len(ppi_proteins) != ppi.shape[1]:
            raise ValueError(
                "PPI protein_id length "
                "does not match matrix columns."
            )

        if not np.array_equal(
            direct_proteins,
            ppi_proteins
        ):
            raise ValueError(
                "Direct-target and PPI protein column "
                "orders do NOT match."
            )

        print(
            "\nDirect-target and PPI protein "
            "column order matches ✓"
        )
