#!/usr/bin/env python3

"""
synergy_model.py

End-to-end multimodal architecture for 3-class drug-combination response
prediction.

Core modules
------------
1. Intrinsic drug encoder
   - Morgan fingerprint
   - direct targets
   - HuRI PPI RWR
   - target/PPI availability state

2. Biological context encoder
   - L1000 landmark basal expression
   - Hallmark pathway activity
   - regulon activity

3. Drug x context conditioner
   - FiLM
   - explicit interactions
   - gated residual

4. Symmetric drug-pair encoder
   - sum
   - absolute difference
   - elementwise product
   - biological context

5. 3-class classifier
   - antagonism
   - no_interaction
   - synergy

6. Optional LINCS L1000 auxiliary decoder
   - predicts an observed single-agent perturbational signature
   - NOT required at inference
   - disabled unless use_l1000_aux=True


IMPORTANT
---------
This file defines architecture only.

Do not globally precompute learned 128-d embeddings before training.
The intrinsic/context encoders and conditioner should be trained jointly
with the synergy objective.

Normalization of raw/context inputs must be fitted on training data only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


# ============================================================
# Constants
# ============================================================

N_CLASSES = 3

CLASS_NAMES = [
    "antagonism",
    "no_interaction",
    "synergy",
]


# ============================================================
# Helper
# ============================================================

def _unwrap_embedding(
    output: Any,
    preferred_keys: Tuple[str, ...] = (
        "embedding",
        "context_embedding",
        "drug_embedding",
        "intrinsic_embedding",
        "context_aware_embedding",
        "output",
    ),
) -> torch.Tensor:
    """
    Make the integration layer tolerant of the existing modules returning
    either:
      - Tensor
      - tuple/list where first element is Tensor
      - dict containing an embedding

    This lets us join previously written modules without rewriting them.
    """

    if torch.is_tensor(output):
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item

        raise TypeError(
            "Module returned tuple/list but no Tensor was found."
        )

    if isinstance(output, dict):
        for key in preferred_keys:
            if key in output and torch.is_tensor(output[key]):
                return output[key]

        tensor_values = [
            value
            for value in output.values()
            if torch.is_tensor(value)
        ]

        if len(tensor_values) == 1:
            return tensor_values[0]

        raise TypeError(
            "Module returned a dict, but embedding Tensor "
            "could not be identified uniquely. "
            f"Available keys: {list(output.keys())}"
        )

    raise TypeError(
        f"Unsupported module output type: {type(output)}"
    )


# ============================================================
# Symmetric pair representation
# ============================================================

class SymmetricPairEncoder(nn.Module):
    """
    Drug-order invariant pair representation.

    For context-aware drug embeddings a and b:

        sum  = a + b
        diff = |a - b|
        prod = a * b

    Then concatenate biological context c:

        pair = [sum, diff, prod, c]

    If all embeddings are 128-d:
        128 * 4 = 512 dimensions.

    Swapping Drug A and Drug B gives exactly the same representation.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = embedding_dim * 4

    def forward(
        self,
        drug_a_context: torch.Tensor,
        drug_b_context: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:

        if drug_a_context.shape != drug_b_context.shape:
            raise ValueError(
                "Drug A and Drug B context-aware embeddings "
                "must have identical shapes. "
                f"Got {drug_a_context.shape} and "
                f"{drug_b_context.shape}."
            )

        if (
            drug_a_context.shape[-1]
            != self.embedding_dim
        ):
            raise ValueError(
                f"Expected drug embedding dimension "
                f"{self.embedding_dim}, got "
                f"{drug_a_context.shape[-1]}."
            )

        if context.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected context dimension "
                f"{self.embedding_dim}, got "
                f"{context.shape[-1]}."
            )

        pair_sum = (
            drug_a_context
            + drug_b_context
        )

        pair_absdiff = torch.abs(
            drug_a_context
            - drug_b_context
        )

        pair_product = (
            drug_a_context
            * drug_b_context
        )

        pair = torch.cat(
            [
                pair_sum,
                pair_absdiff,
                pair_product,
                context,
            ],
            dim=-1,
        )

        return pair


# ============================================================
# Classification head
# ============================================================

class SynergyClassifier(nn.Module):
    """
    512 -> 256 -> 128 -> 3
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        n_classes: int = 3,
        dropout: float = 0.20,
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                embedding_dim,
            ),

            nn.LayerNorm(
                embedding_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                embedding_dim,
                n_classes,
            ),
        )

    def forward(
        self,
        pair_features: torch.Tensor,
    ) -> torch.Tensor:

        return self.net(
            pair_features
        )


# ============================================================
# Optional LINCS decoder
# ============================================================

class L1000AuxiliaryDecoder(nn.Module):
    """
    Optional auxiliary head.

    Predicts a 978-dimensional observed LINCS Level-5
    single-agent perturbational signature from a
    context-aware drug embedding.

        128 -> 256 -> 512 -> 978

    This is TRAINING supervision.

    It is NOT required at inference.

    IMPORTANT:
    Only use SINGLE-AGENT LINCS perturbations.
    Never feed combination signatures as predictors of
    combination synergy.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim_1: int = 256,
        hidden_dim_2: int = 512,
        output_dim: int = 978,
        dropout: float = 0.20,
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim_1,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim_1,
                hidden_dim_2,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim_2,
                output_dim,
            ),
        )

    def forward(
        self,
        context_aware_drug: torch.Tensor,
    ) -> torch.Tensor:

        return self.net(
            context_aware_drug
        )


# ============================================================
# Main multimodal model
# ============================================================

class DrugCombinationModel(nn.Module):
    """
    Full multimodal model.

    Existing trained-together components are supplied to this class:

        intrinsic_drug_encoder
        biological_context_encoder
        context_conditioner

    The SAME intrinsic drug encoder is used for A and B.
    The SAME context conditioner is used for A and B.

    This is important:
      Drug A and Drug B are not semantically different entities.
    """

    def __init__(
        self,
        intrinsic_drug_encoder: nn.Module,
        biological_context_encoder: nn.Module,
        context_conditioner: nn.Module,
        embedding_dim: int = 128,
        classifier_hidden_dim: int = 256,
        dropout: float = 0.20,
        use_l1000_aux: bool = False,
        l1000_dim: int = 978,
    ):
        super().__init__()

        self.intrinsic_drug_encoder = (
            intrinsic_drug_encoder
        )

        self.biological_context_encoder = (
            biological_context_encoder
        )

        self.context_conditioner = (
            context_conditioner
        )

        self.embedding_dim = embedding_dim

        self.pair_encoder = (
            SymmetricPairEncoder(
                embedding_dim=embedding_dim
            )
        )

        self.classifier = (
            SynergyClassifier(
                input_dim=embedding_dim * 4,
                hidden_dim=classifier_hidden_dim,
                embedding_dim=embedding_dim,
                n_classes=N_CLASSES,
                dropout=dropout,
            )
        )

        self.use_l1000_aux = (
            use_l1000_aux
        )

        if use_l1000_aux:

            self.l1000_decoder = (
                L1000AuxiliaryDecoder(
                    input_dim=embedding_dim,
                    output_dim=l1000_dim,
                    dropout=dropout,
                )
            )

        else:

            self.l1000_decoder = None


    # --------------------------------------------------------
    # Existing module adapters
    # --------------------------------------------------------

    def encode_drug(
        self,
        morgan: torch.Tensor,
        direct_targets: torch.Tensor,
        ppi: torch.Tensor,
        availability: torch.Tensor,
    ) -> torch.Tensor:
        """
        Intrinsic drug representation:

        Morgan + direct targets + PPI + availability
            -> 128-d intrinsic drug embedding
        """

        output = self.intrinsic_drug_encoder(
            morgan,
            direct_targets,
            ppi,
            availability,
        )

        embedding = _unwrap_embedding(
            output,
            preferred_keys=(
                "intrinsic_embedding",
                "drug_embedding",
                "embedding",
                "output",
            ),
        )

        return embedding


    def encode_context(
        self,
        expression: torch.Tensor,
        pathways: torch.Tensor,
        regulons: torch.Tensor,
    ) -> torch.Tensor:
        """
        Biological context:

        basal expression + pathway activity + regulon activity
            -> 128-d biological context embedding
        """

        output = self.biological_context_encoder(
            expression,
            pathways,
            regulons,
        )

        embedding = _unwrap_embedding(
            output,
            preferred_keys=(
                "context_embedding",
                "embedding",
                "output",
            ),
        )

        return embedding


    def condition_drug(
        self,
        drug_embedding: torch.Tensor,
        context_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Context-condition one intrinsic drug embedding.

        Exact implementation is delegated to the existing
        context_conditioning.py module:

            projection
            FiLM gamma/beta
            explicit interaction
            gated residual
            LayerNorm
        """

        output = self.context_conditioner(
            drug_embedding,
            context_embedding,
        )

        embedding = _unwrap_embedding(
            output,
            preferred_keys=(
                "context_aware_embedding",
                "conditioned_embedding",
                "drug_context_embedding",
                "embedding",
                "output",
            ),
        )

        return embedding


    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,

        # Drug A
        drug_a_morgan: torch.Tensor,
        drug_a_targets: torch.Tensor,
        drug_a_ppi: torch.Tensor,
        drug_a_availability: torch.Tensor,

        # Drug B
        drug_b_morgan: torch.Tensor,
        drug_b_targets: torch.Tensor,
        drug_b_ppi: torch.Tensor,
        drug_b_availability: torch.Tensor,

        # Biological context
        context_expression: torch.Tensor,
        context_pathways: torch.Tensor,
        context_regulons: torch.Tensor,

        # Optional flags
        return_embeddings: bool = False,
        predict_l1000_a: bool = False,
        predict_l1000_b: bool = False,

    ) -> Dict[str, torch.Tensor]:


        # ====================================================
        # 1. Biological context
        # ====================================================

        context = self.encode_context(
            expression=context_expression,
            pathways=context_pathways,
            regulons=context_regulons,
        )


        # ====================================================
        # 2. Intrinsic drug representations
        # ====================================================

        drug_a = self.encode_drug(
            morgan=drug_a_morgan,
            direct_targets=drug_a_targets,
            ppi=drug_a_ppi,
            availability=drug_a_availability,
        )


        drug_b = self.encode_drug(
            morgan=drug_b_morgan,
            direct_targets=drug_b_targets,
            ppi=drug_b_ppi,
            availability=drug_b_availability,
        )


        # ====================================================
        # 3. Context conditioning
        # ====================================================

        drug_a_context = self.condition_drug(
            drug_embedding=drug_a,
            context_embedding=context,
        )


        drug_b_context = self.condition_drug(
            drug_embedding=drug_b,
            context_embedding=context,
        )


        # ====================================================
        # 4. Symmetric pair representation
        # ====================================================

        pair_features = self.pair_encoder(
            drug_a_context=drug_a_context,
            drug_b_context=drug_b_context,
            context=context,
        )


        # ====================================================
        # 5. Synergy classification
        # ====================================================

        logits = self.classifier(
            pair_features
        )


        output = {
            "logits": logits,
        }


        # ====================================================
        # 6. Optional L1000 auxiliary predictions
        # ====================================================

        if self.use_l1000_aux:

            if predict_l1000_a:

                output[
                    "l1000_pred_a"
                ] = self.l1000_decoder(
                    drug_a_context
                )


            if predict_l1000_b:

                output[
                    "l1000_pred_b"
                ] = self.l1000_decoder(
                    drug_b_context
                )


        # ====================================================
        # Optional diagnostic embeddings
        # ====================================================

        if return_embeddings:

            output.update(
                {
                    "context_embedding":
                        context,

                    "drug_a_intrinsic":
                        drug_a,

                    "drug_b_intrinsic":
                        drug_b,

                    "drug_a_context":
                        drug_a_context,

                    "drug_b_context":
                        drug_b_context,

                    "pair_features":
                        pair_features,
                }
            )


        return output


# ============================================================
# Loss helper
# ============================================================

@dataclass
class LossOutput:

    total: torch.Tensor

    synergy: torch.Tensor

    l1000_a: Optional[
        torch.Tensor
    ] = None

    l1000_b: Optional[
        torch.Tensor
    ] = None


class MultiTaskLoss(nn.Module):
    """
    Main loss:

        L = L_synergy + lambda_L1000 * L_L1000

    L1000 is optional and masked sample-wise.

    A batch may contain:
      - synergy observations only
      - some observations with LINCS for A
      - some with LINCS for B
      - both
      - neither

    Missing LINCS supervision contributes zero loss rather than
    forcing fake zero vectors.
    """

    def __init__(
        self,
        class_weights: Optional[
            torch.Tensor
        ] = None,
        lambda_l1000: float = 0.10,
        l1000_loss: str = "mse",
    ):
        super().__init__()

        self.synergy_loss = (
            nn.CrossEntropyLoss(
                weight=class_weights
            )
        )

        self.lambda_l1000 = (
            lambda_l1000
        )

        if l1000_loss == "mse":

            self.l1000_loss = (
                nn.MSELoss()
            )

        elif l1000_loss == "smooth_l1":

            self.l1000_loss = (
                nn.SmoothL1Loss()
            )

        else:

            raise ValueError(
                "l1000_loss must be "
                "'mse' or 'smooth_l1'."
            )


    def _masked_l1000_loss(
        self,
        prediction: Optional[
            torch.Tensor
        ],
        target: Optional[
            torch.Tensor
        ],
        mask: Optional[
            torch.Tensor
        ],
    ) -> Optional[
        torch.Tensor
    ]:

        if (
            prediction is None
            or target is None
            or mask is None
        ):
            return None


        mask = mask.bool()


        if mask.sum() == 0:
            return None


        return self.l1000_loss(
            prediction[
                mask
            ],
            target[
                mask
            ],
        )


    def forward(
        self,
        model_output: Dict[
            str,
            torch.Tensor
        ],
        labels: torch.Tensor,

        l1000_target_a: Optional[
            torch.Tensor
        ] = None,

        l1000_mask_a: Optional[
            torch.Tensor
        ] = None,

        l1000_target_b: Optional[
            torch.Tensor
        ] = None,

        l1000_mask_b: Optional[
            torch.Tensor
        ] = None,

    ) -> LossOutput:


        loss_synergy = (
            self.synergy_loss(
                model_output[
                    "logits"
                ],
                labels,
            )
        )


        loss_a = self._masked_l1000_loss(
            prediction=model_output.get(
                "l1000_pred_a"
            ),
            target=l1000_target_a,
            mask=l1000_mask_a,
        )


        loss_b = self._masked_l1000_loss(
            prediction=model_output.get(
                "l1000_pred_b"
            ),
            target=l1000_target_b,
            mask=l1000_mask_b,
        )


        total = loss_synergy


        aux_losses = [
            x
            for x in [
                loss_a,
                loss_b,
            ]
            if x is not None
        ]


        if aux_losses:

            loss_l1000 = torch.stack(
                aux_losses
            ).mean()

            total = (
                total
                + self.lambda_l1000
                * loss_l1000
            )


        return LossOutput(
            total=total,
            synergy=loss_synergy,
            l1000_a=loss_a,
            l1000_b=loss_b,
        )


# ============================================================
# Utilities
# ============================================================

def count_parameters(
    model: nn.Module,
) -> Dict[str, int]:

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return {
        "total": total,
        "trainable": trainable,
    }


def check_pair_symmetry(
    model: DrugCombinationModel,
    batch: Dict[
        str,
        torch.Tensor
    ],
    atol: float = 1e-6,
) -> float:
    """
    Diagnostic check:

    f(A, B, C) should equal f(B, A, C)

    Run in eval mode because dropout would otherwise introduce
    random differences.
    """

    was_training = model.training

    model.eval()


    with torch.no_grad():

        out_ab = model(

            drug_a_morgan=
                batch["drug_a_morgan"],

            drug_a_targets=
                batch["drug_a_targets"],

            drug_a_ppi=
                batch["drug_a_ppi"],

            drug_a_availability=
                batch["drug_a_availability"],


            drug_b_morgan=
                batch["drug_b_morgan"],

            drug_b_targets=
                batch["drug_b_targets"],

            drug_b_ppi=
                batch["drug_b_ppi"],

            drug_b_availability=
                batch["drug_b_availability"],


            context_expression=
                batch["context_expression"],

            context_pathways=
                batch["context_pathways"],

            context_regulons=
                batch["context_regulons"],
        )["logits"]


        out_ba = model(

            drug_a_morgan=
                batch["drug_b_morgan"],

            drug_a_targets=
                batch["drug_b_targets"],

            drug_a_ppi=
                batch["drug_b_ppi"],

            drug_a_availability=
                batch["drug_b_availability"],


            drug_b_morgan=
                batch["drug_a_morgan"],

            drug_b_targets=
                batch["drug_a_targets"],

            drug_b_ppi=
                batch["drug_a_ppi"],

            drug_b_availability=
                batch["drug_a_availability"],


            context_expression=
                batch["context_expression"],

            context_pathways=
                batch["context_pathways"],

            context_regulons=
                batch["context_regulons"],
        )["logits"]


    max_difference = (
        torch.max(
            torch.abs(
                out_ab
                - out_ba
            )
        )
        .item()
    )


    if was_training:
        model.train()


    if max_difference > atol:

        raise RuntimeError(
            "Pair symmetry check FAILED. "
            f"Maximum |f(A,B)-f(B,A)| = "
            f"{max_difference:.8g}"
        )


    return max_difference
