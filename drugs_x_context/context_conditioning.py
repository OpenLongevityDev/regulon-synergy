import torch
import torch.nn as nn


# ============================================================
# Context-conditioned drug representation
# ============================================================

class ContextConditioner(nn.Module):
    """
    Convert an intrinsic drug representation into a
    context-aware drug representation.

    Inputs
    ------
    drug_embedding:
        [batch, drug_dim]

        Intrinsic representation of the drug.
        Example:
            SMILES + direct target/PPI encoder output.

    context_embedding:
        [batch, context_dim]

        Baseline biological-context representation.
        Example:
            expression + pathway + regulon fused embedding.

    Output
    ------
    context_drug_embedding:
        [batch, output_dim]

    Design
    ------
    1. Context generates FiLM parameters gamma and beta.
    2. Drug embedding is context-modulated.
    3. Explicit interaction features are constructed.
    4. A learned gate determines how strongly context
       changes the intrinsic drug representation.
    5. Residual connection preserves intrinsic drug identity.
    """

    def __init__(
        self,
        drug_dim=128,
        context_dim=128,
        hidden_dim=256,
        output_dim=128,
        dropout=0.20,
    ):
        super().__init__()

        self.drug_dim = drug_dim
        self.context_dim = context_dim
        self.output_dim = output_dim


        # ----------------------------------------------------
        # Project both modalities into common latent space
        # ----------------------------------------------------

        self.drug_projection = nn.Sequential(
            nn.Linear(
                drug_dim,
                output_dim
            ),
            nn.LayerNorm(
                output_dim
            ),
            nn.GELU(),
        )

        self.context_projection = nn.Sequential(
            nn.Linear(
                context_dim,
                output_dim
            ),
            nn.LayerNorm(
                output_dim
            ),
            nn.GELU(),
        )


        # ----------------------------------------------------
        # FiLM parameters from context
        #
        # context -> gamma, beta
        #
        # gamma/beta both output_dim dimensional
        # ----------------------------------------------------

        self.film = nn.Sequential(
            nn.Linear(
                output_dim,
                hidden_dim
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_dim,
                2 * output_dim
            ),
        )


        # ----------------------------------------------------
        # Explicit drug × context interaction
        #
        # Features:
        #   drug
        #   context
        #   drug * context
        #   |drug - context|
        #
        # total = output_dim * 4
        # ----------------------------------------------------

        interaction_input_dim = (
            output_dim * 4
        )

        self.interaction_mlp = nn.Sequential(

            nn.Linear(
                interaction_input_dim,
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


        # ----------------------------------------------------
        # Context-dependent residual gate
        #
        # gate ~ 0:
        #   preserve intrinsic drug representation
        #
        # gate ~ 1:
        #   strongly use context-specific interaction
        # ----------------------------------------------------

        self.gate = nn.Sequential(

            nn.Linear(
                output_dim * 2,
                hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                output_dim
            ),

            nn.Sigmoid(),
        )


        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        self.output_norm = nn.LayerNorm(
            output_dim
        )


    def forward(
        self,
        drug_embedding,
        context_embedding,
        return_aux=False,
    ):

        # ----------------------------------------------------
        # Common latent space
        # ----------------------------------------------------

        d = self.drug_projection(
            drug_embedding
        )

        c = self.context_projection(
            context_embedding
        )


        # ----------------------------------------------------
        # FiLM modulation
        # ----------------------------------------------------

        gamma_beta = self.film(
            c
        )

        gamma, beta = torch.chunk(
            gamma_beta,
            chunks=2,
            dim=-1
        )


        # Using (1 + gamma) means the model starts conceptually
        # close to identity modulation rather than requiring
        # gamma ~= 1.
        d_film = (
            d * (1.0 + gamma)
            + beta
        )


        # ----------------------------------------------------
        # Explicit interaction representation
        # ----------------------------------------------------

        interaction_features = torch.cat(

            [
                d_film,
                c,
                d_film * c,
                torch.abs(
                    d_film - c
                ),
            ],

            dim=-1
        )


        interaction = self.interaction_mlp(
            interaction_features
        )


        # ----------------------------------------------------
        # Learn how strongly context should alter each latent
        # dimension of the drug representation
        # ----------------------------------------------------

        gate = self.gate(

            torch.cat(
                [
                    d,
                    c,
                ],
                dim=-1
            )
        )


        # ----------------------------------------------------
        # Residual context conditioning
        # ----------------------------------------------------

        out = (
            d
            +
            gate * interaction
        )

        out = self.output_norm(
            out
        )


        if return_aux:

            return {
                "embedding": out,
                "intrinsic_projected": d,
                "context_projected": c,
                "film_gamma": gamma,
                "film_beta": beta,
                "interaction": interaction,
                "gate": gate,
            }

        return out


# ============================================================
# Shared conditioner for two drugs
# ============================================================

class DrugPairContextConditioner(nn.Module):
    """
    Apply the SAME context-conditioning network to Drug A
    and Drug B.

    Weight sharing is important:
        f(drug_A, context)
        f(drug_B, context)

    use identical functions.
    """

    def __init__(
        self,
        drug_dim=128,
        context_dim=128,
        hidden_dim=256,
        output_dim=128,
        dropout=0.20,
    ):
        super().__init__()

        self.conditioner = ContextConditioner(
            drug_dim=drug_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )


    def forward(
        self,
        drug_a_embedding,
        drug_b_embedding,
        context_embedding,
        return_aux=False,
    ):

        if return_aux:

            a = self.conditioner(
                drug_a_embedding,
                context_embedding,
                return_aux=True,
            )

            b = self.conditioner(
                drug_b_embedding,
                context_embedding,
                return_aux=True,
            )

            return {
                "drug_a": a,
                "drug_b": b,
            }


        drug_a_context = self.conditioner(
            drug_a_embedding,
            context_embedding,
        )

        drug_b_context = self.conditioner(
            drug_b_embedding,
            context_embedding,
        )

        return (
            drug_a_context,
            drug_b_context,
        )


# ============================================================
# Diagnostic test
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )


    model = DrugPairContextConditioner(

        drug_dim=128,
        context_dim=128,
        hidden_dim=256,
        output_dim=128,
        dropout=0.20,

    ).to(
        device
    )


    batch_size = 32

    drug_a = torch.randn(
        batch_size,
        128,
        device=device
    )

    drug_b = torch.randn(
        batch_size,
        128,
        device=device
    )

    context = torch.randn(
        batch_size,
        128,
        device=device
    )


    with torch.no_grad():

        out = model(
            drug_a,
            drug_b,
            context,
            return_aux=True,
        )


    print(
        "\nDrug A context-aware embedding:",
        out[
            "drug_a"
        ][
            "embedding"
        ].shape
    )

    print(
        "Drug B context-aware embedding:",
        out[
            "drug_b"
        ][
            "embedding"
        ].shape
    )


    gate_a = (
        out[
            "drug_a"
        ][
            "gate"
        ]
    )

    print(
        "\nGate diagnostics:"
    )

    print(
        "mean:",
        gate_a.mean().item()
    )

    print(
        "std:",
        gate_a.std().item()
    )

    print(
        "min:",
        gate_a.min().item()
    )

    print(
        "max:",
        gate_a.max().item()
    )


    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "\nTotal parameters:",
        f"{n_params:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable:,}"
    )
