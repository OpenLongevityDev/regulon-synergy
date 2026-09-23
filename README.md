# Drug Combination Response Prediction Across Biological Contexts

A deep-learning framework for predicting **drug combination response** from drug representations and biological context.

The model formulates drug combination response as a **three-class classification problem**:

* **Synergy**
* **No interaction / neutral**
* **Antagonism**

The framework is trained using drug-combination experiments in **cell lines** and is designed to enable transfer to biological contexts derived from **human transcriptomic data**, including single-cell RNA-seq pseudobulk profiles.

---

## Overview

Drug combination effects depend not only on the properties of the two compounds but also on the biological state in which they are applied.

This project models the problem as:

```text
Drug A ─────┐
            │
Drug B ─────┼──► Drug–Context Model ──► P(antagonism)
            │                         ├─► P(neutral)
Context ────┘                         └─► P(synergy)
```

Rather than predicting genome-wide transcriptional responses or relying exclusively on a continuous synergy score, the model learns the probability of three pharmacologically relevant outcomes.

A major objective is to learn biological-context representations that can generalize beyond the cell lines used for training and eventually support inference in **patient-, sample-, and cell-type-specific contexts**.

---

## Model Inputs

The framework integrates representations of both drugs with a multimodal representation of biological context.

### Drug representation

Each drug can be represented using information derived from:

* chemical structure;
* molecular fingerprints / learned chemical embeddings;
* known drug targets;
* target profiles propagated through protein–protein interaction networks.

These features are intended to capture complementary chemical and target-level information for each compound.

### Biological context

The current biological-context representation integrates complementary molecular views:

1. **Gene expression**

   Expression of L1000 landmark genes provides a direct transcriptional representation of the cellular state.

2. **Pathway activity**

   Hallmark pathway activities summarize expression into interpretable biological programs.

3. **Transcription-factor activity**

   TF activities derived from **CollecTRI regulons** provide a regulatory representation of cellular state that is less dependent on individual gene-expression measurements.

4. **Network-level features**

   PPI/network-derived representations can be incorporated to capture functional relationships among genes and drug targets.

Each modality is encoded independently before being combined into a shared biological-context embedding.

---

## Model Architecture

At a high level, the model contains three components:

```text
                 ┌───────────────────┐
Drug A ─────────►│ Drug Encoder      │──┐
                 └───────────────────┘  │
                                        │
                 ┌───────────────────┐  │
Drug B ─────────►│ Drug Encoder      │──┼──► Fusion ─► Classifier
                 └───────────────────┘  │                 │
                                        │                 ▼
                 ┌───────────────────┐  │       Antagonism / Neutral / Synergy
Expression ─────►│                   │  │
Pathways ───────►│ Context Encoder   │──┘
TF activities ──►│                   │
Network state ──►│                   │
                 └───────────────────┘
```

The context encoder uses separate branches for complementary biological representations and combines them into a shared latent representation.

The current implementation uses approximately:

```text
L1000 expression       942 features
Hallmark pathways       50 features
CollecTRI TF activity  771 features
```

Active context branches are projected to latent representations and fused into a **128-dimensional biological-context embedding**.

Input normalization parameters are learned from the training contexts and subsequently reused for validation and inference to avoid information leakage.

---

## Training Data

The model is designed around publicly available large-scale drug-combination screens, including data from resources such as:

* DrugComb
* DrugCombDB

Drug-pair experiments are associated with the corresponding cell-line molecular context.

Available drug-combination response metrics may include:

* ZIP
* Loewe
* Bliss
* HSA

Continuous drug-combination measurements are converted into the three target classes used by the classifier.

```text
Antagonism <────── Neutral ──────> Synergy
```

The resulting training examples have the general form:

```text
Drug A + Drug B + Cell-line context → Response class
```

---

## Transfer to Human Transcriptomic Data

A central motivation of the project is applying a model trained on experimental cell lines to biological states observed in human tissue.

For single-cell RNA-seq datasets, cells can be aggregated into biologically meaningful pseudobulk contexts, for example:

```text
Patient 1 × CD4 T cells
Patient 1 × Monocytes
Patient 1 × NK cells

Patient 2 × CD4 T cells
Patient 2 × Monocytes
Patient 2 × NK cells
```

The same context features used during model training are then calculated for these samples:

```text
scRNA-seq
   │
   ▼
Patient × cell-type pseudobulk
   │
   ├──► L1000 expression
   ├──► Hallmark pathway activity
   ├──► CollecTRI TF activity
   └──► network features
             │
             ▼
       Context Encoder
             │
Drug A ──────┤
Drug B ──────┤
             ▼
P(antagonism), P(neutral), P(synergy)
```

This design allows the model to estimate drug-combination behavior for different molecular states without requiring those exact biological contexts to have appeared during training.

---

## Why Regulatory and Pathway Features?

Direct gene-expression profiles can differ substantially between cell lines and primary human cells.

The framework therefore combines raw transcriptional information with higher-level biological representations.

For example:

```text
Individual genes
      ↓
Transcription-factor programs
      ↓
Pathway / regulatory state
```

Regulon and pathway activities may capture biological programs that are shared across otherwise different cellular systems while also reducing dimensionality and gene-level noise.

This is particularly useful for the intended **cell-line → patient tissue** transfer setting.

---

## Prediction Output

For each drug pair and biological context, the model produces class probabilities:

```python
{
    "antagonism": 0.08,
    "neutral":    0.17,
    "synergy":    0.75
}
```

This makes it possible to rank candidate drug combinations while retaining information about prediction uncertainty.

Predictions can ultimately be generated at different biological resolutions, such as:

```text
Drug pair × patient
Drug pair × sample
Drug pair × cell type
Drug pair × patient × cell type
```

depending on how the transcriptomic context is constructed.

---

## Repository Structure

```text
.
├── data/                  # Input data and processed datasets
├── preprocessing/         # Drug and biological-context preprocessing
├── models/                # Model architectures
├── training/              # Training and evaluation scripts
├── inference/             # Prediction on new biological contexts
├── utils/                 # Shared utility functions
├── configs/               # Model/training configuration
├── notebooks/             # Analysis and exploratory notebooks
└── README.md
```

The exact directory structure may change as the project develops.

---

## Workflow

```text
             TRAINING
                │
DrugComb / DrugCombDB
                │
        ┌───────┴────────┐
        ▼                ▼
   Drug features    Cell-line RNA
                         │
                   Context features
                         │
        ┌────────────────┘
        ▼
   Model training
        │
        ▼
3-class drug-response model


             INFERENCE
                │
         Human scRNA-seq
                │
        Patient × cell type
            pseudobulk
                │
        Context features
                │
                ▼
Drug A + Drug B + context
                │
                ▼
 Antagonism / Neutral / Synergy
```

---

## Project Status

🚧 **Research in progress**

The repository contains an actively developed research framework. Model architecture, preprocessing procedures, feature representations, and evaluation strategies may change during development.

The current focus includes:

* construction of robust multimodal biological-context embeddings;
* evaluation of drug-combination classification;
* assessment of generalization to unseen biological contexts;
* transfer from cell-line training data to human transcriptomic data;
* interpretation of predictions using pathways, transcriptional regulators, and drug-target networks.

---

## Intended Applications

The framework is being developed for research applications including:

* context-specific drug-combination prioritization;
* identification of potentially synergistic or antagonistic combinations;
* patient- and cell-type-specific combination prediction;
* integration of pharmacological knowledge with single-cell transcriptomics;
* mechanistic interpretation of predicted combination effects.

---

## Disclaimer

This project is intended for **research purposes only**.

Predictions generated by the model are computational hypotheses and should not be interpreted as clinical recommendations. Candidate drug combinations require appropriate experimental and clinical validation.

---

## Citation

A manuscript describing the methodology is in preparation.

If you use this repository or build upon this work, please check this section for updated citation information.

---

## License

License information will be added as the project is prepared for public release.
