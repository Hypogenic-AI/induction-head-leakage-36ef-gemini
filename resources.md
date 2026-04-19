# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "Are induction heads a large source of leakage?".

## Papers
Total papers downloaded: 8

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| In-context Learning and Induction Heads | Olsson et al. | 2022 | papers/2209.11895v1_In-context_Learning_and_Induction_Heads.pdf | Seminal paper on induction heads. |
| Induction Head Toxicity | Shuxun Wang et al. | 2025 | papers/Induction_Head_Toxicity_v1.pdf | Mechanism for repetition curse. |
| The Evolution of Statistical Induction Heads | Edelman et al. | 2024 | papers/2402.11004v1_The_Evolution_of_Statistical_Induction_Heads_In-Co.pdf | ICL Markov Chain task. |
| Next-token pretraining implies ICL | Riechers et al. | 2025 | papers/2505.18373v2_Next-token_pretraining_implies_in-context_learning.pdf | Link between pretraining and ICL. |
| Language Models "Grok" to Copy | Lv et al. | 2024 | papers/2409.09281v2_Language_Models_"Grok"_to_Copy.pdf | Training dynamics of copying. |
| Mechanistic Data Attribution | Chen et al. | 2026 | papers/2601.21996v1_Mechanistic_Data_Attribution_Tracing_the_Training_.pdf | Role of repetitive structural data. |
| Overthinking the Truth | Halawi et al. | 2023 | papers/2307.09476_Overthinking_the_Truth.pdf | False induction heads. |
| Repetition Neurons and Induction Heads | Unknown | 2025 | papers/2507.07810_Repetition_Neurons_Induction_Heads.pdf | Controlling repetition. |

## Datasets
Total datasets downloaded/identified: 2

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| WikiText-2 | HuggingFace | 4.3K | LM Evaluation | datasets/wikitext-2/ | Standard benchmark. |
| MarkovICL | Synthetic | Generated | ICL Markov Chain | code/MarkovICL/ | Code for generation. |

## Code Repositories
Total repositories cloned: 5

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| TransformerLens | github.com/neelnanda-io/TransformerLens | Interpretability tools | code/TransformerLens/ | Industry standard. |
| GIM | github.com/ejkim47/generalized-induction-head | Generalized induction | code/GIM/ | Retrieval-based module. |
| Evolution-of-SIH | github.com/EzraEdelman/Evolution-of-Statistical-Induction-Heads | Markov Chain Task | code/Evolution-of-Statistical-Induction-Heads/ | Excellent for data generation. |
| MarkovICL | github.com/Simon-Lepage/MarkovICL | Markov Chain ICL | code/MarkovICL/ | Alternative implementation. |
| Surgeon | github.com/gussand/surgeon | Head repair/intervention | code/surgeon/ | Understanding failures. |

## Recommendations for Experiment Design

1.  **Primary Dataset**: Use the **Evolution-of-Statistical-Induction-Heads** repo to generate Markov Chain sequences with varying degrees of randomness and repetition.
2.  **Baseline Methods**: Compare standard Transformers against those with **Toxic Head Ablation** (using TransformerLens).
3.  **Evaluation Metrics**: Measure the **"Leakage Ratio"** (how often the model copies from context when it should not) and **"Repetition Score"**.
4.  **Code to Adapt**: **TransformerLens** for identifying induction heads and performing causal interventions.
