# Literature Review: Induction Heads and Leakage/Repetition

## Research Area Overview
Induction heads are specialized attention heads in transformer models that implement a copying mechanism to complete sequences like `[A][B] ... [A] -> [B]`. While they are foundational to in-context learning (ICL), recent research suggests they can also be a source of unintended leakage and repetitive behavior, particularly when they become "toxic" or dominate the output logits inappropriately.

## Key Papers

### 1. In-context Learning and Induction Heads (Olsson et al., 2022)
-   **Authors**: Olsson et al.
-   **Year**: 2022
-   **Key Contribution**: Seminal paper identifying induction heads as the primary mechanism for ICL.
-   **Relevance to Our Research**: Foundation for understanding what induction heads are and how they operate via a two-layer copying mechanism.

### 2. Induction Head Toxicity Mechanistically Explains Repetition Curse in Large Language Models (Wang et al., 2025)
-   **Authors**: Shuxun Wang, Qingyu Yin, et al.
-   **Year**: 2025
-   **Key Contribution**: Introduces the concept of "induction head toxicity," where these heads dominate the model's output, causing repetitive and cyclic generation.
-   **Relevance to Our Research**: Directly supports the hypothesis that induction heads cause leakage and struggle with non-repetitive (random) generation.

### 3. Understanding and Controlling Repetition Neurons and Induction Heads in In-Context Learning (2025)
-   **Year**: 2025
-   **Key Contribution**: Explores how repetition neurons and induction heads interact to drive ICL but also cause repetitive failures.
-   **Relevance to Our Research**: Provides mechanisms to control or mitigate the leakage/repetition issues.

### 4. Overthinking the Truth: Understanding how Language Models Process False Demonstrations (Halawi et al., 2023)
-   **Authors**: Halawi et al.
-   **Year**: 2023
-   **Key Contribution**: Identifies "false induction heads" that attend to and copy incorrect information from demonstrations.
-   **Relevance to Our Research**: Highlights how induction heads can leak "incorrect" or "distractor" patterns from the context.

### 5. The Evolution of Statistical Induction Heads: In-Context Learning Markov Chains (Edelman et al., 2024)
-   **Authors**: Edelman et al.
-   **Year**: 2024
-   **Key Contribution**: Shows that transformers learn in-context statistics (unigrams, then bigrams) before forming induction heads.
-   **Relevance to Our Research**: Suggests that "leakage" might start as a statistical preference before being mechanized by induction heads.

## Common Methodologies
-   **Mechanistic Interpretability**: Using tools like path patching, activation patching, and attention map visualization to identify specific heads.
-   **Synthetic Tasks**: Copying tasks, Markov Chain tasks, and "Off-by-One" addition tasks are used to isolate induction head behavior.
-   **Head Ablation**: Causally measuring the impact of removing specific heads on the "repetition curse" or ICL performance.

## Standard Baselines
-   **N-gram Models**: Often used as a lower-bound for ICL performance.
-   **Attention-Only Models**: Used to study the emergence of induction heads in isolation from MLP layers.
-   **Llama-3.1, GPT-2, Pythia**: Standard model families for interpretability research.

## Gaps and Opportunities
-   **Randomness Evaluation**: While many papers study copying, fewer focus specifically on the model's *failure* to generate random tokens in the presence of patterns (the "leakage" effect).
-   **Mitigation Strategies**: Most research focuses on understanding; more work is needed on "detoxifying" induction heads without breaking ICL.

## Recommendations for Our Experiment
1.  **Primary Dataset**: Use synthetic Markov Chain sequences (as in Edelman et al.) and specifically designed "Random Interference" tasks where a pattern exists in context but the model should generate random tokens.
2.  **Baseline Methods**: Compare standard Transformers against those with "Repetition Neurons" or "Toxic Head" ablation.
3.  **Evaluation Metrics**: Measure "repetition score" and "pattern leakage" (how much of the context is copied vs. original generation).
4.  **Code to Adapt**: Use **TransformerLens** for head identification and ablation, and the **Evolution-of-Statistical-Induction-Heads** repo for data generation.
