# Are Induction Heads a Large Source of Leakage?

This project investigates whether induction heads in language models cause unintended pattern leakage even when the model is instructed to generate random or unrelated tokens.

## Key Findings
- **Induction heads drive leakage**: In GPT-2 Small, the top induction heads (L5H1, L5H5) can cause a **3800x increase** in the probability of a pattern-completing token even when "randomness" is requested.
- **Causal Evidence**: Ablating these heads reduces pattern leakage by **over 90%** in "Random Interference" tasks.
- **L5H1 is the primary driver**: Ablating this single head alone reduces leakage by **~68%**.
- **Entropy tradeoff**: Removing induction heads decreases overall output entropy, as the model reverts to simpler unigram/bigram patterns (like predicting common punctuation or digits).

## How to Reproduce
1.  Set up the environment:
    ```bash
    uv venv
    source .venv/bin/activate
    uv add transformer-lens torch matplotlib pandas seaborn
    ```
2.  Run the induction head identification:
    ```bash
    python src/identify_heads.py
    ```
3.  Run the leakage experiment:
    ```bash
    python src/experiment_v2.py
    ```
4.  Run the single head ablation study:
    ```bash
    python src/single_head_ablation.py
    ```
5.  Analyze results:
    ```bash
    python src/analyze_results.py
    ```

## File Structure
- `src/`: Python scripts for experiments and analysis.
- `results/`: CSV files and plots generated during research.
- `REPORT.md`: Full research report.
- `planning.md`: Initial research plan and motivation.
- `literature_review.md`: Contextual background from literature.

## Acknowledgements
Uses `TransformerLens` by Neel Nanda for mechanistic interpretability. Inspired by research on induction head toxicity (Wang et al., 2025).
