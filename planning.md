# Research Plan: Are Induction Heads a Large Source of Leakage?

## Motivation & Novelty Assessment

### Why This Research Matters
Large Language Models (LLMs) are known to struggle with generating truly random sequences and often fall into repetitive loops (the "repetition curse"). Understanding the mechanistic cause of this behavior is crucial for improving model reliability, reducing "hallucinations" (when they copy irrelevant context), and enabling better control over stochastic generation. If induction heads—the very mechanism that enables in-context learning—are also responsible for this leakage, it points to a fundamental trade-off in transformer architectures.

### Gap in Existing Work
Existing literature (Olsson et al., 2022; Edelman et al., 2024) focuses heavily on the *success* of induction heads in enabling pattern matching and ICL. While recent work (Wang et al., 2025) has introduced "induction head toxicity" in the context of repetition, there is limited research specifically investigating "leakage" in non-repetitive contexts—i.e., when a model *should* be generating random or novel tokens but is "pulled" towards copying existing patterns by overactive induction heads.

### Our Novel Contribution
We propose to explicitly measure the "leakage" effect of induction heads in a "Random Interference" task. Unlike standard repetition tasks, our task will present a strong pattern in the context but require a random output. We will quantify the causal link between induction head activity and the failure to generate random outputs by using mechanistic interpretability tools (TransformerLens) to ablate these heads and observe the change in output entropy and copying probability.

### Experiment Justification
- **Experiment 1: IH Identification & Baseline Leakage**: Establish which heads are induction heads and measure their "natural" tendency to copy in random contexts.
- **Experiment 2: Causal Ablation**: Verify if removing these heads reduces leakage and increases output randomness, proving they are the drivers of this behavior.
- **Experiment 3: Pattern Strength Sensitivity**: Test how the "pull" of induction heads scales with the strength/length of the pattern in context.

---

## Research Question
Do induction heads in language models cause unintended leakage of patterned information from the text, even when the model should not be copying, and does this contribute to the model's struggle to generate random outputs?

## Hypothesis Decomposition
- **H1**: Induction heads are active and produce high logits for "pattern-completing" tokens even when the ground truth or task context suggests randomness.
- **H2**: Ablating induction heads significantly reduces the probability of copying from context in "Random Interference" tasks.
- **H3**: The "leakage" effect increases with the number of repetitions of a pattern in the prompt (induction head reinforcement).

## Proposed Methodology

### Approach
We will use **GPT-2 Small** (via TransformerLens) as our primary model due to its well-mapped induction heads. We will design synthetic prompts that contain an "induction trigger" (a repeating sequence) followed by a request for a random token.

### Experimental Steps
1. **Model & Head Setup**: Load GPT-2 Small and identify induction heads using the standard "induction score" (matching `[A][B] ... [A]` -> `[B]`).
2. **Task Design**: Create "Random Interference" prompts:
   - Example: `The sequence is 1, 2, 1, 2, 1, 2. The next random number is:`
   - Here, `2` is the "induction" token, but the prompt asks for a "random" number.
3. **Logit Analysis**: Measure the logit assigned to the "induction token" vs. other tokens.
4. **Causal Intervention**: Perform "zero-ablation" on identified induction heads and measure the change in:
   - **Copying Probability**: $P(\text{token} = B | \text{pattern } AB...A)$
   - **Output Entropy**: $H(P) = -\sum p_i \log p_i$
5. **Scale Study**: Vary the number of repetitions in the trigger (e.g., `1, 2` repeated 1, 2, 5, 10 times) and measure the leakage.

### Baselines
- **Random Baseline**: Uniform distribution over the token vocabulary.
- **Unigram/Bigram Baseline**: Frequency-based prediction from the context.
- **Ablated Model**: The same model with induction heads removed.

### Evaluation Metrics
- **Leakage Score**: The ratio of the probability of the "pattern" token to the average probability of non-pattern tokens.
- **Entropy**: To measure the "randomness" of the output distribution.
- **Induction Score**: To verify the heads being ablated are indeed induction heads.

### Statistical Analysis Plan
- T-tests to compare entropy and leakage scores between the base model and the ablated model.
- Correlation analysis between IH induction scores and their individual leakage contributions.

## Expected Outcomes
- We expect to find that GPT-2 Small assigns significantly higher probability to the "pattern" token `2` in the example above, even when "random" is requested.
- We expect that ablating the top induction heads will significantly increase the entropy of the output distribution and decrease the leakage score.

## Timeline and Milestones
- **Phase 1: Planning & Setup**: 1 hour
- **Phase 2: Data Generation & Baseline**: 1 hour
- **Phase 3: IH Identification & Ablation Experiments**: 2 hours
- **Phase 4: Analysis & Visualization**: 1 hour
- **Phase 5: Documentation**: 1 hour

## Potential Challenges
- **Identifying "Leakage" vs. "Correct Prediction"**: In natural language, patterns are often the "correct" next token. We must use synthetic tasks where the "correct" next token is explicitly *not* the pattern.
- **Head Interactions**: Ablating one head might lead to other heads compensating. We will use group ablation of the top $N$ induction heads.

## Success Criteria
- Successful identification of induction heads in GPT-2 Small.
- Demonstration of a statistically significant "leakage" effect in the Random Interference task.
- Clear causal evidence that ablating IH reduces this leakage.
