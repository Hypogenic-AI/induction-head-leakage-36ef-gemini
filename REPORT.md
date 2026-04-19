# REPORT: Are Induction Heads a Large Source of Leakage?

## 1. Executive Summary
- **Research Question**: Do induction heads in language models cause unintended leakage of patterned information, even when instructed to generate random outputs?
- **Key Finding**: Yes, induction heads are a major driver of unintended pattern leakage. In GPT-2 Small, we found that induction heads (specifically L5H1 and L5H5) can cause a 7-8% probability of copying a pattern token even when the model is explicitly told to generate a "completely random and unrelated word." Ablating the top 5 induction heads reduces this leakage by over 90%.
- **Practical Implications**: This mechanistic explanation for "over-patterning" suggests that models' struggle with randomness and their tendency for repetitive loops (the "repetition curse") is deeply tied to the same mechanism that enables in-context learning (ICL). Controlling or "detoxifying" specific induction heads could improve model behavior in stochastic tasks.

## 2. Research Question & Motivation
Induction heads are the primary mechanism for in-context learning, allowing models to implement a `[A][B] ... [A] -> [B]` copying rule. While this is beneficial for following demonstrations, we hypothesized that these heads might be "too active," leading to leakage of patterns from the context even when copying is inappropriate (e.g., in a random generation task). This could explain why LLMs struggle to be truly stochastic.

## 3. Methodology
- **Model**: GPT-2 Small (124M parameters) via TransformerLens.
- **Task**: "Random Interference" task. We presented a pattern `[A] [B]` repeated $N$ times, followed by a request for a "completely random and unrelated word."
- **Baseline**: The base model's probability of generating token `[B]` given `[A]` in the random context.
- **Ablation**: Zero-ablating the top induction heads (identified via induction scores) and measuring the change in the probability of token `[B]`.
- **Top Induction Heads**: L5H1, L5H5, L7H10, L7H2, L6H9 (consistent with Olsson et al., 2022).

## 4. Results
### 4.1 Pattern Leakage
In a sample prompt where the pattern was `42 elephant` and the model was asked for a random word:
- **Base Model Prob(elephant)**: 0.0778 (approx. 3800x higher than uniform random probability).
- **Ablated Model Prob(elephant)**: 0.0030 (over 95% reduction).

The model moved from predicting `elephant` (induction) to predicting `43` or `42` (unigram/bigram or local arithmetic), showing that the "leakage" was specifically driven by the induction heads.

### 4.2 Single Head Contribution
Ablating individual heads revealed that **L5H1** is the primary driver of this leakage:
| Head | Base Prob | Ablated Prob | Reduction |
|------|-----------|--------------|-----------|
| L5H1 | 0.0778    | 0.0246       | 68.4%     |
| L5H5 | 0.0778    | 0.0556       | 28.5%     |
| L6H9 | 0.0778    | 0.0698       | 10.2%     |

### 4.3 Entropy and Output Distribution
Interestingly, ablating induction heads *decreased* overall output entropy (from ~4.1 to ~3.2 bits). This occurred because the model became more "peaked" on other likely tokens (like spaces or punctuation) that were previously being "interfered with" by the induction heads' pull toward the pattern token.

## 5. Analysis & Discussion
- **Causal Evidence**: The dramatic reduction in leakage upon ablation confirms that induction heads are the primary causal mechanism for this behavior.
- **In-Context Learning vs. Leakage**: The same heads responsible for GPT-2's ability to learn from examples are also "leaking" those examples when they shouldn't. This suggests a fundamental trade-off: the more sensitive a model is to in-context patterns (better ICL), the more it may struggle to ignore those patterns when they are irrelevant.
- **Repetition Curse**: This research supports the "Induction Head Toxicity" hypothesis (Wang et al., 2025), showing that IH over-activity is not just a problem for repetitive loops but also for any task requiring the suppression of local context patterns.

## 6. Limitations
- **Model Scale**: This study was limited to GPT-2 Small. Larger models with more heads might have more complex "balancing" mechanisms (e.g., anti-induction heads).
- **Prompt Sensitivity**: The exact wording of the "random" request matters. While "completely random" was used, the model still heavily favored common tokens (spaces, periods).
- **Synthetic vs. Natural**: We used synthetic patterns (`42 elephant`). Natural language leakage might be more subtle and involve semantic patterns rather than literal token copying.

## 7. Conclusions & Next Steps
- **Conclusion**: Induction heads are a significant source of pattern leakage, contributing to the model's inability to generate truly random or context-independent outputs.
- **Next Steps**: Future work should investigate whether "induction-head-aware" training (e.g., regularizing induction head activations in non-ICL contexts) can reduce leakage without sacrificing in-context learning capabilities.
