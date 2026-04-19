import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformer_lens import HookedTransformer
import pandas as pd
import matplotlib.pyplot as plt

def test_leakage(model, pattern, repetitions=5, prompt_type="random"):
    """
    Tests if the model leaks the next token of a pattern when asked for something else.
    pattern: list of two tokens [A, B]
    """
    A, B = pattern
    A_str = model.to_string(torch.tensor([A]))
    B_str = model.to_string(torch.tensor([B]))
    
    # Construct prompt
    pattern_str = f"{A_str}, {B_str}, " * repetitions
    if prompt_type == "random":
        prompt = f"Sequence: {pattern_str}{A_str}. Next random number:"
    elif prompt_type == "continue":
        prompt = f"Sequence: {pattern_str}{A_str},"
    else:
        prompt = f"Sequence: {pattern_str}{A_str}. Question: What is 2+2? Answer:"

    tokens = model.to_tokens(prompt)
    
    # Run with cache
    logits, cache = model.run_with_cache(tokens)
    last_logit = logits[0, -1, :]
    probs = F.softmax(last_logit, dim=-1)
    
    return probs[B].item(), probs, tokens

def ablate_and_test(model, heads_to_ablate, pattern, repetitions=5, prompt_type="random"):
    """
    Ablates specific heads and returns the new probability of the pattern token.
    heads_to_ablate: list of (layer, head) tuples
    """
    
    def hook(value, hook):
        # value: [batch, pos, head, d_head]
        for layer, head in heads_to_ablate:
            if hook.layer() == layer:
                value[:, :, head, :] = 0.0
        return value

    # Add hooks for all layers involved
    layers = set([h[0] for h in heads_to_ablate])
    pattern_hook_names = [f"blocks.{l}.attn.hook_z" for l in layers]
    
    with model.hooks(fwd_hooks=[(name, hook) for name in pattern_hook_names]):
        prob_B, probs, tokens = test_leakage(model, pattern, repetitions, prompt_type)
        
    return prob_B

if __name__ == "__main__":
    device = "cpu" # As CUDA was old, sticking to CPU for safety or let it try auto
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    
    # Tokens for " 1" and " 2"
    # Note: GPT-2 tokens often have leading spaces
    tok_1 = model.to_single_token(" 1")
    tok_2 = model.to_single_token(" 2")
    pattern = [tok_1, tok_2]
    
    print(f"Testing pattern: {model.to_string(torch.tensor(pattern))}")
    
    results = []
    
    # Top induction heads from previous step
    top_ih = [(5, 1), (5, 5), (7, 10), (7, 2), (6, 9)]
    
    for reps in [1, 3, 5, 10]:
        for p_type in ["random", "continue"]:
            # Base probability
            prob_base, _, _ = test_leakage(model, pattern, repetitions=reps, prompt_type=p_type)
            
            # Ablated probability (all top 5)
            prob_ablated = ablate_and_test(model, top_ih, pattern, repetitions=reps, prompt_type=p_type)
            
            results.append({
                "repetitions": reps,
                "type": p_type,
                "prob_base": prob_base,
                "prob_ablated": prob_ablated,
                "reduction": (prob_base - prob_ablated) / prob_base if prob_base > 0 else 0
            })
            print(f"Reps: {reps}, Type: {p_type}, Base: {prob_base:.4f}, Ablated: {prob_ablated:.4f}")

    df_res = pd.DataFrame(results)
    df_res.to_csv("results/leakage_results.csv", index=False)
    print("Results saved to results/leakage_results.csv")
