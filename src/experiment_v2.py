import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
import pandas as pd
import numpy as np

def get_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-10)).item()

def run_trial(model, pattern, reps, prompt_type="random", heads_to_ablate=None):
    A, B = pattern
    A_str = model.to_string(torch.tensor([A]))
    B_str = model.to_string(torch.tensor([B]))
    
    if prompt_type == "random":
        # Using a prompt that specifically asks for a random word
        prompt = f"Here is a list: {A_str} {B_str}, " * reps + f"{A_str}. Now, a completely random and unrelated word is:"
    elif prompt_type == "induction":
        prompt = f"Here is a list: {A_str} {B_str}, " * reps + f"{A_str}"
    
    tokens = model.to_tokens(prompt)
    
    def hook(value, hook):
        for layer, head in (heads_to_ablate or []):
            if hook.layer() == layer:
                value[:, :, head, :] = 0.0
        return value

    layers = set([h[0] for h in (heads_to_ablate or [])])
    pattern_hook_names = [f"blocks.{l}.attn.hook_z" for l in layers]
    
    if heads_to_ablate:
        with model.hooks(fwd_hooks=[(name, hook) for name in pattern_hook_names]):
            logits = model(tokens)
    else:
        logits = model(tokens)
        
    last_logit = logits[0, -1, :]
    probs = F.softmax(last_logit, dim=-1)
    
    prob_B = probs[B].item()
    entropy = get_entropy(probs)
    
    return prob_B, entropy

if __name__ == "__main__":
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    
    # Try multiple random patterns
    patterns = [
        [model.to_single_token(" apple"), model.to_single_token(" sky")],
        [model.to_single_token(" 42"), model.to_single_token(" elephant")],
        [model.to_single_token(" blue"), model.to_single_token(" tree")]
    ]
    
    top_ih = [(5, 1), (5, 5), (7, 10), (7, 2), (6, 9)]
    
    results = []
    
    for i, pattern in enumerate(patterns):
        for reps in [1, 2, 4, 8]:
            for p_type in ["random", "induction"]:
                # Base
                p_base, e_base = run_trial(model, pattern, reps, p_type)
                # Ablated
                p_abl, e_abl = run_trial(model, pattern, reps, p_type, top_ih)
                
                results.append({
                    "pattern_id": i,
                    "reps": reps,
                    "type": p_type,
                    "p_base": p_base,
                    "e_base": e_base,
                    "p_abl": p_abl,
                    "e_abl": e_abl
                })
                print(f"Pat {i}, Reps {reps}, Type {p_type:10}: Base Prob {p_base:.4f}, Abl Prob {p_abl:.4f}")

    df = pd.DataFrame(results)
    df.to_csv("results/experiment_v2_results.csv", index=False)
