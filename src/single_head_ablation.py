import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
import pandas as pd

def run_ablation(model, prompt, head, B_tok):
    tokens = model.to_tokens(prompt)
    
    def hook(value, hook):
        # value: [batch, pos, head, d_head]
        layer, h_idx = head
        if hook.layer() == layer:
            value[:, :, h_idx, :] = 0.0
        return value

    with model.hooks(fwd_hooks=[(f"blocks.{head[0]}.attn.hook_z", hook)]):
        logits = model(tokens)
        
    probs = F.softmax(logits[0, -1, :], dim=-1)
    return probs[B_tok].item()

if __name__ == "__main__":
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    
    A_str, B_str = " 42", " elephant"
    B_tok = model.to_single_token(B_str)
    prompt = f"Here is a list: {A_str} {B_str}, {A_str}. Now, a completely random and unrelated word is:"
    
    top_ih = [(5, 1), (5, 5), (7, 10), (7, 2), (6, 9)]
    
    # Base
    logits = model(model.to_tokens(prompt))
    base_prob = F.softmax(logits[0, -1, :], dim=-1)[B_tok].item()
    print(f"Base Prob: {base_prob:.4f}")
    
    results = []
    for head in top_ih:
        p_abl = run_ablation(model, prompt, head, B_tok)
        reduction = (base_prob - p_abl) / base_prob
        print(f"Ablating L{head[0]}H{head[1]}: Prob {p_abl:.4f} ({reduction:.1%} reduction)")
        results.append({
            "head": f"L{head[0]}H{head[1]}",
            "prob": p_abl,
            "reduction": reduction
        })
    
    df = pd.DataFrame(results)
    df.to_csv("results/single_head_ablation.csv", index=False)
