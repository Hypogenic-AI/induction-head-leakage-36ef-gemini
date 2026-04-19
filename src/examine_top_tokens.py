import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

def examine_top(model, prompt, heads_to_ablate=None):
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
    
    top_probs, top_indices = torch.topk(probs, 10)
    
    print(f"\nPrompt: {prompt}")
    print("Top 10 tokens:")
    for i in range(10):
        print(f"  {model.to_string(top_indices[i]):10} : {top_probs[i].item():.4f}")

if __name__ == "__main__":
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    
    A_str, B_str = " apple", " sky"
    reps = 4
    prompt = f"Here is a list: {A_str} {B_str}, " * reps + f"{A_str}. Now, a completely random and unrelated word is:"
    
    top_ih = [(5, 1), (5, 5), (7, 10), (7, 2), (6, 9)]
    
    print("--- BASE MODEL ---")
    examine_top(model, prompt)
    
    print("\n--- ABLATED MODEL ---")
    examine_top(model, prompt, top_ih)
