import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformer_lens import HookedTransformer
import pandas as pd

def get_induction_heads(model, seq_len=50):
    """
    Identifies induction heads by checking for the [A][B] ... [A] -> [B] pattern.
    """
    # Create a random sequence of tokens
    # Avoid first few tokens as they might be BOS or special
    tokens = torch.randint(100, 1000, (1, seq_len))
    # Repeat the sequence
    full_tokens = torch.cat([tokens, tokens], dim=1)
    
    # Run the model and get attention patterns
    logits, cache = model.run_with_cache(full_tokens, remove_batch_dim=True)
    
    induction_scores = []
    
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            # Attention pattern for this head: [dest_pos, src_pos]
            attn = cache["pattern", layer][head]
            
            # We want to see how much dest_pos i+seq_len attends to src_pos i+1
            # (where i is the index of token A in the first half)
            # The token at i+seq_len is also A (second half).
            # So we check if A (at i+seq_len) attends to B (at i+1).
            
            score = 0
            count = 0
            for i in range(seq_len - 1):
                dest_pos = i + seq_len
                src_pos = i + 1
                score += attn[dest_pos, src_pos].item()
                count += 1
            
            avg_score = score / count if count > 0 else 0
            induction_scores.append({
                "layer": layer,
                "head": head,
                "score": avg_score
            })
            
    return pd.DataFrame(induction_scores)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    
    print("Identifying induction heads...")
    df = get_induction_heads(model)
    
    # Sort by score
    top_heads = df.sort_values("score", ascending=False).head(20)
    print("Top 20 Induction Heads:")
    print(top_heads)
    
    # Save to CSV
    df.to_csv("results/induction_heads_gpt2.csv", index=False)
