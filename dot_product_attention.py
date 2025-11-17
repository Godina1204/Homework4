# attention.py
"""
Implement scaled dot-product attention and test with random Q,K,V.
Run: python attention.py
"""

import torch
import torch.nn.functional as F
import math
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Q: (B, T_q, d_k)
    K: (B, T_k, d_k)
    V: (B, T_k, d_v)
    returns: output (B, T_q, d_v), attn_weights (B, T_q, T_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (B, T_q, T_k)
    # numerical stability check: subtract max before softmax
    scores_max = scores.max(dim=-1, keepdim=True).values
    scores_safe = scores - scores_max
    attn = F.softmax(scores_safe, dim=-1)
    out = torch.matmul(attn, V)
    return out, attn

def test():
    torch.manual_seed(0)
    B, Tq, Tk, d_k, d_v = 2, 3, 4, 8, 8
    Q = torch.randn(B, Tq, d_k)
    K = torch.randn(B, Tk, d_k)
    V = torch.randn(B, Tk, d_v)

    # Without scaling (dangerous)
    raw_scores = torch.matmul(Q, K.transpose(-2,-1))
    print("Raw score scale mean:", raw_scores.mean().item(), "std:", raw_scores.std().item())

    out, attn = scaled_dot_product_attention(Q, K, V)
    print("Attention weights (batch 0):")
    print(attn[0])
    print("Output vectors (batch 0):")
    print(out[0])
    # softmax sanity: rows sum to 1
    print("Attn row sums (should be 1):", attn[0].sum(dim=-1))

if __name__ == '__main__':
    test()
