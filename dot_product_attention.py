# attention.py
"""
Implement scaled dot-product attention and test with random Q,K,V.
Run: python attention.py
"""

import torch
import torch.nn.functional as F
import math


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Compute scaled dot-product attention.

    Args:
        Q: (..., seq_len_q, d_k)
        K: (..., seq_len_k, d_k)
        V: (..., seq_len_v, d_v)
        mask: (..., seq_len_q, seq_len_k) optional boolean mask (True = keep, False = mask out)

    Returns:
        output: (..., seq_len_q, d_v)
        attn_weights: (..., seq_len_q, seq_len_k)
    """
    d_k = Q.size(-1)
    # raw scores
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # numerical stability & scaling
    # Show stability: compute softmax before and after scaling in the test harness
    scaled_scores = scores / math.sqrt(d_k)

    if mask is not None:
        # mask should be additive (0 keep, -inf mask) or boolean; here assume boolean True=keep
        scaled_scores = scaled_scores.masked_fill(~mask, float('-inf'))

    attn = F.softmax(scaled_scores, dim=-1)
    output = torch.matmul(attn, V)
    return output, attn


if __name__ == '__main__':
    torch.manual_seed(0)
    # test small random tensors
    batch = 1
    seq_q = 3
    seq_k = 4
    d_k = 8
    d_v = 6

    Q = torch.randn(batch, seq_q, d_k)
    K = torch.randn(batch, seq_k, d_k)
    V = torch.randn(batch, seq_k, d_v)

    # compute attention without scaling (for stability demonstration)
    raw_scores = torch.matmul(Q, K.transpose(-2, -1))
    try:
        raw_attn = F.softmax(raw_scores, dim=-1)
    except Exception as e:
        raw_attn = None

    out_scaled, attn_scaled = scaled_dot_product_attention(Q, K, V)

    print("Raw scores (before scaling) sample:\n", raw_scores[0])
    if raw_attn is not None:
        print("Softmax on raw scores (may be unstable) sample:\n", raw_attn[0])
    print("Scaled scores (scores / sqrt(d_k)) sample:\n", (raw_scores / math.sqrt(d_k))[0])
    print("Softmax after scaling (attn weights):\n", attn_scaled[0])
    print("Output vectors shape:", out_scaled.shape)
    print("Output vectors sample:\n", out_scaled[0])

