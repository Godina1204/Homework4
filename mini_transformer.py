# mini_transformer.py
"""
Mini Transformer Encoder that takes a small batch of sentences and computes contextual embeddings.
Produces attention heatmaps.
Run: python mini_transformer.py
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Positional Encoding ----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ---------------- Multi-head Self-Attention ----------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.size()
        Q = self.q_lin(x).view(B, T, self.num_heads, self.d_k).transpose(1,2)  # (B, H, T, d_k)
        K = self.k_lin(x).view(B, T, self.num_heads, self.d_k).transpose(1,2)
        V = self.v_lin(x).view(B, T, self.num_heads, self.d_k).transpose(1,2)

        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)  # (B,H,T,T)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)  # (B, H, T, d_k)
        context = context.transpose(1,2).contiguous().view(B, T, D)
        out = self.out_lin(context)
        return out, attn  # return attention for visualization

# ---------------- Transformer Block ----------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden=256):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, attn_weights = self.attn(x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x, attn_weights

# ---------------- Simple tokenizer & demo ----------------
def simple_tokenizer(sentences):
    # whitespace tokenizer + build vocab
    tokens = [s.strip().split() for s in sentences]
    vocab = {}
    idx = 0
    for sent in tokens:
        for w in sent:
            if w not in vocab:
                vocab[w] = idx; idx+=1
    inv_vocab = {v:k for k,v in vocab.items()}
    token_ids = [[vocab[w] for w in s] for s in tokens]
    return token_ids, vocab, inv_vocab

def pad_batch(token_ids, pad_id=0):
    maxlen = max(len(t) for t in token_ids)
    batch = []
    for t in token_ids:
        padded = t + [pad_id]*(maxlen - len(t))
        batch.append(padded)
    return torch.tensor(batch, dtype=torch.long)

def main():
    sentences = [
        "I love natural language processing",
        "This class is interesting",
        "I love learning about transformers",
        "Do you like transformers"
    ]
    token_ids, vocab, inv_vocab = simple_tokenizer(sentences)
    pad_id = 0
    vocab_size = len(vocab)
    # ensure pad token exists
    if '<pad>' not in vocab:
        vocab['<pad>'] = vocab_size; pad_id = vocab_size; vocab_size += 1
        inv_vocab[pad_id] = '<pad>'

    batch = pad_batch(token_ids, pad_id=pad_id)
    B, T = batch.size()
    d_model = 64
    emb = nn.Embedding(vocab_size, d_model)
    pos = PositionalEncoding(d_model, max_len=100)
    block = TransformerBlock(d_model, num_heads=4)
    x = emb(batch)  # (B,T,D)
    x = pos(x)
    out, attn = block(x)  # attn: (B, H, T, T)
    print("Input tokens:")
    for i, s in enumerate(sentences):
        print(i, s)
    print("Final contextual embeddings shape:", out.shape)

    # visualize attention of head 0 for first sentence
    attn_np = attn[0, 0].detach().numpy()  # (T,T) for first batch, head 0
    plt.imshow(attn_np, interpolation='nearest')
    plt.title("Attention heatmap (batch0, head0)")
    plt.xlabel("Key position")
    plt.ylabel("Query position")
    plt.colorbar()
    plt.savefig('attention_heatmap.png')
    print("Saved attention_heatmap.png")

if __name__ == '__main__':
    main()
