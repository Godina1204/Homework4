# mini_transformer.py
"""
Mini Transformer Encoder that takes a small batch of sentences and computes contextual embeddings.
Produces attention heatmaps.
Run: python mini_transformer.py
"""

import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # last column will remain zero for odd dim
            pe[:, 1::2] = torch.cos(position * div_term[:(d_model//2)])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x


class SimpleMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch, seq_len, d_model = x.size()
        Q = self.q_lin(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1,2)  # (b, h, seq, d_k)
        K = self.k_lin(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1,2)
        V = self.v_lin(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1,2)

        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (b, h, seq, d_k)
        out = out.transpose(1,2).contiguous().view(batch, seq_len, d_model)
        return self.out_lin(out), attn


class MiniTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=64, n_heads=4, dim_ff=256, dropout=0.1):
        super().__init__()
        self.mha = SimpleMultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        mha_out, attn = self.mha(x)
        x = self.norm1(x + self.dropout(mha_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, attn


# Toy dataset of 10 sentences
SENTENCES = [
    "the cat sat on the mat",
    "a quick brown fox",
    "hello world",
    "transformers are powerful",
    "attention is all you need",
    "i like pizza",
    "we study nlp",
    "this is a test",
    "deep learning rocks",
    "sequence models matter"
]


def simple_tokenize(sentences):
    # map each unique word to an index
    toks = []
    vocab = {"<pad>":0, "<unk>":1}
    for s in sentences:
        words = s.lower().split()
        toks.append(words)
        for w in words:
            if w not in vocab:
                vocab[w] = len(vocab)
    return toks, vocab


if __name__ == '__main__':
    torch.manual_seed(0)
    toks, vocab = simple_tokenize(SENTENCES)
    max_len = max(len(x) for x in toks)
    d_model = 64

    # create input batch
    batch = len(toks)
    input_ids = torch.zeros(batch, max_len, dtype=torch.long)
    for i, words in enumerate(toks):
        for j, w in enumerate(words):
            input_ids[i,j] = vocab.get(w, vocab["<unk>"])

    emb = nn.Embedding(len(vocab), d_model)
    pos = SinusoidalPositionalEncoding(d_model, max_len=500)
    encoder = MiniTransformerEncoderLayer(d_model=d_model, n_heads=4, dim_ff=128)

    x = emb(input_ids)  # (batch, seq, d_model)
    x = pos(x)
    out, attn = encoder(x)

    # print final contextual embeddings for first token of each sentence
    print("Final contextual embeddings (first token) for each sentence:")
    for i in range(batch):
        print(f"Sentence {i+1}: {SENTENCES[i]} -> embedding[:8] =", out[i,0,:8].detach().numpy())

    # attn shape: (batch, heads, seq, seq)
    # We'll visualize averaged attention over heads for the first example
    attn_avg = attn.mean(dim=1).detach().numpy()  # (batch, seq, seq)
    example_idx = 0
    heat = attn_avg[example_idx]

    plt.imshow(heat, cmap='viridis')
    plt.title(f"Attention heatmap (averaged heads) - example {example_idx}")
    plt.xlabel('Key position')
    plt.ylabel('Query position')
    plt.colorbar()
    plt.savefig('attention_heatmap.png')
    print('Saved attention heatmap to attention_heatmap.png')
