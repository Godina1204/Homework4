# Student: 700769576 Name: Veera Ganga Godina

---

# FILE: README.md
# ----------------
# Character-Level RNN Language Model & Mini Transformer Encoder

## Student info
- Student ID: 700769576
- Name: Veera Ganga Godina

## Overview
This repository contains three main components:

1. `char_rnn.py` - trains a character-level language model (Embedding -> RNN -> Linear -> Softmax). Produces training/validation loss curves and temperature-controlled samples.
2. `mini_transformer.py` - a minimal Transformer Encoder (sinusoidal positional encoding, multi-head self-attention, feed-forward, Add & Norm). Runs on a toy set of sentences and saves an attention heatmap.
3. `dot_product_attention.py` - standalone implementation of scaled dot-product attention with stability checks and a test harness that prints attention weights and outputs.

## Requirements
- Python 3.8+
- PyTorch (tested with 1.13+)
- numpy
- matplotlib (for plots and heatmaps)
- tqdm

Install with:
```
python -m pip install torch numpy matplotlib tqdm
```

## Running
### 1) Scaled Dot-Product Attention test
```
python attention.py
```
This prints attention weight matrices, demonstrates softmax stability with and without scaling, and shows the output vectors.

### 2) Mini Transformer Encoder demo
```
python mini_transformer.py
```
This runs on 10 toy sentences, prints final contextual embeddings (first token of each sentence), and writes `attention_heatmap.png` to the current directory.

### 3) Character-level RNN training
```
python train_rnn.py --data_path data/tiny_corpus.txt --epochs 10 --hidden_size 128 --seq_len 100
```
- The script includes a toy "hello/help" corpus by default if `--data_path` is not provided.
- Saved outputs: `rnn_train_loss.png`, `rnn_val_loss.png`, and `rnn_samples.txt` containing three samples at temperatures 0.7, 1.0, and 1.2.

<img width="954" height="1559" alt="image" src="https://github.com/user-attachments/assets/f166841c-98d6-4d5f-a686-00ebc001a9cc" />
