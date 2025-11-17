
# Character-Level RNN Language Model, Mini Transformer Encoder & Dot-Product-Attention

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

OUTPUT :

<img width="576" height="442" alt="image" src="https://github.com/user-attachments/assets/f0b090e3-e809-430c-b5e0-3e9c62ed11d6" />


### 2) Mini Transformer Encoder demo
```
python mini_transformer.py
```
This runs on 10 toy sentences, prints final contextual embeddings (first token of each sentence), and writes `attention_heatmap.png` to the current directory.

OUTPUT :

<img width="1081" height="835" alt="image" src="https://github.com/user-attachments/assets/c7526a65-3ddc-4efd-997d-30e9bc97de31" />


### 3) Character-level RNN training
```
python train_rnn.py --data_path data/tiny_corpus.txt --epochs 10 --hidden_size 128 --seq_len 100
```
- The script includes a toy "hello/help" corpus by default if `--data_path` is not provided.
- Saved outputs: `rnn_train_loss.png`, `rnn_val_loss.png`, and `rnn_samples.txt` containing three samples at temperatures 0.7, 1.0, and 1.2.
-OUTPUT:
- <img width="781" height="685" alt="image" src="https://github.com/user-attachments/assets/e9479590-7cec-45db-bbd9-f7df9cb03573" />
