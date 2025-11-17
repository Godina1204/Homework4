# char_rnn.py
"""
Character-level RNN language model (PyTorch)
Toy & larger text usage. Embedding -> LSTM -> Linear -> Softmax.
Run: python char_rnn.py --data_file small.txt
"""

import argparse
import random
import math
from collections import Counter
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------- Dataset --------------
class CharDataset(Dataset):
    def __init__(self, text, seq_len):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self stoi = {ch:i for i,ch in enumerate(self.chars)}
        self.itos = {i:ch for ch,i in self.stoi.items()}
        self.data = [self.stoi[c] for c in text]
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y

# -------------- Model --------------
class CharRNN(nn.Module):
    def __init__(self, vocab_size, emb_size=64, hidden_size=128, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)                  # (B, T, E)
        out, hidden = self.lstm(emb, hidden)     # out: (B, T, H)
        logits = self.fc(out)                    # (B, T, V)
        return logits, hidden

# -------------- Sampling --------------
def sample(model, dataset, device, start_char=None, length=200, temp=1.0):
    model.eval()
    idx2char = dataset.itos
    char2idx = dataset.stoi
    if start_char is None:
        cur_idx = torch.tensor([[random.randrange(dataset.vocab_size)]], device=device)
    else:
        cur_idx = torch.tensor([[char2idx.get(start_char, 0)]], device=device)

    hidden = None
    output = []
    with torch.no_grad():
        for _ in range(length):
            logits, hidden = model(cur_idx, hidden)
            logits = logits[:, -1, :] / max(1e-8, temp)
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            output.append(idx2char[int(next_idx)])
            cur_idx = next_idx
    return ''.join(output)

# -------------- Training --------------
def train_loop(model, dataloader, val_loader, device, epochs=10, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses = [], []

    for epoch in range(1, epochs+1):
        model.train()
        total, acc_loss = 0, 0.0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            acc_loss += float(loss.item()) * xb.size(0)
            total += xb.size(0)
        train_epoch_loss = acc_loss / total
        train_losses.append(train_epoch_loss)

        # validation
        model.eval()
        with torch.no_grad():
            vtotal, vacc = 0, 0.0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits, _ = model(xb)
                loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
                vacc += float(loss.item()) * xb.size(0)
                vtotal += xb.size(0)
            val_loss = vacc / vtotal if vtotal>0 else 0.0
            val_losses.append(val_loss)

        print(f"Epoch {epoch}/{epochs} | train_loss={train_epoch_loss:.4f} | val_loss={val_loss:.4f}")

    # plot curves
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend()
    plt.title('Loss curves')
    plt.savefig('loss_curves.png')
    print("Saved loss_curves.png")

# -------------- CLI --------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default=None, help='path to text file')
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--emb', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    if args.data_file is None:
        toy = "hello hello help helo hello world help me hello!"
        text = toy * 200  # toy corpus
    else:
        with open(args.data_file, 'r', encoding='utf-8') as f:
            text = f.read()

    # split train/val
    split = int(0.9 * len(text))
    train_text = text[:split]
    val_text = text[split:]

    train_ds = CharDataset(train_text, seq_len=args.seq_len)
    val_ds = CharDataset(val_text, seq_len=args.seq_len)
    # ensure same vocab mapping
    val_ds.stoi = train_ds.stoi
    val_ds.itos = train_ds.itos
    val_ds.vocab_size = train_ds.vocab_size

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CharRNN(train_ds.vocab_size, emb_size=args.emb, hidden_size=args.hidden)
    train_loop(model, train_loader, val_loader, device, epochs=args.epochs)

    # sampling
    for temp in [0.7, 1.0, 1.2]:
        print("=== Sample (temp=", temp, ") ===")
        print(sample(model, train_ds, device, start_char='h', length=300, temp=temp))
        print()

if __name__ == '__main__':
    main()
