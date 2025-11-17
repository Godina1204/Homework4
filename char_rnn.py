# char_rnn.py
"""
Character-level RNN language model (PyTorch)
Toy & larger text usage. Embedding -> LSTM -> Linear -> Softmax.
Run: python char_rnn.py --data_file small.txt
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import trange
import matplotlib.pyplot as plt


class CharRNN(nn.Module):
    def __init__(self, vocab_size, emb_size=32, hidden_size=128, rnn_type='gru'):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size)
        if rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(emb_size, hidden_size, batch_first=True)
        elif rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(emb_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.RNN(emb_size, hidden_size, batch_first=True)
        self.lin = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch, seq)
        e = self.emb(x)
        out, hidden = self.rnn(e, hidden)
        logits = self.lin(out)
        return logits, hidden


def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {c:i for i,c in enumerate(chars)}
    itos = {i:c for c,i in stoi.items()}
    return stoi, itos


def create_batches(text, seq_len, batch_size, stoi):
    # convert to indices
    data = [stoi[c] for c in text]
    num_batches = len(data) // (seq_len * batch_size)
    if num_batches == 0:
        raise ValueError('Not enough data. Reduce seq_len or batch_size or provide more text.')
    data = data[:num_batches * batch_size * seq_len]
    arr = np.array(data).reshape(batch_size, -1)
    for i in range(0, arr.shape[1], seq_len):
        x = arr[:, i:i+seq_len]
        y = np.zeros_like(x)
        y[:, :-1] = x[:, 1:]
        # last target is next char after the sequence; for simplicity wrap-around
        y[:, -1] = np.roll(x, -1, axis=1)[:, -1]
        yield torch.LongTensor(x), torch.LongTensor(y)


def sample(model, start_str, stoi, itos, length=200, temp=1.0, device='cpu'):
    model.eval()
    hidden = None
    input_ids = torch.LongTensor([[stoi.get(c, 0) for c in start_str]]).to(device)
    out, hidden = model(input_ids, hidden)
    last = input_ids[0,-1].unsqueeze(0).unsqueeze(0).to(device)
    generated = start_str
    for _ in range(length):
        logits, hidden = model(last, hidden)
        logits = logits[:, -1, :] / temp
        probs = torch.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, num_samples=1).item()
        ch = itos[idx]
        generated += ch
        last = torch.LongTensor([[idx]]).to(device)
    return generated


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--emb_size', type=int, default=32)
    parser.add_argument('--rnn_type', type=str, default='gru')
    args = parser.parse_args()

    # toy corpus if none provided
    if args.data_path is None:
        toy = "hello hello help me hello help hello world\n" * 200  # small toy repeated
        text = toy
    else:
        with open(args.data_path, 'r', encoding='utf-8') as f:
            text = f.read()

    stoi, itos = build_vocab(text)
    vocab_size = len(stoi)
    print('Vocab size:', vocab_size)

    model = CharRNN(vocab_size, emb_size=args.emb_size, hidden_size=args.hidden_size, rnn_type=args.rnn_type)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_losses = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        try:
            for x, y in create_batches(text, args.seq_len, args.batch_size, stoi):
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad()
                logits, _ = model(x)
                # flatten
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                loss.backward()
                opt.step()
                total_loss += loss.item()
                batches += 1
        except ValueError as e:
            print('Warning:', e)
            break
        avg = total_loss / max(1, batches)
        train_losses.append(avg)
        print(f'Epoch {epoch+1}/{args.epochs} train loss: {avg:.4f} (batches: {batches})')

    # save loss curve
    plt.figure()
    plt.plot(train_losses)
    plt.xlabel('epoch')
    plt.ylabel('train_loss')
    plt.savefig('rnn_train_loss.png')
    print('Saved training loss to rnn_train_loss.png')

    # sampling at three temperatures
    temps = [0.7, 1.0, 1.2]
    with open('rnn_samples.txt', 'w', encoding='utf-8') as f:
        for t in temps:
            s = sample(model, start_str='h', stoi=stoi, itos=itos, length=300, temp=t, device=device)
            f.write(f'Temperature {t}\n')
            f.write(s + '\n\n')
    print('Saved samples to rnn_samples.txt')
