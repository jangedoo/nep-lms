"""
Training benchmark script — measures throughput across devices.
Run on each machine and compare the logs.
"""

import argparse
import logging
import platform
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
log = logging.getLogger(__name__)


class BenchmarkModel(nn.Module):
    def __init__(self, vocab_size=32000, d_model=512, nhead=8, num_layers=6, seq_len=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2048, dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)
        self.seq_len = seq_len

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.head(x)


class SyntheticDataset(Dataset):
    def __init__(self, size=2048, seq_len=256, vocab_size=32000):
        self.size = size
        self.data = torch.randint(0, vocab_size, (size, seq_len + 1))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        row = self.data[idx]
        return row[:-1], row[1:]  # input, target (next-token prediction)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_benchmark(steps=200, batch_size=16, seq_len=256, warmup_steps=10):
    device = get_device()

    log.info("=" * 60)
    log.info(f"Host     : {platform.node()}")
    log.info(f"OS       : {platform.system()} {platform.release()}")
    log.info(f"PyTorch  : {torch.__version__}")
    log.info(f"Device   : {device}")
    if device.type == "cuda":
        log.info(f"GPU      : {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    log.info(f"Batch    : {batch_size}  |  Seq len: {seq_len}  |  Steps: {steps}")
    log.info("=" * 60)

    model = BenchmarkModel(seq_len=seq_len).to(device)
    log.info(f"Parameters: {count_params(model) / 1e6:.2f}M")

    dataset = SyntheticDataset(size=max(steps, 512) * batch_size, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    step = 0
    total_tokens = 0
    losses = []

    # ---- warmup ----
    log.info(f"Warming up for {warmup_steps} steps...")
    for x, y in loader:
        if step >= warmup_steps:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        step += 1

    if device.type == "cuda":
        torch.cuda.synchronize()

    # ---- timed run ----
    log.info("Starting timed benchmark...")
    step = 0
    t_start = time.perf_counter()

    for x, y in loader:
        if step >= steps:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()

        total_tokens += batch_size * seq_len
        losses.append(loss.item())
        step += 1

        if step % 50 == 0:
            elapsed = time.perf_counter() - t_start
            tok_per_sec = total_tokens / elapsed
            log.info(f"  step {step:4d}/{steps} | loss {loss.item():.4f} | {tok_per_sec:,.0f} tokens/sec")

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t_start
    tok_per_sec = total_tokens / elapsed
    avg_loss = sum(losses) / len(losses)

    log.info("=" * 60)
    log.info(f"RESULTS")
    log.info(f"  Total time   : {elapsed:.2f}s")
    log.info(f"  Steps        : {step}")
    log.info(f"  Avg loss     : {avg_loss:.4f}")
    log.info(f"  Tokens/sec   : {tok_per_sec:,.0f}")
    log.info(f"  Samples/sec  : {(step * batch_size) / elapsed:.1f}")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch training benchmark")
    parser.add_argument("--steps", type=int, default=200, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Warmup steps (not timed)")
    args = parser.parse_args()

    run_benchmark(
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        warmup_steps=args.warmup_steps,
    )
