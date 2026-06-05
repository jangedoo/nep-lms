"""
Training benchmark script — measures throughput across devices.
Run on each machine and compare the logs.
"""

import argparse
import platform
import time

import psutil
import torch
import torch.nn as nn
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch.utils.data import DataLoader, Dataset

console = Console()

VOCAB_SIZE = 32000


class BenchmarkModel(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2048, dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, x):
        return self.head(self.transformer(self.embedding(x)))


class SyntheticDataset(Dataset):
    def __init__(self, size, seq_len):
        self.data = torch.randint(0, VOCAB_SIZE, (size, seq_len + 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return row[:-1], row[1:]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def get_memory_stats(device):
    stats = {}
    if device.type == "cuda":
        sync(device)
        stats["GPU allocated"] = f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
        stats["GPU reserved"] = f"{torch.cuda.memory_reserved() / 1e9:.2f} GB"
        stats["GPU peak"] = f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
    elif device.type == "mps":
        stats["MPS allocated"] = f"{torch.mps.current_allocated_memory() / 1e9:.2f} GB"
    stats["RAM"] = f"{psutil.Process().memory_info().rss / 1e9:.2f} GB"
    return stats


def run_benchmark(steps=200, batch_size=16, seq_len=256, warmup_steps=10, fp16=False):
    device = get_device()

    gpu_line = ""
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        gpu_line = f"\n  GPU     : {torch.cuda.get_device_name(0)} ({props.total_memory / 1e9:.1f} GB)"

    precision_label = "FP16 (autocast)" if fp16 else "FP32"
    info = (
        f"  Host      : [bold]{platform.node()}[/bold]\n"
        f"  OS        : {platform.system()} {platform.release()}\n"
        f"  PyTorch   : {torch.__version__}\n"
        f"  Device    : [bold cyan]{device}[/bold cyan]{gpu_line}\n"
        f"  Precision : [bold magenta]{precision_label}[/bold magenta]\n"
        f"  Config    : batch={batch_size}  seq_len={seq_len}  steps={steps}"
    )
    console.print(Panel(info, title="[bold white]Training Benchmark[/bold white]", expand=False))

    model = BenchmarkModel(d_model=384).to(device)
    num_params = count_params(model)
    console.print(f"  Parameters : [bold]{num_params / 1e6:.2f}M[/bold]")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    dataset = SyntheticDataset(size=max(steps + warmup_steps, 512) * batch_size, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()

    console.print(f"  Warming up [dim]({warmup_steps} steps)[/dim]...")
    step = 0
    for x, y in loader:
        if step >= warmup_steps:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
            loss = criterion(model(x).view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        optimizer.step()
        step += 1
    sync(device)

    # Estimated bytes moved per step (FP32 AdamW):
    # forward: read params (1x)
    # backward: read params (1x) + write grads (1x)
    # optimizer: read params+grads+m1+m2 (4x), write params+m1+m2 (3x)
    # total = 10 passes over param bytes
    bytes_per_step = num_params * 4 * 10

    step = 0
    total_tokens = 0
    losses = []
    t_start = time.perf_counter()

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TextColumn("  loss=[white]{task.fields[loss]:.4f}[/white]"),
        TextColumn("  [green]{task.fields[tok_sec]:>8,.0f}[/green] tok/s"),
        TextColumn("  [yellow]{task.fields[gbps]:>5.1f}[/yellow] GB/s"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    task_id = progress.add_task("Training", total=steps, loss=0.0, tok_sec=0.0, gbps=0.0)

    with progress:
        for x, y in loader:
            if step >= steps:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
                loss = criterion(model(x).view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            optimizer.step()

            step += 1
            total_tokens += batch_size * seq_len
            losses.append(loss.item())

            elapsed = time.perf_counter() - t_start
            progress.update(
                task_id,
                advance=1,
                loss=loss.item(),
                tok_sec=total_tokens / elapsed,
                gbps=bytes_per_step * step / elapsed / 1e9,
            )

    sync(device)
    elapsed = time.perf_counter() - t_start
    avg_loss = sum(losses) / len(losses)
    tok_sec = total_tokens / elapsed
    gbps = bytes_per_step * step / elapsed / 1e9

    table = Table(box=box.ROUNDED, show_header=False, title="[bold]Results[/bold]", min_width=40)
    table.add_column("Metric", style="bold dim")
    table.add_column("Value", justify="right")
    table.add_row("Precision", f"[bold magenta]{precision_label}[/bold magenta]")
    table.add_row("Total time", f"{elapsed:.2f} s")
    table.add_row("Steps", str(step))
    table.add_row("Avg loss", f"{avg_loss:.4f}")
    table.add_row("Tokens / sec", f"[bold green]{tok_sec:,.0f}[/bold green]")
    table.add_row("Samples / sec", f"{(step * batch_size) / elapsed:.1f}")
    table.add_row("Est. Parameter bandwidth", f"[bold yellow]{gbps:.1f} GB/s[/bold yellow]")
    for k, v in get_memory_stats(device).items():
        table.add_row(k, v)
    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch training benchmark")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed-precision via autocast")
    args = parser.parse_args()

    run_benchmark(
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
    )
