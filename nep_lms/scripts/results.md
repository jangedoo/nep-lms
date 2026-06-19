## RTX 2070 Super

➜ python nep_lms/scripts/benchmark_training.py 
╭───────────────── Training Benchmark ──────────────────╮
│   Host      : DESKTOP-2T1H4VF                         │
│   OS        : Linux 6.6.114.1-microsoft-standard-WSL2 │
│   PyTorch   : 2.10.0+cu128                            │
│   Device    : cuda                                    │
│   GPU     : NVIDIA GeForce RTX 2070 SUPER (8.6 GB)    │
│   Precision : FP32                                    │
│   Config    : batch=16  seq_len=256  steps=200        │
╰───────────────────────────────────────────────────────╯
  Parameters : 37.62M
  Warming up (10 steps)...
Training ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 200/200   loss=10.4101     32,995 tok/s    12.1 GB/s 0:00:24 0:00:00
                Results                 
╭──────────────────────────┬───────────╮
│ Precision                │      FP32 │
│ Total time               │   24.83 s │
│ Steps                    │       200 │
│ Avg loss                 │   10.4409 │
│ Tokens / sec             │    32,991 │
│ Samples / sec            │     128.9 │
│ Est. Parameter bandwidth │ 12.1 GB/s │
│ GPU allocated            │   0.63 GB │
│ GPU reserved             │   2.88 GB │
│ GPU peak                 │   2.59 GB │
│ RAM                      │   1.04 GB │
╰──────────────────────────┴───────────╯


python nep_lms/scripts/benchmark_training.py --fp16
╭───────────────── Training Benchmark ──────────────────╮
│   Host      : DESKTOP-2T1H4VF                         │
│   OS        : Linux 6.6.114.1-microsoft-standard-WSL2 │
│   PyTorch   : 2.10.0+cu128                            │
│   Device    : cuda                                    │
│   GPU     : NVIDIA GeForce RTX 2070 SUPER (8.6 GB)    │
│   Precision : FP16 (autocast)                         │
│   Config    : batch=16  seq_len=256  steps=200        │
╰───────────────────────────────────────────────────────╯
  Parameters : 37.62M
  Warming up (10 steps)...
Training ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 200/200   loss=10.8942     74,661 tok/s    27.4 GB/s 0:00:10 0:00:00
                   Results                    
╭──────────────────────────┬─────────────────╮
│ Precision                │ FP16 (autocast) │
│ Total time               │         10.98 s │
│ Steps                    │             200 │
│ Avg loss                 │         10.7188 │
│ Tokens / sec             │          74,642 │
│ Samples / sec            │           291.6 │
│ Est. Parameter bandwidth │       27.4 GB/s │
│ GPU allocated            │         0.62 GB │
│ GPU reserved             │         2.68 GB │
│ GPU peak                 │         2.16 GB │
│ RAM                      │         1.08 GB │
╰──────────────────────────┴─────────────────╯

## RTX 5070TI

╭───────────────── Training Benchmark ──────────────────╮
│   Host      : DESKTOP-2T1H4VF                         │
│   OS        : Linux 6.18.33.1-microsoft-standard-WSL2 │
│   PyTorch   : 2.10.0+cu128                            │
│   Device    : cuda                                    │
│   GPU     : NVIDIA GeForce RTX 5070 Ti (17.1 GB)      │
│   Precision : FP32                                    │
│   Config    : batch=16  seq_len=256  steps=200        │
╰───────────────────────────────────────────────────────╯
  Parameters : 37.62M
  Warming up (10 steps)...
Training ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 200/200   loss=10.4128     92,184 tok/s    33.9 GB/s 0:00:08 0:00:00
                Results                 
╭──────────────────────────┬───────────╮
│ Precision                │      FP32 │
│ Total time               │    8.89 s │
│ Steps                    │       200 │
│ Avg loss                 │   10.4426 │
│ Tokens / sec             │    92,149 │
│ Samples / sec            │     360.0 │
│ Est. Parameter bandwidth │ 33.9 GB/s │
│ GPU allocated            │   0.63 GB │
│ GPU reserved             │   2.88 GB │
│ GPU peak                 │   2.59 GB │
│ RAM                      │   1.36 GB │
╰──────────────────────────┴───────────╯


╭───────────────── Training Benchmark ──────────────────╮
│   Host      : DESKTOP-2T1H4VF                         │
│   OS        : Linux 6.18.33.1-microsoft-standard-WSL2 │
│   PyTorch   : 2.10.0+cu128                            │
│   Device    : cuda                                    │
│   GPU     : NVIDIA GeForce RTX 5070 Ti (17.1 GB)      │
│   Precision : FP16 (autocast)                         │
│   Config    : batch=16  seq_len=256  steps=200        │
╰───────────────────────────────────────────────────────╯
  Parameters : 37.62M
  Warming up (10 steps)...
Training ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 200/200   loss=10.9311    156,480 tok/s    57.5 GB/s 0:00:05 0:00:00
                   Results                    
╭──────────────────────────┬─────────────────╮
│ Precision                │ FP16 (autocast) │
│ Total time               │          5.24 s │
│ Steps                    │             200 │
│ Avg loss                 │         10.7193 │
│ Tokens / sec             │         156,394 │
│ Samples / sec            │           610.9 │
│ Est. Parameter bandwidth │       57.5 GB/s │
│ GPU allocated            │         0.63 GB │
│ GPU reserved             │         2.66 GB │
│ GPU peak                 │         2.14 GB │
│ RAM                      │         1.45 GB │
╰──────────────────────────┴─────────────────╯



## RTX PRO 4500 Blackwell


╭──────────────── Training Benchmark ─────────────────╮
│   Host      : e3d9db47ae78                          │
│   OS        : Linux 6.8.0-124-generic               │
│   PyTorch   : 2.10.0+cu128                          │
│   Device    : cuda                                  │
│   GPU     : NVIDIA RTX PRO 4500 Blackwell (33.7 GB) │
│   Precision : FP32                                  │
│   Config    : batch=16  seq_len=256  steps=200      │
╰─────────────────────────────────────────────────────╯
  Parameters : 37.62M
  Warming up (10 steps)...
Training ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 200/200   loss=10.4158     99,792 tok/s    36.7 GB/s 0:00:08 0:00:00
                Results                 
╭──────────────────────────┬───────────╮
│ Precision                │      FP32 │
│ Total time               │    8.21 s │
│ Steps                    │       200 │
│ Avg loss                 │   10.4429 │
│ Tokens / sec             │    99,771 │
│ Samples / sec            │     389.7 │
│ Est. Parameter bandwidth │ 36.7 GB/s │
│ GPU allocated            │   0.62 GB │
│ GPU reserved             │   2.88 GB │
│ GPU peak                 │   2.59 GB │
│ RAM                      │   1.35 GB │
╰──────────────────────────┴───────────╯



oot@e3d9db47ae78:/workspace/nep-lms# poetry run python nep_lms/scripts/benchmark_training.py --fp16
╭──────────────── Training Benchmark ─────────────────╮
│   Host      : e3d9db47ae78                          │
│   OS        : Linux 6.8.0-124-generic               │
│   PyTorch   : 2.10.0+cu128                          │
│   Device    : cuda                                  │
│   GPU     : NVIDIA RTX PRO 4500 Blackwell (33.7 GB) │
│   Precision : FP16 (autocast)                       │
│   Config    : batch=16  seq_len=256  steps=200      │
╰─────────────────────────────────────────────────────╯
  Parameters : 37.62M
  Warming up (10 steps)...
Training ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 200/200   loss=10.9081    209,597 tok/s    77.0 GB/s 0:00:03 0:00:00
                   Results                    
╭──────────────────────────┬─────────────────╮
│ Precision                │ FP16 (autocast) │
│ Total time               │          3.91 s │
│ Steps                    │             200 │
│ Avg loss                 │         10.7194 │
│ Tokens / sec             │         209,498 │
│ Samples / sec            │           818.4 │
│ Est. Parameter bandwidth │       77.0 GB/s │
│ GPU allocated            │         0.63 GB │
│ GPU reserved             │         2.66 GB │
│ GPU peak                 │         2.14 GB │
│ RAM                      │         1.43 GB │
╰──────────────────────────┴─────────────────╯

## RTX PRO 4000 Blackwell

╭──────────────── Training Benchmark ─────────────────╮
│   Host      : sanjaya-desktop                       │
│   OS        : Linux 7.0.0-22-generic                │
│   PyTorch   : 2.10.0+cu128                          │
│   Device    : cuda                                  │
│   GPU     : NVIDIA RTX PRO 4000 Blackwell (25.1 GB) │
│   Precision : FP32                                  │
│   Config    : batch=16  seq_len=256  steps=200      │
╰─────────────────────────────────────────────────────╯
  Parameters : 37.62M
  Warming up (10 steps)...
Training ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 200/200   loss=10.4124     63,586 tok/s    23.4 GB/s 0:00:12 0:00:00
                Results                 
╭──────────────────────────┬───────────╮
│ Precision                │      FP32 │
│ Total time               │   12.89 s │
│ Steps                    │       200 │
│ Avg loss                 │   10.4419 │
│ Tokens / sec             │    63,574 │
│ Samples / sec            │     248.3 │
│ Est. Parameter bandwidth │ 23.4 GB/s │
│ GPU allocated            │   0.62 GB │
│ GPU reserved             │   2.88 GB │
│ GPU peak                 │   2.59 GB │
│ RAM                      │   1.29 GB │
╰──────────────────────────┴───────────╯


oetry run python nep_lms/scripts/benchmark_training.py  --fp16
╭──────────────── Training Benchmark ─────────────────╮
│   Host      : sanjaya-desktop                       │
│   OS        : Linux 7.0.0-22-generic                │
│   PyTorch   : 2.10.0+cu128                          │
│   Device    : cuda                                  │
│   GPU     : NVIDIA RTX PRO 4000 Blackwell (25.1 GB) │
│   Precision : FP16 (autocast)                       │
│   Config    : batch=16  seq_len=256  steps=200      │
╰─────────────────────────────────────────────────────╯
  Parameters : 37.62M
  Warming up (10 steps)...
Training ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 200/200   loss=10.8897    155,191 tok/s    57.0 GB/s 0:00:05 0:00:00
                   Results                    
╭──────────────────────────┬─────────────────╮
│ Precision                │ FP16 (autocast) │
│ Total time               │          5.28 s │
│ Steps                    │             200 │
│ Avg loss                 │         10.7183 │
│ Tokens / sec             │         155,125 │
│ Samples / sec            │           606.0 │
│ Est. Parameter bandwidth │       57.0 GB/s │
│ GPU allocated            │         0.63 GB │
│ GPU reserved             │         2.66 GB │
│ GPU peak                 │         2.14 GB │
│ RAM                      │         1.45 GB │
╰──────────────────────────┴─────────────────╯