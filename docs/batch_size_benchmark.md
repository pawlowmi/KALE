# Batch Size Benchmark — ViT-L-14 on H100 80GB

**Setup:** Single H100 80GB, ViT-L-14 with gradient checkpointing, AdamW optimizer,
polynomial kernel loss (full forward + kernel + backward + optimizer step).

## Throughput vs Batch Size

| BS | samp/s | s/batch | Memory (GB) |
|----|--------|---------|-------------|
| 128 | 63.3 | 2.02 | 12.2 |
| 192 | 63.4 | 3.03 | 16.4 |
| 256 | 63.7 | 4.02 | 21.0 |
| 320 | 63.4 | 5.04 | 25.0 |
| 384 | 63.5 | 6.05 | 29.2 |
| 448 | 63.2 | 7.09 | 33.4 |
| 512 | 63.0 | 8.13 | 37.7 |
| 640 | 63.2 | 10.13 | 45.7 |
| 768 | 63.2 | 12.15 | 53.7 |
| 896 | 63.2 | 14.18 | 62.3 |
| 1024 | OOM | — | — |

## Conclusion

Throughput is **flat at ~63 samp/s** across all batch sizes — bottleneck is ViT
forward+backward, not memory bandwidth or kernel compute.

**Optimal: bs=128** — same throughput as bs=896, uses 5× less memory (12GB vs 62GB),
leaves headroom for NCCL buffers in DDP, and avoids quadratic kernel matrix growth.

With 8× H100s at bs=128: **effective batch = 1024, ~500 samp/s total**.
