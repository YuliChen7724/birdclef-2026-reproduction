# BirdCLEF+ 2026 Reproduction

本项目用于逐步复现 BirdCLEF+ 2026 铜牌方案。

## Planned pipeline

1. Inspect metadata and audio
2. Build a small CNN-SED baseline
3. Extract Perch v2 embeddings
4. Train a distilled SED model
5. Add soundscape pseudo-labels
6. Apply temporal smoothing and rank ensemble

## Environment

- Windows + WSL2 Ubuntu
- Python 3.11
- PyTorch 2.5.1
- NVIDIA RTX 4060 Laptop GPU

