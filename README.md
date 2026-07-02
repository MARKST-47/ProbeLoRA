# ProbeLoRA: Adaptive Rank Allocation via Layer Probing

Adaptive LoRA rank allocation for Vision Transformers, guided by per-layer diagnostic probing accuracy using Logistic Regression. Three strategies (Support Weak / Amplify Strong / Proportional) are compared against a Vanilla LoRA baseline across CIFAR-100, Oxford Pets and CUB-200.

---

## Project Structure

```
.
├── data/
│   ├── datasets.py            ← DataLoader factory (CIFAR-100, Oxford Pets, CUB-200)
│   └── raw/                   ← Datasets download here (Auto-created)
├── src/
|   └── config.py              ← Single dataclass every setting
|   └── train.py               ← Main LoRA fine‑tuning (PEFT)
│   └── backbone.py            ← DINOv2 + CLIP-ViT loading
|   └── probing.py             ← To run probe through layers
├── scripts/                   ← Run scripts on the cluster
├── requirements.txt
└── README.md
```

---

## Environment Setup

---

## Execution Pipeline

---

## Core Strategy Overview
