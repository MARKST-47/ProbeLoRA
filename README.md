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

Our project operates under a strict **fixed parameter budget**. Instead of giving more parameters to the model, we take the standard budget—which normally gives all 12 transformer layers a flat rank of 8 (a total pool of $12 \times 8 = 96$ rank points)—and distribute it dynamically based on our upfront diagnostic "report card."

> ⚠️ **Important Note on Scale:** We keep the scaling value `lora_alpha` completely constant ($\alpha = 16$) across all layers. We are **not** changing the training volume, instead, we are adjusting the **width of the learning highway (the Rank $r$)** for each layer. Widening the rank gives a layer more structural brain capacity to learn complex features, while shrinking it protects existing knowledge.

Here is how our four core allocation strategies work in practice:

### 🏢 1. Vanilla (The Uniform Baseline)

This is the standard industry default. It distributes the parameter budget completely equally across the entire network architecture. Every single transformer layer is assigned an identical, flat rank.

- **Formula:**
  $$r_i = r_{\text{base}}$$
  _(Where $r_{\text{base}} = 8$ across all $N$ layers)_

### 📉 2. Support the Weak (Strategy A)

This strategy targets the layers that performed poorly on the diagnostic probe and gives them the largest ranks. This operates on the theory that highly competent layers do not need modification, so your extra training capacity should be spent helping struggling layers adapt.

- **Formula:**
  First, we invert the probing accuracy scores $S_i$ into a failure metric:
  $$V_i = 1.0 - S_i$$
  Then, we distribute our total rank pool proportionally based on those failure values:
  $$r_i = \max\left(1, \left\lfloor \frac{V_i}{\sum_{j=1}^{N} V_j} \cdot r_{\text{base}} \cdot N \right\rfloor\right)$$

### 📈 3. Amplify the Strong (Strategy B)

This strategy does the exact opposite of Strategy A. It concentrates your parameter budget into the layers that naturally scored the highest on the diagnostic probe, doubling down on the blocks that already display high task competency.

- **Formula:**
  We distribute our total rank pool directly based on the raw passing accuracy scores:
  $$r_i = \max\left(1, \left\lfloor \frac{S_i}{\sum_{j=1}^{N} S_j} \cdot r_{\text{base}} \cdot N \right\rfloor\right)$$

### 📐 4. Proportional Scaling (Strategy C)

This strategy avoids extreme allocation shifts. It maps layer-wise ranks smoothly between a strict floor and ceiling based on where each layer sits relative to the absolute best and worst performing blocks in the model.

- **Formula:**
  We apply Min-Max normalization to the probing scores and map them into a strict rank range between $2$ and $16$:
  $$r_i = \left\lfloor 2 + \left( 14 \cdot \frac{S_i - \min(S)}{\max(S) - \min(S) + 10^{-6}} \right) \right\rfloor$$
