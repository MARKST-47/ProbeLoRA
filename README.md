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
|   └── train.py               ← Main LoRA fine-tuning (PEFT)
│   └── backbone.py            ← DINOv2 + CLIP-ViT loading
|   └── probing.py             ← To run probe through layers
├── scripts/
|   └── clip_matrix.sub        ← CLIP model probe + runs
|   └── dino_matrix.sub        ← DINO model probe + runs
├── requirements.txt
└── README.md
```

---

## Environment Setup

---

## Execution Pipeline

---

## Core Strategy Overview

Our project operates under a strict **fixed parameter budget**. Instead of giving more parameters to the model, we take the standard budget — which normally gives all $N$ transformer layers a flat rank of $r_{\text{base}}$ (a total pool of $r_{\text{base}} \times N$ rank points) — and distribute it dynamically based on our upfront diagnostic "report card."

> **Note on scale:** `lora_alpha` is kept constant ($\alpha = 16$) across all layers for every strategy. We are not changing the training volume; we are adjusting the **width of the learning pathway (the rank $r$)** for each layer. A wider rank gives a layer more structural capacity to learn new features, while a narrower rank preserves more of the pretrained representation.

Here is how our four core allocation strategies work in practice:

### 1. Vanilla (Uniform Baseline)

This is the standard default. It distributes the parameter budget completely equally across the entire network. Every transformer layer is assigned an identical, flat rank.

**Formula:**

$$r_i = r_{\text{base}}$$

(where `r_base = 8` across all `N` layers)

### 2. Support the Weak (Strategy A)

This strategy targets the layers that performed poorly on the diagnostic probe and gives them the largest ranks, on the theory that already-competent layers need little adaptation, so training capacity is better spent helping struggling layers catch up.

**Formula:**

First, invert the probing accuracy scores $S_i$ into a failure metric:

$$V_i = 1.0 - S_i$$

Then distribute the total rank pool proportionally to those failure values:

$$r_i = \max\left(1, \left\lfloor \frac{V_i}{\sum_{j=1}^{N} V_j} \cdot r_{\text{base}} \cdot N \right\rfloor\right)$$

### 3. Amplify the Strong (Strategy B)

This strategy does the opposite of Strategy A. It concentrates the parameter budget into the layers that scored highest on the diagnostic probe, reinforcing the blocks that already display high task competency.

**Formula:**

Distribute the total rank pool directly according to the raw probing scores:

$$r_i = \max\left(1, \left\lfloor \frac{S_i}{\sum_{j=1}^{N} S_j} \cdot r_{\text{base}} \cdot N \right\rfloor\right)$$

### 4. Proportional Scaling (Strategy C)

This strategy avoids extreme allocation shifts. It maps layer-wise ranks smoothly between a floor and ceiling derived from the same rank budget $r_{\text{base}}$ used by the other strategies, based on where each layer sits relative to the best- and worst-performing blocks in the model.

**Formula:**

Define the floor and ceiling relative to the base rank:

$$r_{\min} = \max(1, \lfloor r_{\text{base}} / 2 \rfloor), \qquad r_{\max} = 2 \cdot r_{\text{base}}$$

Apply min-max normalization to the probing scores and map them into $[r_{\min}, r_{\max}]$:

$$r_i = \left\lfloor r_{\min} + \left( (r_{\max} - r_{\min}) \cdot \frac{S_i - \min(S)}{\max(S) - \min(S) + 10^{-6}} \right) \right\rfloor$$

(at `r_base = 8`, this yields the same `[4, 16]` range as Strategies A and B use as their effective span, keeping all three strategies budget-comparable and jointly controllable via `--lora_r`)

All three adaptive strategies share the same total-budget logic as Vanilla — they redistribute an equivalent rank pool across layers rather than increasing it, isolating the effect of **where** capacity is allocated rather than **how much** capacity is used.
