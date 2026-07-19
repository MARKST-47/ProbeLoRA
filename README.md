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
|   └── analyze_results.py     ← Plots and CSV saved to analysis_output folder
├── scripts/
|   └── clip_matrix.sub        ← CLIP model probe + runs
|   └── dino_matrix.sub        ← DINO model probe + runs
|   └── run_in_docker.sh       ← Pipeline setup
├── analysis_output/
├── requirements.txt
└── README.md
```

---

## Environment Setup

This project runs inside a Docker container (`pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel`) on the department's HTCondor cluster, so no local GPU setup is required.

**1. Clone the repository and install dependencies** (for local/offline testing outside the container):

```bash
git clone <repo-url>
cd probelora
pip install -r requirements.txt
```

**2. Cluster access.** Jobs run under the `UidDomain == "cs.uni-saarland.de"` requirement via `scripts/run_in_docker.sh`, which handles environment setup inside the container (installing any packages not baked into the base image, e.g. `peft`, `wandb`, `scikit-learn`).

**3. WandB.** Training and probing both log to WandB. Authenticate once per machine with:

```bash
wandb login
```

Set `wandb_project` / `wandb_entity` in `src/config.py` before the first run.

**4. Datasets.** CIFAR-100, Oxford Pets, and CUB-200 download automatically into `data/raw/` on first use via `data/datasets.py` — no manual download step needed.

---

## Execution Pipeline

The pipeline has three stages, run in order, since each depends on the previous stage's output:

**Stage 1 — Baseline diagnostic probing.** Trains a per-layer logistic regression probe on the frozen backbone's hidden states, producing the "report card" scores used to compute rank allocations:

```bash
python src/probing.py --dataset <dataset> --model_name <model_name>
```

Saves `cached_features/{dataset}_{model_safe_name}_scores.npy`.

**Stage 2 — LoRA fine-tuning.** Trains the backbone with LoRA adapters under one of the four allocation strategies. Non-vanilla strategies load the Stage 1 scores automatically:

```bash
python src/train.py --dataset <dataset> --model_name <model_name> --strategy <strategy> --epochs 10 --lora_r 8 --batch_size 64
```

Saves the adapter checkpoint to `checkpoints/{dataset}_{model_safe_name}_{strategy}/`.

**Stage 3 — Post-adaptation probing.** Re-runs diagnostic probing on the fine-tuned checkpoint, to see how much each layer's representation improved under that strategy's rank allocation:

```bash
python src/probing.py --dataset <dataset> --model_name <model_name> --checkpoint_path checkpoints/{dataset}_{model_safe_name}_{strategy}
```

Saves `cached_features/{dataset}_{model_safe_name}_adapted_{strategy}_scores.npy`.

**Running the full grid on the cluster.** `scripts/dino_matrix.sub` and `scripts/clip_matrix.sub` queue all (dataset × strategy) combinations for each backbone. Since Stage 3 depends on Stage 2's checkpoints, submit training and probing separately and wait for training to finish first:

```bash
condor_submit scripts/dino_matrix.sub   # Stage 1 + Stage 2
condor_wait condor_logs/train.log
condor_submit scripts/dino_matrix.sub   # Stage 3, once checkpoints exist
```

(same for `clip_matrix.sub`).

**Stage 4 — Analysis.** Once all probing scores are on disk, generate the summary tables and plots:

```bash
python src/analyze_results.py
```

Reads directly from `cached_features/`, matching pre- and post-adaptation scores per (backbone, dataset, strategy). Outputs land in `analysis_output/`: `summary_table.csv`, `layer_detail.csv`, and per-config probing-accuracy plots, plus a console leaderboard of the best strategy per (dataset, backbone).

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

---

## Backbones and Datasets

Two ViT-style backbones are evaluated, each with 12 transformer layers:

- `facebook/dinov2-base`
- `openai/clip-vit-base-patch16`

Across three image classification datasets of increasing granularity:

- CIFAR-100 (coarse-grained, 100 classes)
- Oxford Pets (37 classes)
- CUB-200 (fine-grained, 200 classes)

This gives a full 2 × 3 × 4 grid (backbone × dataset × strategy) of 24 training runs, backed by 6 baseline and 24 post-adaptation diagnostic probing runs.

---

## Results

See `analysis_output/` after running `src/analyze_results.py`:

- `summary_table.csv` — average pre-/post-adaptation probing accuracy, probing accuracy gain, and rank-vs-gain correlation, per (backbone, dataset, strategy)
- `layer_detail.csv` — full per-layer breakdown
- `probing_by_layer_<dataset>_<backbone>.png` — per-layer probing accuracy, all strategies vs. the pre-adaptation baseline
- `probing_delta_bars.png`, `rank_vs_delta_corr.png`, `val_acc_leaderboard.png` — aggregate comparisons across strategies
