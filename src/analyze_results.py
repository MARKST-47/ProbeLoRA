import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CACHE_DIR = "./cached_features"
OUT_DIR = "./analysis_output"
os.makedirs(OUT_DIR, exist_ok=True)

BACKBONES = {
    "facebook/dinov2-base": "facebook_dinov2-base",
    "openai/clip-vit-base-patch16": "openai_clip-vit-base-patch16",
}
DATASETS = ["cifar100", "oxford_pets", "cub200"]
STRATEGIES = ["vanilla", "support_weak", "amplify_strong", "proportional"]
BASE_R = 8

# Format: (backbone, dataset, strategy): (best_val_accuracy, best_train_accuracy)
MANUAL_RUN_METRICS = {
    # DINOv2 Base
    ("facebook/dinov2-base", "cifar100", "vanilla"): (0.0, 0.0),
    ("facebook/dinov2-base", "cifar100", "support_weak"): (0.0, 0.0),
    ("facebook/dinov2-base", "cifar100", "amplify_strong"): (0.0, 0.0),
    ("facebook/dinov2-base", "cifar100", "proportional"): (0.0, 0.0),

    ("facebook/dinov2-base", "oxford_pets", "vanilla"): (0.0, 0.0),
    ("facebook/dinov2-base", "oxford_pets", "support_weak"): (0.0, 0.0),
    ("facebook/dinov2-base", "oxford_pets", "amplify_strong"): (0.0, 0.0),
    ("facebook/dinov2-base", "oxford_pets", "proportional"): (0.0, 0.0),

    ("facebook/dinov2-base", "cub200", "vanilla"): (0.0, 0.0),
    ("facebook/dinov2-base", "cub200", "support_weak"): (0.0, 0.0),
    ("facebook/dinov2-base", "cub200", "amplify_strong"): (0.0, 0.0),
    ("facebook/dinov2-base", "cub200", "proportional"): (0.0, 0.0),
    # CLIP Vit-Base-Patch16
    ("openai/clip-vit-base-patch16", "cifar100", "vanilla"): (0.0, 0.0),
    ("openai/clip-vit-base-patch16", "cifar100", "support_weak"): (0.0, 0.0),
    ("openai/clip-vit-base-patch16", "cifar100", "amplify_strong"): (0.0, 0.0),
    ("openai/clip-vit-base-patch16", "cifar100", "proportional"): (0.0, 0.0),

    ("openai/clip-vit-base-patch16", "oxford_pets", "vanilla"): (0.0, 0.0),
    ("openai/clip-vit-base-patch16", "oxford_pets", "support_weak"): (0.0, 0.0),
    ("openai/clip-vit-base-patch16", "oxford_pets", "amplify_strong"): (0.0, 0.0),
    ("openai/clip-vit-base-patch16", "oxford_pets", "proportional"): (0.0, 0.0),

    ("openai/clip-vit-base-patch16", "cub200", "vanilla"): (0.0, 0.0),
    ("openai/clip-vit-base-patch16", "cub200", "support_weak"): (0.0, 0.0),
    ("openai/clip-vit-base-patch16", "cub200", "amplify_strong"): (0.0, 0.0),
    ("openai/clip-vit-base-patch16", "cub200", "proportional"): (0.0, 0.0),
}

plt.rcParams.update({"figure.dpi": 120, "font.size": 9})

def compute_rank_pattern(probing_scores, strategy, base_r=BASE_R):
    num_layers = len(probing_scores)
    if strategy == "vanilla":
        return [base_r] * num_layers
    if strategy == "support_weak":
        inverted = [1.0 - s for s in probing_scores]
        total = sum(inverted)
        raw = [(v / total) * base_r * num_layers for v in inverted]
    elif strategy == "amplify_strong":
        total = sum(probing_scores)
        raw = [(s / total) * base_r * num_layers for s in probing_scores]
    elif strategy == "proportional":
        min_s, max_s = min(probing_scores), max(probing_scores)
        min_r, max_r = max(1, base_r // 2), base_r * 2
        raw = [min_r + (max_r - min_r) * (s - min_s) / (max_s - min_s + 1e-6) for s in probing_scores]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return [max(1, int(r)) for r in raw]

def load_scores(path):
    if not os.path.exists(path):
        return None
    return np.load(path).tolist()

def collect_layer_data():
    rows = []
    for model_name, safe_name in BACKBONES.items():
        for dataset in DATASETS:
            pre_path = os.path.join(CACHE_DIR, f"{dataset}_{safe_name}_scores.npy")
            pre_scores = load_scores(pre_path)
            if pre_scores is None:
                print(f"[skip] Missing baseline pre-adapt scores: {pre_path}")
                continue
            for strategy in STRATEGIES:
                post_path = os.path.join(CACHE_DIR, f"{dataset}_{safe_name}_adapted_{strategy}_scores.npy")
                post_scores = load_scores(post_path)
                if post_scores is None:
                    print(f"[skip] Missing post-adapt scores: {post_path}")
                    continue
                if len(pre_scores) != len(post_scores):
                    print(f"[warn] Layer count mismatch for {dataset}/{safe_name}/{strategy} — skipping")
                    continue
                ranks = compute_rank_pattern(pre_scores, strategy)
                for layer_idx, (pre, post, rank) in enumerate(zip(pre_scores, post_scores, ranks)):
                    rows.append(dict(
                        backbone=model_name, dataset=dataset, strategy=strategy,
                        layer=layer_idx, assigned_rank=rank,
                        pre_acc=pre * 100, post_acc=post * 100,
                    ))
    if not rows:
        raise SystemExit("No valid data pairs collected from your scores folder. Verification failed.")
    df = pd.DataFrame(rows)
    df["delta"] = df["post_acc"] - df["pre_acc"]
    return df

def build_summary(layer_df):
    agg = layer_df.groupby(["backbone", "dataset", "strategy"]).agg(
        avg_pre_acc=("pre_acc", "mean"),
        avg_post_acc=("post_acc", "mean"),
        avg_delta=("delta", "mean"),
        max_layer_delta=("delta", "max"),
    ).reset_index()

    corrs = []
    for (bb, ds, strat), g in layer_df.groupby(["backbone", "dataset", "strategy"]):
        if strat == "vanilla" or g["assigned_rank"].nunique() <= 1:
            corrs.append((bb, ds, strat, np.nan))
        else:
            r = np.corrcoef(g["assigned_rank"], g["delta"])[0, 1]
            corrs.append((bb, ds, strat, r))
    corr_df = pd.DataFrame(corrs, columns=["backbone", "dataset", "strategy", "rank_delta_corr"])
    summary = agg.merge(corr_df, on=["backbone", "dataset", "strategy"])

    val_accs, train_accs = [], []
    for _, row in summary.iterrows():
        key = (row["backbone"], row["dataset"], row["strategy"])
        metrics = MANUAL_RUN_METRICS.get(key, (np.nan, np.nan))
        val_accs.append(metrics[0])
        train_accs.append(metrics[1])
        
    summary["best_val_acc"] = val_accs
    summary["best_train_acc"] = train_accs
    return summary.sort_values(["dataset", "backbone", "strategy"])

# Complete Visualization Suite
def plot_probing_by_layer(layer_df):
    for (dataset, backbone), g in layer_df.groupby(["dataset", "backbone"]):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        baseline = g[g["strategy"] == "vanilla"][["layer", "pre_acc"]].drop_duplicates().sort_values("layer")
        ax.plot(baseline["layer"], baseline["pre_acc"], "k--", label="Pre-adaptation baseline", linewidth=1.5)
        for strategy in STRATEGIES:
            sg = g[g["strategy"] == strategy].sort_values("layer")
            if sg.empty: continue
            ax.plot(sg["layer"], sg["post_acc"], marker="o", markersize=3, label=f"{strategy} (post)")
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Probing accuracy (%)")
        safe_bb = backbone.replace("/", "_")
        ax.set_title(f"{dataset} — {backbone}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"probing_by_layer_{dataset}_{safe_bb}.png"))
        plt.close(fig)

def plot_probing_delta_bars(summary):
    backbones = summary["backbone"].unique()
    fig, axes = plt.subplots(1, len(backbones), figsize=(7 * len(backbones), 4.5), squeeze=False)
    for ax, backbone in zip(axes[0], backbones):
        sub = summary[summary["backbone"] == backbone]
        pivot = sub.pivot(index="dataset", columns="strategy", values="avg_delta").reindex(columns=STRATEGIES)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(f"Avg probing accuracy gain — {backbone}")
        ax.set_ylabel("Avg post - pre probing acc (pp)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "probing_delta_bars.png"))
    plt.close(fig)

def plot_rank_vs_delta_corr(summary):
    non_vanilla = summary[summary["strategy"] != "vanilla"]
    backbones = non_vanilla["backbone"].unique()
    if len(backbones) == 0: return
    fig, axes = plt.subplots(1, len(backbones), figsize=(7 * len(backbones), 4.5), squeeze=False)
    for ax, backbone in zip(axes[0], backbones):
        sub = non_vanilla[non_vanilla["backbone"] == backbone]
        pivot = sub.pivot(index="dataset", columns="strategy", values="rank_delta_corr")
        if pivot.empty: continue
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(f"Correlation(assigned rank, layer delta) — {backbone}")
        ax.set_ylabel("Pearson r")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylim(-1, 1)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "rank_vs_delta_corr.png"))
    plt.close(fig)

def plot_val_acc_leaderboard(summary):
    if "best_val_acc" not in summary.columns or summary["best_val_acc"].isna().all():
        return
    backbones = summary["backbone"].unique()
    fig, axes = plt.subplots(1, len(backbones), figsize=(7 * len(backbones), 4.5), squeeze=False)
    for ax, backbone in zip(axes[0], backbones):
        sub = summary[summary["backbone"] == backbone]
        pivot = sub.pivot(index="dataset", columns="strategy", values="best_val_acc").reindex(columns=STRATEGIES)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(f"Best Val Accuracy Leaderboard — {backbone}")
        ax.set_ylabel("Validation Accuracy (%)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "val_acc_leaderboard.png"))
    plt.close(fig)

def print_findings(summary):
    print("\n" + "=" * 78)
    print("PROBING LEADERBOARD — Best structural configuration via layer gains")
    print("=" * 78)
    for (backbone, dataset), g in summary.groupby(["backbone", "dataset"]):
        best_row = g.loc[g["avg_post_acc"].idxmax()]
        print(f"[{dataset:12s} | {backbone:30s}] best: {best_row['strategy']:15s} (avg_post_acc={best_row['avg_post_acc']:.2f}%)")

    print("\n" + "=" * 78)
    print("NOTABLE PER-LAYER FINDINGS (Biggest individual block jump)")
    print("=" * 78)
    for (backbone, dataset, strategy), g in summary.groupby(["backbone", "dataset", "strategy"]):
        print(f"[{dataset:12s} | {backbone:30s} | {strategy:15s}] "
              f"avg_delta={g['avg_delta'].values[0]:+.2f}pp, max_layer_delta={g['max_layer_delta'].values[0]:+.2f}pp")

def main():
    layer_df = collect_layer_data()
    summary = build_summary(layer_df)

    layer_df.to_csv(os.path.join(OUT_DIR, "layer_detail.csv"), index=False)
    summary.to_csv(os.path.join(OUT_DIR, "summary_table.csv"), index=False)
    
    plot_probing_by_layer(layer_df)
    plot_probing_delta_bars(summary)
    plot_rank_vs_delta_corr(summary)
    plot_val_acc_leaderboard(summary)
    
    print_findings(summary)

if __name__ == "__main__":
    main()