import os
import sys
import json
import torch
import numpy as np
import wandb
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from peft import PeftModel

from config import parse_args_to_config, ExperimentConfig
from data.datasets import get_dataloader
from backbone import get_backbone_model

@torch.no_grad()
def cache_backbone_features(model, dataloader, device):
    """
    Passes the dataset through the model once 
    and collects the hidden states across all transformer blocks.
    """
    model.eval()
    layer_features = [[] for _ in range(13)]
    all_labels = []
    for images, labels in tqdm(dataloader, desc="Caching Backbone Features"):
        images = images.to(device)
        # Extract features depending on whether model is a PeftModel or base model
        if isinstance(model, PeftModel):
            hidden_states = model.base_model.model.extract_hidden_states(images)
        else:
            hidden_states = model.extract_hidden_states(images)
        for layer_idx in range(13):
            current_layer_tensor = hidden_states[layer_idx]
            cls_features = current_layer_tensor[:, 0, :].cpu().numpy()
            layer_features[layer_idx].append(cls_features)
        all_labels.append(labels.numpy())
    combined_layers = []
    for layer_idx in range(13):
        merged_layer_data = np.concatenate(layer_features[layer_idx], axis=0)
        combined_layers.append(merged_layer_data)
    final_layers_array = np.stack(combined_layers, axis=0)
    final_labels_array = np.concatenate(all_labels, axis=0)
    return final_layers_array, final_labels_array

def main():
    # Parse checkpoint_path argument if provided before standard parser
    checkpoint_path = None
    if "--checkpoint_path" in sys.argv:
        idx = sys.argv.index("--checkpoint_path")
        checkpoint_path = sys.argv[idx + 1]
        sys.argv.remove("--checkpoint_path")
        sys.argv.remove(checkpoint_path)
        
    # Load operational parameters
    config = parse_args_to_config()
    
    # Force alignment of the backbone model name using the saved adapter config metadata
    if checkpoint_path:
        adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r") as f:
                adapter_meta = json.load(f)
            # Override model_name if base_model_name_or_path is present in saved json
            if "base_model_name_or_path" in adapter_meta and adapter_meta["base_model_name_or_path"]:
                config.model_name = adapter_meta["base_model_name_or_path"]
                # Re-run post-init alignment logic manually
                config.__post_init__()

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Starting Diagnostic Probing on device: {device} for model: {config.model_name}")
    
    # Extract the strategy from the folder path since config.strategy defaults to vanilla
    resolved_strategy = config.strategy
    if checkpoint_path:
        folder_name = os.path.basename(os.path.normpath(checkpoint_path))
        for strategy_cand in ["vanilla", "support_weak", "amplify_strong", "proportional"]:
            if strategy_cand in folder_name:
                resolved_strategy = strategy_cand
                break

    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        job_type="diagnostic-probing" if not checkpoint_path else "post-adaptation-probing",
        config=config.to_dict()
    )
    train_loader, num_classes = get_dataloader(
        dataset_name=config.dataset,
        backbone_norm=config.backbone_norm,
        split="train",
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    val_loader, _ = get_dataloader(
        dataset_name=config.dataset,
        backbone_norm=config.backbone_norm,
        split="val",
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    # Initialize base model matching the correct architecture
    model = get_backbone_model(config, num_classes=num_classes).to(device)
    # Load LoRA adapters if checkpoint path is provided
    if checkpoint_path:
        print(f"Loading LoRA weights from: {checkpoint_path}")
        with open(os.path.join(checkpoint_path, "adapter_config.json"), "r") as f:
            saved_targets = set(json.load(f).get("target_modules", []))
        expected_targets = set(config.target_modules)
        if saved_targets != expected_targets:
            raise RuntimeError(
                f"Checkpoint at {checkpoint_path} has target_modules={saved_targets}, "
                f"but {config.model_name} expects {expected_targets}. "
                f"This checkpoint was likely saved by a different backbone under the old naming scheme."
            )
        model = PeftModel.from_pretrained(model, checkpoint_path).to(device)
        model.eval()
    os.makedirs(config.cache_dir, exist_ok=True)
    model_safe_name = config.model_name.replace("/", "_")
    if checkpoint_path:
        strategy_suffix = f"_adapted_{resolved_strategy}"
    else:
        strategy_suffix = ""
        
    cache_file_path = os.path.join(config.cache_dir, f"{config.dataset}_{model_safe_name}{strategy_suffix}.npz")
    if not os.path.exists(cache_file_path):
        print(f"No existing feature cache found ({cache_file_path}). Running forward passes...")
        X_train, y_train = cache_backbone_features(model, train_loader, device)
        X_val, y_val = cache_backbone_features(model, val_loader, device)
        print(f"Saving extracted hidden states directly to: {cache_file_path}")
        np.savez(cache_file_path, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    else:
        print(f"Loading pre-cached features from structural storage file: {cache_file_path}")
        cached_tensors = np.load(cache_file_path)
        X_train, y_train = cached_tensors["X_train"], cached_tensors["y_train"]
        X_val, y_val = cached_tensors["X_val"], cached_tensors["y_val"]
    num_total_layers = X_train.shape[0]
    print(f"\nTraining Diagnostic Linear Probes Across {num_total_layers} Layers.")
    probing_accuracies = []
    for block_idx in range(1, num_total_layers):
        X_tr_block = X_train[block_idx]
        X_va_block = X_val[block_idx]
        scaler = StandardScaler()
        X_tr_block = scaler.fit_transform(X_tr_block)
        X_va_block = scaler.transform(X_va_block)
        probe_head = LogisticRegression(max_iter=100, C=0.1, solver="lbfgs", n_jobs=-1)
        probe_head.fit(X_tr_block, y_train)
        accuracy_score = probe_head.score(X_va_block, y_val)
        probing_accuracies.append(accuracy_score)
        # Log to WandB with distinct key names
        metric_name = "diagnostic/post_probing_accuracy" if checkpoint_path else "diagnostic/probing_accuracy"
        wandb.log({"diagnostic/layer_index": block_idx, metric_name: accuracy_score * 100})
        print(f"Transformer Block {block_idx:02d} -> Downstream Probing Validation Accuracy: {accuracy_score * 100:.2f}%")

    print("\nDiagnostic Probe Execution Finalized.")
    scores_file_path = os.path.join(config.cache_dir, f"{config.dataset}_{model_safe_name}{strategy_suffix}_scores.npy")
    np.save(scores_file_path, np.array(probing_accuracies))
    print(f"Automatically saved report card scores to: {scores_file_path}")
    wandb.finish()

if __name__ == "__main__":
    main()