import os
import torch
import numpy as np
import wandb
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import parse_args_to_config, ExperimentConfig
from data.datasets import get_dataloader
from backbone import get_backbone_model

@torch.no_grad()
def cache_backbone_features(model, dataloader, device):
    """
    Passes the dataset through the completely frozen backbone once 
    and collects the hidden states across all transformer blocks.
    """
    model.eval()
    # List of lists to hold features for each of the 13 layers.
    layer_features = [[] for _ in range(13)]
    all_labels = []
    for images, labels in tqdm(dataloader, desc="Caching Backbone Features"):
        images = images.to(device)
        # Ask our backbone wrapper to pull out the 13 raw hidden layers
        hidden_states = model.extract_hidden_states(images)
        # Loop through all 13 layers and pluck out the [CLS] token features
        for layer_idx in range(13):
            current_layer_tensor = hidden_states[layer_idx]
            # Grab index 0 (the [CLS] token vector which summarizes the image)
            cls_features = current_layer_tensor[:, 0, :].cpu().numpy()
            # Save this batch's layer features into our list
            layer_features[layer_idx].append(cls_features)
        # Save the real labels for this batch
        all_labels.append(labels.numpy())
    # Combine all separate batches into single massive arrays using numpy
    combined_layers = []
    for layer_idx in range(13):
        # Merge all separate batches for this specific layer index
        merged_layer_data = np.concatenate(layer_features[layer_idx], axis=0)
        combined_layers.append(merged_layer_data)
    # Stack the 13 layers into one final 3D block: [13, total_images, 768_features]
    final_layers_array = np.stack(combined_layers, axis=0)
    final_labels_array = np.concatenate(all_labels, axis=0)
    return final_layers_array, final_labels_array

def main():
    # Load operational parameters
    config = parse_args_to_config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Starting Diagnostic Probing on device: {device}")
    # Connect this diagnostic run to your shared team WandB space
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        job_type="diagnostic-probing",
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
    model = get_backbone_model(config, num_classes=num_classes).to(device)
    # Handle serialization paths
    os.makedirs(config.cache_dir, exist_ok=True)
    model_safe_name = config.model_name.replace("/", "_")
    cache_file_path = os.path.join(config.cache_dir, f"{config.dataset}_{model_safe_name}.npz")
    # Execute or load cached activation states
    if not os.path.exists(cache_file_path):
        print(f"No existing feature cache found. Running forward passes...")
        X_train, y_train = cache_backbone_features(model, train_loader, device)
        X_val, y_val = cache_backbone_features(model, val_loader, device)
        print(f"Saving extracted hidden states directly to: {cache_file_path}")
        np.savez(cache_file_path, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    else:
        print(f"Loading pre-cached features from structural storage file: {cache_file_path}")
        cached_tensors = np.load(cache_file_path)
        X_train, y_train = cached_tensors["X_train"], cached_tensors["y_train"]
        X_val, y_val = cached_tensors["X_val"], cached_tensors["y_val"]
    # Fit Linear Probes layer by layer
    # Index 0 is the input embedding layer, indices 1-12 are the actual transformer blocks
    num_total_layers = X_train.shape[0]
    print(f"\nTraining Diagnostic Linear Probes Across {num_total_layers} Layers.")
    probing_accuracies = []
    # We skip layer 0 (raw embeddings) and evaluate the 12 transformer blocks
    for block_idx in range(1, num_total_layers):
        X_tr_block = X_train[block_idx]
        X_va_block = X_val[block_idx]
        # Fast feature normalization scaling for linear stability
        scaler = StandardScaler()
        X_tr_block = scaler.fit_transform(X_tr_block)
        X_va_block = scaler.transform(X_va_block)
        # We chose this setup because it's exactly how top research papers check what big models know. 
        # It's incredibly fast, reliable and gives us an accurate 'report card' for each layer without wasting cluster time.
        probe_head = LogisticRegression(max_iter=100, C=0.1, solver="lbfgs", n_jobs=-1)
        probe_head.fit(X_tr_block, y_train)
        accuracy_score = probe_head.score(X_va_block, y_val)
        probing_accuracies.append(accuracy_score)
        # Log to WandB
        wandb.log({"diagnostic/layer_index": block_idx, "diagnostic/probing_accuracy": accuracy_score * 100})
        print(f"Transformer Block {block_idx:02d} -> Downstream Probing Validation Accuracy: {accuracy_score * 100:.2f}%")

    print("\nDiagnostic Probe Execution Finalized.")
    print("Copy and paste this raw array output directly into your train.py file:")
    print([float(np.round(acc, 4)) for acc in probing_accuracies])
    wandb.finish()

if __name__ == "__main__":
    main()