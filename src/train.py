import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import numpy as np

from config import parse_args_to_config
from data.datasets import get_dataloader
from backbone import get_backbone_model
from peft import LoraConfig, get_peft_model

def get_strategy_ranks(strategy: str, dataset: str, model_name: str, base_r: int, cache_dir: str = "./cached_features") -> dict:
    """
    Dynamically loads calculated probing scores from storage disk and 
    computes a non-uniform layer rank dictionary based on the target strategy.
    """
    rank_pattern = {}
    if strategy == "vanilla":
        return rank_pattern  # Empty dict defaults all target modules to base_r
        
    model_safe_name = model_name.replace("/", "_")
    scores_file_path = os.path.join(cache_dir, f"{dataset}_{model_safe_name}_scores.npy")
    if not os.path.exists(scores_file_path):
        raise FileNotFoundError(
            f"\nCould not find the layer report card file at: {scores_file_path}\n"
            f"Run: python src/probing.py --dataset {dataset} --model_name {model_name}"
        )
    # Load the scores cleanly back into a standard Python list
    probing_scores = np.load(scores_file_path).tolist()
    num_layers = len(probing_scores)
    # Define internal layer naming strings depending on the architecture
    is_clip = "clip" in model_name.lower()
    layer_prefix = "layers" if is_clip else "layer"
    if strategy == "support_weak":
        # Strategy A: Worse layer-wise accuracy -> Higher LoRA rank allocation
        # Invert scores: lower accuracy yields a higher multiplier
        inverted = [1.0 - score for score in probing_scores]
        total = sum(inverted)
        scaled_ranks = [int((val / total) * base_r * num_layers) for val in inverted]
        
    elif strategy == "amplify_strong":
        # Strategy B: Better layer-wise accuracy -> Higher LoRA rank allocation
        total = sum(probing_scores)
        scaled_ranks = [int((score / total) * base_r * num_layers) for score in probing_scores]
    
    elif strategy == "proportional":
        # Strategy C: Smooth proportional rank assignment scaling across depth
        # Map values to scale strictly between min rank 2 and max rank 16
        min_s, max_s = min(probing_scores), max(probing_scores)
        scaled_ranks = [
            int(2 + (14 * (score - min_s) / (max_s - min_s + 1e-6))) 
            for score in probing_scores
        ]
    else:
        raise ValueError(f"Unknown allocation strategy: {strategy}")
    # Map the calculated rank array directly to PEFT module suffix names
    for i, rank in enumerate(scaled_ranks):
        target_rank = max(1, rank)
        module_key = f"{layer_prefix}.{i}."
        rank_pattern[module_key] = target_rank
        
    return rank_pattern

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(dataloader, desc="Training Batch Iteration", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        # Display step progress inside the cluster log trace
        progress_bar.set_postfix(loss=loss.item())
        
    return running_loss / total, (correct / total) * 100

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / total, (correct / total) * 100

def main():
    # Parse operational command line variables into standard dataclass parameters
    config = parse_args_to_config()
    torch.manual_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    # Setup Data pipelines (FIXED: Corrected unpacking logic)
    print(f"Loading data: {config.dataset} | Norm mode: {config.backbone_norm}")
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
    base_model = get_backbone_model(config, num_classes=num_classes)
    # Inject Dynamic Rank Configurations using LoraConfig 
    rank_pattern_dict = get_strategy_ranks(
        strategy=config.strategy,
        dataset=config.dataset,
        model_name=config.model_name,
        base_r=config.lora_r,
        cache_dir=config.cache_dir
    )
    print(f"Applying LoRA Strategy Profile: '{config.strategy}'")
    if rank_pattern_dict:
        print(f"Generated Custom Rank Allocation Map: {rank_pattern_dict}")
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=0.05,
        bias="none",
        modules_to_save=["classifier"],  # Classification head is always 100% trainable
        rank_pattern=rank_pattern_dict
    )
    model = get_peft_model(base_model, peft_config).to(device)
    print("\nModel Trainable Parameters Mapping: ")
    model.print_trainable_parameters()
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        config=config.to_dict()
    )
    # Log the structural rank allocation topology mapping as a distinct meta metric
    wandb.config.update({"resolved_rank_pattern": rank_pattern_dict})
    # Optimization 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # Main Fine-Tuning Execution Loop
    print(f"\nStarting optimization loop for {config.epochs} training epochs...")
    best_val_acc = 0.0
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        print(f"Epoch {epoch:02d}/{config.epochs:02d} | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
        # Tracking logs live to dashboard
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "val/best_accuracy": best_val_acc
        })
    print(f"\nOptimization Routine Finalized. Highest Validation Accuracy Reached: {best_val_acc:.2f}%")
    wandb.finish()

if __name__ == "__main__":
    main()