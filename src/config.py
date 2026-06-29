import argparse
from dataclasses import dataclass, field, asdict
from typing import List

@dataclass
class ExperimentConfig:
    """
    Central parameters configuration class for ProbeLoRA experiments.
    """
    # Hardware/Environment Settings
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True

    # Data Settings
    dataset: str = "cifar100" # choices: ['cifar100', 'oxford_pets', 'cub200']
    data_root: str = "./data/raw"
    cache_dir: str = "./cached_features"
    batch_size: int = 128

    # Backbone Settings
    model_name: str = "facebook/dinov2-base" # choices: ['facebook/dinov2-base', 'openai/clip-vit-base-patch16']
    backbone_norm: str = "imagenet" # Inferred automatically in __post_init__

    # Optimization/Training Hyperparameters
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.01

    # LoRA Specific Parameters
    strategy: str = "vanilla" # choices: ['vanilla', 'support_weak', 'amplify_strong', 'proportional']
    target_modules: List[str] = field(default_factory=list) # Inferred automatically
    lora_r: int = 8 # Base target uniform/average rank
    lora_alpha: int = 16

    # Logging Parameters
    wandb_project: str = "ProbeLoRA"
    wandb_entity: str = "markstephenthomas-saarland-university" 

    def __post_init__(self):
        """
        Runs automatically right after initialization to align normalization stats 
        and attention module names with the chosen Hugging Face backbone model.
        """
        model_lower = self.model_name.lower()
        
        if "clip" in model_lower:
            self.backbone_norm = "clip"
            # Hugging Face CLIP vision attention modules use q_proj and v_proj
            if not self.target_modules:
                self.target_modules = ["q_proj", "v_proj"]
        else:
            self.backbone_norm = "imagenet"
            # Hugging Face DINOv2 vision attention modules use query and value
            if not self.target_modules:
                self.target_modules = ["query", "value"]

    def to_dict(self):
        """Standard method to turn the config into a plain dict for WandB logging."""
        return asdict(self)

def parse_args_to_config() -> ExperimentConfig:
    """
    Parses command line inputs directly into the dataclass layout.
    Allows easy automation from cluster SLURM scripts.
    """
    parser = argparse.ArgumentParser(description="ProbeLoRA Experiment Pipeline")
    
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--model_name", type=str, default="facebook/dinov2-base")
    parser.add_argument("--strategy", type=str, default="vanilla")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lora_r", type=int, default=8)

    args = parser.parse_args()
    
    return ExperimentConfig(
        dataset=args.dataset,
        model_name=args.model_name,
        strategy=args.strategy,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lora_r=args.lora_r
    )

if __name__ == "__main__":
    # Test script integrity for both backbones
    dino_config = ExperimentConfig(model_name="facebook/dinov2-base")
    print(f"DINOv2 Check -> Norm: {dino_config.backbone_norm} | Targets: {dino_config.target_modules}")
    
    clip_config = ExperimentConfig(model_name="openai/clip-vit-base-patch16")
    print(f"CLIP Check -> Norm: {clip_config.backbone_norm} | Targets: {clip_config.target_modules}")