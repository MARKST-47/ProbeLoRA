import torch
import torch.nn as nn
from transformers import Dinov2Model, CLIPVisionModel
from config import ExperimentConfig

class ProbeLoRABackbone(nn.Module):
    """
    Unified wrapper for DINOv2 and CLIP-ViT backbones.
    Handles embedding extractions and appends a linear classification head.
    """
    def __init__(self, config: ExperimentConfig, num_classes: int):
        super().__init__()
        self.model_name = config.model_name.lower()
        self.num_classes = num_classes
        print(f"Initializing backbone network: {config.model_name}")
        if "dinov2" in self.model_name:
            self.backbone = Dinov2Model.from_pretrained(config.model_name)
            self.hidden_dim = self.backbone.config.hidden_size  # 768 for base
        elif "clip" in self.model_name:
            # We load only image encoder of CLIP
            self.backbone = CLIPVisionModel.from_pretrained(config.model_name)
            self.hidden_dim = self.backbone.config.hidden_size  # 768 for base
        else:
            raise ValueError(f"Unsupported model architecture: {config.model_name}")
            
        # Append downstream task classification head
        self.classifier = nn.Linear(self.hidden_dim, num_classes)

    def extract_hidden_states(self, pixel_values: torch.Tensor) -> tuple:
        """
        Executes a forward pass through the frozen backbone to extract hidden states 
        for all sequential transformer blocks.
        """
        if "clip" in self.model_name:
            outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        else:
            outputs = self.backbone(pixel_values, output_hidden_states=True)
        # returns a tuple of length 13: (embedding_layer_output, block_1_output, ..., block_12_output)
        return outputs.hidden_states

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Standard forward tracking pass used during final LoRA fine-tuning.
        """
        if "clip" in self.model_name:
            outputs = self.backbone(pixel_values=pixel_values)
            # CLIP pools visual representations into a distinct pooler_output tensor
            pooled_output = outputs.pooler_output
        else:
            outputs = self.backbone(pixel_values)
            # DINOv2 pools representations using the standard first [CLS] token
            pooled_output = outputs.last_hidden_state[:, 0, :]
            
        # Pass through classification linear layer
        logits = self.classifier(pooled_output)
        return logits

def get_backbone_model(config: ExperimentConfig, num_classes: int) -> ProbeLoRABackbone:
    """Helper to build and instantiate wrapped model."""
    return ProbeLoRABackbone(config, num_classes)

if __name__ == "__main__":
    # Quick execution test to verify feature dimensions
    from config import ExperimentConfig
    
    mock_config = ExperimentConfig(model_name="facebook/dinov2-base")
    mock_images = torch.randn(2, 3, 224, 224)
    
    model = get_backbone_model(mock_config, num_classes=10)
    states = model.extract_hidden_states(mock_images)
    logits = model(mock_images)
    
    print(f"Self-Test Passed Successfully!")
    print(f"Extracted Hidden States Layers Count: {len(states)}")
    print(f"Logits Tensor Target Shape: {list(logits.shape)}")