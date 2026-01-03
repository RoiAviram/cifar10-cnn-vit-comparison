"""
Models for CIFAR-10 classification.
- SimpleCNN: 4-layer CNN baseline
- create_vit_model: Vision Transformer factory
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    4-block CNN: Conv→BN→ReLU→Pool repeated 4 times, then FC layers.
    ~2.3M parameters for 64×64 input.
    """

    def __init__(self, num_classes: int = 10, input_size: int = 64):
        super().__init__()

        # Feature extraction: 4 conv blocks
        self.features = nn.Sequential(
            # Block 1: 3→32 channels, 64→32 spatial
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 32→64 channels, 32→16 spatial
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 64→128 channels, 16→8 spatial
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 128→256 channels, 8→4 spatial
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # After 4 MaxPools: size = input_size / 16
        final_size = input_size // 16
        self.flatten_size = 256 * final_size * final_size

        # Classifier: flatten → FC layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),  # Raw logits output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Image batch → class logits."""
        x = self.features(x)           # (B, 3, H, W) → (B, 256, H/16, W/16)
        x = torch.flatten(x, 1)        # (B, 256, 4, 4) → (B, 4096)
        x = self.classifier(x)         # (B, 4096) → (B, num_classes)
        return x


# ============================================================================
# Vision Transformer (ViT) using timm
# ============================================================================

def create_vit_model(
    num_classes: int = 10,
    img_size: int = 64,
    pretrained: bool = True,
    drop_rate: float = 0.1,
) -> nn.Module:
    """
    Create a Vision Transformer model using timm library.

    ViT works by:
    1. Split image into patches (16×16 pixels each)
    2. Flatten each patch and project to embedding dimension
    3. Add position embeddings (so model knows patch locations)
    4. Process through transformer encoder (self-attention layers)
    5. Use [CLS] token output for classification

    Args:
        num_classes: Output classes (10 for CIFAR-10)
        img_size: Input image size (64 or 128)
        pretrained: Use ImageNet pretrained weights
        drop_rate: Dropout rate for regularization

    Returns:
        Configured ViT model (~5.7M parameters)

    Note on patch count:
        - img_size=64, patch_size=16 → 4×4 = 16 patches
        - img_size=128, patch_size=16 → 8×8 = 64 patches
        More patches = better spatial understanding, but slower
    """
    try:
        import timm
    except ImportError:
        raise ImportError("timm not installed. Run: pip install timm")

    # vit_tiny_patch16_224: smallest ViT variant
    # - tiny: 192 embed dim, 12 layers, 3 heads
    # - patch16: 16×16 pixel patches
    # - 224: originally trained on 224×224 (we override with img_size)
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=img_size,      # Override default 224 → our size
        drop_rate=drop_rate,
    )

    return model
