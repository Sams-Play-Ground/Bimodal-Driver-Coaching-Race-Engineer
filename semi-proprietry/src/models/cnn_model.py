"""
cnn_model.py — EfficientNet-B0 pretrained backbone with a custom 3-output
sigmoid classification head, per proposal §3.5.3.

Ported from `final-cnn-module-codes.ipynb` ("8 - Model — DrivingCNN
(EfficientNet-B0)"). Requires `timm` (installed in the notebook via
`pip install timm`).
"""

import timm
import torch
import torch.nn as nn

from src.data.cnn_constants import BACKBONE, NUM_CLASSES, DROPOUT, LR_FINETUNE, CLASS_NAMES


class DrivingCNN(nn.Module):
    """
    Spatial feature extraction CNN for the bimodal driver coaching framework.

    Architecture (proposal §3.5.3):
        EfficientNet-B0 backbone (pretrained on ImageNet, num_classes=0)
        -> 1280-dim feature vector (Global Average Pooling output)
        -> Dropout(0.45)
        -> Linear(1280, 3)
        -> raw logits [off_track, apex_miss, sliding]

    Training protocol:
        Phase 1 (epochs 0-FREEZE_EPOCHS): backbone frozen, only head trained.
        Phase 2 (epochs FREEZE_EPOCHS+):  all layers unfrozen, lr reduced to LR_FINETUNE.
    """

    def __init__(self, backbone: str = BACKBONE, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)

        self.head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(self.backbone.num_features, num_classes),
        )

        nn.init.xavier_uniform_(self.head[1].weight)
        nn.init.zeros_(self.head[1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, H, W) — normalised frame batch
        Returns:
            logits: (batch, num_classes) — raw pre-sigmoid scores.
                    Apply torch.sigmoid() for probabilities; BCEWithLogitsLoss
                    applies sigmoid internally during training.
        """
        return self.head(self.backbone(x))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper: returns sigmoid probabilities in eval mode."""
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self(x))

    def count_parameters(self) -> int:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Trainable parameters: {total:,}")
        return total

    def freeze_backbone(self):
        """Freeze all backbone layers — only head is trained (Phase 1)."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        print("[Model] Phase 1: backbone frozen — training head only.")

    def unfreeze_all(self):
        """Unfreeze all layers for end-to-end fine-tuning (Phase 2)."""
        for p in self.parameters():
            p.requires_grad_(True)
        print(f"[Model] Phase 2: all layers unfrozen — full fine-tuning at LR={LR_FINETUNE}.")


def build_model(device: str = None) -> DrivingCNN:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DrivingCNN().to(device)
    model.count_parameters()
    print(f"[Model] Backbone   : {model.backbone_name}")
    print(f"[Model] Output dim : {model.num_classes} -> {CLASS_NAMES}")
    print(f"[Model] Device     : {device}")
    return model
