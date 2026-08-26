"""
Vision (CNN) engine — ported 1:1 from cnn-module-codes.ipynb Section 8
(DrivingCNN) and Section 14 (inference preprocessing).

IMPORTANT: the notebook backbone is built with timm.create_model(...),
not torchvision.models.efficientnet_b0(...). The state_dict keys differ
between the two, so this MUST use timm or your checkpoint won't load.
Add "timm" to requirements.txt and build.spec's collect_all(...) loop.
"""
import numpy as np
import cv2
from PIL import Image
from typing import Optional
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = None
except ImportError as e:
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(e)

try:
    import timm
    TIMM_AVAILABLE = True
    TIMM_IMPORT_ERROR = None
except ImportError as e:
    TIMM_AVAILABLE = False
    TIMM_IMPORT_ERROR = str(e)

BACKBONE = "efficientnet_b0"
NUM_CLASSES = 3
CLASS_NAMES = ["Off-Track", "Apex Miss", "Sliding"]
MODEL_INPUT_SIZE = 224
DROPOUT = 0.45
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Per-class decision thresholds for the standalone CNN dot overlay
# (Section 5 of the notebook) — deliberately lower for Off-Track since
# missing a track-limit violation is worse than a false positive.
DEFAULT_THRESHOLDS = {
    "Off-Track": 0.35,
    "Apex Miss": 0.47,
    "Sliding": 0.45,
}


def letterbox(frame_rgb: np.ndarray, target_w: int = 640, target_h: int = 288) -> np.ndarray:
    """Must exactly match the letterbox() used during training/inference
    in the notebook (Section 4 / Section 14) — same target size."""
    h, w = frame_rgb.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


if TORCH_AVAILABLE and TIMM_AVAILABLE:

    class DrivingCNN(nn.Module):
        """EfficientNet-B0 (timm) backbone + 3-class multi-label head.
        Outputs raw logits — apply sigmoid for probabilities (BCE-trained,
        NOT softmax — off-track/apex-miss/sliding can co-occur)."""

        def __init__(self, backbone: str = BACKBONE, num_classes: int = NUM_CLASSES):
            super().__init__()
            self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
            self.head = nn.Sequential(
                nn.Dropout(DROPOUT),
                nn.Linear(self.backbone.num_features, num_classes),
            )

        def forward(self, x):
            return self.head(self.backbone(x))


class VisionEngine:
    """Runtime wrapper for the trained CNN. Falls back to a labelled
    'unavailable' state (not a fabricated heuristic) if no checkpoint
    is present, since the app should be honest that vision is offline
    rather than pretending it has a spatial signal. `status_reason` says
    exactly why, instead of leaving that a mystery."""

    def __init__(self, model_path: Optional[Path] = None, device: Optional[str] = None):
        self.status_reason = "OK"

        if not TORCH_AVAILABLE:
            self.available = False
            self.status_reason = f"torch/torchvision not installed ({TORCH_IMPORT_ERROR})"
            return
        if not TIMM_AVAILABLE:
            self.available = False
            self.status_reason = f"'timm' not installed ({TIMM_IMPORT_ERROR}) - run: pip install timm==1.0.9"
            return
        if not model_path:
            self.available = False
            self.status_reason = "no CNN checkpoint path configured"
            return
        if not Path(model_path).exists():
            self.available = False
            self.status_reason = f"checkpoint file not found at {model_path}"
            return

        self.available = True
        try:
            self._load(model_path, device)
        except Exception as exc:
            self.available = False
            self.status_reason = f"failed to load checkpoint: {exc}"

    def _load(self, model_path, device):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = DrivingCNN().to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        # letterbox -> ToTensor -> CenterCrop(224) -> Normalize — MUST match
        # build_inference_transform() in the notebook exactly.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(MODEL_INPUT_SIZE),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def predict_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Returns (3,) sigmoid probabilities [off_track, apex_miss, sliding].
        Zeros if no checkpoint is loaded."""
        if not self.available:
            return np.zeros(3, dtype=np.float32)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        lb = letterbox(rgb, target_w=640, target_h=288)
        pil = Image.fromarray(lb)
        tensor = self.transform(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        return probs

    def generate_gradcam(self, frame_bgr: np.ndarray, target_class: int = 0) -> np.ndarray:
        """Grad-CAM overlay for the annotated-replay screen."""
        if not self.available:
            return frame_bgr

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        lb = letterbox(rgb, target_w=640, target_h=288)
        pil = Image.fromarray(lb)
        tensor = self.transform(pil).unsqueeze(0).to(self.device)
        tensor.requires_grad_()

        target_layer = self.model.backbone.conv_head if hasattr(self.model.backbone, "conv_head") \
            else list(self.model.backbone.children())[-1]
        gradients, activations = [], []

        def fwd_hook(m, i, o): activations.append(o)
        def bwd_hook(m, gi, go): gradients.append(go[0])

        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)

        output = self.model(tensor)
        score = output[0, target_class]
        self.model.zero_grad()
        score.backward()
        h1.remove(); h2.remove()

        pooled = torch.mean(gradients[0], dim=[0, 2, 3])
        act = activations[0][0]
        for i in range(act.shape[0]):
            act[i, :, :] *= pooled[i]

        heatmap = torch.mean(act, dim=0).detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)

        heatmap_resized = cv2.resize(heatmap, (frame_bgr.shape[1], frame_bgr.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame_bgr, 0.6, color_heatmap, 0.4, 0)
