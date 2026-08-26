"""
lstm_model.py — Stacked 2-layer LSTM with temporal attention.

Architecture per proposal §3.5.4:
  - 2-layer stacked LSTM, hidden dim 128
  - 60-timestep input window
  - Single-head soft temporal attention over the hidden state sequence
  - MLP classification head -> 7 binary telemetry event classes

Ported from `final-lstm-module-codes.ipynb` ("LSTM Module - Model").
See notebooks/ for the original, unabridged cell.
"""

import torch
import torch.nn as nn

from src.data.constants import INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT, WINDOW_SIZE


class TemporalAttention(nn.Module):
    """
    Single-head soft attention over the LSTM's hidden state sequence.
    Produces a scalar attention weight per timestep — tells you which
    moment in the 1-second window drove the prediction.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn_linear = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor) -> tuple:
        """
        Args:
            lstm_out: (batch, seq_len, hidden_dim)
        Returns:
            context:      (batch, hidden_dim) — weighted summary
            attn_weights: (batch, seq_len)    — attention distribution
        """
        scores = self.attn_linear(lstm_out)                # (B, T, 1)
        weights = torch.softmax(scores, dim=1)              # (B, T, 1)
        context = (lstm_out * weights).sum(dim=1)           # (B, H)
        return context, weights.squeeze(-1)                 # (B, H), (B, T)


class AttentionLSTM(nn.Module):
    """
    Complete LSTM module for the bimodal driver coaching framework.

    Forward pass returns:
        logits (always):          (batch, num_classes) — pre-sigmoid scores
        attn_weights (optional):  (batch, seq_len)     — for diagnostics

    Use torch.sigmoid(logits) to get probabilities [0, 1].
    Use BCEWithLogitsLoss(logits, targets) / FocalLossWithLogits during
    training — do NOT apply sigmoid before the loss function.
    """

    def __init__(self,
                 input_dim: int = INPUT_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS,
                 num_classes: int = NUM_CLASSES,
                 dropout: float = DROPOUT,
                 window_size: int = WINDOW_SIZE):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.window_size = window_size

        # Optional input batch norm to stabilise training across LFS/AC domains
        self.input_norm = nn.BatchNorm1d(window_size)

        # Stacked LSTM — dropout applied between layers (not after last layer)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Temporal attention
        self.attention = TemporalAttention(hidden_dim)

        # Classification head. `inplace=False` on the ReLU is deliberate —
        # SHAP's DeepExplainer attaches gradient hooks to layer outputs
        # during a modified backward pass, which requires PyTorch to
        # allocate a clean, separate output tensor per layer.
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal init for LSTM weights, Xavier for linear layers."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # Forget gate bias = 1 — helps with long-range dependencies
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> tuple:
        """
        Args:
            x:                (batch, seq_len, input_dim)
            return_attention: if True, also return attention weights

        Returns:
            logits:       (batch, num_classes)
            attn_weights: (batch, seq_len) — only if return_attention=True
        """
        x = self.input_norm(x)                              # (B, T, F)
        lstm_out, _ = self.lstm(x)                           # (B, T, H)
        context, attn_weights = self.attention(lstm_out)     # (B, H), (B, T)
        logits = self.classifier(context)                    # (B, C)

        if return_attention:
            return logits, attn_weights
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper returning sigmoid probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
        return torch.sigmoid(logits)

    def count_parameters(self) -> int:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Trainable parameters: {total:,}")
        return total


def build_model(input_dim: int = INPUT_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS,
                 num_classes: int = NUM_CLASSES,
                 dropout: float = DROPOUT,
                 device: str = None) -> AttentionLSTM:
    """
    Build and print the AttentionLSTM model summary.
    Automatically selects CUDA if available and device is not specified.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AttentionLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    model.count_parameters()
    print(f"[Model] Device: {device}")
    print(f"[Model] Architecture: LSTM({input_dim}->{hidden_dim}x{num_layers}) "
          f"+ Attention -> MLP -> {num_classes} outputs")
    return model
