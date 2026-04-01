"""Enhanced FTIR model architecture.

Key changes vs v4.0:
  - BatchNorm1d → GroupNorm(8, C) throughout CNN (stable with mixed synthetic/reference batches)
  - Output head uses linear activation (model outputs in normalised label space)
  - Per-species label normalisation handled externally by LabelNormalizer
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

class ResBlock1D(nn.Module):
    """Pre-activation 1D residual block with optional stride for downsampling."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 7,
        stride: int = 1,
        dropout: float = 0.1,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.norm1 = nn.GroupNorm(min(num_groups, in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.norm2 = nn.GroupNorm(min(num_groups, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.drop = nn.Dropout(dropout)

        self.skip: nn.Module
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.relu(self.norm1(x)))
        h = self.drop(self.conv2(F.relu(self.norm2(h))))
        return h + self.skip(x)


class SpectralCNN(nn.Module):
    """5-block residual CNN backbone that maps (B, C_in, NPTS) → (B, C, NPTS//16).

    Channel progression: C_in → 32 → 64 → 128 → 256 → 256
    Each block (except last) halves the temporal dimension via stride-2 conv.
    """

    def __init__(self, in_channels: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, padding=7, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResBlock1D(32, 64,  stride=2, dropout=dropout),  # /2
            ResBlock1D(64, 128, stride=2, dropout=dropout),  # /4
            ResBlock1D(128, 256, stride=2, dropout=dropout), # /8
            ResBlock1D(256, 256, stride=2, dropout=dropout), # /16
            ResBlock1D(256, 256, stride=1, dropout=dropout), # same
        )
        self.out_channels = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))  # (B, 256, T//16)


# ---------------------------------------------------------------------------
# Attention / Transformer
# ---------------------------------------------------------------------------

class SelfAttention1D(nn.Module):
    """Multi-head self-attention over the temporal axis."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)


class TransformerBlock(nn.Module):
    """Single Transformer encoder layer with pre-norm."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        out, _ = self.attn(h, h, h)
        x = x + out
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class FTIRModel(nn.Module):
    """FTIR multi-gas concentration regressor with engineered inputs.

    Architecture:
        Input  (B, C, NPTS)    engineered spectral channels
        ├── SpectralCNN         5 ResBlocks → (B, 256, T)
        ├── SelfAttention1D     8 heads     → (B, T, 512)
        ├── GRU                 hidden=512  → (B, T, 1024)
        ├── Pool + MLP refine               → (B, 1024)
        ├── Optional aux branch             → (B, 128)
        └── Linear head                     → (B, n_species)

    Output in log1p(ppmv) space — invert with torch.expm1.
    """

    def __init__(
        self,
        n_species: int = 11,
        in_channels: int = 3,
        aux_features: int = 0,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.n_species = n_species
        self.in_channels = int(in_channels)
        self.aux_features = int(aux_features)

        self.cnn = SpectralCNN(in_channels=self.in_channels, dropout=dropout)
        cnn_ch = self.cnn.out_channels  # 256

        # Project CNN output to GRU input
        self.proj = nn.Linear(cnn_ch, 512)

        # 8-head self-attention (sequence axis = temporal)
        self.attn = SelfAttention1D(512, n_heads=8, dropout=dropout)

        # Bidirectional GRU — hidden 512 per direction → 1024 concat
        self.gru = nn.GRU(512, 512, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=dropout)

        # Post-pool refinement MLP (replaces former single-token Transformer)
        self.refine = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.Dropout(dropout),
        )
        self.refine_norm = nn.LayerNorm(1024)

        self.aux_proj: nn.Module
        head_input_dim = 1024
        if self.aux_features > 0:
            self.aux_proj = nn.Sequential(
                nn.LayerNorm(self.aux_features),
                nn.Linear(self.aux_features, 128),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            head_input_dim += 128
        else:
            self.aux_proj = nn.Identity()

        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_species),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        aux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, NPTS) float32
        aux : (B, F) float32, optional

        Returns
        -------
        out : (B, n_species) float32, log1p(ppmv)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B, C, NPTS), got {tuple(x.shape)}")

        # 1. CNN backbone
        h = self.cnn(x)                    # (B, 256, T)
        h = h.permute(0, 2, 1)            # (B, T, 256)
        h = self.proj(h)                   # (B, T, 512)

        # 2. Self-attention
        h = self.attn(h)                   # (B, T, 512)

        # 3. GRU
        h, _ = self.gru(h)                # h: (B, T, 1024)

        # 4. Global average pool + residual MLP refinement
        h_pool = h.mean(dim=1)             # (B, 1024)
        h_pool = self.refine_norm(h_pool + self.refine(h_pool))  # (B, 1024)

        if self.aux_features > 0:
            if aux is None:
                raise ValueError("Model expects auxiliary prior features but aux=None was passed")
            h_pool = torch.cat([h_pool, self.aux_proj(aux)], dim=-1)

        # 5. Output head — linear activation; outputs are in normalised label space.
        # Non-negativity is enforced after denormalisation at inference/eval time.
        return self.head(h_pool)  # (B, n_species)


    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        aux: torch.Tensor | None = None,
        *,
        n_samples: int = 20,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MC Dropout: run *n_samples* forward passes with dropout active.

        Returns
        -------
        mean : (B, n_species)  mean prediction across MC samples
        std  : (B, n_species)  per-species predictive std (uncertainty)
        """
        was_training = self.training
        self.train()  # enable dropout

        preds: list[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(self.forward(x, aux=aux))

        stacked = torch.stack(preds, dim=0)  # (n_samples, B, n_species)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0)

        if not was_training:
            self.eval()

        return mean, std


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
