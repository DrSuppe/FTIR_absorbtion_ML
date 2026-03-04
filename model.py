"""Thin wrapper around src/ftir_analysis/modeling.py for legacy compatibility."""

from ftir_analysis.modeling import FTIRModel, SpectralCNN, ResBlock1D, TransformerBlock, SelfAttention1D, count_parameters

__all__ = [
    "FTIRModel",
    "SpectralCNN",
    "ResBlock1D",
    "TransformerBlock",
    "SelfAttention1D",
    "count_parameters",
]
