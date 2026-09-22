import math

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 64) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] > self.encoding.shape[1]:
            raise ValueError("Sequence is longer than the configured positional encoding")
        return x + self.encoding[:, : x.shape[1]]


class AttentivePooling(nn.Module):
    """Learned attention pooling over temporal tokens."""
    def __init__(self, d_model: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, D]
        weights = torch.softmax(self.proj(x), dim=1)  # [B, T, 1]
        return (x * weights).sum(dim=1)  # [B, D]


class MusicTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        heads: int = 8,
        layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        norm_first: bool = False,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.position = SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model,
            heads,
            ff_dim,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d_model))
        self.pooling = pooling
        if pooling == "attention":
            self.pooler = AttentivePooling(d_model)
        elif pooling == "mean":
            self.pooler = None
        else:
            raise ValueError(f"Unknown pooling type: {pooling}")

        if norm_first:
            self.output_norm = nn.LayerNorm(d_model)
        else:
            self.output_norm = None

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != 128:
            raise ValueError("Expected Transformer input with shape [B, windows, 128]")
        encoded = self.encoder(self.position(x))
        if self.pooler is not None:
            pooled = self.pooler(encoded)
        else:
            pooled = encoded.mean(dim=1)
        if self.output_norm is not None:
            pooled = self.output_norm(pooled)
        return pooled