from typing import Optional

import torch
from torch import Tensor, nn

from ..config import ModelConfig
from .musicnn_encoder import MusicNNEncoder
from .transformer import MusicTransformer
from .vae import VAE


class AudioRepresentationModel(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None, feature_extractor: Optional[nn.Module] = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.config = config
        self.musicnn = MusicNNEncoder(config.musicnn_feature_dim, config.transformer_d_model, config.freeze_musicnn, feature_extractor)
        norm_first = getattr(config, "transformer_norm_first", False)
        pooling = getattr(config, "transformer_pooling", "mean")
        self.transformer = MusicTransformer(
            config.transformer_d_model,
            config.transformer_heads,
            config.transformer_layers,
            config.transformer_ff_dim,
            config.transformer_dropout,
            norm_first=norm_first,
            pooling=pooling,
        )
        self.vae = VAE(config.vae_input_dim, config.vae_hidden_dim, config.vae_latent_dim, config.vae_dropout)

    def forward(self, windows_or_features: Tensor) -> dict[str, Tensor]:
        tokens = self.musicnn(windows_or_features)
        transformer_embedding = self.transformer(tokens)
        reconstruction, mu, log_var, z = self.vae(transformer_embedding)
        return {"transformer_embedding": transformer_embedding, "reconstruction": reconstruction, "mu": mu, "log_var": log_var, "z": z}


class AudioClassificationModel(nn.Module):
    def __init__(self, num_classes: int, config: Optional[ModelConfig] = None, feature_extractor: Optional[nn.Module] = None, handcrafted_feature_dim: int = 0) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        config = config or ModelConfig()
        self.representation = AudioRepresentationModel(config, feature_extractor)
        self.handcrafted_feature_dim = handcrafted_feature_dim
        in_dim = config.transformer_d_model + config.vae_latent_dim + handcrafted_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(config.transformer_dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, windows_or_features: Tensor, handcrafted_features: Optional[Tensor] = None) -> dict[str, Tensor]:
        outputs = self.representation(windows_or_features)
        classifier_input = torch.cat([outputs["transformer_embedding"], outputs["mu"]], dim=-1)
        if self.handcrafted_feature_dim:
            if handcrafted_features is None or handcrafted_features.shape[-1] != self.handcrafted_feature_dim:
                raise ValueError(f"Expected handcrafted features with dimension {self.handcrafted_feature_dim}")
            classifier_input = torch.cat([classifier_input, handcrafted_features], dim=1)
        outputs["logits"] = self.classifier(classifier_input)
        return outputs