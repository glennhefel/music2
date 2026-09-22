from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    musicnn_model: str = "MTT_musicnn"
    sample_rate: int = 16000
    audio_segment_seconds: float = 12.0
    musicnn_window_seconds: float = 3.0
    musicnn_feature_dim: int = 200  # Official MTT_musicnn penultimate dense output.
    transformer_d_model: int = 128
    transformer_heads: int = 8
    transformer_layers: int = 4
    transformer_ff_dim: int = 512
    transformer_dropout: float = 0.15
    transformer_norm_first: bool = False
    transformer_pooling: str = "mean"
    vae_input_dim: int = 128
    vae_hidden_dim: int = 128
    vae_latent_dim: int = 64
    vae_dropout: float = 0.15
    vae_beta: float = 1.993
    freeze_musicnn: bool = True
    learning_rate: float = 2.64e-4
    random_forest_estimators: int = 300
    random_forest_max_depth: int | None = 16
    random_forest_random_state: int = 42
    classifier_feature: str = "z"

    @property
    def segment_samples(self) -> int:
        return int(self.audio_segment_seconds * self.sample_rate)

    @property
    def window_samples(self) -> int:
        return int(self.musicnn_window_seconds * self.sample_rate)

    @property
    def num_windows(self) -> int:
        return int(self.audio_segment_seconds / self.musicnn_window_seconds)