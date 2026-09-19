"""Optional bridge from the official TensorFlow musicnn model to PyTorch.

The bridge is intentionally inference-only. TensorFlow musicnn and the PyTorch
Transformer cannot share an autograd graph, so the extracted penultimate
features are detached before they enter the PyTorch model.
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn


class TensorFlowMusicNNExtractor(nn.Module):
    """Extract official MTT_musicnn 200-D penultimate features from windows."""

    output_dim = 200

    def __init__(self, musicnn_root: str | Path, model_name: str = "MTT_musicnn") -> None:
        super().__init__()
        self.musicnn_root = Path(musicnn_root)
        self.model_name = model_name
        self._session = None
        self._input = None
        self._training = None
        self._penultimate = None

    def _build(self, frame_count: int) -> None:
        try:
            import librosa
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "TensorFlow musicnn extraction requires librosa and a TensorFlow "
                "version compatible with the official TF1-style musicnn code. "
                "Use cached penultimate features when TensorFlow is unavailable."
            ) from exc

        tf.compat.v1.disable_eager_execution()
        from musicnn import configuration, models

        tf.compat.v1.reset_default_graph()
        self._input = tf.compat.v1.placeholder(tf.float32, [None, frame_count, configuration.N_MELS])
        self._training = tf.compat.v1.placeholder(tf.bool)
        labels = configuration.MTT_LABELS if self.model_name.startswith("MTT") else configuration.MSD_LABELS
        outputs = models.define_model(self._input, self._training, self.model_name, len(labels))
        self._penultimate = outputs[-1]
        self._session = tf.compat.v1.Session()
        self._session.run(tf.compat.v1.global_variables_initializer())
        checkpoint_directory = self.musicnn_root / self.model_name
        # Prefer the local empty-prefix shards. The checked-in metadata points
        # to the original author's Linux path and makes latest_checkpoint log
        # an error before it can discover the usable local files.
        if (checkpoint_directory / ".index").is_file():
            checkpoint = str(checkpoint_directory) + os.sep
        else:
            checkpoint = tf.train.latest_checkpoint(str(checkpoint_directory))
            if checkpoint is not None and not Path(f"{checkpoint}.index").is_file():
                checkpoint = None
        if checkpoint is None:
            raise FileNotFoundError(f"No TensorFlow checkpoint found in {checkpoint_directory}")
        tf.compat.v1.train.Saver().restore(self._session, checkpoint)

    def forward(self, windows: Tensor) -> Tensor:
        if windows.ndim != 3:
            raise ValueError("Expected raw windows with shape [batch, windows, samples]")
        if windows.shape[-1] != 48000:
            raise ValueError("Official musicnn windows must contain 48000 samples at 16000 Hz")
        import librosa

        batch, window_count, _ = windows.shape
        audio = windows.detach().cpu().numpy().reshape(-1, 48000)
        spectrograms = []
        for window in audio:
            mel = librosa.feature.melspectrogram(y=window, sr=16000, n_fft=512, hop_length=256, n_mels=96).T
            spectrograms.append(np.log10(10000.0 * mel + 1.0).astype(np.float32))
        features = np.stack(spectrograms)
        if self._session is None:
            self._build(features.shape[1])
        output = self._session.run(self._penultimate, feed_dict={self._input: features, self._training: False})
        return torch.from_numpy(np.asarray(output, dtype=np.float32)).reshape(batch, window_count, self.output_dim).to(windows.device)

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def __del__(self) -> None:
        self.close()