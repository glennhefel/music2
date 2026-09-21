from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier


class RandomForestEraClassifier:
    """Classify eras from frozen audio-model representations.

    The classifier is deliberately separate from the PyTorch model. Train the
    audio branch first, extract ``z`` or ``transformer_embedding`` with
    ``torch.no_grad()``, then fit this scikit-learn classifier on those vectors.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = None,
        random_state: int = 42,
        feature_name: str = "z",
    ) -> None:
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if feature_name not in {"z", "mu", "fused", "transformer_embedding"}:
            raise ValueError("feature_name must be 'z', 'mu', 'fused', or 'transformer_embedding'")
        self.feature_name = feature_name
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )

    @staticmethod
    def _to_numpy(features: Any) -> np.ndarray:
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        array = np.asarray(features, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("Features must have shape [samples, feature_dim]")
        if not np.isfinite(array).all():
            raise ValueError("Features contain NaN or infinite values")
        return array

    def fit(self, features: Any, labels: Any) -> "RandomForestEraClassifier":
        feature_array = self._to_numpy(features)
        label_array = np.asarray(labels)
        if label_array.ndim != 1 or len(label_array) != len(feature_array):
            raise ValueError("labels must be one-dimensional and match the number of samples")
        self.model.fit(feature_array, label_array)
        return self

    def predict(self, features: Any) -> np.ndarray:
        return self.model.predict(self._to_numpy(features))

    def predict_proba(self, features: Any) -> np.ndarray:
        return self.model.predict_proba(self._to_numpy(features))

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump({"feature_name": self.feature_name, "model": self.model}, path)

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestEraClassifier":
        import joblib

        payload = joblib.load(path)
        classifier = cls(feature_name=payload["feature_name"])
        classifier.model = payload["model"]
        return classifier