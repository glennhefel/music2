import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class PreprocessedAudioDataset(Dataset):
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        with self.manifest_path.open(encoding="utf-8") as file:
            self.records = [json.loads(line) for line in file if line.strip()]
        if not self.records:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        record = self.records[index]
        audio_path = self.root / record["audio_npy"]
        audio = np.load(audio_path).astype(np.float32, copy=False)
        if audio.shape != (4, 48000):
            raise ValueError(f"Expected audio shape [4, 48000], got {audio.shape}: {audio_path}")
        return {
            "audio": torch.from_numpy(audio),
            "era_label": record["era_label"],
            "recording_id": record["recording_id"],
        }