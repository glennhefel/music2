import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class PreprocessedAudioDataset(Dataset):
    def __init__(self, manifest_path: str | Path, handcrafted_csv: str | Path | None = None) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        with self.manifest_path.open(encoding="utf-8") as file:
            self.records = [json.loads(line) for line in file if line.strip()]
        if not self.records:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")
        self.handcrafted_features: dict[tuple[str, int], np.ndarray] | None = None
        self.handcrafted_max_chunk: dict[str, int] = {}
        self.handcrafted_mean = None
        self.handcrafted_std = None
        if handcrafted_csv is not None:
            self.handcrafted_features, self.handcrafted_max_chunk = self._load_handcrafted_features(handcrafted_csv)

    @staticmethod
    def _load_handcrafted_features(path: str | Path) -> tuple[dict[tuple[str, int], np.ndarray], dict[str, int]]:
        features: dict[tuple[str, int], np.ndarray] = {}
        max_chunk: dict[str, int] = {}
        with Path(path).open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                filename = Path(row["filename"]).stem
                recording_stem, chunk_text = filename.rsplit(".", 1)
                chunk_idx = int(chunk_text)
                names = [name for name in row if name not in {"filename", "length", "label"}]
                features[(recording_stem, chunk_idx)] = np.asarray([float(row[name]) for name in names], dtype=np.float32)
                if chunk_idx > max_chunk.get(recording_stem, -1):
                    max_chunk[recording_stem] = chunk_idx
        if not features:
            raise ValueError(f"Handcrafted feature CSV is empty: {path}")
        return features, max_chunk

    def set_handcrafted_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.handcrafted_mean = mean.astype(np.float32)
        self.handcrafted_std = np.maximum(std, 1e-6).astype(np.float32)

    def handcrafted_matrix(self) -> np.ndarray:
        return np.stack([self._handcrafted_for_record(record) for record in self.records])

    def _handcrafted_for_record(self, record: dict) -> np.ndarray:
        if self.handcrafted_features is None:
            raise ValueError("No handcrafted feature CSV was provided")
        recording_stem = record["recording_id"].split("__", 1)[-1]
        segment_index = int(record["segment_id"].rsplit("_", 1)[-1])
        # Each 12-second segment maps to 4 consecutive 3-second rows in features_3_sec.csv.
        # Clamp to last available chunk to handle tracks slightly longer than 30 s.
        chunks = []
        for offset in range(4):
            idx = segment_index * 4 + offset
            max_idx = self.handcrafted_max_chunk.get(recording_stem, idx)
            clamped = min(idx, max_idx)
            chunk = self.handcrafted_features.get((recording_stem, clamped))
            if chunk is not None:
                chunks.append(chunk)
        if not chunks:
            raise KeyError(f"Missing handcrafted features for {recording_stem}, segment {segment_index}")
        values = np.mean(np.stack(chunks), axis=0)
        if self.handcrafted_mean is not None:
            values = (values - self.handcrafted_mean) / self.handcrafted_std
        return values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        record = self.records[index]
        audio_path = self.root / record["audio_npy"]
        audio = np.load(audio_path).astype(np.float32, copy=False)
        if audio.shape != (4, 48000):
            raise ValueError(f"Expected audio shape [4, 48000], got {audio.shape}: {audio_path}")
        item = {
            "audio": torch.from_numpy(audio),
            "era_label": record["era_label"],
            "recording_id": record["recording_id"],
        }
        if self.handcrafted_features is not None:
            item["handcrafted_features"] = torch.from_numpy(self._handcrafted_for_record(record))
        return item