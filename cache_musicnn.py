"""Pre-extract and cache MusicNN features for all splits.

Run this ONCE with the TensorFlow venv:
    .\.venv-tf\Scripts\python.exe cache_musicnn.py

This saves preprocessed/musicnn_cache_{split}.pt for train, validation, test.
Subsequent training runs load from cache and run epochs in ~3-5 seconds each.
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from project.config import ModelConfig
from project.data import PreprocessedAudioDataset
from project.models.musicnn_tensorflow import TensorFlowMusicNNExtractor

DATA_DIR = Path("preprocessed")
MUSICNN_ROOT = Path("musicnn")
BATCH_SIZE = 64


def cache_split(split: str, extractor: TensorFlowMusicNNExtractor) -> None:
    cache_path = DATA_DIR / f"musicnn_cache_{split}.pt"
    if cache_path.is_file():
        print(f"[{split}] Cache already exists at {cache_path}, skipping.")
        return

    manifest = DATA_DIR / f"{split}.jsonl"
    dataset = PreprocessedAudioDataset(manifest)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    total = len(loader)

    all_features = []
    all_labels = []
    all_recording_ids = []

    print(f"\n[{split}] Extracting MusicNN features for {len(dataset)} segments...")
    for i, batch in enumerate(loader, 1):
        audio = batch["audio"]  # [B, 1, 48000]
        with torch.no_grad():
            feats = extractor(audio)  # [B, 1, 200]
        all_features.append(feats.cpu())
        all_labels.extend(batch["era_label"])
        all_recording_ids.extend(batch["recording_id"])
        if i % 10 == 0 or i == total:
            print(f"  [{split}] {i}/{total} batches ({i/total*100:.0f}%)", flush=True)

    features_tensor = torch.cat(all_features, dim=0)  # [N, 1, 200]
    torch.save(
        {"features": features_tensor, "labels": all_labels, "recording_ids": all_recording_ids},
        cache_path,
    )
    size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"[{split}] Saved {len(all_labels)} samples to {cache_path} ({size_mb:.1f} MB)")


def main() -> None:
    print("=== MusicNN Feature Pre-Extraction ===")
    config = ModelConfig()
    extractor = TensorFlowMusicNNExtractor(MUSICNN_ROOT, config.musicnn_model)
    try:
        for split in ("train", "validation", "test"):
            cache_split(split, extractor)
    finally:
        extractor.close()
    print("\nDone! You can now run train_fast.py for instant epochs.")


if __name__ == "__main__":
    main()
