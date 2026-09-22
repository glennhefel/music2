"""Train Random Forest classifiers on top of the learned latent representations.

Extracts representations from the trained AudioRepresentationModel / AudioClassificationModel:
  1. Pure VAE Latent Vectors (64-D mu / z)
  2. Fused Representation (64-D mu + 57-D normalized handcrafted features = 121-D)

Evaluates:
  - Segment-level accuracy
  - Song-level (recording-level) majority-voting accuracy

Saves the fitted models to:
  - rf_latent_classifier.pkl
  - rf_fused_classifier.pkl
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from project.config import ModelConfig
from project.data import PreprocessedAudioDataset
from project.models.audio_model import AudioClassificationModel


def extract_split_representations(
    checkpoint_path: Path,
    data_dir: Path,
    split: str,
    handcrafted_csv: Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Extracts neural latent z/mu and fused representations for a given split."""
    print(f"Extracting representations for '{split}' split...")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig(**ckpt["config"])
    labels = ckpt["labels"]
    num_classes = len(labels)

    # Load dataset for handcrafted normalization & recording metadata
    train_dataset = PreprocessedAudioDataset(data_dir / "train.jsonl", handcrafted_csv)
    train_matrix = train_dataset.handcrafted_matrix()
    feat_mean = train_matrix.mean(axis=0)
    feat_std = train_matrix.std(axis=0)

    dataset = PreprocessedAudioDataset(data_dir / f"{split}.jsonl", handcrafted_csv)
    dataset.set_handcrafted_normalization(feat_mean, feat_std)
    handcrafted_dim = train_matrix.shape[1]

    # Instantiate model without feature_extractor (since input is cached 200-D features)
    model = AudioClassificationModel(num_classes, config, feature_extractor=None, handcrafted_feature_dim=handcrafted_dim)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # Load cached MusicNN features (auto-detect MSD vs MTT)
    is_msd = "MSD" in config.musicnn_model or "msd" in checkpoint_path.stem.lower()
    cache_prefix = "msd_cache" if is_msd else "musicnn_cache"
    cache_path = data_dir / f"{cache_prefix}_{split}.pt"
    if not cache_path.is_file():
        raise FileNotFoundError(f"Missing cache: {cache_path}. Run extraction first.")
    cached = torch.load(cache_path, weights_only=False)
    musicnn_features = cached["features"]  # [N, 4, 200]

    all_mu = []
    all_fused = []
    all_labels = []
    recording_ids = []

    batch_size = 64
    total_samples = len(dataset)

    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_feats = musicnn_features[i : i + batch_size].to(device)
            records = dataset.records[i : i + batch_size]

            batch_handcrafted = torch.stack(
                [torch.from_numpy(dataset._handcrafted_for_record(r)) for r in records]
            ).to(device)

            outputs = model(batch_feats, batch_handcrafted)
            mu = outputs["mu"].cpu().numpy()  # [B, 64]
            handcrafted = batch_handcrafted.cpu().numpy()  # [B, 57]
            fused = np.concatenate([mu, handcrafted], axis=1)  # [B, 121]

            all_mu.append(mu)
            all_fused.append(fused)
            all_labels.extend([r["era_label"] for r in records])
            recording_ids.extend([r["recording_id"] for r in records])

    return {
        "mu": np.concatenate(all_mu, axis=0),
        "fused": np.concatenate(all_fused, axis=0),
        "labels": np.array(all_labels),
        "recording_ids": np.array(recording_ids),
        "class_names": labels,
    }


def evaluate_song_level(
    model: RandomForestClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    recording_ids: np.ndarray,
) -> float:
    """Evaluates accuracy by aggregating segment predictions per song (recording_id)."""
    probabilities = model.predict_proba(features)
    classes = model.classes_

    song_probs: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    song_true_label: dict[str, str] = {}

    for prob, label, rec_id in zip(probabilities, labels, recording_ids):
        song_probs[rec_id].append(prob)
        song_true_label[rec_id] = label

    correct = 0
    for rec_id, prob_list in song_probs.items():
        avg_prob = np.mean(prob_list, axis=0)
        pred_label = classes[np.argmax(avg_prob)]
        if pred_label == song_true_label[rec_id]:
            correct += 1

    return correct / len(song_probs)


def train_and_eval_rf(
    train_data: dict[str, Any],
    test_data: dict[str, Any],
    feat_key: str,
    name: str,
    output_path: Path,
    n_estimators: int = 300,
    max_depth: int | None = 16,
    seed: int = 42,
) -> tuple[float, float]:
    print(f"\n{'='*60}")
    print(f"Training Random Forest on: {name} ({train_data[feat_key].shape[1]} features)")
    print(f"{'='*60}")

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion="entropy",
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
    )

    X_train = train_data[feat_key]
    y_train = train_data["labels"]
    X_test = test_data[feat_key]
    y_test = test_data["labels"]

    clf.fit(X_train, y_train)

    # Segment-level evaluation
    y_pred = clf.predict(X_test)
    segment_acc = float(accuracy_score(y_test, y_pred))

    # Song-level evaluation (majority voting across 12-second segments)
    song_acc = evaluate_song_level(clf, X_test, y_test, test_data["recording_ids"])

    print(f"Segment-Level Test Accuracy: {segment_acc * 100:.2f}%")
    print(f"Song-Level Test Accuracy:    {song_acc * 100:.2f}%  (Averaging across clips of same song)")
    print("\nClassification Report (Segment-Level):")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save classifier
    joblib.dump(
        {
            "model": clf,
            "feature_name": feat_key,
            "feature_dim": X_train.shape[1],
            "segment_accuracy": segment_acc,
            "song_accuracy": song_acc,
            "classes": clf.classes_,
        },
        output_path,
    )
    print(f"Saved Random Forest model to {output_path}")
    return segment_acc, song_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Random Forest on latent vectors")
    parser.add_argument("--checkpoint", type=Path, default=Path("audio_classifier_fused_test.pt"), help="Path to trained PyTorch checkpoint")
    parser.add_argument("--data-dir", type=Path, default=Path("preprocessed"), help="Path to preprocessed directory")
    parser.add_argument("--handcrafted-csv", type=Path, default=Path("Data/features_3_sec.csv"), help="Handcrafted feature CSV")
    parser.add_argument("--estimators", type=int, default=300, help="Random Forest tree count")
    parser.add_argument("--max-depth", type=int, default=16, help="Random Forest max depth")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_data = extract_split_representations(args.checkpoint, args.data_dir, "train", args.handcrafted_csv, args.device)
    test_data = extract_split_representations(args.checkpoint, args.data_dir, "test", args.handcrafted_csv, args.device)

    is_msd = "msd" in str(args.checkpoint).lower()
    suffix = "_msd" if is_msd else ""

    # 1. Random Forest on pure 64-D VAE latent space
    train_and_eval_rf(
        train_data,
        test_data,
        feat_key="mu",
        name="64-D Pure VAE Latent (mu)",
        output_path=Path(f"rf_latent_classifier{suffix}.pkl"),
        n_estimators=args.estimators,
        max_depth=args.max_depth,
        seed=args.seed,
    )

    # 2. Random Forest on 121-D Fused space (64-D mu + 57-D handcrafted)
    train_and_eval_rf(
        train_data,
        test_data,
        feat_key="fused",
        name="121-D Fused Representation (VAE mu + Handcrafted)",
        output_path=Path(f"rf_fused_classifier{suffix}.pkl"),
        n_estimators=args.estimators,
        max_depth=args.max_depth,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
