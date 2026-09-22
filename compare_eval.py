"""Head-to-head comparison on the untouched Test Split (150 songs, 450 clips).

Compares:
  1. Neural End-to-End Linear Head (nn.Linear on latent space)
  2. Random Forest on Pure VAE Latent (mu)
  3. Random Forest on Fused Latent (mu + HC)

Automatically detects whether the checkpoint uses the MSD or MTT backbone,
loads the appropriate pre-cached features (msd_cache vs musicnn_cache),
and uses or fits corresponding Random Forest classifiers.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from project.config import ModelConfig
from project.data import PreprocessedAudioDataset
from project.models.audio_model import AudioClassificationModel


def get_latest_checkpoint(default_candidates: list[str]) -> Path:
    existing = [Path(p) for p in default_candidates if Path(p).is_file()]
    if not existing:
        raise FileNotFoundError(f"No checkpoint found among: {default_candidates}")
    # Sort by modification time, newest first
    existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return existing[0]


def train_rf_models(
    model: AudioClassificationModel,
    train_ds: PreprocessedAudioDataset,
    train_feats: torch.Tensor,
    device: str,
    latent_rf_path: Path,
    fused_rf_path: Path,
    seed: int = 42,
) -> tuple[RandomForestClassifier, RandomForestClassifier]:
    print("Training matching Random Forest classifiers on train split representations...")
    model.eval()
    all_mu = []
    all_fused = []
    y_train = []
    batch_size = 64

    with torch.no_grad():
        for i in range(0, len(train_ds), batch_size):
            b_feats = train_feats[i : i + batch_size].to(device)
            records = train_ds.records[i : i + batch_size]
            b_hc = torch.stack(
                [torch.from_numpy(train_ds._handcrafted_for_record(r)) for r in records]
            ).to(device)

            out = model(b_feats, b_hc)
            mu = out["mu"].cpu().numpy()
            hc = b_hc.cpu().numpy()
            all_mu.append(mu)
            all_fused.append(np.concatenate([mu, hc], axis=1))
            y_train.extend([r["era_label"] for r in records])

    X_mu = np.concatenate(all_mu, axis=0)
    X_fused = np.concatenate(all_fused, axis=0)
    y_train = np.array(y_train)

    rf_latent = RandomForestClassifier(
        n_estimators=300,
        max_depth=16,
        criterion="entropy",
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf_latent.fit(X_mu, y_train)
    joblib.dump({"model": rf_latent, "classes": rf_latent.classes_}, latent_rf_path)

    rf_fused = RandomForestClassifier(
        n_estimators=300,
        max_depth=16,
        criterion="entropy",
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf_fused.fit(X_fused, y_train)
    joblib.dump({"model": rf_fused, "classes": rf_fused.classes_}, fused_rf_path)

    print(f"Saved fitted RF models to {latent_rf_path.name} and {fused_rf_path.name}")
    return rf_latent, rf_fused


def run_comparison(
    ckpt_path: Path | None = None,
    data_dir: Path = Path("preprocessed"),
    handcrafted_csv: Path = Path("Data/features_3_sec.csv"),
    retrain_rf: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if ckpt_path is None:
        ckpt_path = get_latest_checkpoint([
            "audio_classifier_msd.pt",
            "audio_classifier_fused_test.pt",
            "audio_classifier.pt",
        ])

    print(f"\nEvaluating Checkpoint: {ckpt_path.name} (device={device})")

    # 1. Load trained neural checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ModelConfig(**ckpt["config"])
    labels = ckpt["labels"]

    # Detect backbone
    is_msd = "MSD" in config.musicnn_model or "msd" in ckpt_path.stem.lower()
    cache_prefix = "msd_cache" if is_msd else "musicnn_cache"
    rf_suffix = "_msd" if is_msd else ""
    latent_rf_path = Path(f"rf_latent_classifier{rf_suffix}.pkl")
    fused_rf_path = Path(f"rf_fused_classifier{rf_suffix}.pkl")

    print(f"Detected Backbone:    {config.musicnn_model} (using {cache_prefix}_*.pt)")

    # 2. Load dataset & normalize handcrafted features
    train_ds = PreprocessedAudioDataset(data_dir / "train.jsonl", handcrafted_csv)
    train_mat = train_ds.handcrafted_matrix()
    mean = train_mat.mean(axis=0)
    std = train_mat.std(axis=0)
    train_ds.set_handcrafted_normalization(mean, std)

    test_ds = PreprocessedAudioDataset(data_dir / "test.jsonl", handcrafted_csv)
    test_ds.set_handcrafted_normalization(mean, std)

    model = AudioClassificationModel(
        len(labels), config, feature_extractor=None, handcrafted_feature_dim=train_mat.shape[1]
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # 3. Load or train Random Forest classifiers matching this checkpoint
    need_rf_train = retrain_rf or not latent_rf_path.is_file() or not fused_rf_path.is_file()
    if not need_rf_train:
        # Also check if checkpoint is newer than RF models
        ckpt_mtime = ckpt_path.stat().st_mtime
        if latent_rf_path.stat().st_mtime < ckpt_mtime:
            need_rf_train = True

    if need_rf_train:
        train_cache = torch.load(data_dir / f"{cache_prefix}_train.pt", weights_only=False)
        rf_latent_clf, rf_fused_clf = train_rf_models(
            model, train_ds, train_cache["features"], device, latent_rf_path, fused_rf_path
        )
    else:
        print(f"Loading existing RF models: {latent_rf_path.name}, {fused_rf_path.name}")
        rf_latent_clf = joblib.load(latent_rf_path)["model"]
        rf_fused_clf = joblib.load(fused_rf_path)["model"]

    # 4. Load cached features for test
    test_cache_path = data_dir / f"{cache_prefix}_test.pt"
    if not test_cache_path.is_file():
        raise FileNotFoundError(f"Missing test cache: {test_cache_path}")
    cached_test = torch.load(test_cache_path, weights_only=False)
    test_features = cached_test["features"]  # [450, 4, 200]

    # 5. Extract neural outputs and representations
    neural_probs = []
    all_mu = []
    all_fused = []
    y_true = []
    rec_ids = []

    with torch.no_grad():
        for i, record in enumerate(test_ds.records):
            feats = test_features[i : i + 1].to(device)
            hc = torch.from_numpy(test_ds._handcrafted_for_record(record)).unsqueeze(0).to(device)

            out = model(feats, hc)
            prob = torch.softmax(out["logits"], dim=-1).cpu().numpy()[0]
            mu = out["mu"].cpu().numpy()[0]
            fused = np.concatenate([mu, hc.cpu().numpy()[0]], axis=0)

            neural_probs.append(prob)
            all_mu.append(mu)
            all_fused.append(fused)
            y_true.append(record["era_label"])
            rec_ids.append(record["recording_id"])

    neural_probs = np.array(neural_probs)
    all_mu = np.array(all_mu)
    all_fused = np.array(all_fused)
    y_true = np.array(y_true)

    # 6. Evaluate Neural Head
    neural_preds = [labels[idx] for idx in np.argmax(neural_probs, axis=1)]
    neural_seg_acc = np.mean([p == t for p, t in zip(neural_preds, y_true)])

    song_neural_probs = collections.defaultdict(list)
    song_true = {}
    for p, t, r in zip(neural_probs, y_true, rec_ids):
        song_neural_probs[r].append(p)
        song_true[r] = t

    neural_song_correct = sum(
        labels[np.argmax(np.mean(plist, axis=0))] == song_true[r]
        for r, plist in song_neural_probs.items()
    )
    neural_song_acc = neural_song_correct / len(song_true)

    # 7. Evaluate RF on Latent
    rf_latent_probs = rf_latent_clf.predict_proba(all_mu)
    rf_latent_preds = rf_latent_clf.predict(all_mu)
    rf_latent_seg_acc = np.mean(rf_latent_preds == y_true)

    song_rf_lat_probs = collections.defaultdict(list)
    for p, r in zip(rf_latent_probs, rec_ids):
        song_rf_lat_probs[r].append(p)

    rf_lat_classes = rf_latent_clf.classes_
    rf_lat_song_correct = sum(
        rf_lat_classes[np.argmax(np.mean(plist, axis=0))] == song_true[r]
        for r, plist in song_rf_lat_probs.items()
    )
    rf_lat_song_acc = rf_lat_song_correct / len(song_true)

    # 8. Evaluate RF on Fused
    rf_fused_probs = rf_fused_clf.predict_proba(all_fused)
    rf_fused_preds = rf_fused_clf.predict(all_fused)
    rf_fused_seg_acc = np.mean(rf_fused_preds == y_true)

    song_rf_fused_probs = collections.defaultdict(list)
    for p, r in zip(rf_fused_probs, rec_ids):
        song_rf_fused_probs[r].append(p)

    rf_fused_classes = rf_fused_clf.classes_
    rf_fused_song_correct = sum(
        rf_fused_classes[np.argmax(np.mean(plist, axis=0))] == song_true[r]
        for r, plist in song_rf_fused_probs.items()
    )
    rf_fused_song_acc = rf_fused_song_correct / len(song_true)

    # Print Table
    title = f"HEAD-TO-HEAD COMPARISON: {ckpt_path.name} ({config.musicnn_model})"
    print("\n" + "=" * 70)
    print(title)
    print("TEST SPLIT: 150 songs, 450 clips")
    print("=" * 70)
    print(f"{'Classifier Head':<35} | {'Segment Accuracy':<16} | {'Song-Level (Voting)':<20}")
    print("-" * 70)
    print(f"{'1. Neural Head (MLP)':<35} | {neural_seg_acc * 100:6.2f}%          | {neural_song_acc * 100:6.2f}%")
    print(f"{'2. Random Forest on VAE Latent (mu)':<35} | {rf_latent_seg_acc * 100:6.2f}%          | {rf_lat_song_acc * 100:6.2f}%")
    print(f"{'3. Random Forest on Fused (mu + HC)':<35} | {rf_fused_seg_acc * 100:6.2f}%          | {rf_fused_song_acc * 100:6.2f}%")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Head-to-head comparison on test split")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint (.pt). If omitted, automatically selects the most recent checkpoint.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("preprocessed"),
        help="Directory containing preprocessed split and cache files",
    )
    parser.add_argument(
        "--handcrafted-csv",
        type=Path,
        default=Path("Data/features_3_sec.csv"),
        help="Path to handcrafted features CSV",
    )
    parser.add_argument(
        "--retrain-rf",
        action="store_true",
        help="Force re-training of Random Forest classifiers on train split representations",
    )
    args = parser.parse_args()

    run_comparison(
        ckpt_path=args.checkpoint,
        data_dir=args.data_dir,
        handcrafted_csv=args.handcrafted_csv,
        retrain_rf=args.retrain_rf,
    )


if __name__ == "__main__":
    main()

