"""Optuna Hyperparameter Tuning for Musical Era / Genre Classification.

Supports two tuning modes:
  1. `rf` (Random Forest baseline on handcrafted audio features CSV):
     Fast CPU tuning of tree count, depth, split thresholds, and leaf sizes.
  2. `neural` (PyTorch MusicNN + Transformer + VAE AudioClassificationModel):
     Tunes learning rate, VAE beta, dropout, batch size, and VAE loss weight.
     Strictly preserves the fixed architecture from message.txt:
       - d_model = 128, heads = 8, layers = 4, ff_dim = 512
       - vae_input_dim = 128, vae_hidden_dim = 128, vae_latent_dim = 64
     
     NOTE ON EFFICIENCY:
       Because MusicNN is frozen (freeze_musicnn=True), its 200-D feature
       representations never change. This script automatically pre-caches the
       200-D features on the first pass (or loads them if cached), so epochs
       train in ~0.5s instead of 4 minutes!

Usage:
  # Random Forest baseline:
  python optuna_tune.py --mode rf --n-trials 50 --csv Data/features_30_sec.csv

  # Neural Classifier (ultra-fast with cached MusicNN features):
  python optuna_tune.py --mode neural --n-trials 20 --epochs 5 --data-dir preprocessed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# =====================================================================
# Safe Optuna Import (guards against local 'optuna' git folder collision)
# =====================================================================
def _import_optuna():
    cwd = os.path.abspath(os.getcwd())
    optuna_folder = os.path.join(cwd, "optuna")
    if os.path.isdir(optuna_folder) and not os.path.isfile(os.path.join(optuna_folder, "__init__.py")):
        saved_path = list(sys.path)
        sys.path = [p for p in sys.path if os.path.abspath(p) != cwd]
        try:
            import optuna
            return optuna
        except ImportError as err:
            raise ImportError(
                "Optuna is not installed in your Python environment.\n"
                "Install it using:\n"
                "    uv pip install --python .venv\\Scripts\\python.exe optuna\n"
            ) from err
        finally:
            sys.path = saved_path
    else:
        try:
            import optuna
            return optuna
        except ImportError as err:
            raise ImportError(
                "Optuna is not installed in your Python environment.\n"
                "Install it using:\n"
                "    uv pip install --python .venv\\Scripts\\python.exe optuna\n"
            ) from err

optuna = _import_optuna()


# =====================================================================
# Random Forest Hyperparameter Tuning
# =====================================================================
def objective_random_forest(trial: optuna.Trial, csv_path: Path, seed: int) -> float:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from project.training.train_random_forest import load_feature_csv

    features, labels = load_feature_csv(csv_path)

    n_estimators = trial.suggest_int("n_estimators", 50, 400, step=25)
    max_depth_choice = trial.suggest_categorical("has_max_depth", [True, False])
    max_depth = trial.suggest_int("max_depth", 5, 40) if max_depth_choice else None
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 6)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        random_state=seed,
        n_jobs=-1,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, features, labels, cv=skf, scoring="accuracy")
    mean_score = float(scores.mean())
    print(f"  [Trial {trial.number}] Accuracy: {mean_score:.4f} (estimators={n_estimators}, depth={max_depth}, criterion={criterion})", flush=True)
    return mean_score


def tune_random_forest(csv_path: Path, n_trials: int, seed: int, output_file: Path) -> dict[str, Any]:
    print(f"=== Starting Optuna Study for Random Forest ({n_trials} trials) ===")
    print(f"Feature CSV: {csv_path}\n")

    study = optuna.create_study(
        direction="maximize",
        study_name="random_forest_tuning",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(lambda trial: objective_random_forest(trial, csv_path, seed), n_trials=n_trials)

    print("\n=== Random Forest Optimization Complete ===")
    print(f"Best Accuracy: {study.best_value:.4f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    results = {
        "mode": "random_forest",
        "best_accuracy": study.best_value,
        "best_params": study.best_params,
    }
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved best parameters to {output_file}")
    return results


# =====================================================================
# Pre-cached Dataset for Ultra-Fast Neural Training
# =====================================================================
class FastFeatureDataset:
    """Stores pre-extracted MusicNN features [N, 4, 200] in RAM."""

    def __init__(self, features, labels: list[int], handcrafted=None):
        import torch
        self.features = features if torch.is_tensor(features) else torch.from_numpy(features)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.handcrafted = None
        if handcrafted is not None:
            self.handcrafted = handcrafted if torch.is_tensor(handcrafted) else torch.from_numpy(handcrafted)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        item = {
            "features": self.features[idx],
            "label": self.labels[idx],
        }
        if self.handcrafted is not None:
            item["handcrafted"] = self.handcrafted[idx]
        return item


def load_or_cache_features(data_dir: Path, split: str, musicnn_root: Path, batch_size: int = 32):
    """Extracts MusicNN features once and caches them to disk."""
    import torch
    from torch.utils.data import DataLoader
    from project.data import PreprocessedAudioDataset
    from project.models.musicnn_tensorflow import TensorFlowMusicNNExtractor
    from project.config import ModelConfig

    cache_path = data_dir / f"musicnn_cache_{split}.pt"
    manifest_path = data_dir / f"{split}.jsonl"

    if cache_path.is_file():
        print(f"Loading cached MusicNN features from {cache_path}...")
        data = torch.load(cache_path, weights_only=False)
        return data["features"], data["labels"]

    print(f"\n[One-time setup] Pre-extracting MusicNN 200-D features for '{split}' split...")
    print("This runs only ONCE. Subsequent epochs and trials will run in ~1 second.\n")

    dataset = PreprocessedAudioDataset(manifest_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    config = ModelConfig()
    extractor = TensorFlowMusicNNExtractor(musicnn_root, config.musicnn_model)

    all_features = []
    all_labels = []

    try:
        total_batches = len(loader)
        for i, batch in enumerate(loader):
            audio = batch["audio"]  # [B, 4, 48000]
            with torch.no_grad():
                feats = extractor(audio)  # [B, 4, 200]
            all_features.append(feats.cpu())
            all_labels.extend(batch["era_label"])
            if (i + 1) % 5 == 0 or (i + 1) == total_batches:
                print(f"  Extracting batch {i+1}/{total_batches} ({(i+1)/total_batches*100:.0f}%)...", flush=True)

        features_tensor = torch.cat(all_features, dim=0)
        torch.save({"features": features_tensor, "labels": all_labels}, cache_path)
        print(f"Saved {len(all_labels)} extracted samples to {cache_path} ({cache_path.stat().st_size / (1024*1024):.1f} MB)\n")
        return features_tensor, all_labels
    finally:
        extractor.close()


# =====================================================================
# Neural Classifier Hyperparameter Tuning
# =====================================================================
def run_fast_epoch(model, loader, optimizer, device, beta: float, vae_weight: float):
    import torch
    from torch import nn
    from project.training.losses import audio_vae_loss

    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_items = 0

    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        handcrafted = batch.get("handcrafted")
        if handcrafted is not None:
            handcrafted = handcrafted.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        # features are [B, 4, 200] -> model skips raw extractor and runs directly
        outputs = model(features, handcrafted)
        vae_total, _, _ = audio_vae_loss(outputs, beta)
        classification_loss = nn.functional.cross_entropy(outputs["logits"], labels)
        loss = classification_loss + vae_weight * vae_total

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(labels)
        total_correct += (outputs["logits"].argmax(dim=1) == labels).sum().item()
        total_items += len(labels)

    return total_loss / total_items, total_correct / total_items


def objective_neural(
    trial: optuna.Trial,
    train_dataset: FastFeatureDataset,
    val_dataset: FastFeatureDataset,
    num_classes: int,
    epochs_per_trial: int,
    device: str,
) -> float:
    import torch
    from torch.utils.data import DataLoader
    from project.config import ModelConfig
    from project.models.audio_model import AudioClassificationModel

    # Search space for training & regularization hyperparameters
    # Fixed architecture dimensions (d_model=128, 4 layers, 8 heads) remain locked
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    beta = trial.suggest_float("vae_beta", 0.1, 4.0)
    dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.05)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    vae_loss_weight = trial.suggest_float("vae_loss_weight", 0.02, 0.3)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    config = ModelConfig(
        learning_rate=lr,
        vae_beta=beta,
        transformer_dropout=dropout,
        vae_dropout=dropout,
    )

    # feature_extractor=None because input is already pre-extracted 200-D features!
    model = AudioClassificationModel(num_classes, config, feature_extractor=None).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    best_val_acc = 0.0
    for epoch in range(1, epochs_per_trial + 1):
        train_loss, train_acc = run_fast_epoch(model, train_loader, optimizer, device, beta, vae_loss_weight)
        with torch.no_grad():
            val_loss, val_acc = run_fast_epoch(model, val_loader, None, device, beta, vae_loss_weight)

        print(
            f"  [Trial {trial.number:2d}] Epoch {epoch}/{epochs_per_trial} | "
            f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.3f} | "
            f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.3f}",
            flush=True,
        )

        trial.report(val_acc, step=epoch)
        if trial.should_prune():
            print(f"  [Trial {trial.number:2d}] Pruned at epoch {epoch} (low val_acc: {val_acc:.3f})", flush=True)
            raise optuna.exceptions.TrialPruned()

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc


def tune_neural(
    data_dir: Path,
    musicnn_root: Path,
    n_trials: int,
    epochs: int,
    device: str,
    output_file: Path,
) -> dict[str, Any]:
    print(f"=== Preparing Optuna Study for Neural Classifier ({n_trials} trials, {epochs} epochs/trial) ===")
    print(f"Device: {device}\n")

    # Step 1: Extract or load cached features
    train_feats, train_labels_raw = load_or_cache_features(data_dir, "train", musicnn_root)
    val_feats, val_labels_raw = load_or_cache_features(data_dir, "validation", musicnn_root)

    # Create label mapping
    all_label_names = sorted(set(train_labels_raw))
    label_to_idx = {name: i for i, name in enumerate(all_label_names)}
    num_classes = len(all_label_names)

    train_labels = [label_to_idx[l] for l in train_labels_raw]
    val_labels = [label_to_idx[l] for l in val_labels_raw]

    train_dataset = FastFeatureDataset(train_feats, train_labels)
    val_dataset = FastFeatureDataset(val_feats, val_labels)

    print(f"Dataset ready: {len(train_dataset)} train, {len(val_dataset)} validation ({num_classes} classes)")
    print(f"Features loaded in RAM. Starting ultra-fast tuning...\n")

    study = optuna.create_study(
        direction="maximize",
        study_name="neural_classifier_tuning",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2),
    )
    study.optimize(
        lambda trial: objective_neural(
            trial, train_dataset, val_dataset, num_classes, epochs, device
        ),
        n_trials=n_trials,
    )

    print("\n=== Neural Model Optimization Complete ===")
    print(f"Best Validation Accuracy: {study.best_value:.4f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    results = {
        "mode": "neural",
        "best_val_accuracy": study.best_value,
        "best_params": study.best_params,
    }
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved best parameters to {output_file}")
    return results


# =====================================================================
# CLI Entrypoint
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Tune hyperparameters using Optuna")
    parser.add_argument("--mode", choices=["rf", "neural"], default="rf", help="Tuning target: rf (Random Forest) or neural (AudioModel)")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=Path, default=Path("best_params.json"), help="Output JSON path for best parameters")

    # RF options
    parser.add_argument("--csv", type=Path, default=Path("Data/features_30_sec.csv"), help="Feature CSV path for Random Forest")

    # Neural options
    parser.add_argument("--data-dir", type=Path, default=Path("preprocessed"), help="Directory containing train/val/test .jsonl manifests")
    parser.add_argument("--musicnn-root", type=Path, default=Path("musicnn"), help="Directory with musicnn weights/checkpoints")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs per neural trial")

    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "rf":
        tune_random_forest(args.csv, args.n_trials, args.seed, args.output)
    else:
        tune_neural(
            args.data_dir,
            args.musicnn_root,
            args.n_trials,
            args.epochs,
            device,
            args.output,
        )


if __name__ == "__main__":
    main()
