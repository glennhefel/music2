"""Train and evaluate the upgraded MusicTransformer (Pre-LN + Attentive Pooling).

Uses cached MusicNN features for instant training (~10-15 seconds for 18 epochs).
Saves the new model to: audio_classifier_attn_trans.pt
Evaluates on the untouched test split and prints a side-by-side comparison with the baseline.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from project.config import ModelConfig
from project.data import PreprocessedAudioDataset
from project.models.audio_model import AudioClassificationModel
from project.training.losses import audio_vae_loss


class FastCachedDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: list[int], handcrafted: np.ndarray, rec_ids: list[str]):
        self.features = features
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.handcrafted = torch.from_numpy(handcrafted).float()
        self.rec_ids = rec_ids

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {
            "features": self.features[idx],
            "label": self.labels[idx],
            "handcrafted": self.handcrafted[idx],
            "rec_id": self.rec_ids[idx],
        }


def evaluate_song_level(probs: np.ndarray, labels: list[str], y_true: list[str], rec_ids: list[str]) -> float:
    song_probs = collections.defaultdict(list)
    song_true = {}
    for p, t, r in zip(probs, y_true, rec_ids):
        song_probs[r].append(p)
        song_true[r] = t

    correct = sum(
        labels[np.argmax(np.mean(plist, axis=0))] == song_true[r]
        for r, plist in song_probs.items()
    )
    return correct / len(song_true)


def run_epoch(model, loader, optimizer, scheduler, device, beta: float, vae_weight: float):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_items = 0

    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        handcrafted = batch["handcrafted"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

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

    if training and scheduler is not None:
        scheduler.step()

    return total_loss / total_items, total_correct / total_items


def evaluate_model_on_test(model, test_loader, labels, device):
    model.eval()
    all_probs = []
    all_mu = []
    all_fused = []
    y_true_indices = []
    rec_ids = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            handcrafted = batch["handcrafted"].to(device)

            outputs = model(features, handcrafted)
            probs = torch.softmax(outputs["logits"], dim=-1).cpu().numpy()
            mu = outputs["mu"].cpu().numpy()
            hc = handcrafted.cpu().numpy()
            fused = np.concatenate([mu, hc], axis=1)

            all_probs.append(probs)
            all_mu.append(mu)
            all_fused.append(fused)
            y_true_indices.extend(batch["label"].numpy())
            rec_ids.extend(batch["rec_id"])

    all_probs = np.concatenate(all_probs, axis=0)
    all_mu = np.concatenate(all_mu, axis=0)
    all_fused = np.concatenate(all_fused, axis=0)

    preds_str = [labels[idx] for idx in np.argmax(all_probs, axis=1)]
    true_str = [labels[idx] for idx in y_true_indices]

    seg_acc = accuracy_score(true_str, preds_str)
    song_acc = evaluate_song_level(all_probs, labels, true_str, rec_ids)

    return {
        "seg_acc": seg_acc,
        "song_acc": song_acc,
        "mu": all_mu,
        "fused": all_fused,
        "probs": all_probs,
        "y_true": true_str,
        "rec_ids": rec_ids,
    }


def main():
    parser = argparse.ArgumentParser(description="Train and benchmark upgraded Attentive Transformer")
    parser.add_argument("--epochs", type=int, default=18, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2.64e-4, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout rate")
    parser.add_argument("--beta", type=float, default=1.993, help="VAE beta")
    parser.add_argument("--vae-weight", type=float, default=0.23, help="VAE loss weight")
    parser.add_argument("--layers", type=int, default=4, help="Number of transformer layers (2 or 4)")
    parser.add_argument("--pooling", choices=["attention", "mean"], default="attention", help="Pooling type")
    parser.add_argument("--norm-first", action="store_true", default=True, help="Use Pre-LN")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path("preprocessed")
    handcrafted_csv = Path("Data/features_3_sec_generated.csv")
    if not handcrafted_csv.is_file():
        handcrafted_csv = Path("Data/features_3_sec.csv")

    print("=" * 70)
    print("UPGRADED TRANSFORMER TRAINING & BENCHMARK")
    print(f"Architecture: Pre-LN={args.norm_first}, Pooling={args.pooling.upper()}, Layers={args.layers}")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}")
    print(f"Device: {args.device}")
    print("=" * 70)

    # 1. Load Handcrafted Normalization Stats
    train_ds = PreprocessedAudioDataset(data_dir / "train.jsonl", handcrafted_csv)
    val_ds = PreprocessedAudioDataset(data_dir / "validation.jsonl", handcrafted_csv)
    test_ds = PreprocessedAudioDataset(data_dir / "test.jsonl", handcrafted_csv)

    hc_mat = train_ds.handcrafted_matrix()
    hc_mean = hc_mat.mean(axis=0)
    hc_std = hc_mat.std(axis=0)
    hc_dim = hc_mat.shape[1]

    train_ds.set_handcrafted_normalization(hc_mean, hc_std)
    val_ds.set_handcrafted_normalization(hc_mean, hc_std)
    test_ds.set_handcrafted_normalization(hc_mean, hc_std)

    # 2. Extract Labels
    labels = sorted(set(r["era_label"] for r in train_ds.records))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    num_classes = len(labels)

    # 3. Load Cached MusicNN Features
    cached_train = torch.load(data_dir / "musicnn_cache_train.pt", weights_only=False)["features"]
    cached_val = torch.load(data_dir / "musicnn_cache_validation.pt", weights_only=False)["features"]
    cached_test = torch.load(data_dir / "musicnn_cache_test.pt", weights_only=False)["features"]

    train_hc = np.stack([train_ds._handcrafted_for_record(r) for r in train_ds.records])
    val_hc = np.stack([val_ds._handcrafted_for_record(r) for r in val_ds.records])
    test_hc = np.stack([test_ds._handcrafted_for_record(r) for r in test_ds.records])

    train_dataset = FastCachedDataset(
        cached_train,
        [label_to_idx[r["era_label"]] for r in train_ds.records],
        train_hc,
        [r["recording_id"] for r in train_ds.records],
    )
    val_dataset = FastCachedDataset(
        cached_val,
        [label_to_idx[r["era_label"]] for r in val_ds.records],
        val_hc,
        [r["recording_id"] for r in val_ds.records],
    )
    test_dataset = FastCachedDataset(
        cached_test,
        [label_to_idx[r["era_label"]] for r in test_ds.records],
        test_hc,
        [r["recording_id"] for r in test_ds.records],
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 4. Instantiate Upgraded Model
    config = ModelConfig(
        transformer_layers=args.layers,
        transformer_dropout=args.dropout,
        transformer_norm_first=args.norm_first,
        transformer_pooling=args.pooling,
        vae_dropout=args.dropout,
        vae_beta=args.beta,
        learning_rate=args.lr,
    )

    model = AudioClassificationModel(
        num_classes=num_classes,
        config=config,
        feature_extractor=None,
        handcrafted_feature_dim=hc_dim,
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 20
    )

    print("\nTraining upgraded model on cached features...")
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer, scheduler, args.device, args.beta, args.vae_weight
        )
        with torch.no_grad():
            val_loss, val_acc = run_epoch(
                model, val_loader, None, None, args.device, args.beta, args.vae_weight
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 3 == 0 or epoch == args.epochs:
            print(
                f"  Epoch {epoch:2d}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:5.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:5.2f}%"
            )

    # Load best model weights from validation
    model.load_state_dict({k: v.to(args.device) for k, v in best_state.items()})

    # Save upgraded checkpoint
    save_path = Path("audio_classifier_attn_trans.pt")
    torch.save(
        {
            "model": best_state,
            "labels": labels,
            "config": config.__dict__,
            "val_accuracy": best_val_acc,
        },
        save_path,
    )
    print(f"\nSaved best upgraded checkpoint to: {save_path} (Val Acc: {best_val_acc*100:.2f}%)")

    # 5. Evaluate on Untouched Test Split
    upgraded_eval = evaluate_model_on_test(model, test_loader, labels, args.device)

    # 6. Evaluate Baseline Checkpoint for Direct Head-to-Head Comparison
    baseline_path = Path("audio_classifier_fused_test.pt")
    baseline_eval = None
    if baseline_path.is_file():
        base_ckpt = torch.load(baseline_path, map_location=args.device, weights_only=False)
        base_cfg = ModelConfig(**base_ckpt["config"])
        base_model = AudioClassificationModel(
            num_classes=num_classes,
            config=base_cfg,
            feature_extractor=None,
            handcrafted_feature_dim=hc_dim,
        ).to(args.device)
        base_model.load_state_dict(base_ckpt["model"])
        baseline_eval = evaluate_model_on_test(base_model, test_loader, labels, args.device)

    # 7. Print Comparative Benchmark
    print("\n" + "=" * 76)
    print("TRANSFORMER ARCHITECTURE HEAD-TO-HEAD BENCHMARK (TEST SPLIT: 450 clips, 150 songs)")
    print("=" * 76)
    print(f"{'Architecture':<42} | {'Segment Acc':<13} | {'Song-Level (Voting)':<20}")
    print("-" * 76)
    if baseline_eval:
        print(
            f"{'Baseline: Post-LN + Mean Pooling (4 layers)':<42} | "
            f"{baseline_eval['seg_acc'] * 100:6.2f}%      | "
            f"{baseline_eval['song_acc'] * 100:6.2f}%"
        )
    print(
        f"{f'Upgraded: Pre-LN + Attentive Pooling ({args.layers} layers)':<42} | "
        f"{upgraded_eval['seg_acc'] * 100:6.2f}%      | "
        f"{upgraded_eval['song_acc'] * 100:6.2f}%"
    )
    print("=" * 76)

    # Delta reporting
    if baseline_eval:
        seg_diff = (upgraded_eval['seg_acc'] - baseline_eval['seg_acc']) * 100
        song_diff = (upgraded_eval['song_acc'] - baseline_eval['song_acc']) * 100
        print(f"\nDifference: Segment Acc {seg_diff:+5.2f}%, Song-Level Acc {song_diff:+5.2f}%")


if __name__ == "__main__":
    main()
