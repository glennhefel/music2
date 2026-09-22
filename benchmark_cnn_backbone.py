"""Benchmark CNN Backbones: Million Song Dataset (MSD) vs MagnaTagATune (MTT).

Trains the full classification pipeline on MSD_musicnn features and compares test performance
head-to-head with the MTT_musicnn baseline on the untouched test split (150 songs, 450 clips).
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


def evaluate_model(model, loader, labels, device):
    model.eval()
    all_probs = []
    y_true_indices = []
    rec_ids = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            handcrafted = batch["handcrafted"].to(device)

            outputs = model(features, handcrafted)
            probs = torch.softmax(outputs["logits"], dim=-1).cpu().numpy()

            all_probs.append(probs)
            y_true_indices.extend(batch["label"].numpy())
            rec_ids.extend(batch["rec_id"])

    all_probs = np.concatenate(all_probs, axis=0)
    preds_str = [labels[idx] for idx in np.argmax(all_probs, axis=1)]
    true_str = [labels[idx] for idx in y_true_indices]

    seg_acc = accuracy_score(true_str, preds_str)
    song_acc = evaluate_song_level(all_probs, labels, true_str, rec_ids)
    return seg_acc, song_acc


def main():
    parser = argparse.ArgumentParser(description="Benchmark MSD vs MTT CNN Backbones")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2.64e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--beta", type=float, default=1.993)
    parser.add_argument("--vae-weight", type=float, default=0.23)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path("preprocessed")
    handcrafted_csv = Path("Data/features_3_sec_generated.csv")
    if not handcrafted_csv.is_file():
        handcrafted_csv = Path("Data/features_3_sec.csv")

    # 1. Dataset metadata
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

    labels = sorted(set(r["era_label"] for r in train_ds.records))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    num_classes = len(labels)

    train_hc = np.stack([train_ds._handcrafted_for_record(r) for r in train_ds.records])
    val_hc = np.stack([val_ds._handcrafted_for_record(r) for r in val_ds.records])
    test_hc = np.stack([test_ds._handcrafted_for_record(r) for r in test_ds.records])

    # 2. Load MSD cached features
    print("Loading MSD_musicnn cached features...")
    msd_train_feats = torch.load(data_dir / "msd_cache_train.pt", weights_only=False)["features"]
    msd_val_feats = torch.load(data_dir / "msd_cache_validation.pt", weights_only=False)["features"]
    msd_test_feats = torch.load(data_dir / "msd_cache_test.pt", weights_only=False)["features"]

    msd_train_loader = DataLoader(
        FastCachedDataset(msd_train_feats, [label_to_idx[r["era_label"]] for r in train_ds.records], train_hc, [r["recording_id"] for r in train_ds.records]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    msd_val_loader = DataLoader(
        FastCachedDataset(msd_val_feats, [label_to_idx[r["era_label"]] for r in val_ds.records], val_hc, [r["recording_id"] for r in val_ds.records]),
        batch_size=args.batch_size,
        shuffle=False,
    )
    msd_test_loader = DataLoader(
        FastCachedDataset(msd_test_feats, [label_to_idx[r["era_label"]] for r in test_ds.records], test_hc, [r["recording_id"] for r in test_ds.records]),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # 3. Train model with MSD backbone
    print("\n" + "=" * 70)
    print("TRAINING WITH MILLION SONG DATASET (MSD_musicnn) CNN BACKBONE")
    print("=" * 70)
    config = ModelConfig(
        musicnn_model="MSD_musicnn",
        transformer_dropout=args.dropout,
        vae_dropout=args.dropout,
        vae_beta=args.beta,
        learning_rate=args.lr,
    )

    model_msd = AudioClassificationModel(
        num_classes=num_classes,
        config=config,
        feature_extractor=None,
        handcrafted_feature_dim=hc_dim,
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        [p for p in model_msd.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 20
    )

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model_msd, msd_train_loader, optimizer, scheduler, args.device, args.beta, args.vae_weight)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(model_msd, msd_val_loader, None, None, args.device, args.beta, args.vae_weight)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model_msd.state_dict().items()}

        if epoch % 3 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:2d}/{args.epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc*100:5.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc*100:5.2f}%")

    model_msd.load_state_dict({k: v.to(args.device) for k, v in best_state.items()})
    torch.save(
        {"model": best_state, "labels": labels, "config": config.__dict__, "val_accuracy": best_val_acc},
        "audio_classifier_msd.pt",
    )
    print(f"\nSaved MSD model to audio_classifier_msd.pt (Val Acc: {best_val_acc*100:.2f}%)")

    # 4. Evaluate MSD Model on Test Set
    msd_seg_acc, msd_song_acc = evaluate_model(model_msd, msd_test_loader, labels, args.device)

    # 5. Evaluate MTT Baseline Model on Test Set
    base_seg_acc, base_song_acc = 0.0, 0.0
    if Path("audio_classifier_fused_test.pt").is_file():
        mtt_test_feats = torch.load(data_dir / "musicnn_cache_test.pt", weights_only=False)["features"]
        mtt_test_loader = DataLoader(
            FastCachedDataset(mtt_test_feats, [label_to_idx[r["era_label"]] for r in test_ds.records], test_hc, [r["recording_id"] for r in test_ds.records]),
            batch_size=args.batch_size,
            shuffle=False,
        )
        base_ckpt = torch.load("audio_classifier_fused_test.pt", map_location=args.device, weights_only=False)
        base_cfg = ModelConfig(**base_ckpt["config"])
        base_model = AudioClassificationModel(num_classes, base_cfg, None, hc_dim).to(args.device)
        base_model.load_state_dict(base_ckpt["model"])
        base_seg_acc, base_song_acc = evaluate_model(base_model, mtt_test_loader, labels, args.device)

    # 6. Benchmark Comparison Table
    print("\n" + "=" * 76)
    print("CNN BACKBONE HEAD-TO-HEAD BENCHMARK (TEST SPLIT: 450 clips, 150 songs)")
    print("=" * 76)
    print(f"{'CNN Feature Extractor Backbone':<42} | {'Segment Acc':<13} | {'Song-Level (Voting)':<20}")
    print("-" * 76)
    print(f"{'1. MTT_musicnn (MagnaTagATune - 25k songs)':<42} | {base_seg_acc * 100:6.2f}%      | {base_song_acc * 100:6.2f}%")
    print(f"{'2. MSD_musicnn (Million Song Dataset - 200k songs)':<42} | {msd_seg_acc * 100:6.2f}%      | {msd_song_acc * 100:6.2f}%")
    print("=" * 76)
    diff_seg = (msd_seg_acc - base_seg_acc) * 100
    diff_song = (msd_song_acc - base_song_acc) * 100
    print(f"\nNet Difference: Segment Acc {diff_seg:+5.2f}%, Song-Level Acc {diff_song:+5.2f}%\n")


if __name__ == "__main__":
    main()
