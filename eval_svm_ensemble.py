"""Train and evaluate SVM (RBF Kernel) and Soft Voting / Blending Ensemble.

Compares:
  1. Neural Head (MLP) from audio_classifier_fused_test.pt
  2. SVM (RBF Kernel) on pure VAE Latent (mu)
  3. SVM (RBF Kernel) on Fused representation (mu + Handcrafted)
  4. Soft Voting / Blending Ensemble (Neural Head + SVM Fused)
"""

from __future__ import annotations

import collections
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

from project.config import ModelConfig
from project.data import PreprocessedAudioDataset
from project.models.audio_model import AudioClassificationModel


def extract_features(
    model: AudioClassificationModel,
    split: str,
    data_dir: Path,
    handcrafted_csv: Path,
    device: str,
    handcrafted_mean: np.ndarray | None = None,
    handcrafted_std: np.ndarray | None = None,
) -> dict:
    dataset = PreprocessedAudioDataset(data_dir / f"{split}.jsonl", handcrafted_csv)
    if handcrafted_mean is not None and handcrafted_std is not None:
        dataset.set_handcrafted_normalization(handcrafted_mean, handcrafted_std)

    is_msd = "MSD" in getattr(model.config, "musicnn_model", "")
    cache_prefix = "msd_cache" if is_msd else "musicnn_cache"
    cache_path = data_dir / f"{cache_prefix}_{split}.pt"
    cached = torch.load(cache_path, weights_only=False)
    musicnn_feats = cached["features"]

    all_mu = []
    all_fused = []
    all_neural_probs = []
    y_true = []
    rec_ids = []

    batch_size = 64
    total = len(dataset)

    with torch.no_grad():
        for i in range(0, total, batch_size):
            b_feats = musicnn_feats[i : i + batch_size].to(device)
            records = dataset.records[i : i + batch_size]
            b_hc = torch.stack(
                [torch.from_numpy(dataset._handcrafted_for_record(r)) for r in records]
            ).to(device)

            out = model(b_feats, b_hc)
            probs = torch.softmax(out["logits"], dim=-1).cpu().numpy()
            mu = out["mu"].cpu().numpy()
            hc = b_hc.cpu().numpy()
            fused = np.concatenate([mu, hc], axis=1)

            all_neural_probs.append(probs)
            all_mu.append(mu)
            all_fused.append(fused)
            y_true.extend([r["era_label"] for r in records])
            rec_ids.extend([r["recording_id"] for r in records])

    return {
        "mu": np.concatenate(all_mu, axis=0),
        "fused": np.concatenate(all_fused, axis=0),
        "neural_probs": np.concatenate(all_neural_probs, axis=0),
        "y": np.array(y_true),
        "rec_ids": np.array(rec_ids),
    }


def evaluate_song_level(probs: np.ndarray, classes: list | np.ndarray, y_true: np.ndarray, rec_ids: np.ndarray) -> float:
    song_probs = collections.defaultdict(list)
    song_true = {}
    for p, t, r in zip(probs, y_true, rec_ids):
        song_probs[r].append(p)
        song_true[r] = t

    correct = sum(
        classes[np.argmax(np.mean(plist, axis=0))] == song_true[r]
        for r, plist in song_probs.items()
    )
    return correct / len(song_true)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate SVM and Blending Ensemble")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint (.pt). Defaults to newest checkpoint.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path("preprocessed")
    handcrafted_csv = Path("Data/features_3_sec_generated.csv")
    if not handcrafted_csv.is_file():
        handcrafted_csv = Path("Data/features_3_sec.csv")

    ckpt_candidates = [Path("audio_classifier_msd.pt"), Path("audio_classifier_fused_test.pt")]
    existing_ckpts = [p for p in ckpt_candidates if p.is_file()]
    existing_ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    ckpt_path = args.checkpoint or (existing_ckpts[0] if existing_ckpts else Path("audio_classifier_fused_test.pt"))

    print(f"Loading checkpoint: {ckpt_path}")
    print(f"Handcrafted CSV:   {handcrafted_csv}")
    print(f"Device:            {device}")

    # 1. Load model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ModelConfig(**ckpt["config"])
    labels = ckpt["labels"]
    num_classes = len(labels)

    # 2. Handcrafted stats from train set
    train_ds_raw = PreprocessedAudioDataset(data_dir / "train.jsonl", handcrafted_csv)
    train_hc_mat = train_ds_raw.handcrafted_matrix()
    hc_mean = train_hc_mat.mean(axis=0)
    hc_std = train_hc_mat.std(axis=0)

    model = AudioClassificationModel(
        num_classes,
        config,
        feature_extractor=None,
        handcrafted_feature_dim=train_hc_mat.shape[1],
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # 3. Extract train, validation, and test representations
    print("\nExtracting representations...")
    train_data = extract_features(model, "train", data_dir, handcrafted_csv, device, hc_mean, hc_std)
    val_data = extract_features(model, "validation", data_dir, handcrafted_csv, device, hc_mean, hc_std)
    test_data = extract_features(model, "test", data_dir, handcrafted_csv, device, hc_mean, hc_std)

    y_train = train_data["y"]
    y_test = test_data["y"]
    test_rec_ids = test_data["rec_ids"]

    # -------------------------------------------------------------
    # 1. Neural Head Baseline
    # -------------------------------------------------------------
    neural_probs_test = test_data["neural_probs"]
    neural_preds_test = [labels[idx] for idx in np.argmax(neural_probs_test, axis=1)]
    neural_seg_acc = accuracy_score(y_test, neural_preds_test)
    neural_song_acc = evaluate_song_level(neural_probs_test, labels, y_test, test_rec_ids)

    # -------------------------------------------------------------
    # 2. SVM (RBF) on Pure VAE Latent (mu)
    # -------------------------------------------------------------
    print("\nTraining SVM (RBF) on 64-D Pure VAE Latent (mu)...")
    scaler_mu = StandardScaler()
    X_train_mu = scaler_mu.fit_transform(train_data["mu"])
    X_test_mu = scaler_mu.transform(test_data["mu"])

    # Fast validation search for C
    best_c_mu = 1.0
    best_val_mu = 0.0
    X_val_mu = scaler_mu.transform(val_data["mu"])
    for c_cand in [0.5, 1.0, 2.0, 5.0, 10.0]:
        clf_cand = SVC(kernel="rbf", C=c_cand, gamma="scale", random_state=42)
        clf_cand.fit(X_train_mu, y_train)
        score = clf_cand.score(X_val_mu, val_data["y"])
        if score > best_val_mu:
            best_val_mu = score
            best_c_mu = c_cand

    svm_mu = SVC(kernel="rbf", C=best_c_mu, gamma="scale", probability=True, random_state=42)
    svm_mu.fit(X_train_mu, y_train)
    svm_mu_probs = svm_mu.predict_proba(X_test_mu)
    svm_mu_preds = svm_mu.predict(X_test_mu)
    svm_mu_seg_acc = accuracy_score(y_test, svm_mu_preds)
    svm_mu_song_acc = evaluate_song_level(svm_mu_probs, svm_mu.classes_, y_test, test_rec_ids)
    print(f"  Best C={best_c_mu} (val_acc={best_val_mu:.3f}) -> Test Seg: {svm_mu_seg_acc*100:.2f}%, Song: {svm_mu_song_acc*100:.2f}%")

    # -------------------------------------------------------------
    # 3. SVM (RBF) on Fused (mu + Handcrafted)
    # -------------------------------------------------------------
    print("\nTraining SVM (RBF) on 121-D Fused (mu + Handcrafted)...")
    scaler_fused = StandardScaler()
    X_train_fused = scaler_fused.fit_transform(train_data["fused"])
    X_test_fused = scaler_fused.transform(test_data["fused"])
    X_val_fused = scaler_fused.transform(val_data["fused"])

    best_c_fused = 2.0
    best_val_fused = 0.0
    for c_cand in [0.5, 1.0, 2.0, 5.0, 10.0]:
        clf_cand = SVC(kernel="rbf", C=c_cand, gamma="scale", random_state=42)
        clf_cand.fit(X_train_fused, y_train)
        score = clf_cand.score(X_val_fused, val_data["y"])
        if score > best_val_fused:
            best_val_fused = score
            best_c_fused = c_cand

    svm_fused = SVC(kernel="rbf", C=best_c_fused, gamma="scale", probability=True, random_state=42)
    svm_fused.fit(X_train_fused, y_train)
    svm_fused_probs = svm_fused.predict_proba(X_test_fused)
    svm_fused_preds = svm_fused.predict(X_test_fused)
    svm_fused_seg_acc = accuracy_score(y_test, svm_fused_preds)
    svm_fused_song_acc = evaluate_song_level(svm_fused_probs, svm_fused.classes_, y_test, test_rec_ids)
    print(f"  Best C={best_c_fused} (val_acc={best_val_fused:.3f}) -> Test Seg: {svm_fused_seg_acc*100:.2f}%, Song: {svm_fused_song_acc*100:.2f}%")

    # -------------------------------------------------------------
    # 4. Soft Voting / Blending Ensemble
    # -------------------------------------------------------------
    # Ensure classes alignment between Neural Head and SVM
    svm_classes = list(svm_fused.classes_)
    # Map neural probs columns to match svm_classes order
    neural_class_indices = [labels.index(c) for c in svm_classes]
    neural_probs_aligned = neural_probs_test[:, neural_class_indices]

    # Evaluate multiple blend ratios on validation to find best alpha
    val_svm_fused_probs = svm_fused.predict_proba(X_val_fused)
    val_neural_probs_aligned = val_data["neural_probs"][:, neural_class_indices]

    best_w = 0.5
    best_blend_val_acc = 0.0
    for w in np.linspace(0.1, 0.9, 9):
        blended_val = w * val_neural_probs_aligned + (1 - w) * val_svm_fused_probs
        pred_val = [svm_classes[idx] for idx in np.argmax(blended_val, axis=1)]
        acc = accuracy_score(val_data["y"], pred_val)
        if acc > best_blend_val_acc:
            best_blend_val_acc = acc
            best_w = w

    # Apply best weight to test set
    ensemble_probs = best_w * neural_probs_aligned + (1 - best_w) * svm_fused_probs
    ensemble_preds = [svm_classes[idx] for idx in np.argmax(ensemble_probs, axis=1)]
    ensemble_seg_acc = accuracy_score(y_test, ensemble_preds)
    ensemble_song_acc = evaluate_song_level(ensemble_probs, svm_classes, y_test, test_rec_ids)

    # 50/50 Simple average for comparison
    avg_probs = 0.5 * neural_probs_aligned + 0.5 * svm_fused_probs
    avg_preds = [svm_classes[idx] for idx in np.argmax(avg_probs, axis=1)]
    avg_seg_acc = accuracy_score(y_test, avg_preds)
    avg_song_acc = evaluate_song_level(avg_probs, svm_classes, y_test, test_rec_ids)

    # -------------------------------------------------------------
    # Benchmark Summary Table
    # -------------------------------------------------------------
    print("\n" + "=" * 76)
    print("HEAD-TO-HEAD BENCHMARK: NEURAL HEAD vs SVM (RBF) vs BLENDING ENSEMBLE")
    print("=" * 76)
    print(f"{'Model / Architecture':<42} | {'Segment Acc':<13} | {'Song-Level (Voting)':<20}")
    print("-" * 76)
    print(f"{'1. Neural Head (MLP)':<42} | {neural_seg_acc * 100:6.2f}%      | {neural_song_acc * 100:6.2f}%")
    print(f"{'2. SVM (RBF) on VAE Latent (mu)':<42} | {svm_mu_seg_acc * 100:6.2f}%      | {svm_mu_song_acc * 100:6.2f}%")
    print(f"{'3. SVM (RBF) on Fused (mu + HC)':<42} | {svm_fused_seg_acc * 100:6.2f}%      | {svm_fused_song_acc * 100:6.2f}%")
    print(f"{'4. Soft Blending (50% Neural + 50% SVM)':<42} | {avg_seg_acc * 100:6.2f}%      | {avg_song_acc * 100:6.2f}%")
    print(f"{f'5. Tuned Blend ({best_w*100:.0f}% Neural + {(1-best_w)*100:.0f}% SVM)':<42} | {ensemble_seg_acc * 100:6.2f}%      | {ensemble_song_acc * 100:6.2f}%")
    print("=" * 76)

    # Save trained models
    joblib.dump(
        {"model": svm_fused, "scaler": scaler_fused, "classes": svm_classes, "test_acc": svm_fused_seg_acc},
        "svm_fused_classifier.pkl",
    )
    print("\nSaved trained SVM model to 'svm_fused_classifier.pkl'")


if __name__ == "__main__":
    main()
