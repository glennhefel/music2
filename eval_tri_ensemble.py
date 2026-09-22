"""Tri-Model Ensemble: Neural Head + Kernel SVM (RBF) + Random Forest.

Evaluates on the untouched test split (150 songs, 450 clips):
  1. Neural Head (MLP)
  2. Random Forest on Fused (mu + Handcrafted)
  3. SVM (RBF Kernel) on Fused (mu + Handcrafted)
  4. Tri-Model Soft Voting (Equal Weights: 33% / 33% / 33%)
  5. Optimal Tuned Soft Blend (Validation Grid Search over w1, w2, w3)

Computes both Segment-Level Accuracy and Song-Level Majority-Voting Accuracy.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from project.config import ModelConfig
from project.data import PreprocessedAudioDataset
from project.models.audio_model import AudioClassificationModel


def evaluate_song_level(
    probs: np.ndarray, classes: list[str], y_true: list[str], rec_ids: list[str]
) -> float:
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


def extract_features(
    model: AudioClassificationModel,
    split: str,
    data_dir: Path,
    handcrafted_csv: Path,
    device: str,
    cache_prefix: str = "musicnn_cache",
    handcrafted_mean: np.ndarray | None = None,
    handcrafted_std: np.ndarray | None = None,
) -> dict:
    dataset = PreprocessedAudioDataset(data_dir / f"{split}.jsonl", handcrafted_csv)
    if handcrafted_mean is not None and handcrafted_std is not None:
        dataset.set_handcrafted_normalization(handcrafted_mean, handcrafted_std)

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
        "rec_ids": rec_ids,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Tri-Model Ensemble")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint (.pt). Defaults to newest checkpoint.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("preprocessed"))
    parser.add_argument("--handcrafted-csv", type=Path, default=Path("Data/features_3_sec.csv"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt_candidates = [Path("audio_classifier_msd.pt"), Path("audio_classifier_fused_test.pt")]
    existing = [p for p in ckpt_candidates if p.is_file()]
    existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    ckpt_path = args.checkpoint or (existing[0] if existing else Path("audio_classifier_fused_test.pt"))

    print(f"Loading checkpoint: {ckpt_path.name} (device={args.device})")
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    config = ModelConfig(**ckpt["config"])
    labels = ckpt["labels"]
    num_classes = len(labels)
    is_msd = "MSD" in config.musicnn_model or "msd" in ckpt_path.stem.lower()

    # Handcrafted feature normalization stats from train set
    train_raw = PreprocessedAudioDataset(args.data_dir / "train.jsonl", args.handcrafted_csv)
    train_hc_mat = train_raw.handcrafted_matrix()
    hc_mean = train_hc_mat.mean(axis=0)
    hc_std = train_hc_mat.std(axis=0)

    model = AudioClassificationModel(
        num_classes,
        config,
        feature_extractor=None,
        handcrafted_feature_dim=train_hc_mat.shape[1],
    )
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    cache_prefix = "msd_cache" if is_msd else "musicnn_cache"

    # 1. Extract representations for Train, Val, Test
    print(f"Extracting representations across splits (using {cache_prefix}_*.pt)...")
    train_data = extract_features(model, "train", args.data_dir, args.handcrafted_csv, args.device, cache_prefix, hc_mean, hc_std)
    val_data = extract_features(model, "validation", args.data_dir, args.handcrafted_csv, args.device, cache_prefix, hc_mean, hc_std)
    test_data = extract_features(model, "test", args.data_dir, args.handcrafted_csv, args.device, cache_prefix, hc_mean, hc_std)

    y_train = train_data["y"]
    y_val = val_data["y"]
    y_test = test_data["y"]
    test_rec_ids = test_data["rec_ids"]

    # Canonical class order
    class_order = labels
    class_to_idx = {c: i for i, c in enumerate(class_order)}

    def align_probs(probs: np.ndarray, model_classes: list[str]) -> np.ndarray:
        indices = [list(model_classes).index(c) for c in class_order]
        return probs[:, indices]

    # Model 1: Neural Head
    p_val_neural = align_probs(val_data["neural_probs"], labels)
    p_test_neural = align_probs(test_data["neural_probs"], labels)

    neural_seg_acc = accuracy_score(y_test, [class_order[i] for i in np.argmax(p_test_neural, axis=1)])
    neural_song_acc = evaluate_song_level(p_test_neural, class_order, y_test, test_rec_ids)

    # Model 2: Random Forest on Fused
    rf_suffix = "_msd" if is_msd else ""
    rf_fused_path = Path(f"rf_fused_classifier{rf_suffix}.pkl")
    if rf_fused_path.is_file() and rf_fused_path.stat().st_mtime >= ckpt_path.stat().st_mtime:
        print(f"Loading existing RF: {rf_fused_path.name}")
        rf = joblib.load(rf_fused_path)["model"]
    else:
        print("Training Random Forest on Fused representations...")
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=16,
            criterion="entropy",
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        rf.fit(train_data["fused"], y_train)
        joblib.dump({"model": rf, "classes": rf.classes_}, rf_fused_path)

    p_val_rf = align_probs(rf.predict_proba(val_data["fused"]), rf.classes_)
    p_test_rf = align_probs(rf.predict_proba(test_data["fused"]), rf.classes_)

    rf_seg_acc = accuracy_score(y_test, [class_order[i] for i in np.argmax(p_test_rf, axis=1)])
    rf_song_acc = evaluate_song_level(p_test_rf, class_order, y_test, test_rec_ids)

    # Model 3: Kernel SVM (RBF) on Fused (StandardScaled)
    print("Training / Tuning SVM (RBF) on Fused representations...")
    scaler = StandardScaler()
    X_train_fused_sc = scaler.fit_transform(train_data["fused"])
    X_val_fused_sc = scaler.transform(val_data["fused"])
    X_test_fused_sc = scaler.transform(test_data["fused"])

    best_c = 2.0
    best_val_score = 0.0
    for c_cand in [0.5, 1.0, 2.0, 5.0, 10.0]:
        svm_fast = SVC(kernel="rbf", C=c_cand, gamma="scale", random_state=42)
        svm_fast.fit(X_train_fused_sc, y_train)
        score = svm_fast.score(X_val_fused_sc, y_val)
        if score > best_val_score:
            best_val_score = score
            best_c = c_cand

    svm = SVC(kernel="rbf", C=best_c, gamma="scale", probability=True, random_state=42)
    svm.fit(X_train_fused_sc, y_train)

    p_val_svm = align_probs(svm.predict_proba(X_val_fused_sc), svm.classes_)
    p_test_svm = align_probs(svm.predict_proba(X_test_fused_sc), svm.classes_)

    svm_seg_acc = accuracy_score(y_test, [class_order[i] for i in np.argmax(p_test_svm, axis=1)])
    svm_song_acc = evaluate_song_level(p_test_svm, class_order, y_test, test_rec_ids)

    # 4. Equal Weights Tri-Model Ensemble (1/3, 1/3, 1/3)
    p_test_equal = (p_test_neural + p_test_svm + p_test_rf) / 3.0
    equal_seg_acc = accuracy_score(y_test, [class_order[i] for i in np.argmax(p_test_equal, axis=1)])
    equal_song_acc = evaluate_song_level(p_test_equal, class_order, y_test, test_rec_ids)

    # 5. Grid Search for Optimal Blend Weights on Validation Split
    print("Searching for optimal blend weights on validation split...")
    best_weights = (1/3, 1/3, 1/3)
    best_blend_val_acc = 0.0

    steps = 21  # step size 0.05
    for i in range(steps):
        w_neural = i / (steps - 1)
        remaining = 1.0 - w_neural
        for j in range(steps):
            w_svm = (j / (steps - 1)) * remaining
            w_rf = remaining - w_svm
            if w_rf < -1e-6:
                continue

            p_val_blend = w_neural * p_val_neural + w_svm * p_val_svm + w_rf * p_val_rf
            preds = [class_order[idx] for idx in np.argmax(p_val_blend, axis=1)]
            val_acc = accuracy_score(y_val, preds)

            if val_acc > best_blend_val_acc:
                best_blend_val_acc = val_acc
                best_weights = (w_neural, w_svm, w_rf)

    wn, ws, wr = best_weights
    p_test_tuned = wn * p_test_neural + ws * p_test_svm + wr * p_test_rf
    tuned_seg_acc = accuracy_score(y_test, [class_order[i] for i in np.argmax(p_test_tuned, axis=1)])
    tuned_song_acc = evaluate_song_level(p_test_tuned, class_order, y_test, test_rec_ids)

    # Also evaluate pairwise ensembles for completeness
    p_test_neural_svm = 0.5 * p_test_neural + 0.5 * p_test_svm
    pair_ns_seg = accuracy_score(y_test, [class_order[i] for i in np.argmax(p_test_neural_svm, axis=1)])
    pair_ns_song = evaluate_song_level(p_test_neural_svm, class_order, y_test, test_rec_ids)

    p_test_rf_svm = 0.5 * p_test_rf + 0.5 * p_test_svm
    pair_rs_seg = accuracy_score(y_test, [class_order[i] for i in np.argmax(p_test_rf_svm, axis=1)])
    pair_rs_song = evaluate_song_level(p_test_rf_svm, class_order, y_test, test_rec_ids)

    # Summary Benchmark Table
    print("\n" + "=" * 80)
    print(f"TRI-MODEL ENSEMBLE BENCHMARK ON UNSEEN TEST SPLIT (150 songs, 450 clips)")
    print(f"Backbone: {config.musicnn_model} ({ckpt_path.name})")
    print("=" * 80)
    print(f"{'Model / Architecture':<46} | {'Segment Acc':<13} | {'Song-Level (Voting)':<18}")
    print("-" * 80)
    print(f"{'1. Neural Head (End-to-End MLP)':<46} | {neural_seg_acc * 100:6.2f}%      | {neural_song_acc * 100:6.2f}%")
    print(f"{'2. Random Forest on Fused (mu + HC)':<46} | {rf_seg_acc * 100:6.2f}%      | {rf_song_acc * 100:6.2f}%")
    print(f"{f'3. SVM RBF on Fused (C={best_c})':<46} | {svm_seg_acc * 100:6.2f}%      | {svm_song_acc * 100:6.2f}%")
    print("-" * 80)
    print(f"{'4. Neural + SVM (50% / 50%)':<46} | {pair_ns_seg * 100:6.2f}%      | {pair_ns_song * 100:6.2f}%")
    print(f"{'5. RF + SVM (50% / 50%)':<46} | {pair_rs_seg * 100:6.2f}%      | {pair_rs_song * 100:6.2f}%")
    print(f"{'6. Equal Tri-Ensemble (33% Neural / 33% SVM / 33% RF)':<46} | {equal_seg_acc * 100:6.2f}%      | {equal_song_acc * 100:6.2f}%")
    print(f"{f'7. Tuned Tri-Ensemble ({wn*100:.0f}% N / {ws*100:.0f}% S / {wr*100:.0f}% RF)':<46} | {tuned_seg_acc * 100:6.2f}%      | {tuned_song_acc * 100:6.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
