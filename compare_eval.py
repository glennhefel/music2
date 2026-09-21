"""Head-to-head comparison on the untouched Test Split (150 songs, 450 clips).

Compares:
  1. Neural End-to-End Linear Head (nn.Linear on latent space)
  2. Random Forest on Pure VAE Latent (rf_latent_classifier.pkl)
  3. Random Forest on Fused Latent (rf_fused_classifier.pkl)
"""

import collections
from pathlib import Path
import joblib
import numpy as np
import torch

from project.config import ModelConfig
from project.data import PreprocessedAudioDataset
from project.models.audio_model import AudioClassificationModel


def run_comparison():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path("preprocessed")
    handcrafted_csv = Path("Data/features_3_sec.csv")
    ckpt_path = Path("audio_classifier_fused_test.pt")

    # 1. Load trained neural checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ModelConfig(**ckpt["config"])
    labels = ckpt["labels"]
    label_to_idx = {l: i for i, l in enumerate(labels)}

    # 2. Load dataset & normalize handcrafted features
    train_ds = PreprocessedAudioDataset(data_dir / "train.jsonl", handcrafted_csv)
    train_mat = train_ds.handcrafted_matrix()
    mean = train_mat.mean(axis=0)
    std = train_mat.std(axis=0)

    test_ds = PreprocessedAudioDataset(data_dir / "test.jsonl", handcrafted_csv)
    test_ds.set_handcrafted_normalization(mean, std)

    model = AudioClassificationModel(len(labels), config, feature_extractor=None, handcrafted_feature_dim=train_mat.shape[1])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # 3. Load cached MusicNN features for test
    cached_test = torch.load(data_dir / "musicnn_cache_test.pt", weights_only=False)
    test_musicnn = cached_test["features"]  # [450, 4, 200]

    # 4. Extract neural outputs and representations
    neural_probs = []
    all_mu = []
    all_fused = []
    y_true = []
    rec_ids = []

    with torch.no_grad():
        for i, record in enumerate(test_ds.records):
            feats = test_musicnn[i : i + 1].to(device)
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

    # 5. Evaluate Neural Head
    neural_preds = [labels[idx] for idx in np.argmax(neural_probs, axis=1)]
    neural_seg_acc = np.mean([p == t for p, t in zip(neural_preds, y_true)])

    # Neural Song-Level
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

    # 6. Evaluate RF on Latent (rf_latent_classifier.pkl)
    rf_latent = joblib.load("rf_latent_classifier.pkl")
    rf_latent_clf = rf_latent["model"]
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

    # 7. Evaluate RF on Fused (rf_fused_classifier.pkl)
    rf_fused = joblib.load("rf_fused_classifier.pkl")
    rf_fused_clf = rf_fused["model"]
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
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD COMPARISON ON UNSEEN TEST SPLIT (150 songs, 450 clips)")
    print("=" * 70)
    print(f"{'Classifier Head':<35} | {'Segment Accuracy':<16} | {'Song-Level (Voting)':<20}")
    print("-" * 70)
    print(f"{'1. Neural Linear Head (nn.Linear)':<35} | {neural_seg_acc * 100:6.2f}%          | {neural_song_acc * 100:6.2f}%")
    print(f"{'2. Random Forest on VAE Latent (mu)':<35} | {rf_latent_seg_acc * 100:6.2f}%          | {rf_lat_song_acc * 100:6.2f}%")
    print(f"{'3. Random Forest on Fused (mu + HC)':<35} | {rf_fused_seg_acc * 100:6.2f}%          | {rf_fused_song_acc * 100:6.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    run_comparison()
