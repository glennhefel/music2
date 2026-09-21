"""Run the audio branch on a 30-second track split into 3-second windows.

Example:
    python -m project.runmodel Data/genres_original/blues/blues.00000.wav
    python -m project.runmodel track.wav --classifier era_random_forest.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import ModelConfig
from .classifiers import RandomForestEraClassifier
from .data.audio_utils import load_and_segment_audio
from .models.audio_model import AudioRepresentationModel
from .models.musicnn_tensorflow import TensorFlowMusicNNExtractor


def run_track(
    track_path: str | Path,
    musicnn_root: str | Path,
    classifier_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    config = ModelConfig(audio_segment_seconds=30.0, musicnn_window_seconds=3.0)
    windows = load_and_segment_audio(
        track_path,
        sample_rate=config.sample_rate,
        segment_seconds=config.audio_segment_seconds,
        window_seconds=config.musicnn_window_seconds,
    )
    audio = torch.from_numpy(windows).unsqueeze(0)
    extractor = TensorFlowMusicNNExtractor(musicnn_root, config.musicnn_model)
    model = AudioRepresentationModel(config, feature_extractor=extractor).eval()

    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        rep_dict = {k.replace("representation.", ""): v for k, v in state_dict.items() if k.startswith("representation.")}
        if not rep_dict:
            rep_dict = state_dict
        model.load_state_dict(rep_dict, strict=False)
        print(f"Loaded audio branch checkpoint from: {checkpoint_path}")

    try:
        with torch.inference_mode():
            outputs = model(audio)
    finally:
        extractor.close()

    print(f"track: {Path(track_path)}")
    print(f"audio windows: {tuple(audio.shape)}  (batch, 3-second windows, samples)")
    print(f"musicnn features: (1, {audio.shape[1]}, {config.musicnn_feature_dim})")
    for name, value in outputs.items():
        print(f"{name}: {tuple(value.shape)}")
    print(f"transformer embedding mean: {outputs['transformer_embedding'].mean().item():.6f}")
    print(f"latent z mean: {outputs['z'].mean().item():.6f}")

    if classifier_path is not None:
        classifier = RandomForestEraClassifier.load(classifier_path)
        features = outputs[classifier.feature_name]
        expected_features = getattr(classifier.model, "n_features_in_", None)
        if expected_features is not None and features.shape[1] != expected_features:
            raise ValueError(
                f"Classifier expects {expected_features} features, but "
                f"{classifier.feature_name!r} has {features.shape[1]}. "
                "Use a classifier trained on this audio model's representations."
            )
        prediction = classifier.predict(features)[0]
        probabilities = classifier.predict_proba(features)[0]
        confidence = float(probabilities.max())
        print(f"classification: {prediction}")
        print(f"classification confidence: {confidence:.4f}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track", type=Path, help="Path to a WAV/audio track")
    parser.add_argument(
        "--musicnn-root",
        type=Path,
        default=Path("musicnn"),
        help="Directory containing the official MTT_musicnn checkpoint",
    )
    parser.add_argument(
        "--classifier",
        type=Path,
        default=None,
        help="Joblib classifier trained on the selected audio-model representation",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("audio_classifier_fused_test.pt") if Path("audio_classifier_fused_test.pt").is_file() else None,
        help="Path to trained PyTorch audio representation or classifier checkpoint",
    )
    args = parser.parse_args()
    run_track(args.track, args.musicnn_root, args.classifier, args.checkpoint)


if __name__ == "__main__":
    main()