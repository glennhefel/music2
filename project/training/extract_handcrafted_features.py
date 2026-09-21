"""Extract GTZAN-style handcrafted features from 3-second WAV segments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import numpy as np


def summarize(row: dict[str, float], name: str, values: np.ndarray) -> None:
    row[f"{name}_mean"] = float(np.mean(values))
    row[f"{name}_var"] = float(np.var(values))


def extract_segment_features(
    audio: np.ndarray,
    sample_rate: int,
    harmony: np.ndarray | None = None,
    percussion: np.ndarray | None = None,
) -> dict[str, float]:
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    rms = librosa.feature.rms(y=audio)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
    if harmony is None or percussion is None:
        harmony, percussion = librosa.effects.hpss(audio, n_fft=512, hop_length=256)
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)

    row: dict[str, float] = {}
    summarize(row, "chroma_stft", chroma)
    summarize(row, "rms", rms)
    summarize(row, "spectral_centroid", centroid)
    summarize(row, "spectral_bandwidth", bandwidth)
    summarize(row, "rolloff", rolloff)
    summarize(row, "zero_crossing_rate", zero_crossing_rate)
    summarize(row, "harmony", harmony)
    summarize(row, "perceptr", percussion)
    row["tempo"] = float(np.asarray(tempo).reshape(-1)[0])
    for index in range(20):
        summarize(row, f"mfcc{index + 1}", mfcc[index])
    return row


def extract_csv(input_dir: Path, output_path: Path, sample_rate: int, segment_seconds: int, skip_invalid: bool) -> int:
    segment_samples = sample_rate * segment_seconds
    feature_names = [
        "filename", "length", "chroma_stft_mean", "chroma_stft_var", "rms_mean", "rms_var",
        "spectral_centroid_mean", "spectral_centroid_var", "spectral_bandwidth_mean", "spectral_bandwidth_var",
        "rolloff_mean", "rolloff_var", "zero_crossing_rate_mean", "zero_crossing_rate_var",
        "harmony_mean", "harmony_var", "perceptr_mean", "perceptr_var", "tempo",
    ] + [f"mfcc{index}_{stat}" for index in range(1, 21) for stat in ("mean", "var")] + ["label"]
    count = 0
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=feature_names)
        writer.writeheader()
        for wav_path in sorted(input_dir.rglob("*.wav")):
            try:
                audio, source_rate = librosa.load(wav_path, sr=sample_rate, mono=True)
            except Exception:
                if not skip_invalid:
                    raise
                continue
            try:
                harmony, percussion = librosa.effects.hpss(audio, n_fft=512, hop_length=256)
            except Exception:
                if not skip_invalid:
                    raise
                continue
            segment_count = max(1, len(audio) // segment_samples)
            for segment_index in range(segment_count):
                start = segment_index * segment_samples
                segment = audio[start : start + segment_samples]
                if len(segment) < segment_samples:
                    continue
                values = extract_segment_features(
                    segment,
                    sample_rate,
                    harmony[start : start + segment_samples],
                    percussion[start : start + segment_samples],
                )
                values = {"filename": f"{wav_path.stem}.{segment_index}.wav", "length": len(audio), **values, "label": wav_path.parent.name}
                writer.writerow(values)
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("Data/genres_original"))
    parser.add_argument("--output", type=Path, default=Path("Data/features_3_sec_generated.csv"))
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--segment-seconds", type=int, default=3)
    parser.add_argument("--skip-invalid", action="store_true")
    args = parser.parse_args()
    count = extract_csv(args.input_dir, args.output, args.sample_rate, args.segment_seconds, args.skip_invalid)
    print(f"wrote={args.output} rows={count}")


if __name__ == "__main__":
    main()
