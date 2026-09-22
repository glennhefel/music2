"""Extract GTZAN-style handcrafted features from audio using Librosa with multiprocessing."""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np


FEATURE_NAMES = [
    "filename", "length",
    "chroma_stft_mean", "chroma_stft_var",
    "rms_mean", "rms_var",
    "spectral_centroid_mean", "spectral_centroid_var",
    "spectral_bandwidth_mean", "spectral_bandwidth_var",
    "rolloff_mean", "rolloff_var",
    "zero_crossing_rate_mean", "zero_crossing_rate_var",
    "harmony_mean", "harmony_var",
    "perceptr_mean", "perceptr_var",
    "tempo",
] + [f"mfcc{index}_{stat}" for index in range(1, 21) for stat in ("mean", "var")] + ["label"]


def summarize(row: dict[str, float], name: str, values: np.ndarray) -> None:
    row[f"{name}_mean"] = float(np.mean(values))
    row[f"{name}_var"] = float(np.var(values))


def extract_segment_features(
    segment: np.ndarray,
    sample_rate: int,
    harmony: np.ndarray | None = None,
    percussion: np.ndarray | None = None,
) -> dict[str, float]:
    chroma = librosa.feature.chroma_stft(y=segment, sr=sample_rate)
    rms = librosa.feature.rms(y=segment)
    centroid = librosa.feature.spectral_centroid(y=segment, sr=sample_rate)
    bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sample_rate)
    rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sample_rate)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(segment)
    if harmony is None or percussion is None:
        harmony, percussion = librosa.effects.hpss(segment)
    tempo, _ = librosa.beat.beat_track(y=segment, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=20)

    row: dict[str, float] = {}
    summarize(row, "chroma_stft", chroma)
    summarize(row, "rms", rms)
    summarize(row, "spectral_centroid", centroid)
    summarize(row, "spectral_bandwidth", bandwidth)
    summarize(row, "rolloff", rolloff)
    summarize(row, "zero_crossing_rate", zero_crossing_rate)
    summarize(row, "harmony", harmony)
    summarize(row, "perceptr", percussion)
    tempo_val = np.asarray(tempo).reshape(-1)[0] if np.asarray(tempo).size > 0 else 0.0
    row["tempo"] = float(tempo_val)
    for index in range(20):
        summarize(row, f"mfcc{index + 1}", mfcc[index])
    return row


def process_single_audio(
    wav_path_str: str,
    sample_rate: int,
    segment_seconds: int,
    label: str,
) -> list[dict]:
    wav_path = Path(wav_path_str)
    try:
        audio, _ = librosa.load(wav_path, sr=sample_rate, mono=True)
    except Exception as exc:
        print(f"Warning: Failed to load {wav_path}: {exc}")
        return []

    segment_samples = int(sample_rate * segment_seconds)
    segment_count = max(1, len(audio) // segment_samples)
    
    try:
        harmony, percussion = librosa.effects.hpss(audio)
    except Exception:
        harmony, percussion = None, None

    rows = []
    for segment_index in range(segment_count):
        start = segment_index * segment_samples
        segment = audio[start : start + segment_samples]
        if len(segment) < segment_samples:
            continue
        seg_harmony = harmony[start : start + segment_samples] if harmony is not None else None
        seg_percuss = percussion[start : start + segment_samples] if percussion is not None else None
        
        values = extract_segment_features(
            segment,
            sample_rate,
            seg_harmony,
            seg_percuss,
        )
        row = {
            "filename": f"{wav_path.stem}.{segment_index}.wav",
            "length": len(segment),
            **values,
            "label": label,
        }
        rows.append(row)
    return rows


def extract_csv_parallel(
    input_dir: Path,
    output_path: Path,
    sample_rate: int = 22050,
    segment_seconds: int = 3,
    workers: int = 4,
) -> int:
    wav_files = sorted(input_dir.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {input_dir}")

    tasks = [(str(w), sample_rate, segment_seconds, w.parent.name) for w in wav_files]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Extracting features from {len(tasks)} files using {workers} workers...")
    all_rows: list[dict] = []
    completed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_single_audio, path, sr, seg, label): path
            for path, sr, seg, label in tasks
        }
        for future in as_completed(futures):
            rows = future.result()
            all_rows.extend(rows)
            completed += 1
            if completed % 100 == 0 or completed == len(tasks):
                print(f"  Progress: {completed}/{len(tasks)} files processed ({len(all_rows)} segments)")

    # Sort rows by filename for consistent ordering
    all_rows.sort(key=lambda r: r["filename"])

    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FEATURE_NAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Done! Saved {len(all_rows)} rows to {output_path}")
    return len(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("Data/genres_original"))
    parser.add_argument("--output", type=Path, default=Path("Data/features_3_sec_generated.csv"))
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--segment-seconds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = parser.parse_args()

    extract_csv_parallel(
        input_dir=args.input_dir,
        output_path=args.output,
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
