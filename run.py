"""Prepare a recording-aware audio dataset.

Run from the workspace root:
    python run.py

The default input is ``Data/genres_original``. For real era data, provide a
CSV with columns ``recording_id,era_label`` and optionally ``wav_path,midi_path``:
    python run.py --input-dir Data/audio --metadata Data/metadata.csv

The script never modifies source WAV files. It creates ``preprocessed/`` with
one manifest per split and NumPy arrays containing four padded 3-second windows
for every 12-second segment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Recording:
    recording_id: str
    wav_path: Path
    midi_path: str | None
    era_label: str


def recording_id_for(path: Path, root: Path) -> str:
    return path.relative_to(root).with_suffix("").as_posix().replace("/", "__")


def read_metadata(path: Path, input_dir: Path) -> dict[str, dict[str, str | None]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        required = {"recording_id", "era_label"}
        missing = required - set(rows.fieldnames or [])
        if missing:
            raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
        metadata = {}
        for row in rows:
            recording_id = (row.get("recording_id") or "").strip()
            if not recording_id:
                raise ValueError("Every metadata row needs a recording_id")
            wav_path = row.get("wav_path") or None
            midi_path = row.get("midi_path") or None
            metadata[recording_id] = {
                "wav_path": str((input_dir / wav_path).resolve()) if wav_path else None,
                "midi_path": midi_path,
                "era_label": (row.get("era_label") or "").strip(),
            }
        return metadata


def collect_recordings(input_dir: Path, metadata_path: Path | None, use_folder_labels: bool) -> list[Recording]:
    metadata = read_metadata(metadata_path, input_dir) if metadata_path else {}
    wav_files = sorted(input_dir.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found under {input_dir}")

    recordings = []
    seen_ids = set()
    for wav_path in wav_files:
        default_id = recording_id_for(wav_path, input_dir)
        row = metadata.get(default_id, {})
        recording_id = default_id
        era_label = row.get("era_label") or (wav_path.parent.name if use_folder_labels else "")
        if not era_label:
            raise ValueError(
                f"No era_label for {default_id}. Provide --metadata or explicitly use --folder-labels."
            )
        if recording_id in seen_ids:
            raise ValueError(f"Duplicate recording_id: {recording_id}")
        seen_ids.add(recording_id)
        recordings.append(
            Recording(
                recording_id=recording_id,
                wav_path=wav_path,
                midi_path=row.get("midi_path"),
                era_label=era_label,
            )
        )
    unknown = set(metadata) - seen_ids
    if unknown:
        raise ValueError(f"Metadata references WAVs not found in input: {sorted(unknown)[:5]}")
    return recordings


def split_recordings(recordings: list[Recording], seed: int, train_ratio: float, val_ratio: float) -> dict[str, list[Recording]]:
    """Stratified recording-level split.

    Recordings are grouped by ``era_label`` before splitting so that every
    split receives a proportional share of each class.  Within each class the
    order is deterministic given ``seed``.
    """
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("train and validation ratios must be valid and leave room for test")

    # Group recordings by label (sorted for determinism)
    groups: dict[str, list[Recording]] = {}
    for recording in sorted(recordings, key=lambda r: r.recording_id):
        groups.setdefault(recording.era_label, []).append(recording)

    rng = np.random.default_rng(seed)
    splits: dict[str, list[Recording]] = {"train": [], "validation": [], "test": []}

    for label in sorted(groups):
        group = groups[label]
        order = rng.permutation(len(group))
        shuffled = [group[i] for i in order]

        n = len(shuffled)
        train_end = math.floor(n * train_ratio)
        val_end = train_end + math.floor(n * val_ratio)

        # Guarantee at least one recording per split when the group is large enough
        if n >= 3:
            train_end = max(1, min(train_end, n - 2))
            val_end = max(train_end + 1, min(val_end, n - 1))

        splits["train"].extend(shuffled[:train_end])
        splits["validation"].extend(shuffled[train_end:val_end])
        splits["test"].extend(shuffled[val_end:])

    return splits


def load_wav(path: Path, target_rate: int) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        source_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width ({sample_width} bytes): {path}")
    audio = audio.reshape(-1, channels).mean(axis=1)
    if source_rate != target_rate and len(audio):
        target_length = round(len(audio) * target_rate / source_rate)
        source_positions = np.linspace(0, len(audio) - 1, num=len(audio))
        target_positions = np.linspace(0, len(audio) - 1, num=target_length)
        audio = np.interp(target_positions, source_positions, audio).astype(np.float32)
    return audio


def write_split(
    split: str,
    recordings: list[Recording],
    output_dir: Path,
    sample_rate: int,
    segment_seconds: int,
    skip_invalid: bool,
    skipped: list[str],
) -> int:
    segment_samples = sample_rate * segment_seconds
    window_samples = sample_rate * 3
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{split}.jsonl"
    count = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for recording in recordings:
            try:
                audio = load_wav(recording.wav_path, sample_rate)
            except (OSError, ValueError, wave.Error):
                if not skip_invalid:
                    raise
                skipped.append(str(recording.wav_path))
                continue
            segment_count = max(1, math.ceil(len(audio) / segment_samples))
            for segment_index in range(segment_count):
                start = segment_index * segment_samples
                segment = audio[start : start + segment_samples]
                segment = np.pad(segment, (0, max(0, segment_samples - len(segment))))
                windows = segment.reshape(4, window_samples).astype(np.float32)
                filename = f"{recording.recording_id}__segment_{segment_index:03d}.npy"
                np.save(split_dir / filename, windows)
                manifest.write(json.dumps({
                    "recording_id": recording.recording_id,
                    "segment_id": f"segment_{segment_index:03d}",
                    "wav_path": str(recording.wav_path),
                    "midi_path": recording.midi_path,
                    "era_label": recording.era_label,
                    "audio_npy": str((split_dir / filename).relative_to(output_dir)),
                    "shape": list(windows.shape),
                }) + "\n")
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("Data/genres_original"))
    parser.add_argument("--metadata", type=Path, help="CSV containing recording_id and era_label")
    parser.add_argument("--output-dir", type=Path, default=Path("preprocessed"))
    parser.add_argument("--folder-labels", action="store_true", help="Use WAV parent folders as labels (useful for GTZAN smoke tests)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--segment-seconds", type=int, default=12)
    parser.add_argument("--skip-invalid", action="store_true", help="Skip unreadable WAV files and list them in summary.json")
    args = parser.parse_args()
    is_default_gtzan = args.input_dir == Path("Data/genres_original") and args.metadata is None
    use_folder_labels = args.folder_labels or is_default_gtzan
    skip_invalid = args.skip_invalid or is_default_gtzan
    if is_default_gtzan:
        print("No metadata supplied; using GTZAN folder names as labels.")
    recordings = collect_recordings(args.input_dir, args.metadata, use_folder_labels)
    splits = split_recordings(recordings, args.seed, args.train_ratio, args.val_ratio)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    skipped = []
    summary = {"seed": args.seed, "recordings": len(recordings), "segments": {}, "skipped_invalid": skipped}
    for split, split_recordings_list in splits.items():
        summary["segments"][split] = write_split(
            split,
            split_recordings_list,
            args.output_dir,
            args.sample_rate,
            args.segment_seconds,
            skip_invalid,
            skipped,
        )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(recordings)} recordings into {args.output_dir}")
    print("; ".join(f"{split}: {len(splits[split])} recordings, {summary['segments'][split]} segments" for split in splits))
    if skipped:
        print(f"Skipped invalid WAV files: {len(skipped)} (see {args.output_dir / 'summary.json'})")


if __name__ == "__main__":
    main()