import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import ModelConfig
from ..data import PreprocessedAudioDataset
from ..models.audio_model import AudioClassificationModel
from ..models.musicnn_tensorflow import TensorFlowMusicNNExtractor
from .losses import audio_vae_loss


def read_labels(*manifest_paths: Path) -> list[str]:
    labels = set()
    for path in manifest_paths:
        with path.open(encoding="utf-8") as file:
            labels.update(json.loads(line)["era_label"] for line in file if line.strip())
    result = sorted(labels)
    if len(result) < 2:
        raise ValueError("Need at least two classes")
    return result


def run_epoch(model, loader, label_to_index, optimizer, device, beta):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_items = 0
    for batch in loader:
        audio = batch["audio"].to(device)
        handcrafted_features = batch.get("handcrafted_features")
        if handcrafted_features is not None:
            handcrafted_features = handcrafted_features.to(device)
        labels = torch.tensor([label_to_index[label] for label in batch["era_label"]], device=device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        outputs = model(audio, handcrafted_features)
        vae_total, _, _ = audio_vae_loss(outputs, beta)
        classification_loss = nn.functional.cross_entropy(outputs["logits"], labels)
        loss = classification_loss + 0.1 * vae_total
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(labels)
        total_correct += (outputs["logits"].argmax(dim=1) == labels).sum().item()
        total_items += len(labels)
    return total_loss / total_items, total_correct / total_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the audio model with supervised classification")
    parser.add_argument("--data-dir", type=Path, default=Path("preprocessed"))
    parser.add_argument("--musicnn-root", type=Path, default=Path("musicnn"))
    parser.add_argument("--handcrafted-csv", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("audio_classifier.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=ModelConfig().learning_rate)
    parser.add_argument("--beta", type=float, default=ModelConfig().vae_beta)
    parser.add_argument("--dropout", type=float, default=ModelConfig().transformer_dropout)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in the range [0.0, 1.0)")
    paths = [args.data_dir / f"{split}.jsonl" for split in ("train", "validation", "test")]
    labels = read_labels(*paths)
    label_to_index = {label: index for index, label in enumerate(labels)}
    datasets = [PreprocessedAudioDataset(path, args.handcrafted_csv) for path in paths]
    handcrafted_feature_dim = 0
    if args.handcrafted_csv is not None:
        train_features = datasets[0].handcrafted_matrix()
        feature_mean = train_features.mean(axis=0)
        feature_std = train_features.std(axis=0)
        for dataset in datasets:
            dataset.set_handcrafted_normalization(feature_mean, feature_std)
        handcrafted_feature_dim = train_features.shape[1]
    loaders = [DataLoader(dataset, batch_size=args.batch_size, shuffle=index == 0) for index, dataset in enumerate(datasets)]
    config = ModelConfig(
        learning_rate=args.learning_rate,
        vae_beta=args.beta,
        transformer_dropout=args.dropout,
        vae_dropout=args.dropout,
    )
    extractor = TensorFlowMusicNNExtractor(args.musicnn_root, config.musicnn_model)
    model = AudioClassificationModel(len(labels), config, extractor, handcrafted_feature_dim).to(args.device)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate)
    best_validation_loss = float("inf")
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = run_epoch(model, loaders[0], label_to_index, optimizer, args.device, args.beta)
            with torch.no_grad():
                validation_loss, validation_accuracy = run_epoch(model, loaders[1], label_to_index, None, args.device, args.beta)
            print(f"epoch={epoch} train_loss={train_loss:.6f} train_accuracy={train_accuracy:.4f} validation_loss={validation_loss:.6f} validation_accuracy={validation_accuracy:.4f}")
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save({"model": model.state_dict(), "labels": labels, "config": config.__dict__}, args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        with torch.no_grad():
            test_loss, test_accuracy = run_epoch(model, loaders[2], label_to_index, None, args.device, args.beta)
        print(f"test_loss={test_loss:.6f} test_accuracy={test_accuracy:.4f}")
        print(f"classes={labels}")
        print(f"saved={args.checkpoint}")
    finally:
        extractor.close()


if __name__ == "__main__":
    main()