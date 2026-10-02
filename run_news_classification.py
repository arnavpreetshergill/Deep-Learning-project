"""Train and compare lightweight text-only and metadata-fusion HuffPost classifiers.

The script is deliberately compact enough for a course project while still
keeping the train/validation/test split and preprocessing free of leakage.
It writes all generated charts and result files to ``outputs/``.

Full-data default run (designed for under 30 minutes on the available GPU):
    python run_news_classification.py --models all

Use ``--max-samples`` only when a smaller development sample is wanted.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset, default_collate
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


RANDOM_SEED = 42
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")


@dataclass
class Config:
    csv_path: Path
    output_dir: Path
    models: list[str]
    epochs: int
    bert_epochs: int
    batch_size: int
    bert_batch_size: int
    max_samples: int
    word_max_length: int
    bert_max_length: int
    vocab_size: int
    tfidf_features: int
    svd_components: int
    bert_model: str
    patience: int
    learning_rate: float
    bert_learning_rate: float
    bert_gradient_accumulation: int
    bert_warmup_ratio: float
    bert_weight_decay: float
    class_weight_power: float
    seed: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="News_Category_Dataset_v3.csv")
    parser.add_argument("--output-dir", default="outputs_fast_full")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["all", "mlp", "cnn", "bilstm", "bert"],
        help="Architectures to run. Each runs text-only and text+metadata versions.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Epochs for MLP/CNN/BiLSTM.")
    parser.add_argument("--bert-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bert-batch-size", type=int, default=64)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional stratified sample size; 0 (the default) uses the full dataset.",
    )
    parser.add_argument(
        "--word-max-length",
        type=int,
        default=96,
        help="Maximum word-token sequence length for CNN and BiLSTM.",
    )
    parser.add_argument(
        "--bert-max-length",
        type=int,
        default=96,
        help="Maximum BERT token length for the fast benchmark.",
    )
    parser.add_argument("--vocab-size", type=int, default=20000)
    parser.add_argument("--tfidf-features", type=int, default=12000)
    parser.add_argument("--svd-components", type=int, default=128)
    parser.add_argument(
        "--bert-model",
        default="google/bert_uncased_L-4_H-256_A-4",
        help="A compact pretrained BERT that fits comfortably on an 8 GB GPU.",
    )
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--bert-learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--bert-gradient-accumulation",
        type=int,
        default=1,
        help="Number of BERT mini-batches to accumulate before an optimizer step.",
    )
    parser.add_argument(
        "--bert-warmup-ratio",
        type=float,
        default=0.10,
        help="Fraction of BERT optimizer steps used for linear warm-up.",
    )
    parser.add_argument("--bert-weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=0.5,
        help="0 disables balancing; 0.5 uses square-root class weights; 1 uses inverse weights.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    models = ["mlp", "cnn", "bilstm", "bert"] if "all" in args.models else args.models
    return Config(
        csv_path=Path(args.csv),
        output_dir=Path(args.output_dir),
        models=models,
        epochs=args.epochs,
        bert_epochs=args.bert_epochs,
        batch_size=args.batch_size,
        bert_batch_size=args.bert_batch_size,
        max_samples=args.max_samples,
        word_max_length=args.word_max_length,
        bert_max_length=args.bert_max_length,
        vocab_size=args.vocab_size,
        tfidf_features=args.tfidf_features,
        svd_components=args.svd_components,
        bert_model=args.bert_model,
        patience=args.patience,
        learning_rate=args.learning_rate,
        bert_learning_rate=args.bert_learning_rate,
        bert_gradient_accumulation=max(1, args.bert_gradient_accumulation),
        bert_warmup_ratio=max(0.0, args.bert_warmup_ratio),
        bert_weight_decay=args.bert_weight_decay,
        class_weight_power=max(0.0, args.class_weight_power),
        seed=args.seed,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def make_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "figures": output_dir / "figures",
        "models": output_dir / "models",
        "reports": output_dir / "reports",
        "data": output_dir / "data",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def clean_dataset(csv_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load, minimally clean, and build the compact metadata feature set."""
    required_columns = {"link", "headline", "category", "short_description", "authors", "date"}
    df = pd.read_csv(csv_path)
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    audit = {"raw_rows": len(df)}
    df = df.drop_duplicates().copy()
    audit["after_exact_duplicates"] = len(df)

    for column in ["headline", "short_description", "authors", "link"]:
        df[column] = df[column].fillna("").astype(str).str.strip()

    # Repeated URL rows are always the same published article. Empty links are retained.
    linked = df["link"].ne("")
    df = pd.concat(
        [df.loc[linked].drop_duplicates(subset="link"), df.loc[~linked]], ignore_index=True
    )
    df = df.drop_duplicates(subset=["headline", "short_description"]).copy()
    audit["after_article_duplicates"] = len(df)

    df = df[df["category"].notna()].copy()
    df["category"] = df["category"].astype(str).str.strip()
    df["authors"] = df["authors"].replace("", "UNKNOWN_AUTHOR")
    df["link"] = df["link"].replace("", "UNKNOWN_URL")
    df["headline"] = df["headline"].replace("", "MISSING_HEADLINE")
    df["short_description"] = df["short_description"].replace("", "MISSING_DESCRIPTION")
    df["text"] = df["headline"] + " [SEP] " + df["short_description"]

    parsed_date = pd.to_datetime(df["date"], errors="coerce")
    df["headline_chars"] = df["headline"].str.len().astype(np.float32)
    df["description_chars"] = df["short_description"].str.len().astype(np.float32)
    df["headline_words"] = df["headline"].str.split().str.len().astype(np.float32)
    df["description_words"] = df["short_description"].str.split().str.len().astype(np.float32)
    df["punctuation_count"] = (
        df["headline"].str.count(r"[^\w\s]") + df["short_description"].str.count(r"[^\w\s]")
    ).astype(np.float32)
    df["author_count"] = np.where(
        df["authors"].eq("UNKNOWN_AUTHOR"),
        0,
        df["authors"].str.count(",").add(1),
    ).astype(np.float32)
    df["publication_year"] = parsed_date.dt.year.fillna(0).astype(np.float32)
    df["url_length"] = df["link"].replace("UNKNOWN_URL", "").str.len().astype(np.float32)
    df["url_hyphen_count"] = df["link"].str.count("-").astype(np.float32)

    audit["final_rows"] = len(df)
    audit["classes"] = int(df["category"].nunique())
    return df.reset_index(drop=True), audit


def stratified_sample(df: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    if max_samples <= 0 or max_samples >= len(df):
        return df
    _, sampled = train_test_split(
        df,
        test_size=max_samples,
        stratify=df["category"],
        random_state=seed,
    )
    return sampled.reset_index(drop=True)


def make_splits(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    train_val, test = train_test_split(
        indices, test_size=0.10, random_state=seed, stratify=labels
    )
    train, validation = train_test_split(
        train_val,
        test_size=1 / 9,
        random_state=seed,
        stratify=labels[train_val],
    )
    return train, validation, test


def plot_data_overview(
    df: pd.DataFrame,
    feature_columns: list[str],
    paths: dict[str, Path],
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    counts = df["category"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(12, 13))
    ax.barh(counts.index, counts.values, color="#3b82f6")
    ax.set_title("HuffPost category distribution")
    ax.set_xlabel("Number of articles")
    ax.set_ylabel("Category")
    fig.tight_layout()
    fig.savefig(paths["figures"] / "01_category_distribution.png", dpi=180)
    plt.close(fig)

    missing = df[["headline", "short_description", "authors", "link", "date"]].isna().sum()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(missing.index, missing.values, color="#f59e0b")
    ax.set_title("Missing values after loading")
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(paths["figures"] / "02_missing_values.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(df["headline_words"], bins=40, ax=axes[0], color="#10b981")
    sns.histplot(df["description_words"], bins=40, ax=axes[1], color="#8b5cf6")
    axes[0].set_title("Headline word-count distribution")
    axes[1].set_title("Description word-count distribution")
    for ax in axes:
        ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(paths["figures"] / "03_text_length_distributions.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        df[feature_columns].corr(), cmap="vlag", center=0, annot=True, fmt=".2f", ax=ax
    )
    ax.set_title("Correlation between engineered metadata features")
    fig.tight_layout()
    fig.savefig(paths["figures"] / "04_metadata_correlation.png", dpi=180)
    plt.close(fig)


def basic_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_vocabulary(texts: Iterable[str], max_words: int) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(basic_tokens(text))
    most_common = counter.most_common(max_words - 2)
    return {word: index + 2 for index, (word, _) in enumerate(most_common)}


def encode_word_sequences(texts: Iterable[str], vocabulary: dict[str, int], max_length: int) -> np.ndarray:
    texts = list(texts)
    encoded = np.zeros((len(texts), max_length), dtype=np.int32)
    for row, text in enumerate(texts):
        tokens = basic_tokens(text)[:max_length]
        if tokens:
            encoded[row, : len(tokens)] = [vocabulary.get(token, 1) for token in tokens]
    return encoded


class DenseDataset(Dataset):
    def __init__(self, text: np.ndarray, metadata: np.ndarray, labels: np.ndarray):
        self.text = torch.from_numpy(text.astype(np.float32, copy=False))
        self.metadata = torch.from_numpy(metadata.astype(np.float32, copy=False))
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.text[index], self.metadata[index], self.labels[index]


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, metadata: np.ndarray, labels: np.ndarray):
        self.sequences = torch.from_numpy(sequences.astype(np.int64, copy=False))
        self.metadata = torch.from_numpy(metadata.astype(np.float32, copy=False))
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.sequences[index], self.metadata[index], self.labels[index]


class BertDataset(Dataset):
    def __init__(
        self,
        input_ids: np.ndarray,
        attention_masks: np.ndarray,
        token_type_ids: np.ndarray,
        metadata: np.ndarray,
        labels: np.ndarray,
    ):
        self.input_ids = torch.from_numpy(input_ids.astype(np.int64, copy=False))
        self.attention_masks = torch.from_numpy(attention_masks.astype(np.int64, copy=False))
        self.token_type_ids = torch.from_numpy(token_type_ids.astype(np.int64, copy=False))
        self.metadata = torch.from_numpy(metadata.astype(np.float32, copy=False))
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return (
            self.input_ids[index],
            self.attention_masks[index],
            self.token_type_ids[index],
            self.metadata[index],
            self.labels[index],
        )


class MetadataBranch(nn.Module):
    def __init__(self, metadata_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(metadata_dim, 32), nn.ReLU(), nn.Dropout(0.15), nn.Linear(32, 32), nn.ReLU()
        )

    def forward(self, metadata: Tensor) -> Tensor:
        return self.layers(metadata)


class MLPClassifier(nn.Module):
    def __init__(self, text_dim: int, metadata_dim: int, n_classes: int, use_metadata: bool):
        super().__init__()
        self.use_metadata = use_metadata
        self.text_branch = nn.Sequential(
            nn.Linear(text_dim, 128), nn.ReLU(), nn.Dropout(0.30), nn.Linear(128, 64), nn.ReLU()
        )
        self.metadata_branch = MetadataBranch(metadata_dim) if use_metadata else None
        classifier_input = 64 + (32 if use_metadata else 0)
        self.classifier = nn.Sequential(nn.Dropout(0.30), nn.Linear(classifier_input, n_classes))

    def forward(self, text: Tensor, metadata: Tensor) -> Tensor:
        encoded = self.text_branch(text)
        if self.use_metadata:
            encoded = torch.cat([encoded, self.metadata_branch(metadata)], dim=1)
        return self.classifier(encoded)


class TextCNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, metadata_dim: int, n_classes: int, use_metadata: bool):
        super().__init__()
        self.use_metadata = use_metadata
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.convolutions = nn.ModuleList([nn.Conv1d(64, 64, kernel_size=k) for k in (3, 4, 5)])
        self.text_branch = nn.Sequential(nn.Linear(192, 64), nn.ReLU(), nn.Dropout(0.30))
        self.metadata_branch = MetadataBranch(metadata_dim) if use_metadata else None
        self.classifier = nn.Linear(64 + (32 if use_metadata else 0), n_classes)

    def forward(self, tokens: Tensor, metadata: Tensor) -> Tensor:
        embedded = self.embedding(tokens).transpose(1, 2)
        pooled = [torch.max(torch.relu(conv(embedded)), dim=2).values for conv in self.convolutions]
        encoded = self.text_branch(torch.cat(pooled, dim=1))
        if self.use_metadata:
            encoded = torch.cat([encoded, self.metadata_branch(metadata)], dim=1)
        return self.classifier(encoded)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, metadata_dim: int, n_classes: int, use_metadata: bool):
        super().__init__()
        self.use_metadata = use_metadata
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.text_branch = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.30))
        self.metadata_branch = MetadataBranch(metadata_dim) if use_metadata else None
        self.classifier = nn.Linear(64 + (32 if use_metadata else 0), n_classes)

    def forward(self, tokens: Tensor, metadata: Tensor) -> Tensor:
        lengths = tokens.ne(0).sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(
            self.embedding(tokens), lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        encoded = self.text_branch(torch.cat([hidden[-2], hidden[-1]], dim=1))
        if self.use_metadata:
            encoded = torch.cat([encoded, self.metadata_branch(metadata)], dim=1)
        return self.classifier(encoded)


class BertFusionClassifier(nn.Module):
    def __init__(
        self, model_name: str, metadata_dim: int, n_classes: int, use_metadata: bool
    ) -> None:
        super().__init__()
        self.use_metadata = use_metadata
        self.encoder = AutoModel.from_pretrained(model_name)
        self.supports_token_type_ids = "token_type_ids" in self.encoder.forward.__code__.co_varnames
        hidden_size = self.encoder.config.hidden_size
        self.metadata_branch = MetadataBranch(metadata_dim) if use_metadata else None
        self.classifier = nn.Sequential(
            nn.Dropout(0.20), nn.Linear(hidden_size + (32 if use_metadata else 0), n_classes)
        )

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor, metadata: Tensor
    ) -> Tensor:
        encoder_inputs: dict[str, Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self.supports_token_type_ids:
            encoder_inputs["token_type_ids"] = token_type_ids
        encoded = self.encoder(**encoder_inputs).last_hidden_state[:, 0]
        if self.use_metadata:
            encoded = torch.cat([encoded, self.metadata_branch(metadata)], dim=1)
        return self.classifier(encoded)


def bert_collate(batch: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]) -> tuple[Tensor, ...]:
    """Trim padded BERT inputs to the longest real article in each batch.

    Inputs are stored at the configured maximum length for easy indexing, then
    dynamically shortened here. This keeps a 256-token ceiling without making
    every short HuffPost brief pay the compute cost of 256 tokens.
    """
    input_ids, attention_masks, token_type_ids, metadata, labels = default_collate(batch)
    batch_max_length = max(2, int(attention_masks.sum(dim=1).max().item()))
    return (
        input_ids[:, :batch_max_length],
        attention_masks[:, :batch_max_length],
        token_type_ids[:, :batch_max_length],
        metadata,
        labels,
    )


def make_loader(
    dataset: Dataset, batch_size: int, shuffle: bool, device: torch.device, dynamic_bert_padding: bool = False
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=bert_collate if dynamic_bert_padding else None,
    )


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_labels: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = [item.to(device, non_blocking=True) for item in batch]
            *inputs, labels = batch
            logits = model(*inputs)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(logits.argmax(dim=1).cpu().numpy())
    labels_np = np.concatenate(all_labels)
    predictions_np = np.concatenate(all_predictions)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        labels_np, predictions_np, average="macro", zero_division=0
    )
    metrics = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(labels_np, predictions_np),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "weighted_f1": f1_score(labels_np, predictions_np, average="weighted", zero_division=0),
    }
    return metrics, labels_np, predictions_np


def train_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    class_weights: Tensor,
    epochs: int,
    learning_rate: float,
    gradient_accumulation: int,
    warmup_ratio: float,
    weight_decay: float,
    patience: int,
    paths: dict[str, Path],
) -> tuple[dict[str, float], dict[str, list[float]], np.ndarray, np.ndarray, float]:
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = math.ceil(len(train_loader) / gradient_accumulation) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = (
        get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        if warmup_steps > 0
        else None
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = {"train_loss": [], "validation_loss": [], "validation_macro_f1": []}
    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    no_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        training_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = [item.to(device, non_blocking=True) for item in batch]
            *inputs, labels = batch
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(*inputs)
                raw_loss = criterion(logits, labels)
                loss = raw_loss / gradient_accumulation
            scaler.scale(loss).backward()
            training_loss += raw_loss.item() * labels.size(0)
            if step % gradient_accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and scaler.get_scale() >= scale_before_step:
                    scheduler.step()

        validation_metrics, _, _ = evaluate(model, validation_loader, device, criterion)
        history["train_loss"].append(training_loss / len(train_loader.dataset))
        history["validation_loss"].append(validation_metrics["loss"])
        history["validation_macro_f1"].append(validation_metrics["macro_f1"])
        print(
            f"{name} | epoch {epoch}/{epochs} | "
            f"train loss {history['train_loss'][-1]:.4f} | "
            f"val macro-F1 {validation_metrics['macro_f1']:.4f}",
            flush=True,
        )

        if validation_metrics["macro_f1"] > best_f1:
            best_f1 = validation_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"{name} | early stopping after epoch {epoch}", flush=True)
                break

    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), paths["models"] / f"{name}.pt")
    test_metrics, test_labels, test_predictions = evaluate(model, test_loader, device, criterion)
    return test_metrics, history, test_labels, test_predictions, training_seconds


def plot_training_history(name: str, history: dict[str, list[float]], paths: dict[str, Path]) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train")
    axes[0].plot(epochs, history["validation_loss"], marker="o", label="Validation")
    axes[0].set_title(f"{name}: loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, history["validation_macro_f1"], marker="o", color="#7c3aed")
    axes[1].set_title(f"{name}: validation macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(paths["figures"] / f"training_{name}.png", dpi=180)
    plt.close(fig)


def plot_final_results(
    results: pd.DataFrame,
    best_labels: np.ndarray,
    best_predictions: np.ndarray,
    class_names: np.ndarray,
    best_name: str,
    paths: dict[str, Path],
) -> None:
    if results.empty:
        return
    ordered = results.sort_values("macro_f1", ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(4, len(ordered) * 0.7)))
    bars = ax.barh(ordered["run"], ordered["macro_f1"], color="#2563eb")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlim(0, min(1, max(0.1, ordered["macro_f1"].max() + 0.08)))
    ax.set_xlabel("Test macro-F1")
    ax.set_title("Final model comparison")
    fig.tight_layout()
    fig.savefig(paths["figures"] / "05_final_model_comparison.png", dpi=180)
    plt.close(fig)

    pivot = results.pivot(index="architecture", columns="input", values="macro_f1")
    if {"Text only", "Text + metadata"}.issubset(pivot.columns):
        gain = (pivot["Text + metadata"] - pivot["Text only"]).sort_values()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = ["#16a34a" if value >= 0 else "#dc2626" for value in gain.values]
        bars = ax.bar(gain.index, gain.values, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.bar_label(bars, fmt="%+.3f", padding=3)
        ax.set_ylabel("Change in test macro-F1")
        ax.set_title("Effect of adding engineered metadata")
        fig.tight_layout()
        fig.savefig(paths["figures"] / "06_metadata_fusion_gain.png", dpi=180)
        plt.close(fig)

    matrix = confusion_matrix(best_labels, best_predictions, labels=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(20, 17))
    sns.heatmap(matrix, cmap="Blues", cbar=True, xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"Confusion matrix: {best_name}")
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Actual category")
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    ax.tick_params(axis="y", labelrotation=0, labelsize=6)
    fig.tight_layout()
    fig.savefig(paths["figures"] / "07_best_model_confusion_matrix.png", dpi=200)
    plt.close(fig)


def tokenise_for_bert(
    headlines: list[str], descriptions: list[str], tokenizer: AutoTokenizer, max_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    encoded = tokenizer(
        headlines,
        descriptions,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="np",
    )
    token_type_ids = encoded.get("token_type_ids", np.zeros_like(encoded["input_ids"]))
    return (
        encoded["input_ids"].astype(np.int32),
        encoded["attention_mask"].astype(np.int8),
        token_type_ids.astype(np.int8),
    )


def run_experiment(
    architecture: str,
    use_metadata: bool,
    data: dict[str, object],
    config: Config,
    device: torch.device,
    paths: dict[str, Path],
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    input_label = "Text + metadata" if use_metadata else "Text only"
    short_name = f"{architecture}_{'fusion' if use_metadata else 'text_only'}"
    metadata_dim = data["metadata"].shape[1]  # type: ignore[index]
    labels = data["labels"]  # type: ignore[assignment]
    split = data["split"]  # type: ignore[assignment]
    n_classes = data["n_classes"]  # type: ignore[assignment]
    class_weights = data["class_weights"].to(device)  # type: ignore[union-attr]

    if architecture == "mlp":
        text_data = data["svd_text"]  # type: ignore[assignment]
        train_dataset = DenseDataset(text_data[split[0]], data["metadata"][split[0]], labels[split[0]])
        val_dataset = DenseDataset(text_data[split[1]], data["metadata"][split[1]], labels[split[1]])
        test_dataset = DenseDataset(text_data[split[2]], data["metadata"][split[2]], labels[split[2]])
        model = MLPClassifier(text_data.shape[1], metadata_dim, n_classes, use_metadata)
        epochs, lr, batch_size = config.epochs, config.learning_rate, config.batch_size
    elif architecture in {"cnn", "bilstm"}:
        sequences = data["sequences"]  # type: ignore[assignment]
        train_dataset = SequenceDataset(sequences[split[0]], data["metadata"][split[0]], labels[split[0]])
        val_dataset = SequenceDataset(sequences[split[1]], data["metadata"][split[1]], labels[split[1]])
        test_dataset = SequenceDataset(sequences[split[2]], data["metadata"][split[2]], labels[split[2]])
        if architecture == "cnn":
            model = TextCNNClassifier(data["word_vocab_size"], metadata_dim, n_classes, use_metadata)
        else:
            model = BiLSTMClassifier(data["word_vocab_size"], metadata_dim, n_classes, use_metadata)
        epochs, lr, batch_size = config.epochs, config.learning_rate, config.batch_size
    elif architecture == "bert":
        ids = data["bert_ids"]  # type: ignore[assignment]
        masks = data["bert_masks"]  # type: ignore[assignment]
        token_types = data["bert_token_types"]  # type: ignore[assignment]
        train_dataset = BertDataset(
            ids[split[0]], masks[split[0]], token_types[split[0]], data["metadata"][split[0]], labels[split[0]]
        )
        val_dataset = BertDataset(
            ids[split[1]], masks[split[1]], token_types[split[1]], data["metadata"][split[1]], labels[split[1]]
        )
        test_dataset = BertDataset(
            ids[split[2]], masks[split[2]], token_types[split[2]], data["metadata"][split[2]], labels[split[2]]
        )
        model = BertFusionClassifier(config.bert_model, metadata_dim, n_classes, use_metadata)
        epochs, lr, batch_size = config.bert_epochs, config.bert_learning_rate, config.bert_batch_size
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    model = model.to(device)
    print(f"\nStarting {short_name} on {device} ...", flush=True)
    train_loader = make_loader(train_dataset, batch_size, True, device, architecture == "bert")
    val_loader = make_loader(val_dataset, batch_size, False, device, architecture == "bert")
    test_loader = make_loader(test_dataset, batch_size, False, device, architecture == "bert")
    test_metrics, history, test_labels, test_predictions, seconds = train_model(
        short_name,
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        class_weights,
        epochs,
        lr,
        config.bert_gradient_accumulation if architecture == "bert" else 1,
        config.bert_warmup_ratio if architecture == "bert" else 0.0,
        config.bert_weight_decay if architecture == "bert" else 1e-4,
        config.patience,
        paths,
    )
    plot_training_history(short_name, history, paths)

    model_params = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "run": short_name,
        "architecture": architecture.upper() if architecture != "bilstm" else "BiLSTM",
        "input": input_label,
        "accuracy": test_metrics["accuracy"],
        "macro_precision": test_metrics["macro_precision"],
        "macro_recall": test_metrics["macro_recall"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
        "train_seconds": seconds,
        "parameters": model_params,
    }
    report = classification_report(
        test_labels,
        test_predictions,
        labels=np.arange(n_classes),
        target_names=data["class_names"],
        output_dict=True,
        zero_division=0,
    )
    with (paths["reports"] / f"{short_name}_classification_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    del model, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, test_labels, test_predictions


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    if not config.csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {config.csv_path.resolve()}")
    paths = make_dirs(config.output_dir)
    with (paths["root"] / "run_config.json").open("w", encoding="utf-8") as handle:
        serializable_config = asdict(config)
        serializable_config["csv_path"] = str(config.csv_path)
        serializable_config["output_dir"] = str(config.output_dir)
        json.dump(serializable_config, handle, indent=2)

    print("Loading and cleaning dataset ...", flush=True)
    df, audit = clean_dataset(config.csv_path)
    df = stratified_sample(df, config.max_samples, config.seed)
    audit["rows_used"] = len(df)
    with (paths["data"] / "data_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)

    feature_columns = [
        "headline_chars",
        "description_chars",
        "headline_words",
        "description_words",
        "punctuation_count",
        "author_count",
        "author_train_frequency_log",
        "publication_year",
        "url_length",
        "url_hyphen_count",
    ]

    encoder = LabelEncoder()
    labels = encoder.fit_transform(df["category"])
    train_idx, val_idx, test_idx = make_splits(labels, config.seed)
    # Frequency features are fitted on the training partition only. This preserves
    # the clean separation between train, validation, and test data.
    author_counts = df.loc[train_idx, "authors"].value_counts()
    df["author_train_frequency_log"] = np.log1p(
        df["authors"].map(author_counts).fillna(0)
    ).astype(np.float32)
    plot_data_overview(df, feature_columns, paths)
    np.savez_compressed(
        paths["data"] / "splits.npz", train=train_idx, validation=val_idx, test=test_idx
    )
    pd.DataFrame({"class_id": range(len(encoder.classes_)), "category": encoder.classes_}).to_csv(
        paths["data"] / "label_mapping.csv", index=False
    )

    scaler = StandardScaler()
    metadata = np.zeros((len(df), len(feature_columns)), dtype=np.float32)
    metadata[train_idx] = scaler.fit_transform(df.loc[train_idx, feature_columns]).astype(np.float32)
    metadata[val_idx] = scaler.transform(df.loc[val_idx, feature_columns]).astype(np.float32)
    metadata[test_idx] = scaler.transform(df.loc[test_idx, feature_columns]).astype(np.float32)
    np.save(paths["data"] / "metadata_features.npy", metadata)
    df[feature_columns + ["category"]].describe(include="all").to_csv(
        paths["data"] / "feature_summary.csv"
    )

    class_counts = np.bincount(labels[train_idx], minlength=len(encoder.classes_))
    class_weights = (len(train_idx) / (len(encoder.classes_) * class_counts)) ** config.class_weight_power
    class_weights = class_weights / class_weights.mean()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {len(df):,} articles across {len(encoder.classes_)} categories on {device}.", flush=True)

    data: dict[str, object] = {
        "metadata": metadata,
        "labels": labels,
        "split": (train_idx, val_idx, test_idx),
        "n_classes": len(encoder.classes_),
        "class_names": encoder.classes_,
        "class_weights": torch.tensor(class_weights, dtype=torch.float32),
    }
    texts = df["text"].tolist()

    if "mlp" in config.models:
        print("Building TF-IDF/SVD representation for MLP ...", flush=True)
        vectorizer = TfidfVectorizer(
            max_features=config.tfidf_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            sublinear_tf=True,
        )
        train_tfidf = vectorizer.fit_transform([texts[index] for index in train_idx])
        all_tfidf = vectorizer.transform(texts)
        components = min(config.svd_components, train_tfidf.shape[0] - 1, train_tfidf.shape[1] - 1)
        svd = TruncatedSVD(n_components=components, n_iter=5, random_state=config.seed)
        svd.fit(train_tfidf)
        svd_text = svd.transform(all_tfidf).astype(np.float32)
        text_scaler = StandardScaler()
        svd_text[train_idx] = text_scaler.fit_transform(svd_text[train_idx]).astype(np.float32)
        svd_text[val_idx] = text_scaler.transform(svd_text[val_idx]).astype(np.float32)
        svd_text[test_idx] = text_scaler.transform(svd_text[test_idx]).astype(np.float32)
        np.save(paths["data"] / "mlp_svd_text.npy", svd_text)
        data["svd_text"] = svd_text
        print(f"MLP representation: {components} SVD features.", flush=True)

    if set(config.models).intersection({"cnn", "bilstm"}):
        print("Building word-token sequences for CNN/BiLSTM ...", flush=True)
        vocabulary = build_vocabulary((texts[index] for index in train_idx), config.vocab_size)
        sequences = encode_word_sequences(texts, vocabulary, config.word_max_length)
        np.save(paths["data"] / "word_sequences.npy", sequences)
        with (paths["data"] / "word_vocabulary.json").open("w", encoding="utf-8") as handle:
            json.dump(vocabulary, handle)
        data["sequences"] = sequences
        data["word_vocab_size"] = len(vocabulary) + 2
        print(f"CNN/BiLSTM vocabulary: {len(vocabulary):,} words.", flush=True)

    if "bert" in config.models:
        print(f"Tokenising texts for pretrained BERT: {config.bert_model} ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(config.bert_model)
        bert_ids, bert_masks, bert_token_types = tokenise_for_bert(
            df["headline"].tolist(),
            df["short_description"].tolist(),
            tokenizer,
            config.bert_max_length,
        )
        np.save(paths["data"] / "bert_input_ids.npy", bert_ids)
        np.save(paths["data"] / "bert_attention_masks.npy", bert_masks)
        np.save(paths["data"] / "bert_token_type_ids.npy", bert_token_types)
        data["bert_ids"] = bert_ids
        data["bert_masks"] = bert_masks
        data["bert_token_types"] = bert_token_types

    results: list[dict[str, object]] = []
    best_name = ""
    best_score = -1.0
    best_labels: np.ndarray | None = None
    best_predictions: np.ndarray | None = None
    for architecture in config.models:
        for use_metadata in (False, True):
            result, test_labels, test_predictions = run_experiment(
                architecture, use_metadata, data, config, device, paths
            )
            results.append(result)
            if result["macro_f1"] > best_score:
                best_score = float(result["macro_f1"])
                best_name = str(result["run"])
                best_labels, best_predictions = test_labels, test_predictions
            pd.DataFrame(results).sort_values("macro_f1", ascending=False).to_csv(
                paths["root"] / "metrics.csv", index=False
            )

    results_frame = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    results_frame.to_csv(paths["root"] / "metrics.csv", index=False)
    assert best_labels is not None and best_predictions is not None
    plot_final_results(
        results_frame,
        best_labels,
        best_predictions,
        encoder.classes_,
        best_name,
        paths,
    )
    with (paths["root"] / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "articles_used": len(df),
                "classes": len(encoder.classes_),
                "best_run": best_name,
                "best_macro_f1": best_score,
            },
            handle,
            indent=2,
        )
    print("\nFinished. Results are in:", paths["root"].resolve(), flush=True)
    print(results_frame[["run", "accuracy", "macro_f1", "weighted_f1"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
