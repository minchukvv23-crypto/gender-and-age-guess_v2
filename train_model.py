"""Train a neural network for Telegram author profile classification.

Task: multiclass prediction of a joint class: gender + age group.
Example classes: male_18-24, female_35-44.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.cuda.manual_seed_all(seed)


class TextProfileMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sparse_to_tensor(matrix) -> torch.Tensor:
    return torch.tensor(matrix.toarray(), dtype=torch.float32)


def plot_training(history: dict[str, list[float]], out_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_confusion(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/telegram_author_profile_synthetic.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-features", type=int, default=4000)
    args = parser.parse_args()

    set_seed(SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    required_columns = {"text", "class_label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    x_train_text, x_test_text, y_train_labels, y_test_labels = train_test_split(
        df["text"].astype(str),
        df["class_label"].astype(str),
        test_size=0.2,
        random_state=SEED,
        stratify=df["class_label"],
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=args.max_features,
        min_df=2,
    )
    x_train = vectorizer.fit_transform(x_train_text)
    x_test = vectorizer.transform(x_test_text)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_labels)
    y_test = label_encoder.transform(y_test_labels)

    x_train_tensor = sparse_to_tensor(x_train)
    x_test_tensor = sparse_to_tensor(x_test)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(x_train_tensor, y_train_tensor),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = TextProfileMLP(input_dim=x_train_tensor.shape[1], num_classes=len(label_encoder.classes_))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        model.eval()
        with torch.no_grad():
            val_logits = model(x_test_tensor)
            val_loss = criterion(val_logits, y_test_tensor).item()
            y_pred = val_logits.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_test, y_pred)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        print(f"Epoch {epoch + 1:02d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_accuracy={val_acc:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(x_test_tensor)
        y_pred = logits.argmax(dim=1).cpu().numpy()

    labels = list(label_encoder.classes_)
    report_dict = classification_report(y_test, y_pred, target_names=labels, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, target_names=labels, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    torch.save(model.state_dict(), args.out_dir / "author_profile_mlp.pt")
    joblib.dump(vectorizer, args.out_dir / "tfidf_vectorizer.joblib")
    joblib.dump(label_encoder, args.out_dir / "label_encoder.joblib")

    (args.out_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    with (args.out_dir / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(args.out_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(args.out_dir / "training_history.csv", index=False, encoding="utf-8-sig")

    plot_training(history, args.out_dir / "training_curve.png")
    plot_confusion(cm, labels, args.out_dir / "confusion_matrix.png")

    print("\nFinal classification report:\n")
    print(report_text)
    print(f"Artifacts saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
