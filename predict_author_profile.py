"""Predict gender + age group class for new Telegram-style text."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import torch
from torch import nn


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Text to classify")
    parser.add_argument("--model-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    vectorizer = joblib.load(args.model_dir / "tfidf_vectorizer.joblib")
    label_encoder = joblib.load(args.model_dir / "label_encoder.joblib")
    model = TextProfileMLP(input_dim=len(vectorizer.get_feature_names_out()), num_classes=len(label_encoder.classes_))
    model.load_state_dict(torch.load(args.model_dir / "author_profile_mlp.pt", map_location="cpu"))
    model.eval()

    x = vectorizer.transform([args.text]).toarray()
    x_tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        probs = torch.softmax(model(x_tensor), dim=1).cpu().numpy()[0]
    best_idx = int(probs.argmax())
    label = label_encoder.inverse_transform([best_idx])[0]
    gender, age_group = label.split("_", 1)

    print(f"Predicted class: {label}")
    print(f"Gender: {gender}")
    print(f"Age group: {age_group}")
    print(f"Confidence: {probs[best_idx]:.3f}")


if __name__ == "__main__":
    main()
