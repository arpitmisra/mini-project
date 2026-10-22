"""MLX-only linear probe training on cached CLS vectors."""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .common import file_digest, write_json
from .evaluation import binary_metrics


class BinaryClassifier(nn.Module):
    def __init__(self, dimension=384):
        super().__init__()
        self.linear = nn.Linear(dimension, 2)

    def __call__(self, features):
        return self.linear(features)


def weighted_cross_entropy(model, x, y, class_weights):
    losses = nn.losses.cross_entropy(model(x), y, reduction="none")
    weights = class_weights[y]
    return mx.sum(losses * weights) / mx.sum(weights)


def predict_probabilities(model, features, batch_size=256):
    model.eval()
    parts = []
    for offset in range(0, len(features), batch_size):
        probs = mx.softmax(model(mx.array(features[offset:offset + batch_size], dtype=mx.float32)), axis=-1)
        mx.eval(probs)
        parts.append(np.asarray(probs))
    return np.concatenate(parts)


def save_classifier(model, path, identity, training_config, seed):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.stem}.{os.getpid()}.safetensors")
    model.save_weights(str(temp))
    os.replace(temp, path)
    write_json(path.with_suffix(".json"), {"format_version": 1, "feature_identity": identity, "dimension": 384, "classes": ["NORMAL", "ABNORMAL"], "training": training_config, "seed": seed, "weights_sha256": file_digest(path)})


def load_classifier(path, expected_identity):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta["feature_identity"] != json.loads(json.dumps(expected_identity)) or meta["classes"] != ["NORMAL", "ABNORMAL"] or meta["dimension"] != 384:
        raise ValueError("Classifier/backbone/preprocessing metadata mismatch")
    if meta["weights_sha256"] != file_digest(path):
        raise ValueError("Classifier checkpoint checksum mismatch")
    model = BinaryClassifier(meta["dimension"])
    model.load_weights(str(path), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


def train_classifier(train, val, config, output_dir, seed=42):
    started = time.perf_counter()
    x_train, y_train = train.load_cls()
    x_val, y_val = val.load_cls()
    if len(np.unique(y_train)) != 2 or len(np.unique(y_val)) != 2:
        raise ValueError("Training and validation require both binary classes")
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = BinaryClassifier()
    optimizer = optim.AdamW(learning_rate=config["learning_rate"], weight_decay=config["weight_decay"])
    weights = len(y_train) / (2 * np.bincount(y_train, minlength=2)) if config["weighted_loss"] else np.ones(2)
    class_weights = mx.array(weights, dtype=mx.float32)
    # One transfer per split; only the small classifier participates in autodiff.
    x, y = mx.array(x_train), mx.array(y_train, dtype=mx.int32)
    xv, yv = mx.array(x_val), mx.array(y_val, dtype=mx.int32)
    mx.eval(model.parameters(), x, y, xv, yv, class_weights)
    initial = np.asarray(model.linear.weight).copy()
    loss_and_grad = nn.value_and_grad(model, weighted_cross_entropy)
    best, stale, history = float("inf"), 0, []
    output_dir = Path(output_dir)
    checkpoint = output_dir / "checkpoints/best_classifier.safetensors"
    for epoch in range(config["epochs"]):
        model.train()
        order = rng.permutation(len(y_train))
        total_loss, count = 0.0, 0
        for offset in range(0, len(order), config["batch_size"]):
            indices = mx.array(order[offset:offset + config["batch_size"]], dtype=mx.int32)
            loss, gradients = loss_and_grad(model, x[indices], y[indices], class_weights)
            optimizer.update(model, gradients)
            mx.eval(loss, model.parameters(), optimizer.state)
            total_loss += float(loss.item()) * len(indices)
            count += len(indices)
        model.eval()
        val_loss = nn.losses.cross_entropy(model(xv), yv, reduction="mean")
        mx.eval(val_loss)
        value = float(val_loss.item())
        if not np.isfinite(value) or not np.isfinite(total_loss):
            raise ValueError("Training diverged: nonfinite loss")
        probabilities = predict_probabilities(model, x_val, config["batch_size"])
        metrics = binary_metrics(y_val, probabilities)
        row = {"epoch": epoch + 1, "train_loss": total_loss / count, "val_loss": value, "val_accuracy": metrics["accuracy"], "val_f1_abnormal": metrics["f1_abnormal"]}
        history.append(row)
        print(f"Epoch {epoch + 1}: train loss={row['train_loss']:.5f}, validation loss={value:.5f}")
        if value < best:
            best, stale = value, 0
            before_save = predict_probabilities(model, x_val[:8])
            save_classifier(model, checkpoint, train.index["identity"], config, seed)
            restored = load_classifier(checkpoint, train.index["identity"])
            np.testing.assert_allclose(before_save, predict_probabilities(restored, x_val[:8]), atol=1e-6, rtol=1e-5)
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    if np.array_equal(initial, np.asarray(model.linear.weight)):
        raise RuntimeError("Classifier weights did not change")
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    best_model = load_classifier(checkpoint, train.index["identity"])
    return best_model, {"training_seconds": time.perf_counter() - started, "epochs_completed": len(history), "best_val_loss": best, "class_weights": weights.tolist(), "checkpoint": str(checkpoint)}
