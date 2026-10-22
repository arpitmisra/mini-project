from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from .common import write_json


def binary_metrics(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if labels.ndim != 1 or not len(labels) or probabilities.shape != (len(labels), 2):
        raise ValueError("Expected nonempty labels [N] and probabilities [N,2]")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid binary labels/probabilities")
    predicted = probabilities.argmax(1)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    both = len(np.unique(labels)) == 2
    return {"task": "binary NORMAL versus ABNORMAL", "count": len(labels), "accuracy": float(accuracy_score(labels, predicted)), "precision_abnormal": float(precision_score(labels, predicted, zero_division=0)), "sensitivity_abnormal": float(recall_score(labels, predicted, zero_division=0)), "specificity_normal": float(tn / (tn + fp)) if tn + fp else None, "f1_abnormal": float(f1_score(labels, predicted, zero_division=0)), "confusion_matrix": matrix.tolist(), "roc_auc": float(roc_auc_score(labels, probabilities[:, 1])) if both else None, "pr_auc_average_precision": float(average_precision_score(labels, probabilities[:, 1])) if both else None, "auc_note": None if both else "Undefined: only one ground-truth class is present", "per_class": classification_report(labels, predicted, labels=[0, 1], target_names=["NORMAL", "ABNORMAL"], output_dict=True, zero_division=0)}


def save_evaluation(directory, labels, probabilities, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metrics = binary_metrics(labels, probabilities)
    write_json(directory / "results.json", metrics)
    report = classification_report(labels, probabilities.argmax(1), labels=[0, 1], target_names=["NORMAL", "ABNORMAL"], zero_division=0)
    (directory / "classification_report.txt").write_text(report, encoding="utf-8")
    with (directory / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "source_split", "original_class", "binary_label", "predicted_label", "probability_abnormal"])
        writer.writeheader()
        for record, label, probability in zip(records, labels, probabilities):
            writer.writerow({**{k: record[k] for k in ("sample_id", "source_split", "original_class")}, "binary_label": int(label), "predicted_label": int(probability.argmax()), "probability_abnormal": float(probability[1])})
    ConfusionMatrixDisplay(np.asarray(metrics["confusion_matrix"]), display_labels=["NORMAL", "ABNORMAL"]).plot(cmap="Blues")
    plt.tight_layout()
    plt.savefig(directory / "confusion_matrix.png", dpi=160)
    plt.close()
    if metrics["roc_auc"] is not None:
        for name, display in (("roc_curve", RocCurveDisplay), ("precision_recall_curve", PrecisionRecallDisplay)):
            display.from_predictions(labels, probabilities[:, 1])
            plt.tight_layout()
            plt.savefig(directory / f"{name}.png", dpi=160)
            plt.close()
    return metrics
