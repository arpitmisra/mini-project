"""Portable contracts for self-supervised adaptation and its native checkpoint."""
import json
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .common import digest, file_digest
from .preprocessing import preprocessing_identity

METHOD = "oct-teacher-student-cls-and-masked-patches-v1"


def training_contract(config, records):
    from .backbone import model_identity
    if len(records) < 2 or any(r.split != "train" or r.source_split != "train" for r in records):
        raise ValueError("Fine-tuning requires at least two original training images; validation/test images are forbidden")
    # Snapshot settings: subsequent caller mutations cannot rewrite provenance.
    try:
        mlx_version = version("mlx")
    except PackageNotFoundError:
        mlx_version = None  # Portable tests/inspection do not install MLX.
    return {
        "method": METHOD, "base_model": model_identity(config["model"]),
        "preprocessing": preprocessing_identity(config["preprocessing"]),
        "finetuning": deepcopy(config["finetuning"]), "seed": config["runtime"]["seed"],
        "mlx_version": mlx_version, "numpy_version": version("numpy"),
        "train_samples_sha256": digest([r.identity() for r in records]),
        "train_count": len(records), "labels_used_for_loss": False,
        "validation_used_for_selection": False, "test_used_for_training": False,
    }


def read_adapted_checkpoint(path, model_config):
    from .backbone import model_identity, validate_hf_config
    path = Path(path)
    sidecar = path.with_suffix(".json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"Fine-tuned backbone unavailable: {path}. Run --stage finetune (or --stage all) first, or supply --checkpoint.")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    if (meta.get("format_version") != 1 or meta.get("status") != "complete"
            or meta.get("method") != METHOD or meta.get("representation_source") != "ema_teacher"):
        raise ValueError("Not a completed, supported self-supervised backbone checkpoint")
    if meta.get("base_model") != model_identity(model_config):
        raise ValueError("Fine-tuned checkpoint base model differs from the pinned backbone")
    if meta.get("weights_sha256") != file_digest(path):
        raise ValueError("Fine-tuned backbone checksum mismatch")
    if meta.get("contract_sha256") != digest(meta.get("contract")):
        raise ValueError("Fine-tuning contract checksum mismatch")
    validate_hf_config(meta["hf_config"])
    return meta


def adapted_identity(path, model_config):
    meta = read_adapted_checkpoint(path, model_config)
    return {**meta["base_model"], "base_weights_sha256": meta["base_model"]["weights_sha256"],
            "weights_sha256": meta["weights_sha256"], "adaptation_method": METHOD,
            "adaptation_contract_sha256": meta["contract_sha256"], "representation_source": "ema_teacher"}
