from __future__ import annotations

import os
import warnings
from collections import Counter, deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from PIL import Image

from .common import file_digest

KAGGLE_DATASET = "paultimothymooney/kermany2018"
CLASSES = ("NORMAL", "CNV", "DME", "DRUSEN")
BINARY_CLASS_MAPPING = {"NORMAL": 0, "CNV": 1, "DME": 1, "DRUSEN": 1}
EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")


def discover_root(path: Path, max_depth=5) -> Path:
    """Bounded breadth-first directory search; never walk image trees."""
    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset directory unavailable: {path}")
    queue = deque([(path, 0)])
    candidates = []
    while queue:
        node, depth = queue.popleft()
        if (node / "train").is_dir():
            candidates.append(node)
            continue
        if depth < max_depth:
            with os.scandir(node) as entries:
                children = sorted(Path(e.path) for e in entries if not e.name.startswith(".") and e.is_dir(follow_symlinks=False) and e.name not in CLASSES)
            queue.extend((child, depth + 1) for child in children)
    if len(candidates) != 1:
        raise ValueError(f"Expected one OCT root below {path}, found {len(candidates)}: {candidates}. Set --dataset-path to the exact root.")
    root = candidates[0]
    missing = [str(root / s / c) for s in SPLITS for c in CLASSES if not (root / s / c).is_dir()]
    if missing:
        raise ValueError("Missing required split/class folders: " + ", ".join(missing))
    return root


def resolve_dataset(config, downloader=None):
    local = config.get("local_path")
    if local and Path(local).exists():
        # An existing malformed directory is a data error, not permission for a large download.
        return discover_root(Path(local)), {"source": "local", "requested_path": str(local)}
    if not config.get("allow_download", True):
        raise FileNotFoundError(f"Local OCT dataset unavailable: {local}. Set --dataset-path or OCT_DATASET_PATH. Dataset downloads are disabled for this run.")
    if local:
        warnings.warn(f"Local dataset path unavailable: {local}; falling back to KaggleHub.")
    handle = config.get("handle", KAGGLE_DATASET)
    if handle != KAGGLE_DATASET:
        raise ValueError(f"This pipeline expects the Kermany dataset handle {KAGGLE_DATASET}")
    version = config.get("version")
    request = f"{handle}/versions/{version}" if version is not None else handle
    if downloader is None:
        try:
            from kagglehub import dataset_download
        except ImportError as exc:
            raise RuntimeError("KaggleHub is missing. Install requirements.txt or supply an existing local dataset.") from exc
        downloader = dataset_download
    try:
        downloaded = Path(downloader(request))
    except Exception as exc:
        raise RuntimeError("KaggleHub dataset retrieval failed. Check network access and free disk space. For 401/403, authenticate with kagglehub.login() or configure KAGGLE_API_TOKEN; accept any dataset terms on Kaggle. Alternatively set --dataset-path. " + str(exc)) from exc
    return discover_root(downloaded), {"source": "kagglehub", "handle": request, "cache_path": str(downloaded)}


@dataclass(frozen=True)
class Record:
    image_path: str
    sample_id: str
    split: str
    source_split: str
    original_class: str
    binary_label: int
    size: int
    mtime_ns: int
    sha256: str | None = None

    def metadata(self):
        return asdict(self)

    def identity(self):
        value = self.metadata()
        value.pop("image_path")
        if self.sha256:
            value.pop("mtime_ns")
        return value


class KermanyOCTDataset:
    def __init__(self, root, split, transform=None, max_per_class=None, hash_images=True, records=None):
        self.root, self.split, self.transform = Path(root), split, transform
        self.available_counts = Counter()
        self.records = [] if records is None else list(records)
        if records is not None:
            self.available_counts.update(r.original_class for r in records)
            return
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split}")
        for name in CLASSES:
            directory = self.root / split / name
            if not directory.is_dir():
                raise FileNotFoundError(f"Missing class directory: {directory}")
            # Enumerate names for deterministic selection/counts; decode only __getitem__ requests.
            paths = sorted(p for p in directory.rglob("*") if p.suffix.lower() in EXTENSIONS and p.is_file() and not p.name.startswith("."))
            if not paths:
                raise ValueError(f"No supported images in {directory}")
            self.available_counts[name] = len(paths)
            for path in paths[:max_per_class]:
                stat = path.stat()
                self.records.append(Record(str(path.resolve()), path.relative_to(self.root).as_posix(), split, split, name, BINARY_CLASS_MAPPING[name], stat.st_size, stat.st_mtime_ns, file_digest(path) if hash_images else None))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        try:
            with Image.open(record.image_path) as source:
                image = source.convert("RGB")
                image = self.transform(image) if self.transform else image.copy()
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read OCT image {record.image_path}: {exc}") from exc
        return {**record.metadata(), "image": image}

    def statistics(self):
        return {"available_original": dict(self.available_counts), "available_binary": {"NORMAL": self.available_counts["NORMAL"], "ABNORMAL": sum(self.available_counts[c] for c in CLASSES if c != "NORMAL")}, "selected_original": dict(Counter(r.original_class for r in self.records)), "selected_binary": dict(Counter(r.binary_label for r in self.records)), "selected_count": len(self)}


def check_split_integrity(datasets):
    seen_paths, seen_content = {}, {}
    for split, dataset in datasets.items():
        for record in dataset.records:
            path = str(Path(record.image_path).resolve())
            if record.source_split == "test" and split != "test":
                raise ValueError("Original test image assigned to training/validation")
            if path in seen_paths and seen_paths[path] != split:
                raise ValueError(f"Cross-split path overlap: {path}")
            seen_paths[path] = split
            if record.sha256:
                previous = seen_content.get(record.sha256)
                if previous and previous[0] != split:
                    raise ValueError(f"Cross-split identical image bytes: {previous[1]} and {path}. Audit the dataset; do not tune on leaked validation/test images.")
                seen_content[record.sha256] = (split, path)


def derive_validation(datasets, fraction, seed):
    """Explicit image-level split; no fabricated patient identifiers."""
    from sklearn.model_selection import train_test_split
    warnings.warn("Using an explicit image-level train-derived validation split. Patient independence is NOT established. Official validation images are excluded.")
    train = datasets["train"]
    a, b = train_test_split(train.records, test_size=fraction, random_state=seed, stratify=[r.original_class for r in train.records])
    result = dict(datasets)
    for split, records in (("train", a), ("val", b)):
        result[split] = KermanyOCTDataset(train.root, split, train.transform, records=sorted((replace(r, split=split) for r in records), key=lambda r: r.sample_id))
    return result
