from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "data" / "OCT2017"

IMAGE_SIZE = 224
BATCH_SIZE = 32

# Keep this 0 initially on Windows.
# Increase later after everything works.
NUM_WORKERS = 0


# ============================================================
# KERMANY OCT BINARY DATASET
# ============================================================

class KermanyOCTBinaryDataset(Dataset):

    """
    Custom PyTorch Dataset for the Kermany OCT dataset.

    Original classes:
        CNV
        DME
        DRUSEN
        NORMAL

    Binary mapping:
        NORMAL  -> 0
        CNV     -> 1
        DME     -> 1
        DRUSEN  -> 1

    Each sample returns:

        image
        binary label
        original class
        image path
    """

    CLASS_NAMES = [
        "CNV",
        "DME",
        "DRUSEN",
        "NORMAL"
    ]

    BINARY_LABELS = {
        "NORMAL": 0,
        "CNV": 1,
        "DME": 1,
        "DRUSEN": 1
    }

    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    }

    def __init__(
        self,
        root_dir,
        split="train",
        transform=None
    ):

        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        # ----------------------------------------------------
        # Find split directory
        # ----------------------------------------------------

        if split == "val":

            possible_dirs = [
                self.root_dir / "val",
                self.root_dir / "validation"
            ]

        elif split == "validation":

            possible_dirs = [
                self.root_dir / "validation",
                self.root_dir / "val"
            ]

        else:

            possible_dirs = [
                self.root_dir / split
            ]

        self.split_dir = None

        for directory in possible_dirs:

            if directory.exists():

                self.split_dir = directory
                break

        if self.split_dir is None:

            raise FileNotFoundError(
                f"\nCould not find '{split}' directory.\n\n"
                f"Expected something like:\n"
                f"    {self.root_dir / 'train'}\n"
                f"    {self.root_dir / 'val'}\n"
                f"    {self.root_dir / 'test'}\n"
            )

        # ----------------------------------------------------
        # Store image information
        # ----------------------------------------------------

        self.samples = []

        self._load_samples()

        if len(self.samples) == 0:

            raise RuntimeError(
                f"No images found in {self.split_dir}"
            )

        print(
            f"[{split.upper()}] "
            f"Loaded {len(self.samples):,} images"
        )

    # ========================================================
    # FIND ALL IMAGES
    # ========================================================

    def _load_samples(self):

        for class_name in self.CLASS_NAMES:

            class_dir = self.split_dir / class_name

            if not class_dir.exists():

                print(
                    f"WARNING: Missing class directory:\n"
                    f"    {class_dir}"
                )

                continue

            binary_label = self.BINARY_LABELS[class_name]

            # rglob also searches subdirectories
            for image_path in class_dir.rglob("*"):

                if not image_path.is_file():
                    continue

                if image_path.suffix.lower() not in self.IMAGE_EXTENSIONS:
                    continue

                self.samples.append({
                    "path": str(image_path),
                    "binary_label": binary_label,
                    "original_class": class_name
                })

    # ========================================================
    # NUMBER OF SAMPLES
    # ========================================================

    def __len__(self):

        return len(self.samples)

    # ========================================================
    # GET ONE SAMPLE
    # ========================================================

    def __getitem__(self, index):

        sample = self.samples[index]

        image_path = sample["path"]

        binary_label = sample["binary_label"]

        original_class = sample["original_class"]

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(image_path).convert("RGB")

        # ----------------------------------------------------
        # Apply transformations
        # ----------------------------------------------------

        if self.transform is not None:

            image = self.transform(image)

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = torch.tensor(
            binary_label,
            dtype=torch.long
        )

        return (
            image,
            label,
            original_class,
            image_path
        )

    # ========================================================
    # DATASET STATISTICS
    # ========================================================

    def get_statistics(self):

        original_counts = Counter()
        binary_counts = Counter()

        for sample in self.samples:

            original_class = sample["original_class"]
            binary_label = sample["binary_label"]

            original_counts[original_class] += 1
            binary_counts[binary_label] += 1

        return original_counts, binary_counts


# ============================================================
# TRANSFORMS
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor()
])


# ============================================================
# CREATE DATASETS
# ============================================================

def create_datasets():

    print("\n" + "=" * 70)
    print("KERMANY OCT DATASET")
    print("=" * 70)

    print(f"\nDataset directory:")
    print(f"    {DATASET_DIR}")

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"\nDataset directory does not exist:\n"
            f"    {DATASET_DIR}\n\n"
            f"Please check your folder structure."
        )

    train_dataset = KermanyOCTBinaryDataset(
        root_dir=DATASET_DIR,
        split="train",
        transform=transform
    )

    val_dataset = KermanyOCTBinaryDataset(
        root_dir=DATASET_DIR,
        split="val",
        transform=transform
    )

    test_dataset = KermanyOCTBinaryDataset(
        root_dir=DATASET_DIR,
        split="test",
        transform=transform
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset
    )


# ============================================================
# PRINT STATISTICS
# ============================================================

def print_statistics(dataset, split_name):

    original_counts, binary_counts = (
        dataset.get_statistics()
    )

    print("\n" + "-" * 60)

    print(f"{split_name.upper()} DATASET")

    print("-" * 60)

    print("\nOriginal classes:")

    for class_name in [
        "CNV",
        "DME",
        "DRUSEN",
        "NORMAL"
    ]:

        print(
            f"  {class_name:<8} : "
            f"{original_counts[class_name]:,}"
        )

    print("\nBinary classes:")

    print(
        f"  NORMAL   (0) : "
        f"{binary_counts[0]:,}"
    )

    print(
        f"  ABNORMAL (1) : "
        f"{binary_counts[1]:,}"
    )


# ============================================================
# CREATE DATALOADERS
# ============================================================

def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset
):

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


# ============================================================
# MAIN TEST
# ============================================================

def main():

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    (
        train_dataset,
        val_dataset,
        test_dataset
    ) = create_datasets()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print_statistics(
        train_dataset,
        "Training"
    )

    print_statistics(
        val_dataset,
        "Validation"
    )

    print_statistics(
        test_dataset,
        "Testing"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader
    ) = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset
    )

    # --------------------------------------------------------
    # Load one batch
    # --------------------------------------------------------

    images, labels, classes, paths = next(
        iter(train_loader)
    )

    # --------------------------------------------------------
    # Print information
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BATCH TEST")
    print("=" * 70)

    print(
        f"\nImage tensor shape:"
        f"\n    {images.shape}"
    )

    print(
        f"\nLabel tensor shape:"
        f"\n    {labels.shape}"
    )

    print(
        f"\nLabels:"
        f"\n    {labels}"
    )

    print(
        f"\nOriginal classes:"
        f"\n    {classes}"
    )

    print("\nFirst 3 image paths:")

    for path in paths[:3]:

        print(
            f"    {path}"
        )

    # --------------------------------------------------------
    # Verify dimensions
    # --------------------------------------------------------

    expected_shape = (
        BATCH_SIZE,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    if images.shape == torch.Size(expected_shape):

        print(
            "\n✓ Image batch shape is correct."
        )

    else:

        print(
            f"\nWARNING: Expected "
            f"{expected_shape}"
        )

    print("\n" + "=" * 70)
    print("DATASET LOADER TEST COMPLETE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()