import os
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_ROOT = r"C:\Users\Arpit\Desktop\OCT_DINO_PROJECT\data\OCT2017"

IMAGE_SIZE = 224
PATCH_SIZE = 16

BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3

NUM_CLASSES = 4

CLASS_NAMES = [
    "CNV",
    "DME",
    "DRUSEN",
    "NORMAL"
]

DEVICE = torch.device("cpu")

print("=" * 70)
print("DINO + OCT + CLASS ACTIVATION MAPPING")
print("=" * 70)
print("Device:", DEVICE)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 3. DATASET
# ============================================================

class KermanyOCTDataset(Dataset):

    CLASS_TO_INDEX = {
        "CNV": 0,
        "DME": 1,
        "DRUSEN": 2,
        "NORMAL": 3
    }

    VALID_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    def __init__(self, root_dir, split="train", transform=None):

        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        self.split_dir = self.root_dir / split

        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist:\n"
                f"{self.split_dir}"
            )

        self.samples = []

        self._load_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {self.split_dir}"
            )

    def _load_samples(self):

        for class_name in CLASS_NAMES:

            class_dir = self.split_dir / class_name

            if not class_dir.exists():

                print(
                    f"WARNING: Missing directory: {class_dir}"
                )

                continue

            label = self.CLASS_TO_INDEX[class_name]

            for image_path in class_dir.iterdir():

                if image_path.suffix.lower() in self.VALID_EXTENSIONS:

                    self.samples.append(
                        (
                            image_path,
                            label,
                            class_name
                        )
                    )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, label, class_name = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:

            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(
                label,
                dtype=torch.long
            ),
            "class_name": class_name,
            "path": str(image_path)
        }


# ============================================================
# 4. IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# 5. CREATE DATASETS
# ============================================================

print("\nLoading datasets...")

train_dataset = KermanyOCTDataset(
    DATASET_ROOT,
    split="train",
    transform=transform
)

val_dataset = KermanyOCTDataset(
    DATASET_ROOT,
    split="val",
    transform=transform
)

test_dataset = KermanyOCTDataset(
    DATASET_ROOT,
    split="test",
    transform=transform
)

print("Train images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))


# ============================================================
# 6. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 7. LOAD PRETRAINED DINO
# ============================================================

print("\nLoading pretrained DINO ViT-S/16...")

dino = torch.hub.load(
    "facebookresearch/dino:main",
    "dino_vits16",
    pretrained=True
)

dino = dino.to(DEVICE)

dino.eval()


# ============================================================
# 8. FREEZE DINO
# ============================================================

for parameter in dino.parameters():

    parameter.requires_grad = False


print("DINO loaded successfully.")

print(
    "DINO embedding dimension:",
    dino.embed_dim
)


# ============================================================
# 9. DINO FEATURE EXTRACTOR
# ============================================================

class DINOFeatureExtractor(nn.Module):

    def __init__(self, dino_model):

        super().__init__()

        self.dino = dino_model

    def forward(self, x):

        # get_intermediate_layers returns tokens
        # from the final transformer block.

        output = self.dino.get_intermediate_layers(
            x,
            n=1
        )[0]

        # Shape:
        #
        # [batch, 197, 384]
        #
        # 197 = 1 CLS token + 196 patch tokens

        cls_token = output[:, 0]

        patch_tokens = output[:, 1:]

        return cls_token, patch_tokens


feature_extractor = DINOFeatureExtractor(dino)

feature_extractor.eval()


# ============================================================
# 10. CLASSIFIER
# ============================================================

classifier = nn.Linear(
    dino.embed_dim,
    NUM_CLASSES
)

classifier = classifier.to(DEVICE)


print("\nClassifier:")
print(classifier)


# ============================================================
# 11. TRAIN CLASSIFIER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    classifier.parameters(),
    lr=LEARNING_RATE
)


print("\nStarting classifier training...")


for epoch in range(NUM_EPOCHS):

    classifier.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:

        images = batch["image"].to(DEVICE)

        labels = batch["label"].to(DEVICE)

        # DINO does NOT receive gradients.
        with torch.no_grad():

            cls_features, _ = feature_extractor(
                images
            )

        logits = classifier(
            cls_features
        )

        loss = criterion(
            logits,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(
            logits,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = (
        running_loss / len(train_loader)
    )

    epoch_accuracy = (
        correct / total
    )

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {epoch_accuracy:.4f}"
    )


# ============================================================
# 12. VALIDATION
# ============================================================

classifier.eval()

correct = 0
total = 0

with torch.no_grad():

    for batch in val_loader:

        images = batch["image"].to(DEVICE)

        labels = batch["label"].to(DEVICE)

        cls_features, _ = feature_extractor(
            images
        )

        logits = classifier(
            cls_features
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


val_accuracy = correct / total

print(
    f"\nValidation Accuracy: "
    f"{val_accuracy:.4f}"
)


# ============================================================
# 13. DINO + CLASSIFIER PREDICTION
# ============================================================

def predict_image(image_tensor):

    classifier.eval()

    with torch.no_grad():

        cls_features, _ = feature_extractor(
            image_tensor.unsqueeze(0)
        )

        logits = classifier(
            cls_features
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

    return (
        predicted_class,
        probabilities[0].cpu().numpy()
    )


# ============================================================
# 14. CLASS ACTIVATION MAPPING FOR DINO
# ============================================================

def generate_dino_cam(
    image_tensor,
    target_class=None
):

    """
    Generate a class-discriminative activation map
    using DINO patch tokens.

    This is a ViT-style gradient CAM, not the
    original CNN CAM formulation.

    Returns:
        cam          -> [14, 14]
        predicted    -> predicted class index
        probabilities -> class probabilities
    """

    # --------------------------------------------------------
    # Make a new tensor that can track gradients.
    # --------------------------------------------------------

    image = image_tensor.unsqueeze(0).to(DEVICE)

    image.requires_grad_(False)

    # --------------------------------------------------------
    # Get final transformer tokens.
    #
    # output shape:
    #
    # [1, 197, 384]
    #
    # 1 CLS + 196 patches
    # --------------------------------------------------------

    tokens = feature_extractor.dino.get_intermediate_layers(
        image,
        n=1
    )[0]

    # We need gradients with respect to patch tokens.
    tokens.retain_grad()

    # --------------------------------------------------------
    # CLS token
    # --------------------------------------------------------

    cls_token = tokens[:, 0]

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    logits = classifier(
        cls_token
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    if target_class is None:

        target_class = predicted_class

    # --------------------------------------------------------
    # Select class score.
    # --------------------------------------------------------

    class_score = logits[
        0,
        target_class
    ]

    # --------------------------------------------------------
    # Clear previous gradients.
    # --------------------------------------------------------

    dino.zero_grad()

    classifier.zero_grad()

    # --------------------------------------------------------
    # Backpropagate class score.
    # --------------------------------------------------------

    class_score.backward()

    # --------------------------------------------------------
    # Gradient of class score with respect to
    # all tokens.
    #
    # Shape:
    #
    # [1, 197, 384]
    # --------------------------------------------------------

    gradients = tokens.grad

    # Only patch tokens.
    patch_gradients = gradients[
        0,
        1:,
        :
    ]

    patch_tokens = tokens[
        0,
        1:,
        :
    ]

    # --------------------------------------------------------
    # Compute importance weight for each feature dimension.
    #
    # Average gradients over spatial patches.
    #
    # Shape:
    #
    # [384]
    # --------------------------------------------------------

    weights = patch_gradients.mean(
        dim=0
    )

    # --------------------------------------------------------
    # Weighted sum of feature dimensions.
    #
    # Each patch receives one importance score.
    #
    # [196,384] × [384]
    #
    # -> [196]
    # --------------------------------------------------------

    cam = torch.matmul(
        patch_tokens,
        weights
    )

    # --------------------------------------------------------
    # ReLU
    #
    # Keep positive evidence for the target class.
    # --------------------------------------------------------

    cam = torch.relu(cam)

    # --------------------------------------------------------
    # Reshape:
    #
    # 196 -> 14 × 14
    # --------------------------------------------------------

    grid_size = IMAGE_SIZE // PATCH_SIZE

    cam = cam.reshape(
        grid_size,
        grid_size
    )

    # --------------------------------------------------------
    # Normalize between 0 and 1.
    # --------------------------------------------------------

    cam = cam.detach().cpu().numpy()

    cam_min = cam.min()

    cam_max = cam.max()

    if cam_max - cam_min > 1e-8:

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min
        )

    else:

        cam = np.zeros_like(cam)

    return (
        cam,
        predicted_class,
        probabilities[
            0
        ].detach().cpu().numpy()
    )


# ============================================================
# 15. VISUALIZE CAM
# ============================================================

def visualize_cam(
    image_path,
    target_class=None
):

    # --------------------------------------------------------
    # Load original image.
    # --------------------------------------------------------

    original_image = Image.open(
        image_path
    ).convert("RGB")

    # --------------------------------------------------------
    # Transform for DINO.
    # --------------------------------------------------------

    image_tensor = transform(
        original_image
    )

    # --------------------------------------------------------
    # Generate CAM.
    # --------------------------------------------------------

    cam, predicted_class, probabilities = (
        generate_dino_cam(
            image_tensor,
            target_class
        )
    )

    # --------------------------------------------------------
    # Resize CAM to image size.
    # --------------------------------------------------------

    cam_image = Image.fromarray(
        np.uint8(cam * 255)
    )

    cam_image = cam_image.resize(
        original_image.size,
        Image.Resampling.BILINEAR
    )

    cam = np.asarray(
        cam_image
    ) / 255.0

    # --------------------------------------------------------
    # Plot.
    # --------------------------------------------------------

    plt.figure(
        figsize=(15, 5)
    )

    # Original
    plt.subplot(1, 3, 1)

    plt.imshow(
        original_image
    )

    plt.title(
        "Original OCT"
    )

    plt.axis("off")

    # CAM
    plt.subplot(1, 3, 2)

    plt.imshow(
        cam,
        cmap="jet"
    )

    plt.title(
        "DINO Class Activation Map"
    )

    plt.axis("off")

    # Overlay
    plt.subplot(1, 3, 3)

    plt.imshow(
        original_image
    )

    plt.imshow(
        cam,
        cmap="jet",
        alpha=0.45
    )

    predicted_name = CLASS_NAMES[
        predicted_class
    ]

    plt.title(
        f"Prediction: {predicted_name}"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()

    print("\nPrediction probabilities:")

    for class_name, probability in zip(
        CLASS_NAMES,
        probabilities
    ):

        print(
            f"{class_name:8s}: "
            f"{probability:.4f}"
        )


# ============================================================
# 16. TEST ON ONE IMAGE
# ============================================================

sample_path = test_dataset.samples[0][0]

print("\n" + "=" * 70)

print(
    "Generating CAM for:"
)

print(
    sample_path
)

print("=" * 70)

visualize_cam(
    str(sample_path)
)