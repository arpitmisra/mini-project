import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

from torch.optim import Adam
from tqdm import tqdm

from dataset_loader import (
    create_datasets,
    create_dataloaders
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 2

NUM_EPOCHS = 3

LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CLASSIFIER
# ============================================================

class OCTBinaryClassifier(nn.Module):

    """
    ResNet-18 based binary classifier.

    Input:
        OCT image

    Output:
        2 logits

    Class 0:
        NORMAL

    Class 1:
        ABNORMAL
    """

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Load ResNet-18
        # ----------------------------------------------------

        self.model = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # ----------------------------------------------------
        # Replace final classification layer
        # ----------------------------------------------------

        input_features = (
            self.model.fc.in_features
        )

        self.model.fc = nn.Linear(
            input_features,
            NUM_CLASSES
        )

    def forward(self, x):

        return self.model(x)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels, _, _ in loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = (
        correct / total
        if total > 0
        else 0
    )

    return accuracy


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    progress_bar = tqdm(
        loader,
        desc="Training"
    )

    for images, labels, _, _ in progress_bar:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(images)

        # ----------------------------------------------------
        # Calculate loss
        # ----------------------------------------------------

        loss = criterion(
            outputs,
            labels
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        running_loss += (
            loss.item() * labels.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        progress_bar.set_postfix(
            loss=loss.item()
        )

    epoch_loss = (
        running_loss / total
        if total > 0
        else 0
    )

    epoch_accuracy = (
        correct / total
        if total > 0
        else 0
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "KERMANY OCT BINARY CLASSIFICATION"
    )

    print("=" * 70)

    print(
        f"\nUsing device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    (
        train_dataset,
        val_dataset,
        test_dataset
    ) = create_datasets()

    # --------------------------------------------------------
    # Create loaders
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
    # Create model
    # --------------------------------------------------------

    print("\nCreating ResNet-18...")

    model = OCTBinaryClassifier()

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Loss function
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(NUM_EPOCHS):

        print(
            f"\nEpoch "
            f"{epoch + 1}/{NUM_EPOCHS}"
        )

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer
            )
        )

        val_accuracy = evaluate(
            model,
            val_loader
        )

        print(
            f"\nTrain Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{val_accuracy:.4f}"
        )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_accuracy = evaluate(
        model,
        test_loader
    )

    print("\n" + "=" * 70)

    print(
        f"Final Test Accuracy: "
        f"{test_accuracy:.4f}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    torch.save(
        model.state_dict(),
        "oct_resnet18_binary.pth"
    )

    print(
        "\nModel saved as:"
        "\n    oct_resnet18_binary.pth"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()