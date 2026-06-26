import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from pathlib import Path
import time
import shutil

# Installation requirements:
"""
pip install torch torchvision opencv-python matplotlib
"""


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()

        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),  # For 28x28 input images
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x


class ArabicDigitsDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
            # Convert label to torch.long
        label = torch.tensor(label, dtype=torch.long)

        return image, label


class ArabicDigitRecognizer:
    def __init__(self, num_classes=10, device=None):
        self.num_classes = num_classes
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = ConvNet(num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.5)

    def train(self, train_loader, val_loader, epochs=20):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        best_val_loss = float('inf')
        best_model_state = None
        patience = 5
        counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_accuracy = correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
                counter = 0
            else:
                counter += 1

            # Early stopping
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            print(f"Epoch {epoch + 1}/{epochs}, "
                  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        }

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy

    def save_model(self, path="/app/models/arabic_digit_model.pth"):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path="/app/models/arabic_digit_model.pth"):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # On GPU, switch the model to FP16 for faster inference. Half precision
        # is skipped on CPU since it is poorly supported / slower there.
        if self.device.type == "cuda":
            self.model.half()
            print("ArabicDigitRecognizer model converted to half precision (FP16)")

        print(f"Model loaded from {path}")

    def predict(self, image):
        self.model.eval()

        if isinstance(image, np.ndarray):
            # Convert numpy array to tensor
            image = torch.from_numpy(image).float()

            # Add batch and channel dimensions if needed
            if len(image.shape) == 2:
                image = image.unsqueeze(0).unsqueeze(0)
            elif len(image.shape) == 3 and image.shape[0] == 1:
                image = image.unsqueeze(0)

        # Match the input dtype to whatever dtype the model is currently in
        # (FP16 on GPU after load_model(), FP32 on CPU) regardless of how the
        # caller built the tensor, so callers never need to know/care.
        model_dtype = next(self.model.parameters()).dtype
        image = image.to(self.device, dtype=model_dtype)

        with torch.no_grad():
            output = self.model(image)
            # Upcast logits to FP32 before softmax for numerical stability.
            probabilities = torch.nn.functional.softmax(output.float(), dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return predicted_class, confidence


class YOLODatasetPreparation:
    @staticmethod
    def preprocess_image(image, target_size=(28, 28)):
        """Preprocess a single image for digit recognition"""
        # Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize
        image = cv2.resize(image, target_size)

        # Normalize
        image = image.astype('float32') / 255.0

        return image

    @staticmethod
    def extract_digits_from_yolo_format(dataset_path, output_path, target_size=(28, 28)):
        """
        Extract digits from images using YOLO format annotations

        Args:
            dataset_path: Path to the YOLO format dataset (with train/valid/test subfolders)
            output_path: Path to save extracted digits
            target_size: Size to resize the extracted digits
        """
        # Create output directory structure
        os.makedirs(output_path, exist_ok=True)
        for i in range(10):
            os.makedirs(os.path.join(output_path, str(i)), exist_ok=True)

        # Process each split (train, valid, test)
        for split in ['train', 'valid', 'test']:
            images_dir = os.path.join(dataset_path, split, 'images')
            labels_dir = os.path.join(dataset_path, split, 'labels')

            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                print(f"Skipping {split} split - directory not found")
                continue

            print(f"Processing {split} split...")

            # Process each image in the split
            for img_file in os.listdir(images_dir):
                if not img_file.endswith(('.png', '.jpg', '.jpeg')):
                    continue

                img_path = os.path.join(images_dir, img_file)
                label_file = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')

                if not os.path.exists(label_file):
                    continue

                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Could not read image: {img_path}")
                    continue

                img_height, img_width = img.shape[:2]

                # Read annotations
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue

                        # YOLO format: class_id x_center y_center width height (normalized)
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * img_width
                        y_center = float(parts[2]) * img_height
                        bbox_width = float(parts[3]) * img_width
                        bbox_height = float(parts[4]) * img_height

                        # Calculate top-left coordinates
                        x1 = int(x_center - bbox_width / 2)
                        y1 = int(y_center - bbox_height / 2)
                        x2 = int(x_center + bbox_width / 2)
                        y2 = int(y_center + bbox_height / 2)

                        # Ensure coordinates are within image boundaries
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(img_width, x2)
                        y2 = min(img_height, y2)

                        # Extract digit
                        digit_img = img[y1:y2, x1:x2]
                        if digit_img.size == 0:
                            print(f"Empty digit region in {img_file}")
                            continue

                        # Preprocess digit
                        digit_img = YOLODatasetPreparation.preprocess_image(digit_img, target_size)

                        # Save digit with split information in filename
                        digit_filename = f"{split}_{os.path.splitext(img_file)[0]}_digit_{class_id}.png"
                        digit_path = os.path.join(output_path, str(class_id), digit_filename)

                        cv2.imwrite(digit_path, (digit_img * 255).astype(np.uint8))

            print(f"Processed {split} split")

        print(f"Digit extraction completed. Digits saved to {output_path}")

    @staticmethod
    def load_processed_dataset(processed_dir):
        """Load the processed dataset into numpy arrays"""
        images = []
        labels = []

        for digit_class in range(10):
            class_dir = os.path.join(processed_dir, str(digit_class))
            if not os.path.exists(class_dir):
                continue

            for img_file in os.listdir(class_dir):
                if not img_file.endswith(('.png', '.jpg', '.jpeg')):
                    continue

                img_path = os.path.join(class_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                # Normalize image
                img = img.astype('float32') / 255.0

                images.append(img)
                labels.append(digit_class)

        return np.array(images), np.array(labels)


def plot_training_history(history):
    """Plot training and validation loss/accuracy"""
    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracies'], label='Train Accuracy')
    plt.plot(history)


# Dataset preparation and training logic
if __name__ == "__main__":
    # Step 1: Prepare the Dataset
    dataset_path = "path/to/yolo/dataset"
    output_path = "path/to/save/processed/digits"
    YOLODatasetPreparation.extract_digits_from_yolo_format(dataset_path, output_path)

    # Step 2: Load the Dataset
    processed_dir = output_path  # Use the same directory
    images, labels = YOLODatasetPreparation.load_processed_dataset(processed_dir)

    # Step 3: Create DataLoaders
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = ArabicDigitsDataset(images, labels, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Step 4: Train the Model
    recognizer = ArabicDigitRecognizer(num_classes=10)
    history = recognizer.train(train_loader, val_loader, epochs=10)

    # Step 5: Evaluate the Model
    test_accuracy = recognizer.evaluate(test_loader)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Step 6: Save the Model
    recognizer.save_model("/app/models/arabic_digit_model.pth")

    # Optional: Plot Training History
    #plot_training_history(history)
