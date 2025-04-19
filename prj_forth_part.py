import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

num_classes = 10
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
model = model.to(device)

BASE_DIR = os.getcwd()
DATASET_PATH = os.path.join(BASE_DIR, "Animals-10.zip")
EXTRACT_PATH = os.path.join(BASE_DIR, "animals_data")
RAW_IMG_PATH = os.path.join(EXTRACT_PATH, "raw-img")
TRAIN_PATH = os.path.join(EXTRACT_PATH, "train")
VAL_PATH = os.path.join(EXTRACT_PATH, "val")


base_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

augmented_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

visualization_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=base_transforms)
val_dataset = datasets.ImageFolder(VAL_PATH, transform=val_transform)
viz_dataset = datasets.ImageFolder(VAL_PATH, transform=visualization_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
viz_loader = DataLoader(viz_dataset, batch_size=64, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[-1].parameters(), lr=0.0003)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=10)


def train_model(model, train_loader, val_loader, device, criterion, optimizer, scheduler, epochs=30):
    model.train()

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        if 10 <= epoch <= 26:
            train_dataset.transform = augmented_transforms
        else:
            train_dataset.transform = base_transforms

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        loop = tqdm(train_loader, leave=True)
        loop.set_description(f"Epoch {epoch + 1}/{epochs}")

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_postfix(loss=running_loss / len(train_loader), acc=100 * correct / total)

        scheduler.step()

        train_epoch_loss = running_loss / len(train_loader)
        train_epoch_acc = 100 * correct / total
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_acc)

        model.eval()
        val_running_loss = 0.0
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_acc = 100 * val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)

    return {
        'train_loss': train_losses,
        'train_acc': train_accuracies,
        'val_loss': val_losses,
        'val_acc': val_accuracies
    }

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


def evaluate_model(model, val_loader, viz_loader, device, classes):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    with torch.no_grad():
        for (images, labels), (viz_images, _) in zip(val_loader, viz_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            mask = (predicted != labels)
            for i, is_wrong in enumerate(mask):
                if is_wrong:
                    misclassified_images.append(viz_images[i].numpy())
                    misclassified_labels.append(labels[i].item())
                    misclassified_preds.append(predicted[i].item())

    accuracy = 100 * correct / total
    print(f"\nValidation Accuracy: {accuracy:.2f}%")

    report = classification_report(all_labels, all_preds, target_names=classes)
    print("\nClassification Report:\n", report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    visualize_misclassified(misclassified_images, misclassified_labels, misclassified_preds, classes)

    report_dict = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    plot_class_metrics(report_dict, classes)

    return accuracy, report_dict


# Функція для візуалізації неправильно класифікованих зображень
def visualize_misclassified(images, true_labels, pred_labels, classes, n=10):
    n = min(n, len(images))
    if n == 0:
        return

    plt.figure(figsize=(15, 2 * ((n + 4) // 5)))
    for i in range(n):
        plt.subplot(((n + 4) // 5), 5, i + 1)

        img = np.transpose(images[i], (1, 2, 0))  # CxHxW -> HxWxC

        plt.imshow(img)
        plt.title(f"T: {classes[true_labels[i]]}\nP: {classes[pred_labels[i]]}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('misclassified_examples.png')
    plt.show()


def plot_class_metrics(report_dict, classes):
    f1_scores = [report_dict[cls]['f1-score'] for cls in classes]
    precision = [report_dict[cls]['precision'] for cls in classes]
    recall = [report_dict[cls]['recall'] for cls in classes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    ax1.bar(classes, f1_scores, color='skyblue')
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('F1-score')
    ax1.set_title('F1-score per class')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y')
    for tick in ax1.get_xticklabels():
        tick.set_rotation(45)

    x = np.arange(len(classes))
    width = 0.35
    ax2.bar(x - width / 2, precision, width, label='Precision', color='lightgreen')
    ax2.bar(x + width / 2, recall, width, label='Recall', color='salmon')
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Score')
    ax2.set_title('Precision and Recall per class')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes)
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(axis='y')
    for tick in ax2.get_xticklabels():
        tick.set_rotation(45)

    plt.tight_layout()
    plt.savefig('class_metrics.png')
    plt.show()


def plot_normalized_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig('normalized_confusion_matrix.png')
    plt.show()

# Function to save model
def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

# Main execution
if __name__ == "__main__":
    classes = train_dataset.classes

    history = train_model(model, train_loader, val_loader, device, criterion, optimizer, scheduler, epochs=45)

    plot_training_history(history)

    accuracy, report_dict = evaluate_model(model, val_loader, viz_loader, device, classes)

    with torch.no_grad():
        all_preds = []
        all_labels = []
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        plot_normalized_confusion_matrix(all_labels, all_preds, classes)

    save_model(model, "best_model_mobilenet_v4.pth")