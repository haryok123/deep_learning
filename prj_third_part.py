import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

BASE_DIR = os.getcwd()
DATASET_PATH = os.path.join(BASE_DIR, "Animals-10.zip")
EXTRACT_PATH = os.path.join(BASE_DIR, "animals_data")
RAW_IMG_PATH = os.path.join(EXTRACT_PATH, "raw-img")
TRAIN_PATH = os.path.join(EXTRACT_PATH, "train")
VAL_PATH = os.path.join(EXTRACT_PATH, "val")
TEST_PATH = os.path.join(EXTRACT_PATH, "test")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(2)
            )

        self.block1 = conv_block(3, 32)
        self.block2 = conv_block(32, 64)
        self.block3 = conv_block(64, 128)
        self.block4 = conv_block(128, 256)
        self.block5 = conv_block(256, 512)

        self.res1 = nn.Conv2d(3, 64, kernel_size=1, stride=4)
        self.res2 = nn.Conv2d(64, 128, kernel_size=1, stride=2)
        self.res3 = nn.Conv2d(128, 256, kernel_size=1, stride=2)
        self.res4 = nn.Conv2d(256, 512, kernel_size=1, stride=2)

        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)


    def forward(self, x):
        identity1 = self.res1(x)
        x = self.block1(x)
        x = self.block2(x)
        x += identity1

        identity2 = self.res2(x)
        x = self.block3(x)
        x += identity2

        identity3 = self.res3(x)
        x = self.block4(x)
        x += identity3

        identity4 = self.res4(x)
        x = self.block5(x)
        x += identity4

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x

base_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
augmented_transforms = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
super_augmented_transforms = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
    transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=base_transforms)
val_dataset = datasets.ImageFolder(VAL_PATH, transform=base_transforms)
test_dataset = datasets.ImageFolder(TEST_PATH, transform=base_transforms)
viz_dataset = datasets.ImageFolder(TEST_PATH, transform=transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()]))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
viz_loader = DataLoader(viz_dataset, batch_size=64, shuffle=False, num_workers=4)

classes = train_dataset.classes
model = CNNModel(num_classes=len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.000001)

# Функція тренування моделі
def train_model(model, train_loader, val_loader, device, criterion, optimizer, scheduler, epochs, start_epoch=0):
    model.train()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(start_epoch, start_epoch + epochs):

        train_dataset.transform = augmented_transforms

        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        loop = tqdm(train_loader, leave=True)
        loop.set_description(f"Epoch {epoch + 1}/{epochs + start_epoch}")

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

        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(100 * correct / total)

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

        val_losses.append(val_running_loss / len(val_loader))
        val_accuracies.append(100 * val_correct / val_total)



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
    print(f"\nВалідаційна точність: {accuracy:.2f}%")

    report = classification_report(all_labels, all_preds, target_names=classes)
    print("\nЗвіт класифікації:\n", report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Прогнозований клас")
    plt.ylabel("Справжній клас")
    plt.title("Матриця плутанини")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    visualize_misclassified(misclassified_images, misclassified_labels, misclassified_preds, classes)

    report_dict = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    plot_class_metrics(report_dict, classes)

    return accuracy, report_dict


def visualize_misclassified(images, true_labels, pred_labels, classes, n=10):
    n = min(n, len(images))

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

    # F1-scores
    ax1.bar(classes, f1_scores, color='skyblue')
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('F1-score')
    ax1.set_title('F1-score by classes')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y')
    for tick in ax1.get_xticklabels():
        tick.set_rotation(45)

    # Precision і Recall
    x = np.arange(len(classes))
    width = 0.35
    ax2.bar(x - width / 2, precision, width, label='Precision', color='lightgreen')
    ax2.bar(x + width / 2, recall, width, label='Recall', color='salmon')
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Values')
    ax2.set_title('Precision і Recall по класах')
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
    plt.xlabel("Прогнозований клас")
    plt.ylabel("Справжній клас")
    plt.title("Нормалізована матриця плутанини")
    plt.tight_layout()
    plt.savefig('normalized_confusion_matrix.png')
    plt.show()


def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

def save_checkpoint(model, optimizer, scheduler, epoch, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch
    }, path)
    print(f"Checkpoint saved to {path}")


if __name__ == "__main__":
    start_epoch = 0
    checkpoint_path = "checkpoint_91.pth"

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']

    history = train_model(
        model, train_loader, val_loader, device, criterion, optimizer, scheduler,
        epochs=1,
        start_epoch=start_epoch
    )

    plot_training_history(history)
    accuracy, report_dict = evaluate_model(model, val_loader, viz_loader, device, classes)

    save_checkpoint(model, optimizer, scheduler, start_epoch + 1, checkpoint_path)

    with torch.no_grad():
        all_preds, all_labels = [], []
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        plot_normalized_confusion_matrix(all_labels, all_preds, classes)


    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * test_correct / test_total:.2f}%")


