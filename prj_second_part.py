import os
import zipfile
import torch
import random
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch import nn, optim
from PIL import ImageFile
from tqdm import tqdm

# Висновки після першої частини роботи
#
# Покращена структура проєкту
# Додано третій Conv2d(64 → 128) шар
# Додано BatchNorm2d для кожного згорткового шару
# Використовується LeakyReLU(0.1) замість ReLU
# Додано Dropout(0.4)
#
# Покращена аугментація даних
# Додано RandomAffine та RandomPerspective
# RandomGrayscale для варіативності
#
# Оптимізація навчання
# Додано ReduceLROnPlateau, який зменшує learning rate при повільному покращенні
# Збільшена кількість епох і збереження моделі після навчання у файл

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Налаштування шляхів до даних
BASE_DIR = os.getcwd()
DATASET_PATH = os.path.join(BASE_DIR, "Animals-10.zip")
EXTRACT_PATH = os.path.join(BASE_DIR, "animals_data")
RAW_IMG_PATH = os.path.join(EXTRACT_PATH, "raw-img")
TRAIN_PATH = os.path.join(EXTRACT_PATH, "train")
VAL_PATH = os.path.join(EXTRACT_PATH, "val")

# Візуалізація розподілу даних по класах
def visualize_data_distribution():
    train_class_counts = {class_name: len(os.listdir(os.path.join(TRAIN_PATH, class_name)))
                          for class_name in os.listdir(TRAIN_PATH)}
    df = pd.DataFrame({"Клас": list(train_class_counts.keys()), "Кількість": list(train_class_counts.values())})

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Клас", y="Кількість", data=df)
    plt.xticks(rotation=45)
    plt.title("Кількість зображень по класах (train)")
    plt.show()


# Створення DataLoader'ів для навчання та валідації
def get_dataloaders():
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_PATH, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    return train_loader, val_loader, train_dataset.classes


# Покращена архітектура CNN-моделі
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Функція для навчання моделі
def train_model(model, train_loader, device, criterion, optimizer, scheduler, epochs=40):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
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

        scheduler.step(running_loss / len(train_loader))

    print(f"\nTrain Accuracy: {100 * correct / total:.2f}%")


# Оцінка точності моделі на валідаційних даних
def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"\nValidation Accuracy: {100 * correct / total:.2f}%")


# Збереження моделі після навчання
def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Модель збережена в {file_path}")


if __name__ == "__main__":
    train_loader, val_loader, class_names = get_dataloaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    train_model(model, train_loader, device, criterion, optimizer, scheduler, epochs=40)
    evaluate_model(model, val_loader, device)
    save_model(model, "trained_model.pth")


