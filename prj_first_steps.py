import os
import zipfile
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torch import nn, optim

# Шлях до датасету
DATASET_PATH = 'Intel_Image_Classification.zip'
EXTRACT_PATH = 'intel_data'

# Розпакування датасету
with zipfile.ZipFile(DATASET_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_PATH)

# Трансформації для зображень
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Завантаження датасету
train_dataset = datasets.ImageFolder(os.path.join(EXTRACT_PATH, 'train'), transform=transform)
valid_dataset = datasets.ImageFolder(os.path.join(EXTRACT_PATH, 'val'), transform=transform)

# DataLoader для навчання та валідації
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Перевірка класів
class_names = train_dataset.classes
print("Класи:", class_names)


# Візуалізація зразків зображень
def imshow(img):
    img = img / 2 + 0.5  # денормалізація
    plt.imshow(img.permute(1, 2, 0))
    plt.axis('off')
    plt.show()


# Отримання батчу зображень
images, labels = next(iter(train_loader))
fig, axes = plt.subplots(1, 6, figsize=(12, 6))
for i, ax in enumerate(axes):
    ax.imshow(images[i].permute(1, 2, 0))
    ax.set_title(class_names[labels[i]])
    ax.axis('off')
plt.show()


# Побудова простої CNN моделі
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, len(class_names))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Ініціалізація моделі
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Навчання моделі
EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}')

# Оцінка моделі
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in valid_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')
