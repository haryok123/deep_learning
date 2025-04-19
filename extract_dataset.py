import os
import zipfile
import random
import shutil
from collections import defaultdict
import matplotlib.pyplot as plt



BASE_DIR = os.getcwd()
DATASET_PATH = os.path.join(BASE_DIR, "Animals-10.zip")
EXTRACT_PATH = os.path.join(BASE_DIR, "animals_data")
RAW_IMG_PATH = os.path.join(EXTRACT_PATH, "raw-img")
TRAIN_PATH = os.path.join(EXTRACT_PATH, "train")
VAL_PATH = os.path.join(EXTRACT_PATH, "val")
TEST_PATH = os.path.join(EXTRACT_PATH, "test")


def plot_class_distribution_from_zip(zip_path):
    class_counts = defaultdict(int)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.startswith("raw-img/") and file.lower().endswith(('.jpg', '.jpeg', '.png')):
                parts = file.split("/")
                if len(parts) > 2:
                    class_name = parts[1]
                    class_counts[class_name] += 1

    classes = list(class_counts.keys())
    counts = [class_counts[cls] for cls in classes]

    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts, color='skyblue')
    plt.title("Number of images by classes")
    plt.ylabel("Number of images")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def extract_dataset():
    if not os.path.exists(EXTRACT_PATH):
        with zipfile.ZipFile(DATASET_PATH, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)

def split_data_balanced(train_ratio=0.7, val_ratio=0.15):
    if os.path.exists(TRAIN_PATH): shutil.rmtree(TRAIN_PATH)
    if os.path.exists(VAL_PATH): shutil.rmtree(VAL_PATH)
    if os.path.exists(TEST_PATH): shutil.rmtree(TEST_PATH)

    os.makedirs(TRAIN_PATH)
    os.makedirs(VAL_PATH)
    os.makedirs(TEST_PATH)

    min_images = float('inf')

    class_image_dict = {}
    for class_name in os.listdir(RAW_IMG_PATH):
        class_path = os.path.join(RAW_IMG_PATH, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            class_image_dict[class_name] = images
            if len(images) < min_images:
                min_images = len(images)


    for class_name, images in class_image_dict.items():
        random.shuffle(images)
        images = images[:min_images]

        train_end = int(train_ratio * min_images)
        val_end = train_end + int(val_ratio * min_images)

        paths = {
            'train': os.path.join(TRAIN_PATH, class_name),
            'val': os.path.join(VAL_PATH, class_name),
            'test': os.path.join(TEST_PATH, class_name),
        }

        for path in paths.values():
            os.makedirs(path, exist_ok=True)

        for img in images[:train_end]:
            shutil.copy(os.path.join(RAW_IMG_PATH, class_name, img), os.path.join(paths['train'], img))
        for img in images[train_end:val_end]:
            shutil.copy(os.path.join(RAW_IMG_PATH, class_name, img), os.path.join(paths['val'], img))
        for img in images[val_end:]:
            shutil.copy(os.path.join(RAW_IMG_PATH, class_name, img), os.path.join(paths['test'], img))


def count_images_per_class_in_folder(folder_path):
    print(f"\nNun of images in folder: {folder_path}")
    for class_name in sorted(os.listdir(folder_path)):
        class_folder = os.path.join(folder_path, class_name)
        if os.path.isdir(class_folder):
            image_count = len(os.listdir(class_folder))
            print(f"{class_name}: {image_count} images")


if __name__ == '__main__':
    extract_dataset()
    split_data_balanced()
    plot_class_distribution_from_zip(DATASET_PATH)

    count_images_per_class_in_folder(TRAIN_PATH)
    count_images_per_class_in_folder(TEST_PATH)
    count_images_per_class_in_folder(VAL_PATH)