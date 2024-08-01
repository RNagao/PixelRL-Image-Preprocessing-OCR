import re
import random
import os
import torch
import matplotlib.pyplot as plt
import chardet
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from typing import Tuple
from pathlib import Path

DATA_PATH = Path("datasets/SROIE2019")
TRAIN_IMAGES_PATH = DATA_PATH / "train/img"
TRAIN_BOX_PATH = DATA_PATH / "train/box"
TEST_IMAGES_PATH = DATA_PATH / "test/img"
TEST_BOX_PATH = DATA_PATH / "test/box"

class ImageCustomDataset(Dataset):
    def __init__(self,
                images_dir: str,
                asserts_dir: str|None,
                transform=None) -> None:
        extensoes = ["*.jpg", "*.bmp", "*.png"]
        self.images_paths = []
        for ext in extensoes:
            self.images_paths.extend(list(Path(images_dir).glob(ext)))
        if asserts_dir is None:
            self.asserts_paths = None
        else:
            self.asserts_paths = list(Path(asserts_dir).glob("*.txt"))
        self.transform = transform

    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns"
        return Image.open(self.images_paths[index])

    def __len__(self) -> int:
        "Returns the total number of samples"
        return len(self.images_paths)

    def extract_text_from_assert(self, index: int) -> str:
        with open(self.asserts_paths[index], "rb") as f:
            bytes_data = f.read()
            encoding = chardet.detect(bytes_data)['encoding']
            text = bytes_data.decode(encoding)
            text = re.sub(r"\d+,\d+,\d+,\d+,\d+,\d+,\d+,\d+,", "", text)
            # text = re.sub(r"\n", " ", text)
        return text

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        "Returns one sample image and it's text assert (X, y)"
        img = self.load_image(index)

        if self.asserts_paths is None:
            item_assert = ""
        else:
            item_assert = self.extract_text_from_assert(index)

        if self.transform:
            return self.transform(img), item_assert

        return img, item_assert


def create_datasets(img_size: tuple[int, int]|None, image_channels:int, num_augments:int=5):
    # Construcao dos datasets
    train_datasets = []
    test_datasets = []

    transforms_list = [
        transforms.Grayscale(num_output_channels=image_channels),
        transforms.ToTensor()
    ]
    if img_size is not None:
        transforms_list.insert(0, transforms.Resize(img_size))

    original_transform = transforms.Compose(transforms_list)

    train_datasets.append(ImageCustomDataset(images_dir=TRAIN_IMAGES_PATH,
                                    asserts_dir=TRAIN_BOX_PATH,
                                    transform=original_transform))
    test_datasets.append(ImageCustomDataset(images_dir=TEST_IMAGES_PATH,
                                            asserts_dir=TEST_BOX_PATH,
                                            transform=original_transform))


    # seeds para reprodutibilidade das tranformações

    list_of_transforms = [
        transforms.GaussianBlur(kernel_size=5, sigma=0.5),
        transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=1),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=1),
        transforms.RandomAutocontrast()
        ]
    size_list_transforms = len(list_of_transforms)

    for _ in range(num_augments):
        transforms_list = [
            transforms.Grayscale(num_output_channels=image_channels),
            *[list_of_transforms[random.randint(0, size_list_transforms - 1)] for _ in range(random.randint(1, 7))],
            transforms.ToTensor()
        ]
        if img_size is not None:
            transforms_list.insert(0, transforms.Resize(img_size))

        random_transform = transforms.Compose(transforms_list)
        print(random_transform)

        train_datasets.append(ImageCustomDataset(images_dir=TRAIN_IMAGES_PATH,
                                        asserts_dir=TRAIN_BOX_PATH,
                                        transform=random_transform))
        test_datasets.append(ImageCustomDataset(images_dir=TEST_IMAGES_PATH,
                                        asserts_dir=TEST_BOX_PATH,
                                        transform=random_transform))

    # Concat datasets
    final_train_dataset = ConcatDataset(train_datasets)
    final_test_dataset = ConcatDataset(test_datasets)
    print(f"Final Train Dataset Size: {len(final_train_dataset)}")
    print(f"Final Test Dataset Size: {len(final_test_dataset)}")

    return  final_train_dataset, final_test_dataset

def display_dataset_image_by_index(dataset: torch.utils.data.Dataset,
                                   index: int):
    # Get image in dataset
    target_img = dataset[index][0]
    print(target_img.shape)

    # Adjust shape for plotting -> [C, H, W] -> [H, W, C]
    target_image_permuted = target_img.permute(1, 2, 0)

    # Plot
    plt.imshow(target_image_permuted)
    plt.axis("off")

def create_dataloaders(train_dataset:torch.utils.data.Dataset,
                        test_dataset:torch.utils.data.Dataset,
                        batch_size:int=32,
                        num_workers:int=os.cpu_count()):

    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers)
    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)

    return train_dataloader, test_dataloader