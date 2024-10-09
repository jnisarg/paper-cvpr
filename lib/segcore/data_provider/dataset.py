import os

import albumentations as A
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SegmentationDataset(Dataset):
    def __init__(self, config_manager, mode="train"):
        super(SegmentationDataset, self).__init__()

        assert mode in ["train", "val", "test"], f"Unknown mode: {mode}.\n"

        self.root = config_manager.get("segcore.dataset.root")
        self.mode = mode
        self.train_size = config_manager.get("segcore.dataset.train_size")
        self.test_size = config_manager.get("segcore.dataset.test_size")
        self.mean = config_manager.get("segcore.dataset.mean")
        self.std = config_manager.get("segcore.dataset.std")
        self.ignore_index = config_manager.get("segcore.dataset.ignore_index")

        self.class_mapping = config_manager.get("segcore.dataset.class_mapping")
        self.class_names = config_manager.get("segcore.dataset.class_names")

        with open(os.path.join(self.root, f"{mode}.txt"), "r") as f:
            self.samples = f.read().splitlines()

        if mode == "train":
            self.augmentation = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomResizedCrop(*self.train_size, scale=(0.5, 1.0)),
                ]
            )
        else:
            self.augmentation = A.Compose([A.Resize(*self.test_size)])

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index].split()

        image = cv2.imread(os.path.join(self.root, image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.root, mask_path), cv2.IMREAD_GRAYSCALE)

        augmented = self.augmentation(image=image, mask=mask)

        image = augmented["image"]
        mask = augmented["mask"]

        image = transforms.ToTensor()(image)
        image = transforms.Normalize(self.mean, self.std)(image)

        temp = mask.copy()
        for k, v in self.class_mapping.items():
            mask[temp == k] = v

        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask

    def __len__(self):
        return len(self.samples)
