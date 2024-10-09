import lightning as L
from torch.utils.data import DataLoader

from lib.segcore.data_provider.dataset import SegmentationDataset


class SegmentationDataModule(L.LightningDataModule):
    def __init__(self, config_manager):
        super(SegmentationDataModule, self).__init__()

        self.config_manager = config_manager

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = SegmentationDataset(self.config_manager, mode="train")

        self.test_dataset = SegmentationDataset(self.config_manager, mode="val")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config_manager.get("segcore.dataloader.train.batch_size"),
            num_workers=self.config_manager.get("segcore.dataloader.train.num_workers"),
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config_manager.get("segcore.dataloader.val.batch_size"),
            num_workers=self.config_manager.get("segcore.dataloader.val.num_workers"),
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )
