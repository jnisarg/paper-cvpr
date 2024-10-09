import lightning as L
import torch

from lib.models.efficientvit import efficientvit_seg_b0
from lib.segcore.utils import Metrics, Criterion, get_scheduler


class SegmentationRunner(L.LightningModule):
    def __init__(self, config_manager):
        super(SegmentationRunner, self).__init__()

        self.example_input_array = torch.randn(
            config_manager.get("segcore.example_input_array")
        )

        self.model = efficientvit_seg_b0(
            n_classes=config_manager.get("segcore.dataset.n_classes"),
            **config_manager.get("segcore.model", {}),
        )
        self.criterion = Criterion(config_manager)
        self.metrics = Metrics(config_manager)

        self.config_manager = config_manager

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        preds = self(images)
        loss = self.criterion(preds, targets)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_start(self):
        self.metrics.reset()

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        preds = self(images)
        loss = self.criterion(preds, targets)

        self.metrics.update(preds, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        self.metrics.collect()

        self.log(
            "val/iou",
            self.metrics.metrics["iou"].mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.print(f"Validation Metrics:\n{self.metrics}\n\n")

    def configure_optimizers(self):
        optimizer = getattr(
            torch.optim, self.config_manager.get("segcore.optimizer.name")
        )(
            self.model.parameters(),
            lr=self.config_manager.get("segcore.optimizer.lr"),
            weight_decay=self.config_manager.get("segcore.optimizer.weight_decay"),
            **self.config_manager.get("segcore.optimizer.kwargs", {}),
        )

        scheduler = get_scheduler(
            self.config_manager.get("segcore.scheduler.name"),
            optimizer,
            max_iter=self.trainer.estimated_stepping_batches,
            **self.config_manager.get("segcore.scheduler.kwargs", {}),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
