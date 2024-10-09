import argparse
import os
import warnings

import lightning as L
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers

from lib.configs import ConfigManager
from lib.segcore.trainer import SegmentationRunner
from lib.segcore.data_provider import SegmentationDataModule

warnings.filterwarnings("ignore")


def train(config_manager):
    dm = SegmentationDataModule(config_manager)
    dm.setup("fit")

    model = SegmentationRunner(config_manager)

    work_dir = os.path.join(
        config_manager.get("work_dir"), config_manager.get("run_name")
    )
    if os.path.exists(work_dir):
        if (
            input(f"Work directory {work_dir} already exists. Overwrite? (y/n): ")
            == "y"
        ):
            os.system(f"rm -rf {work_dir}")
        else:
            exit(0)

    tb_logger = pl_loggers.TensorBoardLogger(
        work_dir,
        name=config_manager.get("tb.name"),
        log_graph=config_manager.get("tb.log_graph"),
    )
    csv_logger = pl_loggers.CSVLogger(work_dir, name=config_manager.get("csv.name"))

    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        dirpath=work_dir,
        filename=config_manager.get("checkpoints.filename"),
        monitor=config_manager.get("checkpoints.monitor"),
        mode=config_manager.get("checkpoints.mode"),
        save_last=True,
        save_top_k=config_manager.get("checkpoints.save_top_k"),
        every_n_epochs=config_manager.get("checkpoints.every_n_epochs"),
    )

    callbacks = [checkpoint_callback]
    if config_manager.get("rich_logging"):
        callbacks.extend(
            (pl_callbacks.RichModelSummary(), pl_callbacks.RichProgressBar())
        )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=config_manager.get("devices", 1),
        precision=config_manager.get("trainer.precision"),
        callbacks=callbacks,
        logger=[tb_logger, csv_logger],
        max_epochs=config_manager.get("trainer.max_epochs"),
        enable_checkpointing=config_manager.get("trainer.enable_checkpointing"),
        enable_progress_bar=True,
        check_val_every_n_epoch=config_manager.get("trainer.check_val_every_n_epoch"),
        benchmark=True,
        default_root_dir=work_dir,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config_manager = ConfigManager(args.config)
    L.seed_everything(42, workers=True)

    train(config_manager)
