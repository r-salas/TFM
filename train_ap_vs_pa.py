#
#
#   Train AP vs PA
#
#

import typer
import pytorch_lightning as pl

from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

from datasets import NIHChestXrays
from utils import get_default_device
from model import ClassificationModel


def main(epochs: int = 15, device: str = typer.Argument(get_default_device)):
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])

    train_dataset = NIHChestXrays(transform=train_transform, split="train")
    val_dataset = NIHChestXrays(transform=val_transform, split="val")

    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=8, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=2, persistent_workers=True)

    model = ClassificationModel()

    trainer = pl.Trainer(
        accelerator=device,
        max_epochs=epochs,
        logger=TensorBoardLogger(save_dir="ap_vs_pa_logs"),
        reload_dataloaders_every_n_epochs=False,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    typer.run(main)
