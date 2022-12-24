#
#
#   Test chest X-ray or not
#
#
import json
import torch
import typer
import pandas as pd
import pytorch_lightning as pl

from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from utils import get_default_device
from train_chest_xray_or_not import AutoEncoderModel
from datasets import PadChest, MiniImagenet


def main(checkpoint_path, meta_path, device: str = "auto"):
    if device == "auto":
        device = get_default_device()

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    true_class_dataset = PadChest(exclude_labels=[0], transform=test_transform)  # exclude AP - vertical
    no_class_dataset = MiniImagenet(transform=test_transform)

    true_class_dataloader = DataLoader(true_class_dataset, batch_size=128, num_workers=10)
    no_class_dataloader = DataLoader(no_class_dataset, batch_size=128, num_workers=10)

    trainer = pl.Trainer(
        accelerator=device,
        logger=False
    )

    with open(meta_path) as fp:
        meta = json.load(fp)

    autoencoder = AutoEncoderModel.load_from_checkpoint(checkpoint_path)

    true_class_errors, no_class_errors = trainer.predict(autoencoder,
                                                         dataloaders=[true_class_dataloader, no_class_dataloader])

    true_class_errors_np = torch.cat(true_class_errors).detach().cpu().numpy()
    true_class_pd = pd.DataFrame({
        "error": true_class_errors_np,
        "is_chest": 1,
        "pred_is_chest": (true_class_errors_np < meta["threshold"]).astype(int)
    })
    no_class_errors_np = torch.cat(no_class_errors).detach().cpu().numpy()
    no_class_pd = pd.DataFrame({
        "error": no_class_errors_np,
        "is_chest": 0,
        "pred_is_chest": (no_class_errors_np < meta["threshold"]).astype(int)
    })
    global_pd = pd.concat([true_class_pd, no_class_pd])

    global_score = accuracy_score(global_pd["is_chest"], global_pd["pred_is_chest"])
    true_score = accuracy_score(true_class_pd["is_chest"], true_class_pd["pred_is_chest"])
    false_score = accuracy_score(no_class_pd["is_chest"], no_class_pd["pred_is_chest"])

    print(f"Global Acc: {global_score}")
    print(f"True Acc: {true_score}")
    print(f"False Acc: {false_score}")


if __name__ == "__main__":
    typer.run(main)
