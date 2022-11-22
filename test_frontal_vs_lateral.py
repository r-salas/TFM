#
#
#   Test frontal vs lateral
#
#

try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    load_dotenv()

import typer
import pytorch_lightning as pl

from model import Model
from torchvision import transforms
from utils import get_default_device
from torch.utils.data import DataLoader
from torchvision.models import DenseNet121_Weights, ResNet50_Weights

from datasets import PadChest
from utils import pil_grayscale_to_rgb, radimagenet_transforms


def main(checkpoint_path, device: str = typer.Argument(get_default_device), num_workers: int = 10):
    model = Model.load_from_checkpoint(checkpoint_path)

    if model.transfer_learning_model == "DenseNet-121-ImageNet":
        transform = transforms.Compose([
            pil_grayscale_to_rgb,
            DenseNet121_Weights.IMAGENET1K_V1.transforms()
        ])
    elif model.transfer_learning_model == "ResNet-50-ImageNet":
        transform = transforms.Compose([
            pil_grayscale_to_rgb,
            ResNet50_Weights.IMAGENET1K_V2.transforms()
        ])
    elif model.transfer_learning_model in ("DenseNet-121-RadImageNet", "ResNet-50-RadImageNet"):
        transform = transforms.Compose([
            pil_grayscale_to_rgb,
            radimagenet_transforms
        ])
    else:
        raise NotImplementedError()

    dataset = PadChest(transform=transform, goal="frontal_vs_lateral", undersampling=True)

    loader = DataLoader(dataset, batch_size=64, num_workers=num_workers)

    trainer = pl.Trainer(
        accelerator=device,
        logger=False
    )

    trainer.test(model, loader)


if __name__ == "__main__":
    typer.run(main)
