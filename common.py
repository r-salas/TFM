#
#
#   Common
#
#

import os
import optuna
import pytorch_lightning as pl

from typing import Optional
from torchvision import transforms
from torch.utils.data import Subset
from optuna.exceptions import TrialPruned
from torch.utils.data import DataLoader, ConcatDataset
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchvision.models import DenseNet121_Weights, ResNet50_Weights

from models import ClassificationModel
from utils import pil_grayscale_to_rgb, radimagenet_transforms
from datasets import NIHChestXrays, CheXpert, ChestXRaysIndianaUniversity


def get_train_transform(transfer_learning_model):
    if transfer_learning_model == "DenseNet-121-ImageNet":
        return transforms.Compose([
            pil_grayscale_to_rgb,
            DenseNet121_Weights.IMAGENET1K_V1.transforms()
        ])
    elif transfer_learning_model == "ResNet-50-ImageNet":
        return transforms.Compose([
            pil_grayscale_to_rgb,
            ResNet50_Weights.IMAGENET1K_V2.transforms()
        ])
    elif transfer_learning_model in ("DenseNet-121-RadImageNet", "ResNet-50-RadImageNet"):
        return transforms.Compose([
            pil_grayscale_to_rgb,
            radimagenet_transforms
        ])
    else:
        raise NotImplementedError()


def get_test_transform(transfer_learning_model):
    if transfer_learning_model == "DenseNet-121-ImageNet":
        return transforms.Compose([
            pil_grayscale_to_rgb,
            DenseNet121_Weights.IMAGENET1K_V1.transforms()
        ])
    elif transfer_learning_model == "ResNet-50-ImageNet":
        return transforms.Compose([
            pil_grayscale_to_rgb,
            ResNet50_Weights.IMAGENET1K_V2.transforms()
        ])
    elif transfer_learning_model in ("DenseNet-121-RadImageNet", "ResNet-50-RadImageNet"):
        return transforms.Compose([
            pil_grayscale_to_rgb,
            radimagenet_transforms
        ])
    else:
        raise NotImplementedError()


def create_objective_func(dataset, num_epochs, device, max_train_samples, max_val_samples, num_workers, save_best,
                          early_stopping):
    def objective(trial: optuna.Trial) -> float:
        optimizer_type = trial.suggest_categorical("optimizer", ["SGD", "Adam", "RMSProp"])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)

        transfer_learning_model = trial.suggest_categorical("transfer_learning_model",
                                                            ["ResNet-50-ImageNet", "DenseNet-121-ImageNet",
                                                             "ResNet-50-RadImageNet", "DenseNet-121-RadImageNet"])
        transfer_learning_technique = trial.suggest_categorical("transfer_learning_technique",
                                                                ["freeze_all", "unfreeze_all"])

        train_transform = get_train_transform(transfer_learning_model)
        val_transform = get_test_transform(transfer_learning_model)

        # FIXME: split is missing test samples
        if dataset == "ap_vs_pa":
            train_dataset_1 = NIHChestXrays(transform=train_transform, split="train", undersampling=True)
            train_dataset_2 = CheXpert(transform=train_transform, goal="ap_vs_pa", split="train",
                                       undersampling=True)

            val_dataset_1 = NIHChestXrays(transform=val_transform, split="val", undersampling=True)
            val_dataset_2 = CheXpert(transform=val_transform, goal="ap_vs_pa", split="val", undersampling=True)

            train_dataset = ConcatDataset([train_dataset_1, train_dataset_2])
            val_dataset = ConcatDataset([val_dataset_1, val_dataset_2])
        elif dataset == "frontal_vs_lateral":
            train_dataset_1 = ChestXRaysIndianaUniversity(transform=train_transform, split="train", undersampling=True)
            train_dataset_2 = CheXpert(transform=train_transform, goal="frontal_vs_lateral", split="train",
                                       undersampling=True)

            val_dataset_1 = ChestXRaysIndianaUniversity(transform=val_transform, split="val", undersampling=True)
            val_dataset_2 = CheXpert(transform=val_transform, goal="frontal_vs_lateral", split="val",
                                     undersampling=True)

            train_dataset = ConcatDataset([train_dataset_1, train_dataset_2])
            val_dataset = ConcatDataset([val_dataset_1, val_dataset_2])
        else:
            raise NotImplementedError()

        if max_train_samples is not None:
            train_samples = min(max_train_samples, len(train_dataset))
            train_dataset = Subset(train_dataset, range(0, train_samples))

        if max_val_samples is not None:
            val_samples = min(max_val_samples, len(val_dataset))
            val_dataset = Subset(val_dataset, range(0, val_samples))

        train_loader = DataLoader(train_dataset, batch_size=64, num_workers=num_workers, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=64, num_workers=num_workers, persistent_workers=True)

        if dataset == "ap_vs_pa":
            names = ["AP", "PA"]
        elif dataset == "frontal_vs_lateral":
            names = ["Frontal", "Lateral"]
        else:
            raise NotImplementedError()

        model = ClassificationModel(optimizer_type, learning_rate, transfer_learning_model,
                                    transfer_learning_technique, names)

        tensorboard_logger = TensorBoardLogger(save_dir=os.path.join("logs", dataset))
        tensorboard_logger.log_hyperparams({
            "optimizer_type": optimizer_type,
            "learning_rate": learning_rate,
            "transfer_learning_model": transfer_learning_model,
            "transfer_learning_technique": transfer_learning_technique,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset)
        })

        callbacks = []

        if save_best:
            callback = ModelCheckpoint(dirpath=os.path.join("runs", dataset, str(tensorboard_logger.version)),
                                       save_top_k=1, monitor="val_loss", save_last=False)
            callbacks.append(callback)

        if early_stopping:
            callback = EarlyStopping(monitor="val_loss", mode="min")
            callbacks.append(callback)

        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator=device,
            reload_dataloaders_every_n_epochs=False,
            logger=tensorboard_logger,
            callbacks=callbacks
        )

        trainer.fit(model, train_loader, val_loader)

        if trainer.interrupted:
            trial.study.stop()
            raise TrialPruned()

        return trainer.callback_metrics["val_acc"].item()

    return objective


def optimize(dataset: str, num_epochs: int, device: str, num_trials: int, max_train_samples: Optional[int],
             max_val_samples: Optional[int], num_workers: int, early_stopping: bool, save_best_every_trial: bool):
    study = optuna.create_study(direction="maximize")
    study.optimize(create_objective_func(dataset, num_epochs, device, max_train_samples, max_val_samples,
                                         num_workers, save_best_every_trial, early_stopping),
                   n_trials=num_trials)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
