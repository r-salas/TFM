#
#
#   Train chest x-ray or not
#
#
import json
import os
import torch
import typer
import optuna
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from models import AutoEncoderModel
from utils import get_default_device

from torchvision import transforms
from optuna.exceptions import TrialPruned
from torch.utils.data import ConcatDataset, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from datasets import NIHChestXrays, ChestXRaysIndianaUniversity, CheXpert, UNIFESPXrayBodyPart, PadChest


def get_metrics_by_threshold(errors, threshold):
    total_false_samples = len(errors[errors["is_chest"] == 0])
    correct_false_samples = len(errors[(errors["is_chest"] == 0) & (errors["error"] > threshold)])
    false_acc = correct_false_samples / total_false_samples

    correct_true_samples = len(errors[(errors["is_chest"] == 1) & (errors["error"] <= threshold)])
    total_true_samples = len(errors[errors["is_chest"] == 1])
    true_acc = correct_true_samples / total_true_samples

    return true_acc, false_acc


def get_metrics_by_each_threshold(errors):
    min_threshold, max_threshold = errors["error"].min(), errors["error"].max()

    possible_thresholds = np.linspace(min_threshold, max_threshold, 10_000)
    metrics_by_threshold = []
    for threshold in possible_thresholds:
        true_acc, false_acc = get_metrics_by_threshold(errors, threshold)
        avg_acc = np.mean([true_acc, false_acc])
        metrics_by_threshold.append((true_acc, false_acc, avg_acc, threshold))
    metrics_by_threshold = pd.DataFrame(metrics_by_threshold, columns=["true_acc", "false_acc", "avg_acc", "threshold"])
    return metrics_by_threshold


def create_objective_func(num_epochs, device, num_workers, save_best, early_stopping):
    def objective(trial: optuna.Trial) -> float:
        loss_type = trial.suggest_categorical("loss_type", ["mae", "mse"])
        latent_vector_size = trial.suggest_int("latent_vector_size", 32, 1024)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)

        train_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # AP, PA
        train_dataset_1 = NIHChestXrays(transform=train_transform, undersampling=True, split="train")
        # Frontal, Lateral
        train_dataset_2 = ChestXRaysIndianaUniversity(transform=train_transform, undersampling=True, split="train")
        # More lateral
        train_dataset_3 = CheXpert(exclude_labels=[0, 1], transform=train_transform, split="train")

        train_dataset = ConcatDataset([train_dataset_1, train_dataset_2, train_dataset_3])

        # AP, PA
        val_dataset_1 = NIHChestXrays(transform=val_transform, undersampling=True, split="val")
        # Frontal, Lateral
        val_dataset_2 = ChestXRaysIndianaUniversity(transform=val_transform, undersampling=True, split="val")
        # More lateral
        val_dataset_3 = CheXpert(exclude_labels=[0, 1], transform=val_transform, split="val")

        val_dataset = ConcatDataset([val_dataset_1, val_dataset_2, val_dataset_3])

        train_dataloader = DataLoader(train_dataset, batch_size=256, num_workers=num_workers, persistent_workers=True)
        val_dataloader = DataLoader(val_dataset, batch_size=256, num_workers=num_workers, persistent_workers=True)

        autoencoder = AutoEncoderModel(latent_vector_size=latent_vector_size, learning_rate=learning_rate,
                                       loss_type=loss_type)

        tensorboard_logger = TensorBoardLogger(save_dir=os.path.join("logs", "chest_xray_or_not"))

        save_dirpath = os.path.join("runs", "chest_xray_or_not", str(tensorboard_logger.version))

        os.makedirs(save_dirpath, exist_ok=False)

        callbacks = []

        if save_best:
            callback = ModelCheckpoint(dirpath=save_dirpath, save_top_k=1, monitor="val_loss", save_last=False)
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

        trainer.fit(
            autoencoder,
            train_dataloader,
            val_dataloader
        )

        if trainer.interrupted:
            trial.study.stop()
            raise TrialPruned()

        predict_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        true_class_dataset = PadChest(exclude_labels=[0], transform=predict_transform)  # !AP-vertical
        no_class_dataset = UNIFESPXrayBodyPart(exclude_labels=[0, 3], transform=predict_transform)  # !abdomen & !chest

        true_class_dataloader = DataLoader(true_class_dataset, batch_size=128, num_workers=num_workers)
        no_class_dataloader = DataLoader(no_class_dataset, batch_size=128, num_workers=num_workers)

        true_class_errors, no_class_errors = trainer.predict(ckpt_path="best",
                                                             dataloaders=[true_class_dataloader, no_class_dataloader])

        true_class_pd = pd.DataFrame({
            "error": torch.cat(true_class_errors).detach().cpu().numpy(),
            "is_chest": 1
        })
        no_class_pd = pd.DataFrame({
            "error": torch.cat(no_class_errors).detach().cpu().numpy(),
            "is_chest": 0
        })
        errors_pd = pd.concat([true_class_pd, no_class_pd])

        metrics_by_threshold = get_metrics_by_each_threshold(errors_pd)

        best_metric_by_threshold = metrics_by_threshold.sort_values("avg_acc", ascending=False).iloc[0]

        with open(os.path.join(save_dirpath, "meta.json"), "w") as fp:
            json.dump({
                "threshold": best_metric_by_threshold["threshold"]
            }, fp)

        return best_metric_by_threshold["avg_acc"]

    return objective


def main(device: str = "auto", num_trials: int = 10, num_epochs: int = 25, save_best_every_trial: bool = True,
         early_stopping: bool = True, num_workers: int = 10):
    if device == "auto":
        device = get_default_device()

    study = optuna.create_study(direction="maximize")
    study.optimize(create_objective_func(num_epochs, device, num_workers, save_best_every_trial, early_stopping),
                   n_trials=num_trials)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    typer.run(main)
