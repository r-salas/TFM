#
#
#   Model
#
#

import torchmetrics
import pytorch_lightning as pl
import torch.nn.functional as F

from torch import optim, nn


class ClassificationModel(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.Flatten(),
            nn.Linear(238_144, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2)
        )

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        x = self.net(x)
        return x

    def training_step(self, batch, batch_idx):
        y_hat = self(batch[0])
        loss = F.cross_entropy(y_hat, batch[1])
        self.train_accuracy(y_hat, batch[1])
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outputs):
        self.log("train_acc", self.train_accuracy)

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch[0])
        loss = F.cross_entropy(y_hat, batch[1])
        self.val_accuracy(y_hat, batch[1])
        self.log("val_loss", loss)

    def validation_epoch_end(self, outputs):
        self.log("val_acc", self.val_accuracy)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
