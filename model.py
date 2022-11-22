#
#
#   Model
#
#

import os
import onnx2torch
import torchmetrics
import pytorch_lightning as pl
import torch.nn.functional as F

from torch import optim, nn
from config import DATA_ROOT_DIR
from torchvision.models import densenet121, resnet50, DenseNet121_Weights, ResNet50_Weights


class Model(pl.LightningModule):

    def __init__(self, optimizer_type, learning_rate, transfer_learning_model, transfer_learning_technique):
        super().__init__()

        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.transfer_learning_model = transfer_learning_model
        self.transfer_learning_technique = transfer_learning_technique

        self.save_hyperparameters()

        radimagenet_models_dir = os.path.join(DATA_ROOT_DIR, "radimagenet")

        if transfer_learning_model == "DenseNet-121-ImageNet":
            base_model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        elif transfer_learning_model == "ResNet-50-ImageNet":
            base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        elif transfer_learning_model == "DenseNet-121-RadImageNet":
            base_model = onnx2torch.convert(os.path.join(radimagenet_models_dir, "RadImageNet-DenseNet121_notop.onnx"))
        elif transfer_learning_model == "ResNet-50-RadImageNet":
            base_model = onnx2torch.convert(os.path.join(radimagenet_models_dir, "RadImageNet-ResNet50_notop.onnx"))
        else:
            raise NotImplementedError()

        if transfer_learning_technique == "unfreeze_all":
            for param in base_model.parameters():
                param.requires_grad = True
        elif transfer_learning_technique == "freeze_all":
            for param in base_model.parameters():
                param.requires_grad = False
        else:
            raise NotImplementedError()

        def build_classifier(in_features):
            return nn.Sequential(
                nn.Linear(in_features, 2048),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(2048, 256),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(256, 2)
            )

        if transfer_learning_model == "DenseNet-121-ImageNet":
            base_model.classifier = build_classifier(base_model.classifier.in_features)
        elif transfer_learning_model == "ResNet-50-ImageNet":
            base_model.fc = build_classifier(base_model.fc.in_features)
        elif transfer_learning_model == "DenseNet-121-RadImageNet":
            base_model = nn.Sequential(
                base_model,
                build_classifier(1024)
            )
        elif transfer_learning_model == "ResNet-50-RadImageNet":
            base_model = nn.Sequential(
                base_model,
                build_classifier(2048)
            )
        else:
            raise NotImplementedError()

        self.net = base_model

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

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

    def test_step(self, batch, batch_idx):
        y_hat = self(batch[0])
        loss = F.cross_entropy(y_hat, batch[1])
        self.test_accuracy(y_hat, batch[1])
        self.log("test_loss", loss)

    def test_epoch_end(self, outputs):
        self.log("test_acc", self.test_accuracy)

    def configure_optimizers(self):
        if self.optimizer_type == "Adam":
            return optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "SGD":
            return optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "RMSProp":
            return optim.RMSprop(self.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError()
