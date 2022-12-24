#
#
#   API
#
#
import json
import os
import torch
import fastdl
import joblib
import numpy as np

from PIL import Image
from torchvision import transforms
from typing import Union, Sequence, Optional

from config import DATA_ROOT_DIR
from common import get_test_transform
from utils import convert_I_to_L, get_default_device
from models import ClassificationModel, AutoEncoderModel

FRONTAL_LATERAL_CHECKPOINT_URL = "https://github.com/r-salas/TFM/releases/download/2022.11.24/frontal_vs_lateral.ckpt"
AP_PA_CHECKPOINT_URL = "https://github.com/r-salas/TFM/releases/download/2022.11.24/ap_vs_pa.ckpt"
CHEST_XRAY_OR_NOT_AUTOENCODER_CHECKPOINT_URL = "https://github.com/r-salas/TFM/releases/download/2022.12.24/chest-xray-or-not__autoencoder.ckpt"
CHEST_XRAY_OR_NOT_META_CHECKPOINT_URL = "https://github.com/r-salas/TFM/releases/download/2022.12.24/chest-xray-or-not__meta.json"


def _load_img_if_needed(img: Union[str, Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(img, str):
        return Image.open(img)
    elif isinstance(img, np.ndarray):
        return Image.fromarray(np.uint8(img))
    return img


def _to_grayscale(img: Image.Image) -> Image.Image:
    if img.mode == "I":
        return convert_I_to_L(img)
    else:
        return img.convert("L")


class _BaseClassifier:

    def __init__(self, checkpoint_path, device: str = "auto"):
        if device == "auto":
            device = get_default_device()

        self.device = device
        self.checkpoint_path = checkpoint_path

        self.model = ClassificationModel.load_from_checkpoint(checkpoint_path).eval().to(device)
        self.transform_fn = transforms.Compose([
            _load_img_if_needed,
            _to_grayscale,
            get_test_transform(self.model.transfer_learning_model)
        ])

    def predict_on_batch(self, imgs: Union[torch.Tensor, Sequence[Union[str, Image.Image, np.ndarray]]]):
        assert len(imgs) > 0

        if isinstance(imgs, torch.Tensor):
            tensor_imgs = imgs
        else:
            tensor_imgs = torch.stack([self.transform_fn(img) for img in imgs])

        tensor_imgs = tensor_imgs.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_imgs)
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().detach().numpy()

        return [{
            "label": np.argmax(probs[i]),
            "proba": probs[i]
        } for i in range(len(imgs))]

    def predict(self, img: Union[str, Image.Image]):
        return self.predict_on_batch([img])[0]


class FrontalLateralClassifier(_BaseClassifier):

    names = ["frontal", "lateral"]

    def __init__(self, checkpoint_path: Optional[str] = None):
        if checkpoint_path is None:
            checkpoint_path = fastdl.download(FRONTAL_LATERAL_CHECKPOINT_URL,
                                              dir_prefix=os.path.join(DATA_ROOT_DIR, "checkpoints", "2022.11.24"))

        super().__init__(checkpoint_path)


class APPAClassifier(_BaseClassifier):

    names = ["ap", "pa"]

    def __init__(self, checkpoint_path: Optional[str] = None):
        if checkpoint_path is None:
            checkpoint_path = fastdl.download(AP_PA_CHECKPOINT_URL,
                                              dir_prefix=os.path.join(DATA_ROOT_DIR, "checkpoints", "2022.11.24"))

        super().__init__(checkpoint_path)


class ChestXrayOrNotClassifier:

    names = ["no", "yes"]

    def __init__(self, autoencoder_checkpoint_path: Optional[str] = None,
                 meta_checkpoint_path: Optional[str] = None, device: str = "auto"):
        if device == "auto":
            device = get_default_device()

        if autoencoder_checkpoint_path is None:
            autoencoder_checkpoint_path = fastdl.download(CHEST_XRAY_OR_NOT_AUTOENCODER_CHECKPOINT_URL,
                                                          dir_prefix=os.path.join(DATA_ROOT_DIR, "checkpoints",
                                                                                  "2022.12.24"))

        if meta_checkpoint_path is None:
            meta_checkpoint_path = fastdl.download(CHEST_XRAY_OR_NOT_META_CHECKPOINT_URL,
                                                   dir_prefix=os.path.join(DATA_ROOT_DIR, "checkpoints", "2022.12.24"))

        self.device = device

        self.autoencoder = AutoEncoderModel.load_from_checkpoint(autoencoder_checkpoint_path).eval().to(device)
        with open(meta_checkpoint_path) as fp:
            self.meta = json.load(fp)

        self.transform_fn = transforms.Compose([
            _load_img_if_needed,
            _to_grayscale,
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        if self.autoencoder.loss_type == "mae":
            self.loss_fn = torch.nn.L1Loss(reduction="none")
        elif self.autoencoder.loss_type == "mse":
            self.loss_fn = torch.nn.MSELoss(reduction="none")
        else:
            raise NotImplementedError()

    def predict_on_batch(self, imgs: Union[torch.Tensor, Sequence[Union[str, Image.Image, np.ndarray]]]):
        assert len(imgs) > 0

        if isinstance(imgs, torch.Tensor):
            tensor_imgs = imgs
        else:
            tensor_imgs = torch.stack([self.transform_fn(img) for img in imgs])

        tensor_imgs = tensor_imgs.to(self.device)

        with torch.no_grad():
            tensor_pred_imgs = self.autoencoder(tensor_imgs)
            errors = self.loss_fn(tensor_pred_imgs, tensor_imgs)
            errors = torch.mean(errors, [1, 2, 3]).detach().cpu().numpy()
            pred_imgs = tensor_pred_imgs.detach().cpu().numpy()

        return [{
            "error": errors[i],
            "error_threshold": self.meta["threshold"],
            "label": int(errors[i] < self.meta["threshold"]),
            "img": Image.fromarray(pred_imgs[i].squeeze() * 255.0).convert("L")
        } for i in range(len(imgs))]

    def predict(self, img: Union[str, Image.Image]):
        return self.predict_on_batch([img])[0]
