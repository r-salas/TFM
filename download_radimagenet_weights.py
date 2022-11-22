#
#
#   Download RadImageNet weights
#
#

import os
import gdown

from config import DATA_ROOT_DIR

densenet121_url = "https://drive.google.com/file/d/1xZKo4n5m6iSHPFNUAyfpvuANrRqBB6eH/view?usp=share_link"
resnet50_url = "https://drive.google.com/file/d/1_AB0p3iC5bSiDSTAjxt0FSl3WA8aIHap/view?usp=share_link"

models_dir = os.path.join(DATA_ROOT_DIR, "radimagenet")

os.makedirs(models_dir, exist_ok=True)

gdown.download(id="1xZKo4n5m6iSHPFNUAyfpvuANrRqBB6eH",
               output=os.path.join(models_dir, "RadImageNet-DenseNet121_notop.h5"))
gdown.download(id="1_AB0p3iC5bSiDSTAjxt0FSl3WA8aIHap",
               output=os.path.join(models_dir, "RadImageNet-ResNet50_notop.h5"))
