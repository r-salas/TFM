#
#
#   Convert RadImageNet models
#
#

import os
import tf2onnx

from config import DATA_ROOT_DIR
from tensorflow.keras.applications import ResNet50, DenseNet121

image_size = 256

models_dir = os.path.join(DATA_ROOT_DIR, "radimagenet")

input_path = os.path.join(models_dir, "RadImageNet-ResNet50_notop.h5")
resnet50 = ResNet50(weights=input_path, input_shape=(image_size, image_size, 3), include_top=False, pooling='avg')

output_path = os.path.join(models_dir, "RadImageNet-ResNet50_notop.onnx")
tf2onnx.convert.from_keras(resnet50, output_path=output_path)

input_path = os.path.join(models_dir, "RadImageNet-DenseNet121_notop.h5")
densenet121 = DenseNet121(weights=input_path, input_shape=(image_size, image_size, 3), include_top=False, pooling='avg')

output_path = os.path.join(models_dir, "RadImageNet-DenseNet121_notop.onnx")
tf2onnx.convert.from_keras(densenet121, output_path=output_path)
