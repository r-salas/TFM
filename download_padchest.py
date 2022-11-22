#
#
#   Download data
#   https://b2drop.bsc.es/index.php/s/BIMCV-PadChest-FULL
#
#

import os
import fastdl
import pandas as pd

from utils import extract_zip, extract_gz
from config import DOWNLOADS_ROOT_DIR, DATA_ROOT_DIR


def url_for_padchest(fname):
    return f"https://b2drop.bsc.es/index.php/s/BIMCV-PadChest-FULL/download?path=%2F&files={fname}"


fastdl.conf["default_dir_prefix"] = os.path.join(DOWNLOADS_ROOT_DIR, "PadChest")

path = fastdl.download(url_for_padchest("PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz"),
                       fname="PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz",
                       extract=False)
extracted_path = extract_gz(path, os.path.join(DATA_ROOT_DIR, "PadChest"))

df = pd.read_csv(extracted_path, low_memory=False)

# Remove ResNet annotated images
df = df[df["MethodProjection"] == "Manual review of DICOM fields"]

# Remove images from duplicated patients
df = df.drop_duplicates("PatientID", keep="first")

# Keep only interesting projections
df = df[df["Projection"].isin(["AP", "AP_horizontal", "L", "PA"])]

bad_images = [
    "216840111366964013076187734852011291090445391_00-196-188.png",
    "216840111366964012819207061112010281134410801_00-129-131.png",
    "216840111366964012989926673512011151082430686_00-157-045.png",
    "216840111366964013076187734852011297123949023_01-001-162.png",
    "216840111366964013686042548532013283123624619_02-086-157.png",
    "216840111366964012989926673512011083134050913_00-168-009.png",
    "216840111366964012989926673512011074122523403_00-163-058.png",
    "216840111366964012373310883942009117084022290_00-064-025.png"
]
df = df[~df["ImageID"].isin(bad_images)]

images_dirs_to_download = df["ImageDir"].unique()
"""
for img_dir in images_dirs_to_download:
    path = fastdl.download(url_for_padchest(f"{img_dir}.zip"), extract=False, fname=f"{img_dir}.zip")
    images_to_extract = df[df["ImageDir"] == img_dir]["ImageID"].tolist()
    extract_zip(path, os.path.join(DATA_ROOT_DIR, "PadChest", str(img_dir)), images_to_extract)
"""

# Preprocess

name_to_label = {
    "AP": 0,
    "PA": 1,
    "AP_horizontal": 2,
    "L": 3
}
df["label"] = df["Projection"].replace(name_to_label)
df["path"] = (os.path.join(DATA_ROOT_DIR, "PadChest") + os.path.sep + df["ImageDir"].astype(str) + os.path.sep +
              df["ImageID"])
df[["path", "label"]].to_pickle(os.path.join(DATA_ROOT_DIR, "PadChest",
                                             "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.pkl"))
