#
#
#   Download data
#   https://b2drop.bsc.es/index.php/s/BIMCV-PadChest-FULL
#
#

import io
import os
import gzip
import h5py
import zipfile
import fastdl
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from utils import convert_I_to_L
from config import DOWNLOADS_ROOT_DIR, DATA_ROOT_DIR


def url_for_padchest(fname):
    return f"https://b2drop.bsc.es/index.php/s/BIMCV-PadChest-FULL/download?path=%2F&files={fname}"


fastdl.conf["default_dir_prefix"] = os.path.join(DOWNLOADS_ROOT_DIR, "PadChest")

path = fastdl.download(url_for_padchest("PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz"),
                       fname="PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz",
                       extract=False)

with gzip.open(path) as archive:
    meta_data = archive.read()
    meta_bytes = io.BytesIO(meta_data)
    meta = pd.read_csv(meta_bytes, low_memory=False)

# Remove ResNet annotated images
meta = meta[meta["MethodProjection"] == "Manual review of DICOM fields"]

# Keep only interesting projections
meta = meta[meta["Projection"].isin(["AP", "AP_horizontal", "L", "PA"])]

bad_images = [
    "216840111366964013076187734852011291090445391_00-196-188.png",
    "216840111366964012819207061112010281134410801_00-129-131.png",
    "216840111366964012989926673512011151082430686_00-157-045.png",
    "216840111366964013076187734852011297123949023_01-001-162.png",
    "216840111366964013686042548532013283123624619_02-086-157.png",
    "216840111366964012989926673512011083134050913_00-168-009.png",
    "216840111366964012989926673512011074122523403_00-163-058.png",
    "216840111366964012373310883942009117084022290_00-064-025.png",
    "267593312931260619142226905522973356507_dfimnx.png",
    "315366742299617648846204862485207109881_beilul.png",
    "83271681649958405552864211916792219179_6anlem.png",
    "216840111366964013076187734852011262112534962_00-134-007.png",
    "216840111366964012283393834152009033140208626_00-059-118.png",
    "216840111366964013686042548532013208193054515_02-026-007.png",
    "216840111366964013686042548532013190085910247_02-033-156.png",
    "216840111366964013590140476722013058110301622_02-056-111.png",
    "216840111366964013590140476722013057095145100_02-065-148.png",
    "216840111366964013962490064942014134093945580_01-178-104.png",
    "216840111366964012819207061112010307142602253_04-014-084.png"
]
meta = meta[~meta["ImageID"].isin(bad_images)]

# Remove images from duplicated patients
meta = meta.drop_duplicates(["PatientID", "Projection"], keep="first")

projection_to_label = {
    "AP": 0,
    "PA": 1,
    "AP_horizontal": 2,
    "L": 3
}
meta["label"] = meta["Projection"].replace(projection_to_label)

meta["path"] = meta["ImageDir"].astype(str) + os.path.sep + meta["ImageID"]

images_dirs_to_download = meta["ImageDir"].unique()

data_path = os.path.join(DATA_ROOT_DIR, "padchest")

os.makedirs(data_path, exist_ok=True)

with (h5py.File(os.path.join(data_path, "padchest.h5"), "w") as hdf5_file):
    data = hdf5_file.create_dataset("data", shape=(len(meta), 256, 256), dtype="uint8")
    labels = hdf5_file.create_dataset("labels", shape=(len(meta)), dtype="int")

    for img_dir in tqdm(images_dirs_to_download):
        path = fastdl.download(url_for_padchest(f"{img_dir}.zip"), extract=False, fname=f"{img_dir}.zip")

        img_dir_meta = meta[meta["ImageDir"] == img_dir]
        for index, row in enumerate(tqdm(img_dir_meta.itertuples(), leave=False, total=len(img_dir_meta))):
            try:
                pil_img = Image.open(os.path.join("/Users/ruben/Documents/college/TFM/TFM/data/PadChest", str(img_dir),
                                                  row.ImageID))
                pil_img = convert_I_to_L(pil_img)
                pil_img = pil_img.resize((256, 256), Image.Resampling.LANCZOS)
            except:
                continue

            data[index] = np.asarray(pil_img).astype("uint8")
            labels[index] = row.label

"""
        with zipfile.ZipFile(path) as archive:
            img_dir_meta = meta[meta["ImageDir"] == img_dir]
            for index, row in enumerate(tqdm(img_dir_meta.itertuples(), leave=False, total=len(img_dir_meta))):
                img_data = archive.read(row.ImageID)
                img_bytes = io.BytesIO(img_data)

                pil_img = Image.open(img_bytes)
                pil_img = convert_I_to_L(pil_img)
                pil_img = pil_img.resize((256, 256), Image.Resampling.LANCZOS)

                data[index] = np.asarray(pil_img).astype("uint8")
                labels[index] = labels[row.label]
"""
