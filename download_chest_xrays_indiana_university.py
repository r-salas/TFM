#
#
#   Download chest x-rays Indiana University
#
#

import os
import io
import h5py
import zipfile
import kaggle.api
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from config import DOWNLOADS_ROOT_DIR, DATA_ROOT_DIR

data_path = os.path.join(DATA_ROOT_DIR, "chest-xrays-indiana-university")
download_path = os.path.join(DOWNLOADS_ROOT_DIR, "raddar", "chest-xrays-indiana-university")

os.makedirs(data_path, exist_ok=True)

kaggle.api.dataset_download_files("raddar/chest-xrays-indiana-university", download_path, quiet=False)

with (zipfile.ZipFile(os.path.join(download_path, "chest-xrays-indiana-university.zip")) as archive,
      h5py.File(os.path.join(data_path, "chest-xrays-indiana-university.h5"), "w") as hdf5_file):
    meta_data = archive.read("indiana_projections.csv")
    meta_bytes = io.BytesIO(meta_data)
    meta = pd.read_csv(meta_bytes)

    name_to_label = {
        "Frontal": 0,
        "Lateral": 1
    }
    meta["label"] = meta["projection"].replace(name_to_label)
    meta["path"] = os.path.join("images", "images_normalized") + os.path.sep + meta["filename"]

    data = hdf5_file.create_dataset("data", shape=(len(meta), 256, 256), dtype="uint8")
    labels = hdf5_file.create_dataset("labels", shape=(len(meta)), dtype="int")

    for index, row in enumerate(tqdm(meta.itertuples(index=False), total=len(meta))):
        img_data = archive.read(row.path)
        img_bytes = io.BytesIO(img_data)

        pil_img = Image.open(img_bytes).convert("L")
        pil_img = pil_img.resize((256, 256), Image.Resampling.LANCZOS)

        data[index] = np.asarray(pil_img).astype("uint8")
        labels[index] = row.label
