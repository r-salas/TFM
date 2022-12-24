#
#
#   Download NIH chest X-rays
#
#
import io
import os

import h5py
import zipfile
import kaggle.api
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image
from config import DOWNLOADS_ROOT_DIR, DATA_ROOT_DIR

data_path = os.path.join(DATA_ROOT_DIR, "nih-chest-xrays")
download_path = os.path.join(DOWNLOADS_ROOT_DIR, "nih-chest-xrays", "data")

os.makedirs(data_path, exist_ok=True)

kaggle.api.dataset_download_files("nih-chest-xrays/data", download_path, quiet=False)

with (zipfile.ZipFile(os.path.join(download_path, "data.zip")) as archive,
      h5py.File(os.path.join(data_path, "nih-chest-xrays.h5"), "w") as hdf5_file):
    meta_data = archive.read("Data_Entry_2017.csv")
    meta_bytes = io.BytesIO(meta_data)
    meta = pd.read_csv(meta_bytes, index_col="Image Index")

    img_paths = []
    for archive_file in archive.infolist():
        if archive_file.filename.endswith(".png"):
            img_paths.append(archive_file.filename)

    data = hdf5_file.create_dataset("data", shape=(len(img_paths), 256, 256), dtype="uint8")
    labels = hdf5_file.create_dataset("labels", shape=(len(img_paths)), dtype="int")

    for index, img_path in enumerate(tqdm(img_paths)):
        img_data = archive.read(img_path)
        img_bytes = io.BytesIO(img_data)

        pil_img = Image.open(img_bytes).convert("L")
        pil_img = pil_img.resize((256, 256), Image.Resampling.LANCZOS)

        data[index] = np.asarray(pil_img).astype("uint8")

        fname = os.path.basename(img_path)
        view_position = meta.loc[fname, "View Position"]

        view_position_to_label = {
            "AP": 0,
            "PA": 1
        }

        labels[index] = view_position_to_label[view_position]
