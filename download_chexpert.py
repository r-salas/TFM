#
#
#   Download CheXpert
#
#

import io
import os
import h5py
import fastdl
import zipfile
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from config import DOWNLOADS_ROOT_DIR, DATA_ROOT_DIR


fastdl.conf["default_dir_prefix"] = os.path.join(DOWNLOADS_ROOT_DIR, "CheXpert")

download_path = fastdl.download("https://us13.mailchimp.com/mctx/clicks?url=http%3A%2F%2Fdownload.cs."
                                "stanford.edu%2Fdeep%2FCheXpert-v1.0-small.zip&h=b3ac4027e89a042f68fe"
                                "5302e9638ff9d946b4d2e53944fd1965273e9c3cc2cd&v=1&xid=77408d5985&uid="
                                "55365305&pool=contact_facing&subject=CheXpert-v1.0%3A+Subscription+Confirmed",
                                fname="data.zip")

with (zipfile.ZipFile(download_path) as archive,
      h5py.File(os.path.join(DATA_ROOT_DIR, "chexpert", "chexpert.h5"), "w") as hdf5_file):
    train_meta_data = archive.read(os.path.join("CheXpert-v1.0-small", "train.csv"))
    valid_meta_data = archive.read(os.path.join("CheXpert-v1.0-small", "valid.csv"))

    train_meta_bytes = io.BytesIO(train_meta_data)
    valid_meta_bytes = io.BytesIO(valid_meta_data)

    train_meta = pd.read_csv(train_meta_bytes)
    valid_meta = pd.read_csv(valid_meta_bytes)

    meta = pd.concat([train_meta, valid_meta])

    meta["patient_id"] = meta["Path"].str.extract(r"patient(\d+)")

    meta = meta.drop_duplicates(["patient_id", "AP/PA"])

    name_to_label = {
        "AP": 0,
        "PA": 1,
        "Lateral": 2
    }
    meta["label"] = meta["AP/PA"].fillna("Lateral").replace(name_to_label)
    meta = meta[meta["label"].isin([0, 1, 2])]

    data = hdf5_file.create_dataset("data", shape=(len(train_meta) + len(valid_meta), 256, 256), dtype="uint8")
    labels = hdf5_file.create_dataset("labels", shape=(len(train_meta) + len(valid_meta)), dtype="int")

    for index, row in enumerate(tqdm(meta.itertuples(index=False), total=len(meta))):
        img_data = archive.read(row.Path)
        img_bytes = io.BytesIO(img_data)

        pil_img = Image.open(img_bytes).convert("L")
        pil_img = pil_img.resize((256, 256), Image.Resampling.LANCZOS)

        data[index] = np.asarray(pil_img).astype("uint8")
        labels[index] = row.label
