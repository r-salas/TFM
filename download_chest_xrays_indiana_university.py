#
#
#   Download chest x-rays Indiana University
#
#

import os
import kaggle.api
import pandas as pd

from utils import extract_zip
from config import DOWNLOADS_ROOT_DIR, DATA_ROOT_DIR

data_path = os.path.join(DATA_ROOT_DIR, "raddar", "chest-xrays-indiana-university")
download_path = os.path.join(DOWNLOADS_ROOT_DIR, "raddar", "chest-xrays-indiana-university")

kaggle.api.dataset_download_files("raddar/chest-xrays-indiana-university", download_path, quiet=False)

extract_zip(os.path.join(download_path, "chest-xrays-indiana-university.zip"), data_path, [
    "indiana_projections.csv",
    "images/"
])

# Preprocess

meta = pd.read_csv(os.path.join(data_path, "indiana_projections.csv"))

name_to_label = {
    "Frontal": 0,
    "Lateral": 1
}
meta["label"] = meta["projection"].replace(name_to_label)
meta["path"] = os.path.join(data_path, "images", "images_normalized") + os.path.sep + meta["filename"]
meta[["path", "label"]].to_pickle(os.path.join(data_path, "indiana_projections.pkl"))
