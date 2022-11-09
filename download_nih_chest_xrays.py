#
#
#   Download NIH chest X-rays
#
#

import os
import glob
import kaggle.api
import pandas as pd

from tqdm import tqdm
from utils import extract_zip
from config import DOWNLOADS_ROOT_DIR, DATA_ROOT_DIR

data_path = os.path.join(DATA_ROOT_DIR, "nih-chest-xrays", "data")
download_path = os.path.join(DOWNLOADS_ROOT_DIR, "nih-chest-xrays", "data")

kaggle.api.dataset_download_files("nih-chest-xrays/data", download_path, quiet=False)

extract_zip(os.path.join(download_path, "data.zip"), data_path, [
    "Data_Entry_2017.csv",
    "images_001/",
    "images_002/",
    "images_003/",
    "images_004/",
    "images_005/",
    "images_006/",
    "images_007/",
    "images_008/",
    "images_009/",
    "images_010/",
    "images_011/",
])

# Preprocess

meta = pd.read_csv(os.path.join(data_path, "Data_Entry_2017.csv"), index_col="Image Index")

images_dirs = [
    "images_001",
    "images_002",
    "images_003",
    "images_004",
    "images_005",
    "images_006",
    "images_007",
    "images_008",
    "images_009",
    "images_010",
    "images_011",
    "images_012",
]

meta["path"] = None
for img_dir in tqdm(images_dirs):
    images_paths = glob.glob(os.path.join(data_path, img_dir, "images", "*.png"))
    for img_path in tqdm(images_paths, leave=False):
        name = os.path.basename(img_path)
        meta.loc[name, "path"] = img_path

meta.dropna(subset="path", inplace=True)

name_to_label = {
    "AP": 0,
    "PA": 1
}
meta["label"] = meta["View Position"].replace(name_to_label)

meta[["path", "label"]].to_pickle(os.path.join(data_path, "Data_Entry_2017.pkl"))
