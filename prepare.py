#
#
#   Prepare datasets
#
#

import os
import glob
import pandas as pd

from tqdm import tqdm

from config import RAW_DATA_ROOT_DIR, PREPARED_DATA_ROOT_DIR

# NIH Chest X-Rays

output_dir = os.path.join(PREPARED_DATA_ROOT_DIR, "nih-chest-xrays")
os.makedirs(output_dir, exist_ok=True)

data_dir = os.path.join(RAW_DATA_ROOT_DIR, "nih-chest-xrays", "data")
prepared_data_dir = os.path.join(PREPARED_DATA_ROOT_DIR, "nih-chest-xrays", "data")

os.makedirs(prepared_data_dir, exist_ok=True)

meta = pd.read_csv(os.path.join(data_dir, "Data_Entry_2017.csv"), index_col="Image Index")

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
    images_paths = glob.glob(os.path.join(data_dir, img_dir, "images", "*.png"))
    for img_path in tqdm(images_paths, leave=False):
        name = os.path.basename(img_path)
        meta.loc[name, "path"] = img_path

meta.dropna(subset="path", inplace=True)

name_to_label = {
    "AP": 0,
    "PA": 1
}
meta["label"] = meta["View Position"].replace(name_to_label)

meta[["path", "label"]].to_pickle(os.path.join(prepared_data_dir, "Data_Entry_2017.pkl"))

# Chest X-Rays Indiana University

data_dir = os.path.join(RAW_DATA_ROOT_DIR, "raddar", "chest-xrays-indiana-university")
prepared_data_dir = os.path.join(PREPARED_DATA_ROOT_DIR, "raddar", "chest-xrays-indiana-university")

os.makedirs(prepared_data_dir, exist_ok=True)

meta = pd.read_csv(os.path.join(data_dir, "indiana_projections.csv"))

name_to_label = {
    "Frontal": 0,
    "Lateral": 1
}
meta["label"] = meta["projection"].replace(name_to_label)
meta["path"] = os.path.join(data_dir, "images", "images_normalized") + os.path.sep + meta["filename"]
meta[["path", "label"]].to_pickle(os.path.join(prepared_data_dir, "indiana_projections.pkl"))
