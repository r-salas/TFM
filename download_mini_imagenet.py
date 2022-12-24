#
#
#   Download Mini-ImageNet
#
#
import io
import os
import h5py
import kaggle
import zipfile
import numpy as np

from PIL import Image
from tqdm import tqdm
from config import DATA_ROOT_DIR, DOWNLOADS_ROOT_DIR

data_path = os.path.join(DATA_ROOT_DIR, "mini-imagenet")
download_path = os.path.join(DOWNLOADS_ROOT_DIR, "lijiyu", "imagenet")

os.makedirs(data_path, exist_ok=True)

kaggle.api.dataset_download_files("lijiyu/imagenet", download_path, quiet=False)

with (zipfile.ZipFile(os.path.join(download_path, "imagenet.zip")) as archive,
      h5py.File(os.path.join(data_path, "mini-imagenet.h5"), "w") as hdf5_file):
    archive_files = archive.infolist()

    data = hdf5_file.create_dataset("data", shape=(len(archive_files), 256, 256), dtype="uint8")

    for index, archive_file in enumerate(tqdm(archive_files, total=len(archive_files))):
        img_data = archive.read(archive_file)
        img_bytes = io.BytesIO(img_data)

        pil_img = Image.open(img_bytes).convert("L")
        pil_img = pil_img.resize((256, 256), Image.Resampling.LANCZOS)

        data[index] = np.asarray(pil_img).astype("uint8")
