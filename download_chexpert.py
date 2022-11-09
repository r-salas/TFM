#
#
#   Download CheXpert
#
#

import os
import fastdl
import pandas as pd

from utils import extract_zip
from config import DOWNLOADS_ROOT_DIR, DATA_ROOT_DIR


fastdl.conf["default_dir_prefix"] = os.path.join(DOWNLOADS_ROOT_DIR, "CheXpert")

download_path = fastdl.download("https://us13.mailchimp.com/mctx/clicks?url=http%3A%2F%2Fdownload.cs."
                                "stanford.edu%2Fdeep%2FCheXpert-v1.0-small.zip&h=b3ac4027e89a042f68fe"
                                "5302e9638ff9d946b4d2e53944fd1965273e9c3cc2cd&v=1&xid=77408d5985&uid="
                                "55365305&pool=contact_facing&subject=CheXpert-v1.0%3A+Subscription+Confirmed",
                                fname="data.zip")

extract_dir = os.path.join(DATA_ROOT_DIR, "CheXpert")
extract_zip(download_path, extract_dir)

train_df = pd.read_csv(os.path.join(extract_dir, "CheXpert-v1.0-small", "train.csv"))
valid_df = pd.read_csv(os.path.join(extract_dir, "CheXpert-v1.0-small", "valid.csv"))

df = pd.concat([train_df, valid_df])

df["patient_id"] = df["Path"].str.extract(r"patient(\d+)")

df = df.drop_duplicates(["patient_id", "AP/PA"])

name_to_label = {
    "AP": 0,
    "PA": 1,
    "Lateral": 2
}
df["label"] = df["AP/PA"].fillna("Lateral").replace(name_to_label)
df = df[df["label"].isin([0, 1, 2])]
df["path"] = os.path.join(extract_dir) + os.path.sep + df["Path"]

df[["path", "label"]].to_pickle(os.path.join(extract_dir, "data.pkl"))
