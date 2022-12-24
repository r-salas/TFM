#
#
#   Config
#
#

try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    load_dotenv()

import os

HERE = os.path.dirname(os.path.realpath(__file__))

DOWNLOADS_ROOT_DIR = os.getenv("DOWNLOADS_ROOT_DIR", os.path.join(HERE, "downloads"))
DATA_ROOT_DIR = os.getenv("DATA_ROOT_DIR", os.path.join(HERE, "data"))
