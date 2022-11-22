#
#
#   Optimize hyperparameters for AP vs PA
#
#

try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    load_dotenv()

import typer

from common import optimize
from typing import Optional
from utils import get_default_device


def main(num_epochs: int = 25, device: str = typer.Argument(get_default_device), num_trials: int = 20,
         max_train_samples: Optional[int] = None, max_val_samples: Optional[int] = None, num_workers: int = 10,
         early_stopping: bool = True, save_best_every_trial: bool = True):
    optimize("ap_vs_pa", num_epochs, device, num_trials, max_train_samples, max_val_samples,
             num_workers, early_stopping, save_best_every_trial)


if __name__ == "__main__":
    typer.run(main)
