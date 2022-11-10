"""
This module operates a user defined pre-trained model on a given dataset.
The results are saved in a npy file to be plotted in a separate script.

To read the data generated by this script, do the following:
```
predictions = np.fromfile("predictions.npy", dtype=np.float32)
predictions = predictions.reshape((-1, len(OUTPUT_COLUMN_NAMES)))
```
"""

import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader

from trained_models.DS4_input_Cp_output_Cp_distribution.ViT.src.models.models import (
    get_model,
    AvailableModels,
)
from trained_models.DS4_input_Cp_output_Cp_distribution.ViT.src.checkpoint import (
    load_checkpoint,
)
from datasets import DatasetConstructor, DatasetType
from running import DefaultRunner, UNetRunner

# hyperparameters
MODEL_NAME = AvailableModels.VIT
PATH_TO_SAVED_MODEL = (
    "trained_models/DS4_input_Cp_output_Cp_distribution/ViT/my_checkpoint.pth.tar"
)

NUM_WORKERS = 0

SELECTED_INDICES = []  # Leave this array empty to select all frames
DATASET_TYPE = DatasetType.DEFAULT

PATH_TO_DATASET = (
    "./dataset_unet.csv" if DATASET_TYPE == DatasetType.UNET else "./dataset.csv"
)

# INPUT_COLUMN_NAMES = ["images_unet"]
INPUT_COLUMN_NAMES = ["images_pressure"]
# INPUT_COLUMN_NAMES = ["images_z_vort"]
# INPUT_COLUMN_NAMES = ["images_vel_x", "images_vel_y"]
# OUTPUT_COLUMN_NAMES = ["Cl", "Cd", "Cm"]
OUTPUT_COLUMN_NAMES = [f"Cp{i}" for i in range(1, 301)]

# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def save_timestamps() -> None:
    try:
        dataset = pd.read_csv(PATH_TO_DATASET)
        times = dataset["times"]
        times.to_numpy().tofile("times.npy")
    except KeyError:
        print("This dataset contains no time information. Ignoring it...")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    model, transform_preprocess = get_model(
        MODEL_NAME,
        num_classes=len(OUTPUT_COLUMN_NAMES),
        num_input_images=len(INPUT_COLUMN_NAMES),
        is_inception=bool(MODEL_NAME == AvailableModels.INCEPTION),
    )

    load_checkpoint(model=model, device=device, filename=PATH_TO_SAVED_MODEL)

    constructor = DatasetConstructor(
        PATH_TO_DATASET,
        INPUT_COLUMN_NAMES,
        OUTPUT_COLUMN_NAMES,
        transform_preprocess,
    )
    dataset = constructor.create(DATASET_TYPE)

    if SELECTED_INDICES:
        dataset = torch.utils.data.Subset(dataset, SELECTED_INDICES)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    if DATASET_TYPE == DatasetType.UNET:
        runner = UNetRunner(model, dataloader, device)
    else:
        runner = DefaultRunner(model, dataloader, device)

    runner.run()


if __name__ == "__main__":
    main()
    save_timestamps()
