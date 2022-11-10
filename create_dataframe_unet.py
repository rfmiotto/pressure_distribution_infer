"""
This module generates the Pandas Dataframe containing the path to every image
of the UNet predictions.

Here, the dataframe contains only those images, because it will be used as input
to the regression models, which were already trained. As the models are already
trained (it is just an inference operation), there is no need for the labels.

If you want to get the labels, then there is another script to generate the
dataframes with the labels and another script to infer the regression model.

The global variable `DATASET_BASE_DIRECTORY` specifies where to look for the
predicted images. It expects the following structure:

DATASET_BASE_DIRECTORY/
    outputs/
        0000.jpg
        0001.jpg
        ...
"""
import numpy as np
import pandas as pd

# This is being imported from the post-process module. Remember to set its path
# in PATHONPATH environment variable for this to work.
from file_searching import search_for_files

DATASET_BASE_DIRECTORY = "/home/miotto/Desktop/CNN_pytorch_coeffs_Unet/outputs"

SELECTED_INDICES = [1500, 2700, 4200]
SELECTED_INDICES = [100, 200, 300, 400]
SELECTED_INDICES = []


def create_dataframe() -> None:
    paths_to_images_pressure = np.asarray(
        search_for_files(DATASET_BASE_DIRECTORY, pattern="*.png")
    )

    dataframe = pd.DataFrame(columns=["images_unet"])

    dataframe["images_unet"] = paths_to_images_pressure

    dataframe.to_csv("dataset_unet.csv")


def select_few_snapshots() -> None:
    full_dataframe = pd.read_csv("dataset_unet.csv")

    dataframe_only_selected = full_dataframe.loc[SELECTED_INDICES]

    dataframe_only_selected.to_csv("dataset_unet.csv")


if __name__ == "__main__":
    create_dataframe()

    if SELECTED_INDICES:
        select_few_snapshots()
