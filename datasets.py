from enum import Enum, auto
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset


class DatasetType(Enum):
    UNET = auto()
    TEST = auto()
    DEFAULT = auto()


class DatasetConstructor:
    def __init__(
        self,
        csv_file: str,
        input_column_names: list,
        output_column_names: list,
        transform=None,
    ):
        self.csv_file = csv_file
        self.transform = transform
        self.input_column_names = input_column_names
        self.output_column_names = output_column_names

    def create(self, dataset_type: DatasetType) -> Dataset:
        if dataset_type == DatasetType.TEST:
            return TestDataset(
                self.csv_file,
                self.input_column_names,
                self.output_column_names,
                self.transform,
            )

        if dataset_type == DatasetType.UNET:
            return UNetImagesDataset(
                self.csv_file,
                self.input_column_names,
                self.transform,
            )

        return FluidDataset(
            self.csv_file,
            self.input_column_names,
            self.output_column_names,
            self.transform,
        )


class FluidDataset(Dataset):
    """
    Custom dataset that loads a CSV file that looks like this:

    images                 Cl     Cd     Cm    etc.
    /input/path/img1.jpg   1.234  0.123  1.23  .
    /input/path/img2.jpg   2.345  1.234  2.34  .
    .                      .
    .                      .

    Args:
        csv_file (string): Path to the csv file with image paths
        transform (callable, optional): Optional transform to be applied
        on a sample
    """

    def __init__(
        self,
        csv_file: str,
        input_column_names: list,
        output_column_names: list,
        transform=None,
    ):
        self.files = pd.read_csv(csv_file)
        self.transform = transform
        self.input_column_names = input_column_names
        self.output_column_names = output_column_names

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        input_img_paths = [
            self.files[column][index] for column in self.input_column_names
        ]
        outputs = [self.files[column][index] for column in self.output_column_names]

        input_images = []
        for path in input_img_paths:
            input_img = Image.open(path).convert("RGB")
            input_images.append(input_img)

        if self.transform:
            for i, image in enumerate(input_images):
                input_images[i] = self.transform(image)

        return input_images, outputs, input_img_paths


class TestDataset(FluidDataset):
    """
    This dataset aims to reproduce the test set used during training.

    Make sure that you are using the same CSV file from training when instantiating
    this class! By "the same" I mean the same cardinality and classes of simulations.
    This is necessary because this class implements the same train-valid-test split
    used when training. So, if the cardinality is different, the data you will get
    will not be the actual testset from the training procedure.
    """

    def __init__(
        self,
        csv_file: str,
        input_column_names: list,
        output_column_names: list,
        transform=None,
    ):
        super().__init__(
            csv_file,
            input_column_names,
            output_column_names,
            transform,
        )

        self.turn_it_into_testset()

    def turn_it_into_testset(self):
        train_proportion = 0.8
        valid_proportion = 0.1
        # test proportion is the remaining to 1

        indices = torch.randperm(
            len(self.files), generator=torch.Generator().manual_seed(42)
        )

        train_size = int(train_proportion * len(self.files))
        valid_size = int(valid_proportion * len(self.files))

        indices_test = indices[(train_size + valid_size) :].tolist()

        self.files = self.files.iloc[indices_test].reset_index(drop=True)


class UNetImagesDataset(Dataset):
    """
    The CSV file have a single column with the path to a given fluid property,
    which is designated by the header. Ex:

    property_john_doe
    /input/path/img1.jpg
    /input/path/img2.jpg
    .
    .

    Then, it returns the list of opened images and the list of paths to the images.

    Args:
        csv_file (string): Path to the csv file with image paths
        transform (callable, optional): Optional transform to be applied
            on a sample
    """

    def __init__(
        self,
        csv_file: str,
        input_column_names: list,
        transform=None,
    ):
        self.files = pd.read_csv(csv_file)
        self.transform = transform
        self.input_column_names = input_column_names

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        input_img_paths = [
            self.files[column][index] for column in self.input_column_names
        ]

        input_images = []
        for path in input_img_paths:
            input_img = Image.open(path).convert("RGB")
            input_images.append(input_img)

        if self.transform:
            for i, image in enumerate(input_images):
                input_images[i] = self.transform(image)

        return input_images, input_img_paths
