from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


class Runner(ABC):
    @abstractmethod
    def run(self):
        """Makes the network infer the entire dataset"""


class UNetRunner(Runner):
    def __init__(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
    ):
        self.model = model
        self.loader = loader
        self.device = device
        self.model.to(device=device)

    def run(self) -> None:
        progress_bar = tqdm(self.loader, total=len(self.loader))

        predicted_samples = []

        self.model.eval()
        with torch.no_grad():
            for _, (inputs, _) in enumerate(progress_bar):

                inputs = torch.hstack(inputs)
                inputs = inputs.to(device=self.device)

                predictions = self.model(inputs)

                predicted_samples.append(predictions.cpu().numpy())

        predicted_samples = np.concatenate(predicted_samples, axis=0)

        predicted_samples.tofile("predictions.npy")


class DefaultRunner(Runner):
    def __init__(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
    ):
        self.model = model
        self.loader = loader
        self.device = device
        self.model.to(device=device)

    def run(self) -> None:
        progress_bar = tqdm(self.loader, total=len(self.loader))

        predicted_samples = []
        target_samples = []

        self.model.eval()
        with torch.no_grad():
            for _, (inputs, targets, _) in enumerate(progress_bar):

                inputs = torch.hstack(inputs)
                inputs = inputs.to(device=self.device)
                targets = torch.t(torch.stack(targets))

                predictions = self.model(inputs)

                predicted_samples.append(predictions.cpu().numpy())
                target_samples.append(targets)

        predicted_samples = np.concatenate(predicted_samples, axis=0)
        target_samples = np.concatenate(target_samples, axis=0)

        predicted_samples.tofile("predictions.npy")
        target_samples.tofile("targets.npy")

        # To retrieve the image paths of a test set, uncomment the line below:
        self.loader.dataset.files.to_csv("loader.csv")
