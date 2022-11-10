from typing import NoReturn
import itertools
import copy
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from trained_models.DS1_input_uv_output_coeffs.ViT.src.models.models import (
    get_model,
    AvailableModels,
)
from trained_models.DS1_input_uv_output_coeffs.ViT.src.checkpoint import (
    load_checkpoint,
)
from trained_models.DS1_input_uv_output_coeffs.ViT.src.loaders import (
    get_dataloaders,
)
from trained_models.DS1_input_uv_output_coeffs.ViT.src.running import (
    Runner,
)
from dataset_with_filenames import SimulationDataset
import texfig


# hyperparameters
MODEL_NAME = AvailableModels.VIT
BATCH_SIZE = 1
NUM_WORKERS = 0
PATH_TO_DATASET = "./dataset.csv"
# INPUT_COLUMN_NAMES = ["images_pressure"]
# INPUT_COLUMN_NAMES = ["images_z_vort"]
INPUT_COLUMN_NAMES = ["images_vel_x", "images_vel_y"]
OUTPUT_COLUMN_NAMES = ["Cl", "Cd", "Cm"]
# OUTPUT_COLUMN_NAMES = [f"Cp{i}" for i in range(1, 301)]
PATH_TO_SAVED_MODEL = (
    "trained_models/DS1_input_uv_output_coeffs/ViT/my_checkpoint.pth.tar"
)
NUM_TEST_DATA_TO_INFER = 100

# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def load_runner(model_name: AvailableModels):

    is_inception = bool(model_name == AvailableModels.INCEPTION)

    model, transform_preprocess = get_model(
        model_name,
        num_classes=len(OUTPUT_COLUMN_NAMES),
        num_input_images=len(INPUT_COLUMN_NAMES),
        is_inception=is_inception,
    )

    _, _, test_loader = get_dataloaders(
        PATH_TO_DATASET,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        input_column_names=INPUT_COLUMN_NAMES,
        output_column_names=OUTPUT_COLUMN_NAMES,
        transform_train=transform_preprocess,
        transform_valid=transform_preprocess,
        transform_test=transform_preprocess,
    )

    test_runner = Runner(test_loader, model, is_inception=is_inception)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_checkpoint(model=model, device=device, filename=PATH_TO_SAVED_MODEL)

    return test_runner, test_loader, transform_preprocess


def _plot_lift_drag_moment_test_samples(
    predicted_labels: np.ndarray, true_labels: np.ndarray
) -> NoReturn:
    plt.close("all")
    fig, ax = texfig.subplots(ratio=0.34, nrows=1, ncols=3)

    ax[0].plot(
        predicted_labels[:, 0],
        "o",
        markersize=2,
        color="royalblue",
        label="Predicted",
    )
    ax[0].plot(true_labels[:, 0], "o", markersize=2, color="tomato", label="True")
    ax[0].set_ylabel(r"$C_l$", rotation="horizontal", labelpad=1.0)
    ax[0].yaxis.set_label_coords(0, 1.02, transform=ax[0].transAxes)

    ax[1].plot(predicted_labels[:, 1], "o", markersize=2, color="royalblue")
    ax[1].plot(true_labels[:, 1], "o", markersize=2, color="tomato")
    ax[1].set_ylabel(r"$C_d$", rotation="horizontal", labelpad=1.0)
    ax[1].yaxis.set_label_coords(0, 1.02, transform=ax[1].transAxes)

    ax[2].plot(-predicted_labels[:, 2], "o", markersize=2, color="royalblue")
    ax[2].plot(-true_labels[:, 2], "o", markersize=2, color="tomato")
    ax[2].set_ylabel(r"$C_m$", rotation="horizontal", labelpad=1.0)
    ax[2].yaxis.set_label_coords(0, 1.02, transform=ax[2].transAxes)

    fig.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.2, wspace=0.25)

    texfig.make_fancy_axis(ax[0])
    texfig.make_fancy_axis(ax[1])
    texfig.make_fancy_axis(ax[2])

    texfig.savefig(
        "Predicted_coeff_from_all_simulations",
        dpi=1000,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close("all")


def _plot_lift_drag_moment(
    predicted_labels: np.ndarray, true_labels: np.ndarray, times: np.ndarray
) -> NoReturn:
    plt.close("all")
    fig, ax = texfig.subplots(ratio=0.34, nrows=1, ncols=3)

    ax[0].plot(times, predicted_labels[:, 0], color="royalblue", label="Predicted")
    ax[0].plot(times, true_labels[:, 0], color="tomato", label="True")
    ax[0].set_xlabel(r"$t$", rotation="horizontal", labelpad=1.0)
    ax[0].set_ylabel(r"$C_l$", rotation="horizontal", labelpad=1.0)
    ax[0].yaxis.set_label_coords(0, 1.02, transform=ax[0].transAxes)

    ax[1].plot(times, predicted_labels[:, 1], color="royalblue", label="Predicted")
    ax[1].plot(times, true_labels[:, 1], color="tomato", label="True")
    ax[1].set_xlabel(r"$t$", rotation="horizontal", labelpad=1.0)
    ax[1].set_ylabel(r"$C_d$", rotation="horizontal", labelpad=1.0)
    ax[1].yaxis.set_label_coords(0, 1.02, transform=ax[1].transAxes)

    ax[2].plot(times, -predicted_labels[:, 2], color="royalblue")
    ax[2].plot(times, -true_labels[:, 2], color="tomato")
    ax[2].set_xlabel(r"$t$", rotation="horizontal", labelpad=1.0)
    ax[2].set_ylabel(r"$C_m$", rotation="horizontal", labelpad=1.0)
    ax[2].yaxis.set_label_coords(0, 1.02, transform=ax[2].transAxes)

    fig.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.2, wspace=0.25)

    texfig.make_fancy_axis(ax[0])
    texfig.make_fancy_axis(ax[1])
    texfig.make_fancy_axis(ax[2])

    ax[0].legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 0.6),
        frameon=False,
        handlelength=0.5,
        handletextpad=0.2,
    )

    texfig.savefig(
        "Predicted_coeff",
        dpi=1000,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close("all")


def _plot_pressure_distribution(
    predicted_labels: np.ndarray, true_labels: np.ndarray, images: np.ndarray
) -> NoReturn:
    plt.close("all")
    fig, ax = texfig.subplots(
        ratio=1.1, nrows=3, ncols=2, gridspec_kw={"width_ratios": [2, 1]}
    )

    suction_side = np.linspace(0, 1, num=len(predicted_labels[0]))

    ax[0, 0].plot(
        suction_side, predicted_labels[0], color="royalblue", label="Predicted"
    )
    ax[0, 0].plot(
        suction_side,
        true_labels[0],
        color="tomato",
        label="True",
    )
    ax[0, 0].set_xlim(xmin=0, xmax=1)
    ax[0, 0].set_xticklabels([])
    ax[0, 0].set_ylabel(r"$C_p$", rotation="horizontal", labelpad=1.0)
    ax[0, 0].yaxis.set_label_coords(0, 1.02, transform=ax[0, 0].transAxes)

    ax[0, 1].imshow(plt.imread(images[0]))
    ax[0, 1].set_axis_off()

    ax[1, 0].plot(suction_side, predicted_labels[1], color="royalblue")
    ax[1, 0].plot(suction_side, true_labels[1], color="tomato")
    ax[1, 0].set_xlim(xmin=0, xmax=1)
    ax[1, 0].set_xticklabels([])

    ax[1, 1].imshow(plt.imread(images[1]))
    ax[1, 1].set_axis_off()

    ax[2, 0].plot(suction_side, predicted_labels[2], color="royalblue")
    ax[2, 0].plot(suction_side, true_labels[2], color="tomato")
    ax[2, 0].set_xlim(xmin=0, xmax=1)
    ax[2, 0].set_xlabel(r"$x/c$", labelpad=1.0)

    ax[2, 1].imshow(plt.imread(images[2]))
    ax[2, 1].set_axis_off()

    ax[0, 0].legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 0.6),
        frameon=False,
        handlelength=0.5,
        handletextpad=0.2,
    )

    @matplotlib.ticker.FuncFormatter
    def major_formatter(x, _):
        label = "{:.2f}".format(0 if round(x, 2) == 0 else x).rstrip("0").rstrip(".")
        return label

    @matplotlib.ticker.FuncFormatter
    def major_formatter2(x, _):
        label = "{:.5f}".format(0 if round(x, 5) == 0 else x).rstrip("0").rstrip(".")
        return label

    # Remove pointless zeros from ticks
    ax[2, 0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(major_formatter))

    fig.subplots_adjust(
        left=0.03, right=0.8, top=0.92, bottom=0.2, hspace=0.2, wspace=0.1
    )

    texfig.make_fancy_axis(ax[0, 0])
    texfig.make_fancy_axis(ax[0, 1])
    texfig.make_fancy_axis(ax[1, 0])
    texfig.make_fancy_axis(ax[1, 1])
    texfig.make_fancy_axis(ax[2, 0])
    texfig.make_fancy_axis(ax[2, 1])

    texfig.savefig(
        "Predicted_Cp_distribution", dpi=1000, bbox_inches="tight", pad_inches=0
    )


def _plot_pressure_distribution_compact(
    predicted_labels: np.ndarray, true_labels: np.ndarray, images: np.ndarray
) -> NoReturn:
    plt.close("all")
    fig, ax = texfig.subplots(
        ratio=0.32,
        nrows=1,
        ncols=3,
    )

    suction_side = np.linspace(0, 1, num=len(predicted_labels[0]))

    ax[0].plot(suction_side, predicted_labels[0], alpha=0.1, linewidth=6, color="white")
    ax[0].plot(suction_side, predicted_labels[0], alpha=0.2, linewidth=5, color="white")
    ax[0].plot(suction_side, predicted_labels[0], alpha=0.3, linewidth=4, color="white")
    ax[0].plot(suction_side, predicted_labels[0], alpha=0.4, linewidth=3, color="white")
    ax[0].plot(suction_side, predicted_labels[0], alpha=0.8, linewidth=2, color="white")
    ax[0].plot(suction_side, true_labels[0], alpha=0.1, linewidth=6, color="white")
    ax[0].plot(suction_side, true_labels[0], alpha=0.2, linewidth=5, color="white")
    ax[0].plot(suction_side, true_labels[0], alpha=0.3, linewidth=4, color="white")
    ax[0].plot(suction_side, true_labels[0], alpha=0.4, linewidth=3, color="white")
    ax[0].plot(suction_side, true_labels[0], alpha=0.8, linewidth=2, color="white")
    ax[0].plot(suction_side, predicted_labels[0], color="royalblue", label="Predicted")
    ax[0].plot(
        suction_side,
        true_labels[0],
        color="tomato",
        label="True",
    )
    ax[0].set_xlim(xmin=0, xmax=1)
    ax[0].set_ylabel(r"$C_p$", rotation="horizontal", labelpad=1.0)
    ax[0].yaxis.set_label_coords(0, 1.02, transform=ax[0].transAxes)
    ax[0].set_ylim([-6, 0])
    ax[0].set_xlabel(r"$x/c$", labelpad=1.0)
    ax[0].set_xticks([0, 0.5, 1])
    ax[0].set_facecolor("none")

    ax_img1 = ax[0].inset_axes([0.0, 0.0, 1.0, 1.0], zorder=-50)
    img = Image.open(images[0])
    img = img.rotate(8)
    img = img.crop((152, 100, 440, 340))
    ax_img1.imshow(img, aspect="equal", alpha=1.0)
    ax_img1.set_axis_off()

    ax[1].plot(suction_side, predicted_labels[1], alpha=0.1, linewidth=6, color="white")
    ax[1].plot(suction_side, predicted_labels[1], alpha=0.2, linewidth=5, color="white")
    ax[1].plot(suction_side, predicted_labels[1], alpha=0.3, linewidth=4, color="white")
    ax[1].plot(suction_side, predicted_labels[1], alpha=0.4, linewidth=3, color="white")
    ax[1].plot(suction_side, predicted_labels[1], alpha=0.8, linewidth=2, color="white")
    ax[1].plot(suction_side, true_labels[1], alpha=0.1, linewidth=6, color="white")
    ax[1].plot(suction_side, true_labels[1], alpha=0.2, linewidth=5, color="white")
    ax[1].plot(suction_side, true_labels[1], alpha=0.3, linewidth=4, color="white")
    ax[1].plot(suction_side, true_labels[1], alpha=0.4, linewidth=3, color="white")
    ax[1].plot(suction_side, true_labels[1], alpha=0.8, linewidth=2, color="white")
    ax[1].plot(suction_side, predicted_labels[1], color="royalblue", label="Predicted")
    ax[1].plot(suction_side, true_labels[1], color="tomato", label="True")
    ax[1].yaxis.set_ticklabels([])
    ax[1].set_ylim([-6, 0])
    ax[1].set_xlabel(r"$x/c$", labelpad=1.0)
    ax[1].set_xlim([0, 1])
    ax[1].set_xticks([0, 0.5, 1])
    ax[1].set_facecolor("none")

    ax_img2 = ax[1].inset_axes([0.0, 0.0, 1.0, 1.0], zorder=-50)
    img = Image.open(images[1])
    img = img.rotate(8)
    img = img.crop((152, 100, 440, 340))
    ax_img2.imshow(img, aspect="equal", alpha=1.0)
    ax_img2.set_axis_off()

    ax[2].plot(suction_side, predicted_labels[2], alpha=0.1, linewidth=6, color="white")
    ax[2].plot(suction_side, predicted_labels[2], alpha=0.2, linewidth=5, color="white")
    ax[2].plot(suction_side, predicted_labels[2], alpha=0.3, linewidth=4, color="white")
    ax[2].plot(suction_side, predicted_labels[2], alpha=0.4, linewidth=3, color="white")
    ax[2].plot(suction_side, predicted_labels[2], alpha=0.8, linewidth=2, color="white")
    ax[2].plot(suction_side, true_labels[2], alpha=0.1, linewidth=6, color="white")
    ax[2].plot(suction_side, true_labels[2], alpha=0.2, linewidth=5, color="white")
    ax[2].plot(suction_side, true_labels[2], alpha=0.3, linewidth=4, color="white")
    ax[2].plot(suction_side, true_labels[2], alpha=0.4, linewidth=3, color="white")
    ax[2].plot(suction_side, true_labels[2], alpha=0.8, linewidth=2, color="white")
    ax[2].plot(suction_side, predicted_labels[2], color="royalblue")
    ax[2].plot(suction_side, true_labels[2], color="tomato")
    ax[2].yaxis.set_ticklabels([])
    ax[2].set_ylim([-6, 0])
    ax[2].set_xlabel(r"$x/c$", labelpad=1.0)
    ax[2].set_xlim([0, 1])
    ax[2].set_xticks([0, 0.5, 1])
    ax[2].set_facecolor("none")

    ax_img3 = ax[2].inset_axes([0.0, 0.0, 1.0, 1.0], zorder=-50)
    img = Image.open(images[2])
    img = img.rotate(8)
    img = img.crop((152, 100, 440, 340))
    ax_img3.imshow(img, aspect="equal", alpha=1.0)
    ax_img3.set_axis_off()

    ax[0].legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 0.6),
        frameon=False,
        handlelength=0.5,
        handletextpad=0.2,
    )

    @matplotlib.ticker.FuncFormatter
    def major_formatter(x, _):
        label = "{:.2f}".format(0 if round(x, 2) == 0 else x).rstrip("0").rstrip(".")
        return label

    @matplotlib.ticker.FuncFormatter
    def major_formatter2(x, _):
        label = "{:.5f}".format(0 if round(x, 5) == 0 else x).rstrip("0").rstrip(".")
        return label

    # Remove pointless zeros from ticks
    ax[0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(major_formatter))
    ax[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(major_formatter))
    ax[2].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(major_formatter))

    fig.subplots_adjust(
        left=0.03, right=0.97, top=0.92, bottom=0.2, hspace=0.2, wspace=0.2
    )

    texfig.make_fancy_axis(ax[0])
    texfig.make_fancy_axis(ax[1])
    texfig.make_fancy_axis(ax[2])

    texfig.savefig(
        "Predicted_Cp_distribution", dpi=1000, bbox_inches="tight", pad_inches=0
    )


def load_different_dataset(transform) -> DataLoader:
    """
    Loads a dataset not necessarily identical to the one the model was trained on.
    This is useful to test our model on simulations with different flow or
    kinematical parameters.

    Notice that the dataset this method will load is specified in the global
    variable `PATH_TO_DATASET`.
    """
    dataset = SimulationDataset(
        PATH_TO_DATASET, INPUT_COLUMN_NAMES, OUTPUT_COLUMN_NAMES, transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader


def handle_plot_lift_drag_moment_test_samples() -> NoReturn:
    test_runner, test_loader, _ = load_runner(MODEL_NAME)

    test_loader_samples = itertools.islice(test_loader, NUM_TEST_DATA_TO_INFER)
    num_batches = NUM_TEST_DATA_TO_INFER
    progress_bar = tqdm(
        test_loader_samples, total=num_batches, desc="Infering test set"
    )

    predicted_samples = []
    target_samples = []

    for _, (inputs, targets) in enumerate(progress_bar):
        predictions = test_runner.infer(inputs)
        targets = torch.t(torch.stack(targets))

        predicted_samples.append(predictions.cpu().numpy())
        target_samples.append(targets)

    predicted_samples = np.concatenate(predicted_samples, axis=0)
    target_samples = np.concatenate(target_samples, axis=0)

    _plot_lift_drag_moment_test_samples(predicted_samples[::2], target_samples[::2])


def handle_plot_lift_drag_moment() -> NoReturn:
    test_runner, _, transform_preprocess = load_runner(MODEL_NAME)

    test_loader = load_different_dataset(transform_preprocess)
    num_batches = len(test_loader)
    progress_bar = tqdm(test_loader, total=num_batches, desc="Infering test set")

    predicted_samples = []
    target_samples = []

    for _, (inputs, targets, _) in enumerate(progress_bar):
        predictions = test_runner.infer(inputs)
        targets = torch.t(torch.stack(targets))

        predicted_samples.append(predictions.cpu().numpy())
        target_samples.append(targets)

    predicted_samples = np.concatenate(predicted_samples, axis=0)
    target_samples = np.concatenate(target_samples, axis=0)

    dataset = pd.read_csv(PATH_TO_DATASET)
    times = dataset["times"]

    predicted_samples.tofile("predictions.npy")
    target_samples.tofile("targets.npy")

    # beg = len(times) // 2
    # times = times - times[beg]
    beg = 0
    _plot_lift_drag_moment(
        predicted_samples[beg::1], target_samples[beg::1], times[beg::1]
    )


def handle_plot_pressure_distribution_from_test_set() -> NoReturn:
    test_runner, test_loader, _ = load_runner(MODEL_NAME)

    test_loader_samples = copy.deepcopy(test_loader)
    indices = [30, 120, 0]  # Select here the snapshot indices from test loader
    loader_indices = [
        test_loader_samples.sampler.data_source.indices[i] for i in indices
    ]
    test_loader_samples.sampler.data_source.indices = loader_indices

    # test_loader_samples = itertools.islice(test_loader, 3)
    num_batches = len(indices)
    progress_bar = tqdm(
        test_loader_samples, total=num_batches, desc="Infering test set"
    )

    input_image_paths = []
    predicted_samples = []
    target_samples = []

    for batch_index, (inputs, targets) in enumerate(progress_bar):
        predictions = test_runner.infer(inputs)
        targets = torch.t(torch.stack(targets))

        index = test_loader_samples.sampler.data_source.indices[batch_index]
        img_path = test_loader_samples.sampler.data_source.dataset.files.loc[index][
            "images_pressure"
        ]
        print(img_path)

        input_image_paths.append(img_path)
        predicted_samples.append(predictions.cpu().numpy())
        target_samples.append(targets)

    predicted_samples = np.concatenate(predicted_samples, axis=0)
    target_samples = np.concatenate(target_samples, axis=0)

    _plot_pressure_distribution_compact(
        predicted_samples, target_samples, input_image_paths
    )


def handle_plot_pressure_distribution() -> NoReturn:
    test_runner, _, transform_preprocess = load_runner(MODEL_NAME)

    selected_indices = [300, 900, 1300]
    # selected_indices = [200, 400, 600]

    dataset = SimulationDataset(
        PATH_TO_DATASET,
        INPUT_COLUMN_NAMES,
        OUTPUT_COLUMN_NAMES,
        transform=transform_preprocess,
    )

    filtered_dataset = torch.utils.data.Subset(dataset, selected_indices)

    dataloader = DataLoader(
        filtered_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    num_batches = len(dataloader)
    progress_bar = tqdm(dataloader, total=num_batches, desc="Infering test set")

    input_image_paths = []
    predicted_samples = []
    target_samples = []

    for _, (inputs, targets, img_paths) in enumerate(progress_bar):
        predictions = test_runner.infer(inputs)
        targets = torch.t(torch.stack(targets))

        input_image_paths.append(img_paths[0][0])
        predicted_samples.append(predictions.cpu().numpy())
        target_samples.append(targets)

    predicted_samples = np.concatenate(predicted_samples, axis=0)
    target_samples = np.concatenate(target_samples, axis=0)

    _plot_pressure_distribution_compact(
        predicted_samples, target_samples, input_image_paths
    )


if __name__ == "__main__":
    # handle_plot_lift_drag_moment_test_samples()
    # handle_plot_pressure_distribution_from_test_set()
    handle_plot_lift_drag_moment()
    # handle_plot_pressure_distribution()
