import os
from typing import NoReturn
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import texfig

BASE_PATH = "./trained_models/DS1_input_uv_output_Cp_distribution"

EFFICIENT_TRAIN_EPOCH_LOSS = pd.read_csv(
    os.path.join(BASE_PATH, "EFFICIENT_train_epoch_loss.csv")
)
EFFICIENT_TRAIN_EPOCH_ACC = pd.read_csv(
    os.path.join(BASE_PATH, "EFFICIENT_train_epoch_acc.csv")
)
EFFICIENT_VALID_EPOCH_LOSS = pd.read_csv(
    os.path.join(BASE_PATH, "EFFICIENT_valid_epoch_loss.csv")
)
EFFICIENT_VALID_EPOCH_ACC = pd.read_csv(
    os.path.join(BASE_PATH, "EFFICIENT_valid_epoch_acc.csv")
)

INCEPTION_TRAIN_EPOCH_LOSS = pd.read_csv(
    os.path.join(BASE_PATH, "INCEPTION_train_epoch_loss.csv")
)
INCEPTION_TRAIN_EPOCH_ACC = pd.read_csv(
    os.path.join(BASE_PATH, "INCEPTION_train_epoch_acc.csv")
)
INCEPTION_VALID_EPOCH_LOSS = pd.read_csv(
    os.path.join(BASE_PATH, "INCEPTION_valid_epoch_loss.csv")
)
INCEPTION_VALID_EPOCH_ACC = pd.read_csv(
    os.path.join(BASE_PATH, "INCEPTION_valid_epoch_acc.csv")
)

RESNET_TRAIN_EPOCH_LOSS = pd.read_csv(
    os.path.join(BASE_PATH, "RESNET_train_epoch_loss.csv")
)
RESNET_TRAIN_EPOCH_ACC = pd.read_csv(
    os.path.join(BASE_PATH, "RESNET_train_epoch_acc.csv")
)
RESNET_VALID_EPOCH_LOSS = pd.read_csv(
    os.path.join(BASE_PATH, "RESNET_valid_epoch_loss.csv")
)
RESNET_VALID_EPOCH_ACC = pd.read_csv(
    os.path.join(BASE_PATH, "RESNET_valid_epoch_acc.csv")
)

VGG_TRAIN_EPOCH_LOSS = pd.read_csv(os.path.join(BASE_PATH, "VGG_train_epoch_loss.csv"))
VGG_TRAIN_EPOCH_ACC = pd.read_csv(os.path.join(BASE_PATH, "VGG_train_epoch_acc.csv"))
VGG_VALID_EPOCH_LOSS = pd.read_csv(os.path.join(BASE_PATH, "VGG_valid_epoch_loss.csv"))
VGG_VALID_EPOCH_ACC = pd.read_csv(os.path.join(BASE_PATH, "VGG_valid_epoch_acc.csv"))

VIT_TRAIN_EPOCH_LOSS = pd.read_csv(os.path.join(BASE_PATH, "VIT_train_epoch_loss.csv"))
VIT_TRAIN_EPOCH_ACC = pd.read_csv(os.path.join(BASE_PATH, "VIT_train_epoch_acc.csv"))
VIT_VALID_EPOCH_LOSS = pd.read_csv(os.path.join(BASE_PATH, "VIT_valid_epoch_loss.csv"))
VIT_VALID_EPOCH_ACC = pd.read_csv(os.path.join(BASE_PATH, "VIT_valid_epoch_acc.csv"))


def main() -> NoReturn:
    plt.close("all")
    fig, ax = texfig.subplots(ratio=0.34, nrows=1, ncols=2)

    (efficient,) = ax[0].plot(
        EFFICIENT_TRAIN_EPOCH_LOSS["Step"],
        EFFICIENT_TRAIN_EPOCH_LOSS["Value"],
        color="green",
        label="EfficientNet-B4",
        zorder=50,
    )
    ax[0].plot(
        EFFICIENT_VALID_EPOCH_LOSS["Step"],
        EFFICIENT_VALID_EPOCH_LOSS["Value"],
        color="green",
        label="EfficientNet-B4",
        alpha=0.2,
        # linestyle="dashed",
        # dashes=(2, 1),
    )
    (inception,) = ax[0].plot(
        INCEPTION_TRAIN_EPOCH_LOSS["Step"],
        INCEPTION_TRAIN_EPOCH_LOSS["Value"],
        color="red",
        label="Inception-V3",
        zorder=50,
    )
    ax[0].plot(
        INCEPTION_VALID_EPOCH_LOSS["Step"],
        INCEPTION_VALID_EPOCH_LOSS["Value"],
        color="red",
        label="Inception-V3",
        alpha=0.2,
        # linestyle="dashed",
        # dashes=(2, 1),
    )
    (vgg,) = ax[0].plot(
        VGG_TRAIN_EPOCH_LOSS["Step"],
        VGG_TRAIN_EPOCH_LOSS["Value"],
        color="peru",
        label="VGG-11-BN",
        zorder=50,
    )
    ax[0].plot(
        VGG_VALID_EPOCH_LOSS["Step"],
        VGG_VALID_EPOCH_LOSS["Value"],
        color="peru",
        label="VGG-11-BN",
        alpha=0.2,
        # linestyle="dashed",
        # dashes=(2, 1),
    )
    (resnet,) = ax[0].plot(
        RESNET_TRAIN_EPOCH_LOSS["Step"],
        RESNET_TRAIN_EPOCH_LOSS["Value"],
        color="darkorchid",
        label="ResNet-50",
        zorder=50,
    )
    ax[0].plot(
        RESNET_VALID_EPOCH_LOSS["Step"],
        RESNET_VALID_EPOCH_LOSS["Value"],
        color="darkorchid",
        label="ResNet-50",
        alpha=0.2,
        # linestyle="dashed",
        # dashes=(2, 1),
    )
    (vit,) = ax[0].plot(
        VIT_TRAIN_EPOCH_LOSS["Step"],
        VIT_TRAIN_EPOCH_LOSS["Value"],
        color="navy",
        label="ViT-B/16",
        zorder=50,
    )
    ax[0].plot(
        VIT_VALID_EPOCH_LOSS["Step"],
        VIT_VALID_EPOCH_LOSS["Value"],
        color="navy",
        label="ViT-B/16",
        alpha=0.2,
        # linestyle="dashed",
        # dashes=(2, 1),
    )

    ax[0].set_xlabel("Epoch", rotation="horizontal", labelpad=1.0)
    ax[0].set_ylabel("Loss", rotation="horizontal", labelpad=1.0)
    ax[0].yaxis.set_label_coords(0, 1.02, transform=ax[0].transAxes)
    ax[0].set_yscale("log")
    ax[0].yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=4))
    ax[0].yaxis.set_minor_locator(
        matplotlib.ticker.LogLocator(numticks=999, subs="auto")
    )

    ax[1].plot(
        EFFICIENT_TRAIN_EPOCH_ACC["Step"],
        EFFICIENT_TRAIN_EPOCH_ACC["Value"],
        color="green",
        label="EfficientNet-B4",
        zorder=50,
    )
    ax[1].plot(
        EFFICIENT_VALID_EPOCH_ACC["Step"],
        EFFICIENT_VALID_EPOCH_ACC["Value"],
        color="green",
        label="EfficientNet-B4",
        alpha=0.2,
        # linestyle="dashed",
        # dashes=(2, 1),
    )
    ax[1].plot(
        INCEPTION_TRAIN_EPOCH_ACC["Step"],
        INCEPTION_TRAIN_EPOCH_ACC["Value"],
        color="red",
        label="Inception-V3",
        zorder=50,
    )
    ax[1].plot(
        INCEPTION_VALID_EPOCH_ACC["Step"],
        INCEPTION_VALID_EPOCH_ACC["Value"],
        color="red",
        label="Inception-V3",
        alpha=0.2,
        # linestyle="dashed",
        # dashes=(2, 1),
    )
    ax[1].plot(
        VGG_TRAIN_EPOCH_ACC["Step"],
        VGG_TRAIN_EPOCH_ACC["Value"],
        color="peru",
        label="VGG-11-BN",
        zorder=50,
    )
    ax[1].plot(
        VGG_VALID_EPOCH_ACC["Step"],
        VGG_VALID_EPOCH_ACC["Value"],
        color="peru",
        label="VGG-11-BN",
        alpha=0.2,
        # linestyle="dashed",
        # dashes=(2, 1),
    )
    ax[1].plot(
        RESNET_TRAIN_EPOCH_ACC["Step"],
        RESNET_TRAIN_EPOCH_ACC["Value"],
        color="darkorchid",
        label="ResNet-50",
        zorder=50,
    )
    ax[1].plot(
        RESNET_VALID_EPOCH_ACC["Step"],
        RESNET_VALID_EPOCH_ACC["Value"],
        color="darkorchid",
        label="ResNet-50",
        alpha=0.2,
        # linestyle="dashed",
        # dashes=(2, 1),
    )
    ax[1].plot(
        VIT_TRAIN_EPOCH_ACC["Step"],
        VIT_TRAIN_EPOCH_ACC["Value"],
        color="navy",
        label="ViT-B/16",
        zorder=50,
    )
    ax[1].plot(
        VIT_VALID_EPOCH_ACC["Step"],
        VIT_VALID_EPOCH_ACC["Value"],
        color="navy",
        label="ViT-B/16",
        alpha=0.2,
        # linestyle="dashed",
        # dashes=(2, 1),
    )

    ax[1].set_xlabel("Epoch", rotation="horizontal", labelpad=1.0)
    ax[1].set_ylabel("MSE", rotation="horizontal", labelpad=1.0)
    ax[1].yaxis.set_label_coords(0, 1.02, transform=ax[1].transAxes)
    ax[1].set_yscale("log")
    ax[1].yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=4))
    ax[1].yaxis.set_minor_locator(
        matplotlib.ticker.LogLocator(numticks=999, subs="auto")
    )

    fig.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.2, wspace=0.25)

    texfig.make_fancy_axis(ax[0])
    texfig.make_fancy_axis(ax[1])

    lgnd1 = ax[0].legend(
        handles=[efficient, inception],
        loc="upper right",
        bbox_to_anchor=(0.72, 1.1),
        frameon=False,
        handlelength=0.5,
        handletextpad=0.2,
    )
    ax[0].add_artist(lgnd1)

    lgnd2 = ax[0].legend(
        handles=[vgg, resnet, vit],
        loc="upper right",
        bbox_to_anchor=(1.1, 1.1),
        frameon=False,
        handlelength=0.5,
        handletextpad=0.2,
    )
    ax[0].add_artist(lgnd2)

    texfig.savefig(
        "Training",
        dpi=1000,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close("all")


if __name__ == "__main__":
    main()
