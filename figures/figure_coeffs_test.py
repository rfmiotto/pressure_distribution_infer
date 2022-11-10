from typing import NoReturn
import numpy as np
import matplotlib.pyplot as plt
import texfig


def plot_lift_drag_moment_test_samples(
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
        "Predicted_coeffs",
        dpi=1000,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close("all")


def main() -> NoReturn:
    predictions = np.fromfile(
        "./outputs/predictions_DS1_Cp_coeffs_train.npy", dtype=np.float32
    ).reshape([-1, 3])

    targets = np.fromfile("./outputs/targets_DS1_train.npy", dtype=np.float64).reshape(
        [-1, 3]
    )

    plot_lift_drag_moment_test_samples(predictions[:50], targets[:50])


if __name__ == "__main__":
    main()
