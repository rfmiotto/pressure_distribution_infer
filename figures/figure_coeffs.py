import numpy as np
import matplotlib.pyplot as plt
import texfig


def plot_single_prediction(
    true_labels: np.ndarray, predicted_labels: np.ndarray, times: np.ndarray
) -> None:
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


def main() -> None:
    predictions = np.fromfile(
        "./outputs/predictions_DS2_17_Cp_coeffs.npy", dtype=np.float32
    ).reshape([-1, 3])

    targets = np.fromfile("./outputs/targets_17_coeffs.npy", dtype=np.float64).reshape(
        [-1, 3]
    )

    times = np.fromfile("./outputs/times_17.npy", dtype=np.float64)

    plot_single_prediction(targets, predictions, times)

    # beg = len(times) // 2
    # times = times - times[beg]

    # plot_single_prediction(targets[beg:], predictions[beg:], times[beg:])


if __name__ == "__main__":
    main()
