from typing import NoReturn
import numpy as np
import matplotlib.pyplot as plt
import texfig


def _plot_lift_drag_moment(
    predicted_labels_1: np.ndarray,
    predicted_labels_2: np.ndarray,
    predicted_labels_3: np.ndarray,
    true_labels: np.ndarray,
    times: np.ndarray,
) -> NoReturn:
    plt.close("all")
    fig, ax = texfig.subplots(ratio=0.34, nrows=1, ncols=3)

    ax[0].plot(
        times,
        predicted_labels_1[:, 0],
        color="royalblue",
        alpha=0.5,
        label=r"$u$- and $v$-velocity",
    )
    ax[0].plot(
        times,
        predicted_labels_2[:, 0],
        color="royalblue",
        label=r"$z$-vorticity",
    )
    ax[0].plot(times, predicted_labels_3[:, 0], color="navy", label=r"$C_p$")
    ax[0].plot(times, true_labels[:, 0], color="tomato", label="True")
    ax[0].set_xlabel(r"$t$", rotation="horizontal", labelpad=1.0)
    ax[0].set_ylabel(r"$C_l$", rotation="horizontal", labelpad=1.0)
    ax[0].yaxis.set_label_coords(0, 1.02, transform=ax[0].transAxes)

    ax[1].plot(
        times,
        predicted_labels_1[:, 1],
        alpha=0.5,
        color="royalblue",
        label=r"$u$- and $v$-velocity",
    )
    ax[1].plot(
        times,
        predicted_labels_2[:, 1],
        color="royalblue",
        label=r"$z$-vorticity",
    )
    ax[1].plot(times, predicted_labels_3[:, 1], color="navy", label=r"$C_p$")
    ax[1].plot(times, true_labels[:, 1], color="tomato", label="True")
    ax[1].set_xlabel(r"$t$", rotation="horizontal", labelpad=1.0)
    ax[1].set_ylabel(r"$C_d$", rotation="horizontal", labelpad=1.0)
    ax[1].yaxis.set_label_coords(0, 1.02, transform=ax[1].transAxes)

    ax[2].plot(times, -predicted_labels_1[:, 2], alpha=0.5, color="royalblue")
    ax[2].plot(times, -predicted_labels_2[:, 2], color="royalblue")
    ax[2].plot(times, -predicted_labels_3[:, 2], color="navy")
    ax[2].plot(times, -true_labels[:, 2], color="tomato")
    ax[2].set_xlabel(r"$t$", rotation="horizontal", labelpad=1.0)
    ax[2].set_ylabel(r"$C_m$", rotation="horizontal", labelpad=1.0)
    ax[2].yaxis.set_label_coords(0, 1.02, transform=ax[2].transAxes)

    fig.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.2, wspace=0.25)

    texfig.make_fancy_axis(ax[0])
    texfig.make_fancy_axis(ax[1])
    texfig.make_fancy_axis(ax[2])

    ax[1].legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 0.4),
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


def handle_plot_lift_drag_moment() -> NoReturn:
    predictions_using_cp = np.fromfile(
        "./outputs/predictions_DS1_11_Cp_coeffs.npy", dtype=np.float32
    ).reshape([-1, 3])
    predictions_using_uv = np.fromfile(
        "./outputs/predictions_DS1_11_uv_coeffs.npy", dtype=np.float32
    ).reshape([-1, 3])
    predictions_using_zvort = np.fromfile(
        "./outputs/predictions_DS1_11_zvort_coeffs.npy", dtype=np.float32
    ).reshape([-1, 3])
    targets = np.fromfile("./outputs/targets_11_coeffs.npy", dtype=np.float64).reshape(
        [-1, 3]
    )

    times = np.fromfile("./outputs/times_11.npy", dtype=np.float64)

    _plot_lift_drag_moment(
        predictions_using_uv[::1],
        predictions_using_zvort[::1],
        predictions_using_cp[::1],
        targets[::1],
        times[::1],
    )


if __name__ == "__main__":
    handle_plot_lift_drag_moment()
