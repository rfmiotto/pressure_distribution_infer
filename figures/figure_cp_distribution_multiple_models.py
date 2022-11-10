from PIL import Image
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import texfig


def white_background_highlight(
    ax: matplotlib.axes.Axes, x: np.ndarray, y: np.ndarray
) -> None:
    ax.plot(x, y, alpha=0.1, linewidth=6, color="white")
    ax.plot(x, y, alpha=0.2, linewidth=5, color="white")
    ax.plot(x, y, alpha=0.3, linewidth=4, color="white")
    ax.plot(x, y, alpha=0.4, linewidth=3, color="white")
    ax.plot(x, y, alpha=0.8, linewidth=2, color="white")


def plot_distribution_multiple_models(
    predicted_labels: list[list[np.ndarray]],
    true_labels: list[np.ndarray],
    image_paths: list[str],
    labels: list[str],
    colors: list[str],
) -> None:
    plt.close("all")
    fig, ax = texfig.subplots(
        ratio=0.32,
        nrows=1,
        ncols=3,
    )

    suction_side = np.linspace(0, 1, num=len(predicted_labels[0][0]))

    white_background_highlight(ax[0], suction_side, true_labels[0])
    for i, prediction in enumerate(predicted_labels):
        white_background_highlight(ax[0], suction_side, prediction[0])
    for i, prediction in enumerate(predicted_labels):
        ax[0].plot(suction_side, prediction[0], color=colors[i], label=labels[i])
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
    img = Image.open(image_paths[0])
    img = img.rotate(8)
    img = img.crop((152, 100, 440, 340))
    ax_img1.imshow(img, aspect="equal", alpha=1.0)
    ax_img1.set_axis_off()

    white_background_highlight(ax[1], suction_side, true_labels[1])
    for i, prediction in enumerate(predicted_labels):
        white_background_highlight(ax[1], suction_side, prediction[1])
    for i, prediction in enumerate(predicted_labels):
        ax[1].plot(suction_side, prediction[1], color=colors[i], label=labels[i])
    ax[1].plot(suction_side, true_labels[1], color="tomato", label="True")
    ax[1].yaxis.set_ticklabels([])
    ax[1].set_ylim([-6, 0])
    ax[1].set_xlabel(r"$x/c$", labelpad=1.0)
    ax[1].set_xlim([0, 1])
    ax[1].set_xticks([0, 0.5, 1])
    ax[1].set_facecolor("none")

    ax_img2 = ax[1].inset_axes([0.0, 0.0, 1.0, 1.0], zorder=-50)
    img = Image.open(image_paths[1])
    img = img.rotate(8)
    img = img.crop((152, 100, 440, 340))
    ax_img2.imshow(img, aspect="equal", alpha=1.0)
    ax_img2.set_axis_off()

    white_background_highlight(ax[2], suction_side, true_labels[2])
    for i, prediction in enumerate(predicted_labels):
        white_background_highlight(ax[2], suction_side, prediction[2])
    for i, prediction in enumerate(predicted_labels):
        ax[2].plot(suction_side, prediction[2], color=colors[i], label=labels[i])
    ax[2].plot(suction_side, true_labels[2], color="tomato")
    ax[2].yaxis.set_ticklabels([])
    ax[2].set_ylim([-6, 0])
    ax[2].set_xlabel(r"$x/c$", labelpad=1.0)
    ax[2].set_xlim([0, 1])
    ax[2].set_xticks([0, 0.5, 1])
    ax[2].set_facecolor("none")

    ax_img3 = ax[2].inset_axes([0.0, 0.0, 1.0, 1.0], zorder=-50)
    img = Image.open(image_paths[2])
    img = img.rotate(8)
    img = img.crop((152, 100, 440, 340))
    ax_img3.imshow(img, aspect="equal", alpha=1.0)
    ax_img3.set_axis_off()

    ax[0].legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 0.8),
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


def main() -> None:
    dataframe = pd.read_csv(
        # "./individual_dataframes/dataset_M02_Re60k_span01_g480_heave_ramp_k005.csv"
        "dataset.csv"
    )

    number_of_measurements_suction_side = 300

    predictions_cp_model_1 = np.fromfile(
        "./outputs/predictions_DS1_2_uv_Cp_distribution.npy", dtype=np.float32
    ).reshape([-1, number_of_measurements_suction_side])

    predictions_cp_model_2 = np.fromfile(
        "./outputs/predictions_DS1_2_Cp_Cp_distribution.npy", dtype=np.float32
    ).reshape([-1, number_of_measurements_suction_side])

    targets = np.fromfile(
        "./outputs/targets_2_Cp_distribution.npy", dtype=np.float64
    ).reshape([-1, number_of_measurements_suction_side])

    selected_indices = [100, 600, 900]  #
    selected_indices = [400, 510, 524]  # Fig 8
    selected_indices = [100, 600, 850]  # Fig 9
    selected_indices = [100, 750, 950]  # Fig 8 ou 9 sei la
    selected_indices = [180, 510, 600]
    selected_indices = [400, 450, 510]

    selected_predictions_model_1 = []
    selected_predictions_model_2 = []
    selected_targets = []
    selected_image_paths = []

    for index in selected_indices:
        selected_predictions_model_1.append(predictions_cp_model_1[index])
        selected_predictions_model_2.append(predictions_cp_model_2[index])
        selected_targets.append(targets[index])
        selected_image_paths.append(dataframe["images_pressure"][index])

    selected_predictions = [selected_predictions_model_1, selected_predictions_model_2]
    labels = ["Velocity", "Pressure"]
    colors = ["royalblue", "navy"]

    plot_distribution_multiple_models(
        selected_predictions, selected_targets, selected_image_paths, labels, colors
    )


if __name__ == "__main__":
    main()
