import os
from multiprocessing import Queue, Process
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

from metrics import Metrics
from file_searching import search_for_files
from read_flow_cgns import read_flow_in_cgns
from read_grid_cgns import read_grid_in_cgns
from properties import compute_z_vorticity
from aerodynamic_coefficients import compute_pressure_coefficient
from cmaps import custom_magma


PATH_ALL_SIMULATIONS = [
    # Periodic cases reduced freq. 0.25
    "/media/miotto/Seagate Backup Plus Drive/SD7003/Brener_M01_k025_span04/output",  #
    "/media/miotto/3B712DB11C683E49/SD7003/Brener_M02_k025_span04/proc/output",
    "/media/miotto/3B712DB11C683E49/SD7003/Brener_M04_k025_span04/proc/output",
    # Periodic cases reduced freq. 0.5
    "/media/miotto/Backup Plus1/SD7003/Brener_M01_k050_span04/output",  #
    "/media/miotto/Backup Plus1/SD7003/Brener_M04_k050_span04/proc/output",  #
    # Ramp cases Mach 0.1 and Reynolds 60,000
    "/media/miotto/3B712DB11C683E49/SD7003/M01_Re60k_span01_g480/proc_pitch_ramp_k005/output",
    "/media/miotto/Seagate Backup Plus Drive/SD7003/M01_Re60k_span01_g480/proc_pitch_ramp_k010/output",  #
    "/media/miotto/Seagate Backup Plus Drive/SD7003/M01_Re60k_span01_g480/proc_heave_ramp_k005/output",
    "/media/miotto/Seagate Backup Plus Drive/SD7003/M01_Re60k_span01_g480/proc_heave_ramp_k010/output",  #
    # Ramp cases Mach 0.2 and Reynolds 60,000
    "/media/miotto/3B712DB11C683E49/SD7003/M02_Re60k_span01_g480/proc_heave_ramp_k005/output",
    "/media/miotto/Backup Plus/SD7003/M02_Re60k_span01_g480/proc_heave_ramp_k010/output",
    "/media/miotto/Backup Plus/SD7003/M02_Re60k_span01_g480/proc_pitch_ramp_k010/output",
    # Ramp cases Mach 0.4 and Reynolds 60,000
    "/media/miotto/3B712DB11C683E49/SD7003/M04_Re60k_span01_g480/proc_pitch_ramp_k005/output",
    "/media/miotto/Seagate Backup Plus Drive/SD7003/M04_Re60k_span01_g480/proc_pitch_ramp_k010/output",  #
    "/media/miotto/3B712DB11C683E49/SD7003/M04_Re60k_span01_g480/proc_heave_ramp_k005/output",
    "/media/miotto/Seagate Backup Plus Drive/SD7003/M04_Re60k_span01_g480/proc_heave_ramp_k010/output",  #
    # Ramp cases Mach 0.1 and Reynolds 200,000
    "/media/miotto/Backup Plus1/SD7003/M01_Re200k_g720/proc_pitch_ramp_k005/output",  #
    "/media/miotto/Backup Plus1/SD7003/M01_Re200k_g720/proc_pitch_ramp_k010/output",  #
    "/media/miotto/3B712DB11C683E49/SD7003/M01_Re200k_span01_g720/proc_heave_ramp_k005/output",
    "/media/miotto/Backup Plus1/SD7003/M01_Re200k_g720/proc_heave_ramp_k010/output",  #
]

MAP_DIRECTORY_TO_SAVE_RESULTS = [
    # Periodic cases reduced freq. 0.25
    "M01_Re60k_span04_g480_heave_periodic_k025",
    "M02_Re60k_span04_g480_heave_periodic_k025",
    "M04_Re60k_span04_g480_heave_periodic_k025",
    # Periodic cases reduced freq. 0.5
    "M01_Re60k_span04_g480_heave_periodic_k050",
    "M04_Re60k_span04_g480_heave_periodic_k050",
    # Ramp cases Mach 0.1 and Reynolds 60,000
    "M01_Re60k_span01_g480_pitch_ramp_k005",
    "M01_Re60k_span01_g480_pitch_ramp_k010",
    "M01_Re60k_span01_g480_heave_ramp_k005",
    "M01_Re60k_span01_g480_heave_ramp_k010",
    # Ramp cases Mach 0.2 and Reynolds 60,000
    "M02_Re60k_span01_g480_heave_ramp_k005",
    "M02_Re60k_span01_g480_heave_ramp_k010",
    "M02_Re60k_span01_g480_pitch_ramp_k010",
    # Ramp cases Mach 0.4 and Reynolds 60,000
    "M04_Re60k_span01_g480_pitch_ramp_k005",
    "M04_Re60k_span01_g480_pitch_ramp_k010",
    "M04_Re60k_span01_g480_heave_ramp_k005",
    "M04_Re60k_span01_g480_heave_ramp_k010",
    # Ramp cases Mach 0.1 and Reynolds 200,000
    "M01_Re200k_span01_g720_pitch_ramp_k005",
    "M01_Re200k_span01_g720_pitch_ramp_k010",
    "M01_Re200k_span01_g720_heave_ramp_k005",
    "M01_Re200k_span01_g720_heave_ramp_k010",
]

MAP_REYNOLDS = [
    # Periodic cases reduced freq. 0.25
    6e4,
    6e4,
    6e4,
    # Periodic cases reduced freq. 0.5
    6e4,
    6e4,
    # Ramp cases Mach 0.1 and Reynolds 60,000
    6e4,
    6e4,
    6e4,
    6e4,
    # Ramp cases Mach 0.2 and Reynolds 60,000
    6e4,
    6e4,
    6e4,
    # Ramp cases Mach 0.4 and Reynolds 60,000
    6e4,
    6e4,
    6e4,
    6e4,
    # Ramp cases Mach 0.1 and Reynolds 200,000
    2e5,
    2e5,
    2e5,
    2e5,
]

MAP_MACH = [
    # Periodic cases reduced freq. 0.25
    0.1,
    0.2,
    0.4,
    # Periodic cases reduced freq. 0.5
    0.1,
    0.4,
    # Ramp cases Mach 0.1 and Reynolds 60,000
    0.1,
    0.1,
    0.1,
    0.1,
    # Ramp cases Mach 0.2 and Reynolds 60,000
    0.2,
    0.2,
    0.2,
    # Ramp cases Mach 0.4 and Reynolds 60,000
    0.4,
    0.4,
    0.4,
    0.4,
    # Ramp cases Mach 0.1 and Reynolds 200,000
    0.1,
    0.1,
    0.1,
    0.1,
]


X_LIMITS = [-0.5, 1.5]
Y_LIMITS = [-1.0, -1.0 + np.diff(X_LIMITS)[0]]

GAMMA = 1.4
REFERENCE_DENSITY = 1.0
STATIC_PRESSURE = 1.0 / GAMMA

BASE_PATH_TO_SAVE_DATA = (
    "/home/miotto/Desktop/CNN_PyTorch_coeffs_results/trained_models/dataset"
)


FIGURE_WIDTH_IN_PIXELS = 600
FIGURE_HEIGHT_IN_PIXELS = 600
DPI = 96


def make_plot(
    ax, x, y, variable, vmax, vmin, cmap, full_path_output_folder, index_qout
):
    norm = matplotlib.cm.colors.Normalize(vmax=vmax, vmin=vmin)
    levels = np.linspace(vmin, vmax, 512)
    ax.contourf(
        x,
        y,
        variable,
        levels,
        norm=norm,
        cmap=cmap,
        extend="both",
    )
    airfoil = [[x[_, 0], y[_, 0]] for _ in range(x.shape[0])]
    airfoil = np.array(airfoil)
    hull = ConvexHull(airfoil)
    ax.fill(
        airfoil[hull.vertices, 0],
        airfoil[hull.vertices, 1],
        color="#bfbfbf",
        linewidth=0.2,
    )
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*Y_LIMITS)
    ax.axis("off")
    ax.set_aspect("equal")

    folder_exist = os.path.exists(full_path_output_folder)
    if not folder_exist:
        try:  # need a try-catch block due to concurrency with multiprocessing
            os.makedirs(full_path_output_folder)
        except FileExistsError:
            pass
    filename = os.path.join(full_path_output_folder, f"{index_qout:04d}.jpg")
    plt.savefig(filename, dpi=DPI)
    plt.cla()


def main(
    ax: plt.Axes,
    qout_files: Queue,
    mach_number: float,
    x: np.ndarray,
    y: np.ndarray,
    metrics: Metrics,
    output_path: str,
):
    for index_qout, qout_file in iter(qout_files.get, "STOP"):
        print(
            f"{index_qout + 1}/{qout_files.qsize()} -- {os.path.split(output_path)[1]} / {os.path.split(qout_file)[1]}"
        )

        q_vector, _ = read_flow_in_cgns(qout_file)

        pressure = q_vector[3]

        # PLOT PRESSURE COEFFICIENT

        pressure_coeff = compute_pressure_coefficient(
            pressure, STATIC_PRESSURE, REFERENCE_DENSITY, mach_number
        )

        vmin, vmax = -6.0, 0.0
        cmap = custom_magma()

        make_plot(
            ax,
            x,
            y,
            pressure_coeff,
            vmax=vmax,
            vmin=vmin,
            cmap=cmap,
            full_path_output_folder=os.path.join(
                output_path, f"pressure_coeff_range_{vmin}_{vmax}"
            ),
            index_qout=index_qout,
        )

        # PLOT VELOCITY X

        velocity_x = q_vector[1] / q_vector[0] / mach_number

        vmin, vmax = -2.0, 2.0
        cmap = "bwr"

        make_plot(
            ax,
            x,
            y,
            velocity_x,
            vmax=vmax,
            vmin=vmin,
            cmap=cmap,
            full_path_output_folder=os.path.join(
                output_path, f"velocity_x_range_{vmin}_{vmax}"
            ),
            index_qout=index_qout,
        )

        # PLOT VELOCITY Y

        velocity_y = q_vector[2] / q_vector[0] / mach_number

        make_plot(
            ax,
            x,
            y,
            velocity_y,
            vmax=vmax,
            vmin=vmin,
            cmap=cmap,
            full_path_output_folder=os.path.join(
                output_path, f"velocity_y_range_{vmin}_{vmax}"
            ),
            index_qout=index_qout,
        )

        # PLOT Z-VORTICITY

        z_vorticity = compute_z_vorticity(velocity_x, velocity_y, metrics)

        vmin, vmax = -5.0, 5.0
        cmap = "bwr"

        make_plot(
            ax,
            x,
            y,
            z_vorticity,
            vmax=vmax,
            vmin=vmin,
            cmap=cmap,
            full_path_output_folder=os.path.join(
                output_path, f"z_vorticity_range_{vmin}_{vmax}"
            ),
            index_qout=index_qout,
        )


def loop_through_simulations():
    for i, path in enumerate(PATH_ALL_SIMULATIONS):

        mach_number = MAP_MACH[i]

        output_path = os.path.join(
            BASE_PATH_TO_SAVE_DATA, MAP_DIRECTORY_TO_SAVE_RESULTS[i]
        )

        folder_exist = os.path.exists(output_path)
        if not folder_exist:
            os.makedirs(output_path)

        fig = plt.figure(frameon=False)
        fig.set_size_inches(
            (FIGURE_WIDTH_IN_PIXELS / DPI, FIGURE_HEIGHT_IN_PIXELS / DPI)
        )
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)

        qout_files = search_for_files(path, pattern="qout2Davg*.cgns")
        grid_file = search_for_files(path, pattern="grid2D.cgns")[0]

        x, y = read_grid_in_cgns(grid_file)

        metrics = Metrics(x, y)

        queue_of_files = Queue()
        for index_qout, qout_file in enumerate(qout_files):
            queue_of_files.put((index_qout, qout_file))

        all_processes = []
        num_workers = 4
        for _ in range(num_workers):
            process = Process(
                target=main,
                args=(
                    ax,
                    queue_of_files,
                    mach_number,
                    x,
                    y,
                    metrics,
                    output_path,
                ),
            )
            queue_of_files.put("STOP")  # signals end of tasks for workers
            all_processes.append(process)
            process.start()  # start worker
        for process in all_processes:
            process.join()  # inside main process, wait for worker to finish tasks


if __name__ == "__main__":
    loop_through_simulations()
