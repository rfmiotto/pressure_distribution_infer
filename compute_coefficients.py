import os
import numpy as np

from metrics import Metrics
from file_searching import search_for_files
from read_flow_cgns import read_flow_in_cgns
from read_grid_cgns import read_grid_in_cgns
from airfoil_sides import get_airfoil_suction_and_pressure_side_indices
from flow_forces import compute_wall_pressure_forces, compute_wall_viscous_forces
from aerodynamic_coefficients import (
    compute_lift_and_drag_coefficients,
    compute_pitch_moment_coefficient,
    compute_pressure_coefficient,
)


PATH_ALL_SIMULATIONS = [
    # Periodic cases reduced freq. 0.25
    "/media/miotto/Seagate Backup Plus Drive/SD7003/Brener_M01_k025_span04/output",  #
    # "/media/miotto/3B712DB11C683E49/SD7003/Brener_M02_k025_span04/proc/output",
    # "/media/miotto/3B712DB11C683E49/SD7003/Brener_M04_k025_span04/proc/output",
    # Periodic cases reduced freq. 0.5
    "/media/miotto/Backup Plus1/SD7003/Brener_M01_k050_span04/output",  #
    "/media/miotto/Backup Plus1/SD7003/Brener_M04_k050_span04/proc/output",  #
    # Ramp cases Mach 0.1 and Reynolds 60,000
    # "/media/miotto/3B712DB11C683E49/SD7003/M01_Re60k_span01_g480/proc_pitch_ramp_k005/output",
    "/media/miotto/Seagate Backup Plus Drive/SD7003/M01_Re60k_span01_g480/proc_pitch_ramp_k010/output",  #
    # "/media/miotto/Seagate Backup Plus Drive/SD7003/M01_Re60k_span01_g480/proc_heave_ramp_k005/output",
    "/media/miotto/Seagate Backup Plus Drive/SD7003/M01_Re60k_span01_g480/proc_heave_ramp_k010/output",  #
    # Ramp cases Mach 0.2 and Reynolds 60,000
    # "/media/miotto/3B712DB11C683E49/SD7003/M02_Re60k_span01_g480/proc_heave_ramp_k005/output",
    # "/media/miotto/Backup Plus/SD7003/M02_Re60k_span01_g480/proc_heave_ramp_k010/output",
    # "/media/miotto/Backup Plus/SD7003/M02_Re60k_span01_g480/proc_pitch_ramp_k010/output",
    # Ramp cases Mach 0.4 and Reynolds 60,000
    # "/media/miotto/3B712DB11C683E49/SD7003/M04_Re60k_span01_g480/proc_pitch_ramp_k005/output",
    "/media/miotto/Seagate Backup Plus Drive/SD7003/M04_Re60k_span01_g480/proc_pitch_ramp_k010/output",  #
    # "/media/miotto/3B712DB11C683E49/SD7003/M04_Re60k_span01_g480/proc_heave_ramp_k005/output",
    "/media/miotto/Seagate Backup Plus Drive/SD7003/M04_Re60k_span01_g480/proc_heave_ramp_k010/output",  #
    # Ramp cases Mach 0.1 and Reynolds 200,000
    "/media/miotto/Backup Plus1/SD7003/M01_Re200k_g720/proc_pitch_ramp_k005/output",  #
    "/media/miotto/Backup Plus1/SD7003/M01_Re200k_g720/proc_pitch_ramp_k010/output",  #
    # "/media/miotto/3B712DB11C683E49/SD7003/M01_Re200k_span01_g720/proc_heave_ramp_k005/output",
    "/media/miotto/Backup Plus1/SD7003/M01_Re200k_g720/proc_heave_ramp_k010/output",  #
]

MAP_DIRECTORY_TO_SAVE_RESULTS = [
    # Periodic cases reduced freq. 0.25
    "M01_Re60k_span04_g480_heave_periodic_k025",
    # "M02_Re60k_span04_g480_heave_periodic_k025",
    # "M04_Re60k_span04_g480_heave_periodic_k025",
    # Periodic cases reduced freq. 0.5
    "M01_Re60k_span04_g480_heave_periodic_k050",
    "M04_Re60k_span04_g480_heave_periodic_k050",
    # Ramp cases Mach 0.1 and Reynolds 60,000
    # "M01_Re60k_span01_g480_pitch_ramp_k005",
    "M01_Re60k_span01_g480_pitch_ramp_k010",
    # "M01_Re60k_span01_g480_heave_ramp_k005",
    "M01_Re60k_span01_g480_heave_ramp_k010",
    # Ramp cases Mach 0.2 and Reynolds 60,000
    # "M02_Re60k_span01_g480_heave_ramp_k005",
    # "M02_Re60k_span01_g480_heave_ramp_k010",
    # "M02_Re60k_span01_g480_pitch_ramp_k010",
    # Ramp cases Mach 0.4 and Reynolds 60,000
    # "M04_Re60k_span01_g480_pitch_ramp_k005",
    "M04_Re60k_span01_g480_pitch_ramp_k010",
    # "M04_Re60k_span01_g480_heave_ramp_k005",
    "M04_Re60k_span01_g480_heave_ramp_k010",
    # Ramp cases Mach 0.1 and Reynolds 200,000
    "M01_Re200k_span01_g720_pitch_ramp_k005",
    "M01_Re200k_span01_g720_pitch_ramp_k010",
    # "M01_Re200k_span01_g720_heave_ramp_k005",
    "M01_Re200k_span01_g720_heave_ramp_k010",
]

MAP_REYNOLDS = [
    # Periodic cases reduced freq. 0.25
    6e4,
    # 6e4,
    # 6e4,
    # Periodic cases reduced freq. 0.5
    6e4,
    6e4,
    # Ramp cases Mach 0.1 and Reynolds 60,000
    # 6e4,
    6e4,
    # 6e4,
    6e4,
    # Ramp cases Mach 0.2 and Reynolds 60,000
    # 6e4,
    # 6e4,
    # 6e4,
    # Ramp cases Mach 0.4 and Reynolds 60,000
    # 6e4,
    6e4,
    # 6e4,
    6e4,
    # Ramp cases Mach 0.1 and Reynolds 200,000
    2e5,
    2e5,
    # 2e5,
    2e5,
]

MAP_MACH = [
    # Periodic cases reduced freq. 0.25
    0.1,
    # 0.2,
    # 0.4,
    # Periodic cases reduced freq. 0.5
    0.1,
    0.4,
    # Ramp cases Mach 0.1 and Reynolds 60,000
    # 0.1,
    0.1,
    # 0.1,
    0.1,
    # Ramp cases Mach 0.2 and Reynolds 60,000
    # 0.2,
    # 0.2,
    # 0.2,
    # Ramp cases Mach 0.4 and Reynolds 60,000
    # 0.4,
    0.4,
    # 0.4,
    0.4,
    # Ramp cases Mach 0.1 and Reynolds 200,000
    0.1,
    0.1,
    # 0.1,
    0.1,
]


PIVOT_POINT = [0.25 * np.cos(np.deg2rad(8)), -0.25 * np.sin(np.deg2rad(8))]
STATIC_AOA_IN_DEG = 8

MACH_NUMBER = 0.1
REYNOLDS_NUMBER = 6e4

GAMMA = 1.4
REFERENCE_DENSITY = 1.0
REFERENCE_VISCOSITY = 1.0 / REYNOLDS_NUMBER
STATIC_PRESSURE = 1.0 / GAMMA

BASE_PATH_TO_SAVE_DATA = (
    "/home/miotto/Desktop/CNN_PyTorch_coeffs_results/trained_models/dataset_pln"
)


def main(
    path: str,
    mach_number: float,
    reference_viscosity: float,
    output_path: str,
):
    qout_files = search_for_files(path, pattern="qout2Dpln*.cgns")
    grid_file = search_for_files(path, pattern="grid2D.cgns")[0]

    x, y = read_grid_in_cgns(grid_file)

    metrics = Metrics(x, y)

    suction_side_indices, _ = get_airfoil_suction_and_pressure_side_indices(
        x[:, 0], y[:, 0], STATIC_AOA_IN_DEG, metrics
    )

    straight_airfoil_x = (
        np.cos(np.deg2rad(STATIC_AOA_IN_DEG)) * x
        - np.sin(np.deg2rad(STATIC_AOA_IN_DEG)) * y
    )

    # straight_airfoil_y = (
    #     np.sin(np.deg2rad(STATIC_AOA_IN_DEG)) * x
    #     + np.cos(np.deg2rad(STATIC_AOA_IN_DEG)) * y
    # )

    # import matplotlib.pyplot as plt

    # plt.plot(x[:, 0], y[:, 0], "-o", color="b")
    # plt.plot(straight_airfoil_x[:, 0], straight_airfoil_y[:, 0], "-o", color="r")
    # plt.show()

    lift = []
    drag = []
    pitch_moment = []
    timestamps = []
    wall_pressure_distribution = []

    for index_qout, qout_file in enumerate(qout_files):
        print(
            f"{index_qout + 1}/{len(qout_files)} -- {os.path.split(output_path)[1]} / {os.path.split(qout_file)[1]}"
        )

        q_vector, time = read_flow_in_cgns(qout_file)

        velocity_x = q_vector[1] / q_vector[0]
        velocity_y = q_vector[2] / q_vector[0]
        pressure = q_vector[3]

        wall_pressure_forces = compute_wall_pressure_forces(pressure, metrics)
        wall_viscous_forces = compute_wall_viscous_forces(
            velocity_x, velocity_y, reference_viscosity, metrics
        )

        wall_pressure_forces = np.array(wall_pressure_forces, dtype=np.ndarray)
        wall_viscous_forces = np.array(wall_viscous_forces, dtype=np.ndarray)

        instantaneous_lift, instantaneous_drag = compute_lift_and_drag_coefficients(
            wall_pressure_forces, wall_viscous_forces, REFERENCE_DENSITY, mach_number
        )

        instantaneous_pitch_moment = compute_pitch_moment_coefficient(
            wall_pressure_forces,
            wall_viscous_forces,
            REFERENCE_DENSITY,
            mach_number,
            metrics,
            PIVOT_POINT,
        )

        instantaneous_wall_pressure_distribution = compute_pressure_coefficient(
            pressure[:, 0], STATIC_PRESSURE, REFERENCE_DENSITY, mach_number
        )[suction_side_indices]

        lift.append(instantaneous_lift)
        drag.append(instantaneous_drag)
        pitch_moment.append(instantaneous_pitch_moment)
        timestamps.append(time)
        wall_pressure_distribution.append(instantaneous_wall_pressure_distribution)

    lift = np.array(lift)
    drag = np.array(drag)
    pitch_moment = np.array(pitch_moment)
    timestamps = np.array(timestamps)
    wall_pressure_distribution = np.array(wall_pressure_distribution)
    x_coord_suction_side = straight_airfoil_x[suction_side_indices, 0]

    # Create timestamps based on simulation timestep
    # start_time = 0
    # timestamps = start_time + np.arange(len(qout_files)) * SIMULATION_DT

    lift.tofile(os.path.join(output_path, "coeff_lift.npy"))
    drag.tofile(os.path.join(output_path, "coeff_drag.npy"))
    pitch_moment.tofile(os.path.join(output_path, "coeff_moment.npy"))
    timestamps.tofile(os.path.join(output_path, "times.npy"))
    wall_pressure_distribution.tofile(
        os.path.join(output_path, "wall_pressure_distribution.npy")
    )
    x_coord_suction_side.tofile(os.path.join(output_path, "x_coord_suction_side.npy"))


def loop_through_simulations():
    for i, path in enumerate(PATH_ALL_SIMULATIONS):

        mach_number = MAP_MACH[i]
        reynolds_number = MAP_REYNOLDS[i]

        reference_viscosity = 1.0 / reynolds_number

        output_path = os.path.join(
            BASE_PATH_TO_SAVE_DATA, MAP_DIRECTORY_TO_SAVE_RESULTS[i]
        )

        folder_exist = os.path.exists(output_path)
        if not folder_exist:
            os.makedirs(output_path)

        main(path, mach_number, reference_viscosity, output_path)

        # break


if __name__ == "__main__":
    loop_through_simulations()
