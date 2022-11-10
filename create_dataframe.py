"""
This module generates the Pandas Dataframe used to train the neural net.
This Dataframe contains the path to every image of the simulations and
their respective aerodynamic coefficients.

Here I create one dataframe for each simulation. Then, a function stacks
multiple dataframes that will form the dataset used by the neural net. In this
function, you can specify which simulation (dataset) should be included.

The global variable `DATASET_BASE_DIRECTORY` specifies where to look for the
images and aerodynamic coefficients. It expects the following structure:

DATASET_BASE_DIRECTORY/
    simulation1/
        coeff_lift.npy
        coeff_drag.npy
        coeff_moment.npy
        times.npy
        wall_pressure_distribution.npy
        x_coord_suction_side.npy
        ...
        pressure_coeff_range_-6.0_0.0/
            0000.jpg
            0001.jpg
            ...
        some_other_property/
            0000.jpg
            0001.jpg
            ...
    simulation2/
        ...
"""
import os
import numpy as np
import pandas as pd
from scipy import interpolate

# This is being imported from the post-process module. Remember to set its path
# in PATHONPATH environment variable for this to work.
from file_searching import search_for_files

# Number of points used to interpolate the pressure distribution over the airfoil
# suction side uniformly.
NUM_INTERP_POINTS = 301

# TODO: WARNING: I removed the first point from the interpolated Cp distribution.
# TODO: It seems like an outlier, but it is actually physical. However, the network
# TODO: might have issues with this discontinuity and I want to remove it...
# TODO: It is important to remember that when you plot the Cp distribution. The x
# TODO: should NOT start at zero. Use `x = np.linspace(0, 1, NUM_INTERP_POINTS)[1:]`
# TODO: instead.

# This flag indicates whether it should create a dataframe for all simulations
CREATE_ALL_DATAFRAMES = False

# This is the base path where to loop for the simulation data (images and coeff)
DATASET_BASE_DIRECTORY = (
    "/home/miotto/Desktop/CNN_PyTorch_coeffs_results/trained_models/dataset"
)

# This dict indicates the path inside the base directory where each individual
# simulation data is stored (key). The value of the dict tells whether this
# specific simulation should be included in the dataframe that the neural network
# will use.
SIMULATION_DIRS = {
    # Periodic cases reduced freq. 0.25
    "M01_Re60k_span04_g480_heave_periodic_k025": False,
    "M02_Re60k_span04_g480_heave_periodic_k025": True,
    "M04_Re60k_span04_g480_heave_periodic_k025": False,
    # Periodic cases reduced freq. 0.5
    "M01_Re60k_span04_g480_heave_periodic_k050": False,
    "M04_Re60k_span04_g480_heave_periodic_k050": False,
    # Ramp cases Mach 0.1 and Reynolds 60,000
    "M01_Re60k_span01_g480_pitch_ramp_k005": False,
    "M01_Re60k_span01_g480_pitch_ramp_k010": False,
    "M01_Re60k_span01_g480_heave_ramp_k005": False,
    "M01_Re60k_span01_g480_heave_ramp_k010": False,
    # Ramp cases Mach 0.2 and Reynolds 60,000
    "M02_Re60k_span01_g480_heave_ramp_k005": False,
    "M02_Re60k_span01_g480_heave_ramp_k010": False,
    "M02_Re60k_span01_g480_pitch_ramp_k010": False,
    # Ramp cases Mach 0.4 and Reynolds 60,000
    "M04_Re60k_span01_g480_pitch_ramp_k005": False,
    "M04_Re60k_span01_g480_pitch_ramp_k010": False,
    "M04_Re60k_span01_g480_heave_ramp_k005": False,
    "M04_Re60k_span01_g480_heave_ramp_k010": False,
    # Ramp cases Mach 0.1 and Reynolds 200,000
    "M01_Re200k_span01_g720_pitch_ramp_k005": False,
    "M01_Re200k_span01_g720_pitch_ramp_k010": False,
    "M01_Re200k_span01_g720_heave_ramp_k005": False,
    "M01_Re200k_span01_g720_heave_ramp_k010": False,
}


# SELECTED_INDICES = [1500, 2700, 4200]
# SELECTED_INDICES = [100, 200, 300, 400]
SELECTED_INDICES = []  # Leave this array empty to select all frames


def interpolate_pressure_distribution(
    x_coord_suction_side: np.ndarray, wall_pressure_distribution: np.ndarray
) -> np.ndarray:
    number_of_snapshots = wall_pressure_distribution.shape[0]

    interpolated_pressure_dist = np.empty((number_of_snapshots, NUM_INTERP_POINTS))

    for i in range(number_of_snapshots):
        instantaneous_interp_dist = interpolate_instantaneous_distribution(
            x_coord_suction_side, wall_pressure_distribution[i]
        )
        interpolated_pressure_dist[i] = instantaneous_interp_dist

    return interpolated_pressure_dist


def interpolate_instantaneous_distribution(
    x_coord: np.ndarray, distribution: np.ndarray
) -> np.ndarray:
    # tck = interpolate.splrep(x_coord, distribution, s=0)
    approximate_fn = interpolate.interp1d(x_coord, distribution)
    x_coord_new = np.linspace(0.0, 1.0, num=NUM_INTERP_POINTS)
    # instantaneous_interp_dist = interpolate.splev(x_coord_new, tck, der=0)
    instantaneous_interp_dist = approximate_fn(x_coord_new)
    return instantaneous_interp_dist


def create_individual_dataframe(full_path_to_data) -> None:
    path = os.path.join(full_path_to_data, "pressure_coeff_range_-6.0_0.0")
    images_pressure = np.asarray(search_for_files(path, pattern="*.jpg"))

    path = os.path.join(full_path_to_data, "velocity_x_range_-2.0_2.0")
    images_vel_x = np.asarray(search_for_files(path, pattern="*.jpg"))

    path = os.path.join(full_path_to_data, "velocity_y_range_-2.0_2.0")
    images_vel_y = np.asarray(search_for_files(path, pattern="*.jpg"))

    path = os.path.join(full_path_to_data, "z_vorticity_range_-5.0_5.0")
    images_z_vort = np.asarray(search_for_files(path, pattern="*.jpg"))

    times = np.fromfile(os.path.join(full_path_to_data, "times.npy"))
    coeff_lift = np.fromfile(os.path.join(full_path_to_data, "coeff_lift.npy"))
    coeff_drag = np.fromfile(os.path.join(full_path_to_data, "coeff_drag.npy"))
    coeff_moment = np.fromfile(os.path.join(full_path_to_data, "coeff_moment.npy"))
    x_coord_suction_side = np.fromfile(
        os.path.join(full_path_to_data, "x_coord_suction_side.npy")
    )
    wall_pressure_distribution = np.fromfile(
        os.path.join(full_path_to_data, "wall_pressure_distribution.npy")
    ).reshape((-1, len(x_coord_suction_side)))

    interpolated_wall_pressure_distribution = interpolate_pressure_distribution(
        x_coord_suction_side, wall_pressure_distribution
    )

    dataframe_column_names_map = {
        "images_pressure": "images_pressure",
        "images_vel_x": "images_vel_x",
        "images_vel_y": "images_vel_y",
        "images_z_vort": "images_z_vort",
        "times": "times",
        "coeff_lift": "Cl",
        "coeff_drag": "Cd",
        "coeff_moment": "Cm",
        "wall_pressure_distribution": [f"Cp{i}" for i in range(NUM_INTERP_POINTS)],
    }

    def flatten_list(nested_list):
        flatten_list = []
        for item in nested_list:
            if isinstance(item, list):
                flatten_list.extend(item)
            else:
                flatten_list.append(item)
        return flatten_list

    dataframe_column_names = flatten_list(list(dataframe_column_names_map.values()))

    dataframe = pd.DataFrame(columns=dataframe_column_names)

    dataframe["images_pressure"] = images_pressure
    dataframe["images_vel_x"] = images_vel_x
    dataframe["images_vel_y"] = images_vel_y
    dataframe["images_z_vort"] = images_z_vort
    dataframe["times"] = times
    dataframe["Cl"] = coeff_lift
    dataframe["Cd"] = coeff_drag
    dataframe["Cm"] = coeff_moment
    for i_loc, label in enumerate(
        dataframe_column_names_map["wall_pressure_distribution"]
    ):
        dataframe[label] = interpolated_wall_pressure_distribution[:, i_loc]

    # TODO: Here is where I remove the first point:
    dataframe = dataframe.drop("Cp0", axis=1)

    simulation_dir = os.path.split(full_path_to_data)[1]
    filename = f"dataset_{simulation_dir}.csv"
    dataframe.to_csv(os.path.join("individual_dataframes", filename))


def create_dataframes_of_all_simulations() -> None:
    print("Creating a CSV file for each simulation")

    folder_exist = os.path.exists("individual_dataframes")
    if not folder_exist:
        os.makedirs("individual_dataframes")

    for simulation_dir in SIMULATION_DIRS:
        print(simulation_dir)
        path = os.path.join(DATASET_BASE_DIRECTORY, simulation_dir)
        create_individual_dataframe(path)


def concatenate_selected_dataframes() -> None:
    """
    Produce a single dataframe that will be used to train different neural
    network models.
    """
    print("Creating CSV file for the neural network")

    final_dataframe = pd.DataFrame()

    for simulation_dir, select_simulation in SIMULATION_DIRS.items():
        if select_simulation:
            print(simulation_dir)

            filename = f"dataset_{simulation_dir}.csv"
            dataframe = pd.read_csv(
                os.path.join("individual_dataframes", filename), index_col=0
            )
            final_dataframe = pd.concat([final_dataframe, dataframe])

    final_dataframe.to_csv("dataset.csv")


def select_few_snapshots() -> None:
    full_dataframe = pd.read_csv("dataset.csv")

    dataframe_only_selected = full_dataframe.loc[SELECTED_INDICES]

    dataframe_only_selected.to_csv("dataset.csv")


if __name__ == "__main__":
    if CREATE_ALL_DATAFRAMES:
        create_dataframes_of_all_simulations()

    concatenate_selected_dataframes()

    if SELECTED_INDICES:
        select_few_snapshots()

    # import matplotlib.pyplot as plt

    # full_path_to_data = os.path.join(
    #     DATASET_BASE_DIRECTORY, "M01_Re60k_span01_g480_heave_ramp_k005"
    # )

    # x_coord_suction_side = np.fromfile(
    #     os.path.join(full_path_to_data, "x_coord_suction_side.npy")
    # )
    # wall_pressure_distribution = np.fromfile(
    #     os.path.join(full_path_to_data, "wall_pressure_distribution.npy")
    # ).reshape((-1, len(x_coord_suction_side)))

    # dataframe = pd.read_csv(
    #     os.path.join(
    #         "individual_dataframes", "dataset_M01_Re60k_span01_g480_heave_ramp_k005.csv"
    #     ),
    #     index_col=0,
    # )

    # # dataframe = dataframe.drop("Cp0", axis=1)  # REMOVE THE FIRST POINT BECAUSE IT LOOKS LIKE AN OUTLIER

    # y = dataframe.loc[:, dataframe.columns.str.startswith("Cp")].to_numpy()
    # x = np.linspace(0, 1, NUM_INTERP_POINTS)
    # x = x[1:]

    # plt.plot(x_coord_suction_side, wall_pressure_distribution[200, :], "-o")
    # plt.plot(x, y[200, :], "-o")
    # plt.show()
