"""Perform inference using SINDy (PySINDy) for benchmarked ODE systems."""

import os
import random
import time

import numpy as np
import pandas as pd
import pysindy as ps
from utils import copy_data_files_to_results_dir, time_saving

# Seed for reproducibility
np.random.seed(1998)
random.seed(1998)

# Number of top candidates to save
top_n = 10

# Feature library for each system
system_features = {
    "sir": ["poly", "sqrt"],
    "lotka-volterra": ["poly"],
    "brusselator": ["poly"],
    "fairen-velarde": ["poly", "div"],
}

# Create results/sindy directory if it doesn't exist
sindy_file_dir = "results/sindy"
if not os.path.exists(sindy_file_dir):
    os.makedirs(sindy_file_dir)

# Walk through FluxDisco data directory and copy data files to SINDy results directories
copy_data_files_to_results_dir(data_dir="../data", target_dir=sindy_file_dir)

for root, _, files in os.walk(sindy_file_dir):
    # Go to level where folders are named experiment_{noise_level}
    if "experiment_" in root:
        print(f"Processing {root}...")
        # Check for estimated_data.csv to avoid re-processing
        if os.path.exists(f"{root}/estimated_data.csv"):
            print(f"Estimated data already exists for {root}, skipping...")
            continue

        experiment_start_time = time.time()

        # Get system label from the path
        for system in system_features.keys():
            if system in root:
                system_feature = system_features[system]
                break

        # Get data
        noisy_realisations = np.load(f"{root}/data_plot.npy", allow_pickle=True)[0]
        noiseless_data = pd.read_csv(f"{root}/noiseless_data.csv")

        # Get time and states from the realisation
        state_variables = [var for var in noisy_realisations.keys() if var != "time"]
        print(f"State variables: {state_variables}")
        state_trajectories = np.stack(
            [noisy_realisations[var] for var in state_variables], axis=1
        )
        times = noisy_realisations["time"]

        # Build feature library
        libraries = []

        # Polynomials
        if "poly" in system_feature:
            poly_lib = ps.PolynomialLibrary(degree=3, include_bias=True)
            libraries.append(poly_lib)

        # Square root
        if "sqrt" in system_feature:
            # Safe sqrt (mcgs/utils/protected_expr)
            sqrt_functions = [lambda x: np.sqrt(np.abs(x))]
            sqrt_names = [lambda x: f"sqrt({x})"]
            sqrt_lib = ps.CustomLibrary(
                library_functions=sqrt_functions, function_names=sqrt_names
            )
            libraries.append(sqrt_lib)

        # 1/x functions
        if "div" in system_feature:
            # Safe division (mcgs/utils/protected_expr)
            tolerance = 1e-8
            div_functions = [
                lambda x: 1
                / (np.sqrt(x**2 + tolerance**2))
                * (np.sign(x) + (1 - np.abs(np.sign(x))))
            ]
            div_names = [lambda x: f"1/{x}"]
            div_lib = ps.CustomLibrary(
                library_functions=div_functions, function_names=div_names
            )
            libraries.append(div_lib)

        if len(libraries) > 1:
            feature_library = ps.GeneralizedLibrary(libraries)
        else:
            feature_library = libraries[0]

        # pySINDy
        optimiser = ps.STLSQ(
            threshold=0.03, alpha=0.05, max_iter=50
        )  # True min coeff = 0.06 (SIR sqrt)
        model = ps.SINDy(feature_library=feature_library, optimizer=optimiser)
        model.fit(state_trajectories, t=times, feature_names=state_variables)

        print(f"Total features: {len(model.get_feature_names())}")
        print(f"Feature names: {model.get_feature_names()}")

        # Store the top equations
        with open(f"{root}/top_equation.txt", "w") as f:
            equations = model.equations(precision=5)
            for state, eq in zip(state_variables, equations):
                f.write(f"({state})' = {eq}\n")

        # Initial conditions for prediction
        noiseless_data_state_names = [
            f"true_{state.upper()}" for state in state_variables
        ]
        initial_conditions = noiseless_data[noiseless_data_state_names].iloc[0].values

        # Predict trajectory using the top candidate
        try:
            pred_trajectory = model.simulate(
                initial_conditions, times, integrator="solve_ivp"
            )
        except Exception as e:
            print(f"Error occurred while simulating: {e}")
            pred_trajectory = model.simulate(
                initial_conditions, times, integrator="odeint"
            )

        print(f"Predicted trajectory shape: {pred_trajectory.shape}")

        # Write estimated_data.csv
        estimated_data = pd.DataFrame(
            {
                "time": times,
                **{
                    f"estimated_{state.upper()}": pred_trajectory[:, i]
                    for i, state in enumerate(state_variables)
                },
            }
        )
        estimated_data.to_csv(f"{root}/estimated_data.csv", index=False)

        # Time saving
        experiment_end_time = time.time()
        time_saving(
            experiment_end_time=experiment_end_time,
            experiment_start_time=experiment_start_time,
            save_dir=root,
        )
