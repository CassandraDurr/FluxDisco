"""Perform inference using PySR for benchmarked ODE systems."""

import os
import random
import time

import numpy as np
import pandas as pd
import sympy
from pysr import PySRRegressor
from scipy.integrate import odeint
from utils import copy_data_files_to_results_dir, time_saving

# Seed for reproducibility
np.random.seed(1998)
random.seed(1998)

# Define system allowed operators
binary_operators = {
    "sir": ["+", "-", "*"],
    "lotka-volterra": ["+", "-", "*"],
    "brusselator": ["+", "-", "*"],
    "fairen-velarde": ["+", "-", "*", "/"],  # Allow rational polynomials
}

unary_operators = {
    "sir": ["sqrt"],
    "lotka-volterra": [],
    "brusselator": [],
    "fairen-velarde": [],
}

# Create results/pysr directory if it doesn't exist
pysr_file_dir = "results/pysr"
if not os.path.exists(pysr_file_dir):
    os.makedirs(pysr_file_dir)

# Walk through FluxDisco data directory and copy data files to pySR results directories
copy_data_files_to_results_dir(data_dir="../data", target_dir=pysr_file_dir)


for root, _, _ in os.walk(pysr_file_dir):
    # Go to level where folders are named experiment_{noise_level}
    if "experiment_" in root:
        print(f"Processing {root}...")
        # Check for estimated_data.csv to avoid re-processing
        if os.path.exists(f"{root}/estimated_data.csv"):
            # Check if there is no missing data in estimated_data.csv
            estimated_data = pd.read_csv(f"{root}/estimated_data.csv")
            if estimated_data.isnull().values.any():
                print(f"Estimated data has missing values for {root}, re-processing...")
            else:
                print(f"Estimated data already exists for {root}, skipping...")
                continue

        experiment_start_time = time.time()

        # Establish the binary and unary operators for the current system
        binary_ops = ["+", "-", "*"]  # Default to polynomials
        unary_ops = []
        for system in binary_operators.keys():
            if system in root:
                binary_ops = binary_operators[system]
                unary_ops = unary_operators[system]
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

        # Get derivatives using finite differences
        dt = times[1] - times[0]
        derivatives = np.gradient(state_trajectories, dt, axis=0)

        # pySR
        model = PySRRegressor(
            binary_operators=binary_ops,
            unary_operators=unary_ops,
            denoise=True,
            verbosity=0,
        )

        model.fit(state_trajectories, derivatives, variable_names=state_variables)

        # Best equations
        for i, state in enumerate(state_variables):
            state_df = model.equations_[i].copy()
            state_df = state_df.sort_values(by="loss", ascending=False)
            state_df.to_csv(f"{root}/state_{state}_top_equations.csv", index=False)

        # Sympy expressions
        sympy_exprs = model.sympy()
        # Store these in a file for later use
        with open(f"{root}/sympy_expressions.txt", "w") as f:
            for i, state in enumerate(state_variables):
                f.write(f"Derivative of {state}:\n")
                f.write(str(sympy_exprs[i]) + "\n\n")

        # Create a function to compute the derivatives
        pred_eq_func = sympy.lambdify(state_variables, sympy_exprs, "numpy")

        # Define the system of ODEs for the solver
        def predicted_system(y, t):
            """Calculate derivative for given states."""
            return pred_eq_func(*y)

        # Initial conditions for prediction
        noiseless_data_state_names = [
            f"true_{state.upper()}" for state in state_variables
        ]
        initial_conditions = noiseless_data[noiseless_data_state_names].iloc[0].values

        # Simulate the system using the found equations and clean initial conditions
        try:
            pred_trajectory = odeint(predicted_system, initial_conditions, times)
        except Exception as e:
            print(f"Error occurred while simulating the system for {root}: {e}")
            continue

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
