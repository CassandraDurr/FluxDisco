"""Perform inference using Symbolic Physics Learner (SPL) for benchmarked ODE systems."""

import os
import random
import time

import numpy as np
import pandas as pd
import sympy
from pysindy.differentiation import SmoothedFiniteDifference
from scipy.integrate import odeint
from scipy.signal import savgol_filter
from spl_base import SplBase
from spl_score import score_with_est
from utils import copy_data_files_to_results_dir, time_saving

# Seed for reproducibility
np.random.seed(1998)
random.seed(1998)

# Number of top candidates to save
top_n = 10

# System grammars
system_grammars = {
    "sir": ["A->(A+A)", "A->(A-A)", "A->(A*A)", "A->sqrt(A)"],  # Allow sqrt
    "lotka-volterra": ["A->(A+A)", "A->(A-A)", "A->(A*A)"],
    "brusselator": ["A->(A+A)", "A->(A-A)", "A->(A*A)"],
    "fairen-velarde": [
        "A->(A+A)",
        "A->(A-A)",
        "A->(A*A)",
        "A->A/A",
    ],  # Allow rational polynomials
}

# Create results/spl directory if it doesn't exist
spl_file_dir = "results/spl"
if not os.path.exists(spl_file_dir):
    os.makedirs(spl_file_dir)

# Walk through FluxDisco data directory and copy data files to Symbolic Physics Leaner results dir
copy_data_files_to_results_dir(data_dir="../data", target_dir=spl_file_dir)

for root, _, files in os.walk(spl_file_dir):
    # Go to level where folders are named experiment_{noise_level}
    if "experiment_" in root:
        print(f"Processing {root}...")
        noise_level = root.split("experiment_")[-1]
        # Check for estimated_data.csv to avoid re-processing
        if os.path.exists(f"{root}/estimated_data.csv"):
            # Check if there is no missing data in estimated_data.csv
            estimated_data = pd.read_csv(f"{root}/estimated_data.csv")
            if estimated_data.isnull().values.any():
                print(f"Estimated data has missing values for {root}, re-processing...")
                # Try change seed and re-run the experiment
                np.random.seed(123)
                random.seed(123)
            else:
                print(f"Estimated data already exists for {root}, skipping...")
                continue

        experiment_start_time = time.time()

        # Get system grammar
        system_grammar = ["A->(A+A)", "A->(A-A)", "A->(A*A)"]
        for system in system_grammars.keys():
            if system in root:
                system_grammar = system_grammars[system]
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

        if noise_level == "no_noise":
            # Don't need to smooth
            derivatives = np.zeros_like(state_trajectories)
            for i, state in enumerate(state_variables):
                derivatives[:, i] = np.gradient(state_trajectories[:, i], times)
        else:
            # Use finite differences to estimate derivatives
            # Use method followed in:
            # https://github.com/isds-neu/SymbolicPhysicsLearner/blob/main/dynamics_task/dp_makedata.ipynb
            sfd = SmoothedFiniteDifference(smoother_kws={"window_length": 5})
            derivatives = sfd._differentiate(state_trajectories, times)
            print(f"Estimated derivatives shape: {derivatives.shape}")
            print(f"State trajectories shape: {state_trajectories.shape}")

            # Use Savitzky-Golay filter to smooth the trajectories
            for i in range(state_trajectories.shape[1]):
                state_trajectories[:, i] = savgol_filter(
                    state_trajectories[:, i], window_length=5, polyorder=2
                )

        # List to store the right-hand side (RHS) of the discovered differential equations
        rhs_exprs = []

        # Map state variables to 'x', 'y', 'z' as expected by score_with_est
        mapped_vars = [chr(ord("x") + j) for j in range(len(state_variables))]

        # Run the model on each state variable separately
        for i, state in enumerate(state_variables):
            print(f"Processing state variable: {state}...")
            # Prepare data for SPL
            data_X = state_trajectories.T
            data_y = derivatives[:, i].reshape(1, -1)
            data_sample = np.vstack([data_X, data_y])

            # Grammar
            math_operators = system_grammar
            terminal_nodes = [f"A->{var}" for var in mapped_vars]
            terminal_nodes.append("A->C")
            base_grammars = math_operators + terminal_nodes
            nt_nodes = ["A"]

            # Initialise the model
            spl_model = SplBase(
                data_sample=data_sample,
                base_grammars=base_grammars,
                aug_grammars=[],
                nt_nodes=nt_nodes,
                max_len=50,
                max_module=10,
                aug_grammars_allowed=5,
                func_score=score_with_est,
            )

            # Run the MCTS Search
            # num_episodes = np.floor(100 / len(state_variables)).astype(int)
            num_episodes = 100
            reward_his, best_solution, good_modules = spl_model.run(
                num_episodes=num_episodes
            )

            # best_solution is a tuple: (equation_string, reward)
            best_eq_str = best_solution[0]
            if best_eq_str == "nothing":
                best_eq_str = "0"  # Handle the case where no equation is found
            sympy_equation = sympy.sympify(best_eq_str)
            rhs_exprs.append(sympy_equation)

            print(f"Discovered Equation for {state}: d({state})/dt = {sympy_equation}")

        # Save the best models to a CSV file
        equations_df = pd.DataFrame(
            {
                "State_Variable": state_variables,
                "Discovered_Equation": [str(expr) for expr in rhs_exprs],
            }
        )
        equations_df.to_csv(f"{root}/top_equations.csv", index=False)

        # Initial conditions for prediction
        noiseless_data_state_names = [
            f"true_{state.upper()}" for state in state_variables
        ]
        initial_conditions = noiseless_data[noiseless_data_state_names].iloc[0].values

        # Predict trajectory using the top candidate
        pred_eq_func = sympy.lambdify(mapped_vars, rhs_exprs, "numpy")

        # Define the system of ODEs for the solver
        def predicted_system(y, t):
            """Calculate derivative for given states."""
            return pred_eq_func(*y)

        # Simulate the system using the found equations and clean initial conditions
        pred_trajectory = odeint(predicted_system, initial_conditions, times)

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
