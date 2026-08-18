"""Perform inference using ProGED for benchmarked ODE systems."""

import csv
import os
import random
import time

import numpy as np
import pandas as pd
import sympy
from ProGED import EqDisco
from ProGED.generators.grammar import GeneratorGrammar
from ProGED.generators.grammar_construction import construct_production
from scipy.integrate import odeint
from utils import copy_data_files_to_results_dir, time_saving

# Seed for reproducibility
np.random.seed(1998)
random.seed(1998)

# ProGED hyperparameters
sample_size = 1000

# Number of top candidates to save
top_n = 10


# Construct grammars
# Unable to specify different rules per ODE in ODE system
def custom_grammar(
    p_S=[1, 1],  # S -> Rational | Expression
    p_E=[1, 1],  # E -> E + M | M
    p_M=[1, 1],  # M -> M * F | F
    p_F=[1, 1, 1],  # F -> V | 'C' | sqrt(E)
    p_vars=[1, 1],  # Probs per variable
    variables=["'x'", "'y'"],  # Variables
):
    # [S]tart: Rational or (standard) Expression
    grammar = construct_production(
        left="S", items=["'(' E ')' '/' '(' E ')'", "E"], probs=p_S
    )
    # [E]xpression: Addition
    grammar += construct_production(left="E", items=["E '+' M", "M"], probs=p_E)
    # [M]onomial/Term: Multiplication
    grammar += construct_production(left="M", items=["M '*' F", "F"], probs=p_M)
    # [F]actor: Variable, Constant, or sqrt of an expression
    grammar += construct_production(
        left="F", items=["V", "'C'", "'sqrt(' E ')'"], probs=p_F
    )
    # [V]ariable
    grammar += construct_production(left="V", items=variables, probs=p_vars)

    return grammar


system_grammars = {
    # SIR: allow sqrt but not rational
    "sir": GeneratorGrammar(
        custom_grammar(
            variables=["'s'", "'i'", "'r'"],
            p_vars=[1, 1, 1],
            p_S=[0, 1],  # no rational
            p_F=[1, 1, 1],  # allow sqrt
        )
    ),
    # LV: only polynomial
    "lotka-volterra": GeneratorGrammar(
        custom_grammar(
            variables=["'x'", "'y'"],
            p_vars=[1, 1],
            p_S=[0, 1],  # no rational
            p_F=[1, 1, 0],  # no sqrt
        )
    ),
    # LV: only polynomial
    "brusselator": GeneratorGrammar(
        custom_grammar(
            variables=["'x'", "'y'"],
            p_vars=[1, 1],
            p_S=[0, 1],  # no rational
            p_F=[1, 1, 0],  # no sqrt
        )
    ),
    # LV: allow rational but not sqrt
    "fairen-velarde": GeneratorGrammar(
        custom_grammar(
            variables=["'x'", "'y'"],
            p_vars=[1, 1],
            p_S=[1, 1],  # allow rational
            p_F=[1, 1, 0],  # no sqrt
        )
    ),
}

# Create results/proged directory if it doesn't exist
proged_file_dir = "results/proged"
if not os.path.exists(proged_file_dir):
    os.makedirs(proged_file_dir)

# Walk through FluxDisco data directory and copy data files to ProGED results directories
copy_data_files_to_results_dir(data_dir="../data", target_dir=proged_file_dir)

for root, _, files in os.walk(proged_file_dir):
    # Go to level where folders are named experiment_{noise_level}
    if "experiment_" in root:
        print(f"Processing {root}...")
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
        system_grammar = None
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

        # Get model (use default hyperparameters for now)
        data = pd.DataFrame(noisy_realisations, columns=["time"] + state_variables)
        # Rename time to t
        data.rename(columns={"time": "t"}, inplace=True)

        ED = EqDisco(
            data=data,
            lhs_vars=state_variables,
            rhs_vars=state_variables,  # Can add t here if we think the DEs are a function of time
            task_type="differential",
            sample_size=sample_size,
            system_size=len(state_variables),
            generator=system_grammar if system_grammar else "grammar",
            generator_template_name=None if system_grammar else "polynomial",
            verbosity=0,
            strategy_settings={"max_repeat": 100},
            success_threshold=1e-6,
        )

        # Update max_constants
        ED.estimation_settings["parameter_estimation"]["max_constants"] = 10

        # Fit the model to the trajectory data
        print("Generating and fitting models...")
        print(ED.generate_models())
        print(ED.fit_models())

        # Store the top candidates
        sorted_models = sorted(ED.models, key=lambda m: m.get_error())
        if not sorted_models or sorted_models[0].get_error() == np.inf:
            print(
                "No valid models were found; fitting process may have failed for all candidates."
            )
            print("Try increasing verbosity to debug the parameter estimation step.")

        best_models = ED.get_results(N=top_n)
        print(best_models)
        # Save the best models to a CSV file
        results_file_path = os.path.join(root, f"top_{top_n}_models.csv")
        header = ["rank", "equations", "error"]
        with open(results_file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i, model in enumerate(best_models):
                row = [
                    str(i + 1),
                    str(model.get_full_expr()),
                    f"{model.get_error():.6f}",
                ]
                writer.writerow(row)

        # Initial conditions for prediction
        noiseless_data_state_names = [
            f"true_{state.upper()}" for state in state_variables
        ]
        initial_conditions = noiseless_data[noiseless_data_state_names].iloc[0].values

        # Predict trajectory using the top candidate
        rhs_exprs = best_models[0].get_full_expr()  # List of sympy expressions
        print(f"Top predicted equations: {rhs_exprs}")

        # Create a function to compute the derivatives
        pred_eq_func = sympy.lambdify(state_variables, rhs_exprs, "numpy")

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
