"""Perform inference using ODEFormer for benchmarked ODE systems."""

import io
import os
import random
import time
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import sympy
from odeformer.model import SymbolicTransformerRegressor
from odeformer_param_opt import ConstantOptimizer
from utils import copy_data_files_to_results_dir, time_saving

# --------------------------------
# ODEFormer specific functions
# --------------------------------


def get_odeformer_model(
    beam_size: int = 50, beam_temperature: float = 0.1, rescale: bool = True
) -> SymbolicTransformerRegressor:
    """Return a pre-trained ODEFormer model with specified beam parameters.

    Args:
        beam_size (int, optional): Beam size. Defaults to 50.
        beam_temperature (float, optional): Beam temperature. Defaults to 0.1.
        rescale (bool, optional): Whether to rescale. Defaults to True.

    Returns:
        SymbolicTransformerRegressor: Pre-trained ODEFormer model.
    """
    # Initialise the SymbolicTransformerRegressor with pre-trained weights
    dstr = SymbolicTransformerRegressor(from_pretrained=True, rescale=rescale)
    dstr.set_model_args({"beam_size": beam_size, "beam_temperature": beam_temperature})

    return dstr


def get_odeformer_equations(dstr: SymbolicTransformerRegressor, n_predictions=1) -> str:
    """Return the ODEFormer model equations from the ODEFormer output."""
    f = io.StringIO()
    with redirect_stdout(f):
        dstr.print(n_predictions=n_predictions)
    output = f.getvalue()
    return output


def odeformer_equations_csv(
    root: str, dstr: SymbolicTransformerRegressor, n_states: int, n_predictions=1
) -> None:
    # Get the top candidate equations as text
    top_candidates_text = get_odeformer_equations(dstr, n_predictions=n_predictions)
    equation_lines = [
        line.strip()
        for line in top_candidates_text.splitlines()
        if line.strip().startswith("x_")
    ]

    rows = []

    for rank in range(n_predictions):
        start = rank * n_states
        block = equation_lines[start : start + n_states]
        if len(block) < n_states:
            break

        rows.append(
            {
                "rank": rank + 1,
                "equations": "; ".join(block),
            }
        )

    pd.DataFrame(rows).to_csv(f"{root}/top_{n_predictions}_candidates.csv", index=False)


# --------------------------------
# ODEFormer experiment code
# --------------------------------

# Seed for reproducibility
np.random.seed(1998)
random.seed(1998)

# Number of top candidates to save
top_n = 10

# Create results directories if they don't exist
odeformer_result_dirs = ["results/odeformer", "results/odeformer_opt"]
for odeformer_result_dir in odeformer_result_dirs:
    if not os.path.exists(odeformer_result_dir):
        os.makedirs(odeformer_result_dir)

# Walk through FluxDisco data directory and copy data files to ODEFormer results directories
for odeformer_result_dir in odeformer_result_dirs:
    copy_data_files_to_results_dir(data_dir="../data", target_dir=odeformer_result_dir)

for odeformer_result_dir in odeformer_result_dirs:
    for root, _, _ in os.walk(odeformer_result_dir):
        # Go to level where folders are named experiment_{noise_level}
        if "experiment_" in root:
            print(f"Processing {root}...")
            # Check for estimated_data.csv to avoid re-processing
            if os.path.exists(f"{root}/estimated_data.csv"):
                print(f"Estimated data already exists for {root}, skipping...")
                continue

            experiment_start_time = time.time()

            # Get data
            noisy_realisations = np.load(f"{root}/data_plot.npy", allow_pickle=True)[0]
            noiseless_data = pd.read_csv(f"{root}/noiseless_data.csv")

            # Get time and states from the realisation
            state_variables = [
                var for var in noisy_realisations.keys() if var != "time"
            ]
            print(f"State variables: {state_variables}")
            state_trajectories = np.stack(
                [noisy_realisations[var] for var in state_variables], axis=1
            )
            print(f"State trajectories shape: {state_trajectories.shape}")
            times = noisy_realisations["time"]

            # Get model (use default hyperparameters)
            dstr = get_odeformer_model(rescale=True)

            # Fit the model to the trajectory data
            dstr.fit(times=times, trajectories=state_trajectories)

            # Store the top candidates
            odeformer_equations_csv(
                root=root, dstr=dstr, n_states=len(state_variables), n_predictions=top_n
            )

            # Initial conditions for prediction
            noiseless_data_state_names = [
                f"true_{state.upper()}" for state in state_variables
            ]
            initial_conditions = (
                noiseless_data[noiseless_data_state_names].iloc[0].values
            )

            # Prediction using the top candidate
            if "opt" in odeformer_result_dir:
                # Get the top predicted equation
                try:
                    equations_file = os.path.join(root, f"top_{top_n}_candidates.csv")
                    equations_df = pd.read_csv(equations_file)
                    raw_top_eq = equations_df.loc[
                        equations_df["rank"] == 1, "equations"
                    ].iloc[0]

                    # Format the top equation for ConstantOptimizer
                    rhs_exprs = [
                        eq.split("=")[1].strip() for eq in raw_top_eq.split(";")
                    ]
                    pred_eq = " | ".join([str(sympy.parse_expr(e)) for e in rhs_exprs])

                    # Run the ConstantOptimizer
                    print("Optimising constants...")
                    param_optimiser = ConstantOptimizer(
                        eq=pred_eq,
                        y0=initial_conditions,
                        time=times,
                        observed_trajectory=state_trajectories,
                        init_random=False,
                        optimization_objective="mse",
                        eval_objective="mse",
                        track_eval_history=True,
                    )

                    # Get the optimised parameters and trajectory
                    final_eq, _, pred_trajectory = param_optimiser.optimize()
                    print(f"Original Equations: {pred_eq}")
                    print(f"Optimised Equations: {final_eq}")

                    # Save the optimised equations to a csv file
                    final_eq_save = "; ".join(
                        [
                            f"x_{i}' = {eq.strip()}"
                            for i, eq in enumerate(final_eq.split(" | "))
                        ]
                    )
                    with open(
                        os.path.join(root, "top_1_optimised_equation.csv"), "w"
                    ) as f:
                        f.write("rank,equations\n")
                        f.write(f"1,{final_eq_save}\n")

                except Exception as e:
                    print(
                        f"WARNING: Optimisation failed. Using default prediction. Error: {e}"
                    )
                    pred_trajectory = dstr.predict(times, initial_conditions)

            else:
                # Predict trajectory using the top candidate without constant optimisation
                pred_trajectory = dstr.predict(times, initial_conditions)

            if pred_trajectory is not None:
                print(f"Predicted trajectory shape: {pred_trajectory.shape}")
                print(f"Initial conditions shape: {initial_conditions.shape}")
                if pred_trajectory.shape != state_trajectories.shape:
                    print(
                        f"WARNING: Predicted trajectory has shape {pred_trajectory.shape} but expected {state_trajectories.shape}.\n"  # noqa: E501
                    )
                    # Try manually predict the trajectory using the equations and initial conditions
                    try:
                        from scipy.integrate import odeint

                        # Get the top predicted equations
                        try:
                            equations_file = os.path.join(
                                root, "top_1_optimised_equation.csv"
                            )
                            equations_df = pd.read_csv(equations_file)
                        except FileNotFoundError:
                            equations_file = os.path.join(
                                root, f"top_{top_n}_candidates.csv"
                            )
                            equations_df = pd.read_csv(equations_file)
                        raw_top_eq = equations_df.loc[
                            equations_df["rank"] == 1, "equations"
                        ].iloc[0]
                        rhs_exprs = [
                            eq.split("=")[1].strip() for eq in raw_top_eq.split(";")
                        ]
                        # Get ODEFormer states
                        odeformer_states = [
                            f"x_{idx}" for idx in range(len(state_variables))
                        ]
                        # Create sympy expressions for the predicted equations
                        pred_eqs = [
                            sympy.parse_expr(
                                eq,
                                local_dict={
                                    s: sympy.Symbol(s) for s in odeformer_states
                                },
                            )
                            for eq in rhs_exprs
                        ]
                        # Create a function to compute the derivatives
                        pred_eq_func = sympy.lambdify(
                            odeformer_states, pred_eqs, "numpy"
                        )

                        # Define the system of ODEs for the solver
                        def predicted_system(y, t):
                            """Calculate derivative for given states."""
                            return pred_eq_func(*y)

                        # Simulate the system using the found equations and clean initial conditions
                        pred_trajectory = odeint(
                            predicted_system, initial_conditions, times
                        )
                        print(
                            f"Manually predicted trajectory shape: {pred_trajectory.shape}"
                        )
                    except Exception as e:
                        print(
                            f"ERROR: Failed to get trajectories from the top predicted equation: {e}"
                        )

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

            else:
                print(
                    "WARNING: ODEFormer failed to predict a trajectory for this experiment."
                )
