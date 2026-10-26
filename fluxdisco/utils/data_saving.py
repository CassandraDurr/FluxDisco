"""Helper functions to save the results of the FluxDisco run."""

import os

import numpy as np
import pandas as pd
import sympy
from scipy.integrate import odeint


def estimation_saving(
    top_str_fluxes: list[str],
    states: list[str],
    state_variables: list[sympy.Symbol],
    initial_conditions_matrix: np.ndarray,
    times: np.ndarray,
    system_structure: dict[str, list[int]],
    save_dir: str,
) -> None:
    """Save estimated state trajectory data.

    Args:
        top_str_fluxes (list[str]): Flux expressions in string format (from the top solution).
        states (list[str]): List of state variable names.
        state_variables (list[sympy.Symbol]): SymPy symbols representing the state variables.
        initial_conditions_matrix (np.ndarray): Initial-condition matrix with shape
            (num_states, num_realisations).
        times (np.ndarray): Array of time points.
        system_structure (dict[str, list[int]]): Stoichiometry.
        save_dir (str): Directory where the estimated data will be saved.
    """
    # Convert the numerical strings back into SymPy expressions
    top_flux_exprs = [sympy.sympify(fs) for fs in top_str_fluxes]

    # Build the full system equations from the top fluxes
    system_eqs = {}
    for state_key in system_structure.keys():
        state_eq = sympy.Float(0.0)
        for i_flux, flux_coeff in enumerate(system_structure[state_key]):
            if flux_coeff != 0:
                state_eq += flux_coeff * top_flux_exprs[i_flux]
        system_eqs[state_key] = state_eq

    # Convert the numerical SymPy equations to functions
    dot_funcs = {}
    for state_key in states:
        dot_funcs[state_key] = sympy.lambdify(
            state_variables, system_eqs[state_key], "numpy"
        )

    # Integrate to get the estimated states
    def estimated_system(y, t):
        """Calculate derivative for given states."""
        return [dot_funcs[state](*y) for state in states]

    if initial_conditions_matrix.shape[0] != len(states):
        raise ValueError("Initial-condition matrix does not match the state count.")

    estimated_solutions = []
    for realisation in range(initial_conditions_matrix.shape[1]):
        estimated_solutions.append(
            odeint(
                estimated_system,
                initial_conditions_matrix[:, realisation],
                times,
            )
        )
    estimated_sol = np.stack(estimated_solutions, axis=-1)

    # Store estimated trajectories for every realisation.
    estimated_data = {"time": times}
    for idx, state in enumerate(states):
        for realisation in range(estimated_sol.shape[2]):
            estimated_data[f"estimated_{state.upper()}_{realisation}"] = estimated_sol[
                :, idx, realisation
            ]

    estimated_df = pd.DataFrame(estimated_data)
    estimated_df.to_csv(f"{save_dir}/estimated_data.csv", index=False)
    np.save(f"{save_dir}/estimated_initial_conditions.npy", initial_conditions_matrix)
    initial_conditions_df = pd.DataFrame(
        initial_conditions_matrix.T, columns=[f"initial_{state}" for state in states]
    )
    initial_conditions_df.index.name = "realisation"
    initial_conditions_df.to_csv(f"{save_dir}/estimated_initial_conditions.csv")
    print(f"Saved estimated data to {save_dir}/estimated_data.csv")


def time_saving(
    experiment_start_time: float, experiment_end_time: float, save_dir: str
) -> None:
    """Save the time it took to run an experiment.

    Args:
        experiment_start_time (float): Start time of the experiment (in seconds).
        experiment_end_time (float): End time of the experiment (in seconds).
        save_dir (str): Directory where the timing data will be saved.
    """
    elapsed = experiment_end_time - experiment_start_time
    elapsed_minutes = elapsed / 60.0
    elapsed_hours = elapsed / 3600.0

    print(
        f"Time taken: {elapsed:.2f} seconds ({elapsed_minutes:.2f} min, {elapsed_hours:.2f} hr)\n"
    )

    # Create save dir if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save timing to text file
    timing_path = os.path.join(save_dir, "timing.txt")
    with open(timing_path, "w") as f:
        f.write(f"Time taken (seconds): {elapsed:.2f}\n")
        f.write(f"Time taken (minutes): {elapsed_minutes:.2f}\n")
        f.write(f"Time taken (hours): {elapsed_hours:.2f}\n")
