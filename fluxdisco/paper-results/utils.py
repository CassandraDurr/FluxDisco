"""Utility functions for running paper experiments."""

import os
import random

import numpy as np
import pandas as pd
from scipy.integrate import odeint

from fluxdisco import run_fluxdisco


def sir_odes(
    par: list[float], time: np.ndarray, constants: tuple, infection_type: str
) -> list[float]:
    """SIR ODE model for epidemic spread.

    Args:
        par (list[float]): Initial conditions for s, i, and r.
        time (np.ndarray): Time points for the solution.
        constants (tuple): Constants for the SIR model.
        infection_type (str): Type of infection dynamics to use (Standard, Squared, Sqrt).

    Raises:
        NotImplementedError: Infection type is not implemented.

    Returns:
        list[float]: Derivatives of s, i, and r.
    """
    s, i, _ = par  # s = susceptible, i = infected, r = recovered
    beta, gamma = constants

    # --- Fluxes ---
    # Infection rate
    if infection_type == "Standard":
        j0 = beta * s * i
    elif infection_type == "Squared":
        j0 = beta * s * (i**2)
    elif infection_type == "Sqrt":
        j0 = beta * s * np.sqrt(i)
    else:
        raise NotImplementedError(
            f"Unknown infection type: {infection_type}. Allowed types: Standard, Squared, Sqrt"
        )
    # Recovery rate
    j1 = gamma * i

    # --- ODEs ---
    dsdt = -j0
    didt = j0 - j1
    drdt = j1
    return [dsdt, didt, drdt]


def lotka_volterra_odes(
    par: list[float], time: np.ndarray, constants: tuple[float, float, float]
) -> list[float]:
    """Lotka-Volterra system of ODEs.

    Args:
        par (list[float]): Initial conditions for x and y.
        time (np.ndarray): Time points for the solution.
        constants (tuple[float, float, float]): Constants for the Lotka-Volterra system.

    Returns:
        list[float]: Derivatives of x and y.
    """
    a, b, c = constants
    x, y = par
    dxdt = a * x - b * x * y
    dydt = -c * y + b * x * y
    return [dxdt, dydt]


def brusselator_odes(
    par: list[float], time: np.ndarray, constants: tuple[float, float]
) -> list[float]:
    """Brusselator system of ODEs.

    Args:
        par (list[float]): Initial conditions for x and y.
        time (np.ndarray): Time points for the solution.
        constants (tuple[float, float]): Constants for the Brusselator system.

    Returns:
        list[float]: Derivatives of x and y.
    """
    a, b = constants
    x, y = par
    dxdt = x**2 * y - b * x - x + a
    dydt = -(x**2) * y + b * x
    return [dxdt, dydt]


def fairen_velarde_odes(
    par: list[float], time: np.ndarray, constants: tuple[float, float, float]
) -> list[float]:
    """Dimensionless Fairen-Velarde model for bacterial respiration.

    Args:
        par (list[float]): Initial conditions for x and y.
        time (np.ndarray): Time points for the solution.
        constants (tuple[float, float, float]): Constants for the Fairen-Velarde model.

    Returns:
        list[float]: Derivatives of x and y.
    """
    x, y = par  # x = oxygen, y = nutrients
    B, A, q = constants  # Dimensionless constants (notation from original paper)

    # --- Fluxes ---
    # Consumption
    j_cons = (x * y) / (1 + q * x**2)
    # Oxygen dynamics
    j_oxy = B - x  # Input + diffusion/ loss
    # Nutrient dynamics
    j_nut = A  # No losses

    # --- ODEs ---
    dxdt = j_oxy - j_cons
    dydt = j_nut - j_cons

    return [dxdt, dydt]


def generate_data(
    system_name: str,
    regime: str,
    noise_level: str,
    data_seed: int,
    num_realisations: int,
    configurations: dict,
) -> tuple[list[dict], list[dict], list[list]]:
    """Generate data for a given system (model) x regime x noise level.

    Args:
        system_name (str): ODE system name.
        regime (str): Regime name.
        noise_level (str): Noise level name.
        data_seed (int): Data seed.
        num_realisations (int): Number of realisations to simulate.
        configurations (dict): Configurations dictionary.

    Returns:
        tuple[list[dict], list[dict], list[list], np.ndarray]: Generated data, plotting data, initial conditions, noiseless solution.
    """  # noqa: E501
    # Set seed
    random.seed(data_seed)
    np.random.seed(data_seed)

    # Generate data realisations
    data_X = []
    data_plot = []
    all_initial_conditions = []

    # States of the model
    model_config = configurations[system_name]
    states = model_config["states"]

    for _ in range(num_realisations):
        all_initial_conditions.append(model_config["initial_conditions"][regime])

        # Noiseless solution
        constants = model_config["regimes"][regime]
        if system_name == "sir":
            solver_args = (constants, regime)
        else:
            solver_args = (constants,)
        noiseless_solution = odeint(
            model_config["ode_func"],
            model_config["initial_conditions"][regime],
            model_config["times"],
            args=solver_args,
        )

        # Add noise
        noise_val = model_config["noise_levels"][noise_level]
        solution_range = np.ptp(noiseless_solution, axis=0)
        noisy_solution = (
            noiseless_solution
            + noise_val
            * solution_range
            * np.random.normal(size=noiseless_solution.shape)
        )

        # Ensure positivity
        noisy_solution = np.maximum(noisy_solution, 0.0)

        if system_name == "sir":
            # SIR: Ensure proportions sum to 1
            row_sums = noisy_solution.sum(axis=1, keepdims=True)
            noisy_solution = noisy_solution / row_sums

        # Model data
        data_X.append(
            {
                "time": model_config["times"],
                "initial_conditions": model_config["initial_conditions"][regime],
                "states": {
                    state: noisy_solution[:, idx] for idx, state in enumerate(states)
                },
            }
        )

        # Plotting data
        data_plot.append(
            {
                "time": model_config["times"],
                **{state: noisy_solution[:, idx] for idx, state in enumerate(states)},
            }
        )

    return data_X, data_plot, all_initial_conditions, noiseless_solution


def run_single_experiment(args):
    """Run a single FluxDisco experiment for a given system x regime x noise level."""
    system_name, config, noise_level, regime, results_dir = args

    # Results path
    noise_dir_name = noise_level.lower().replace(" ", "_")
    regime_suffix = f"_{regime.lower()}" if regime != "Default" else ""
    experiment_name = f"{system_name}{regime_suffix}/experiment_{noise_dir_name}"
    save_dir = os.path.join(results_dir, experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    # Build system config dict
    system_config = {
        "states": config["states"],
        "state_variables": config["state_variables"],
        "state_to_variable": dict(zip(config["states"], config["state_variables"])),
        "num_fluxes": config["num_fluxes"],
        "system_structure": config["system_structure"],
        "grammar": config["grammar"],
        "flux_priors": config["flux_priors"],
    }
    # Fairen-Velarde system has custom initial state
    if "initial_state" in config:
        system_config["initial_state"] = config["initial_state"]

    # Run FluxDisco
    system_data = config["data"][(regime, noise_level)]
    top_equations, results_df = run_fluxdisco(
        data=system_data["data_X"],
        system_config=system_config,
        search_params=config.get("search_params", {}),
        experiment_name=experiment_name,
        save_dir=results_dir,
    )

    # Save data
    np.save(f"{save_dir}/data_plot.npy", system_data["data_plot"])
    np.save(f"{save_dir}/all_data_X.npy", system_data["data_X"])
    np.save(
        f"{save_dir}/all_initial_conditions.npy", system_data["all_initial_conditions"]
    )
    noiseless_data = {
        "time": config["times"],
    }
    for idx, state in enumerate(config["states"]):
        noiseless_data[f"true_{state.upper()}"] = system_data["noiseless_solution"][
            :, idx
        ]
    noiseless_df = pd.DataFrame(noiseless_data)
    noiseless_df.to_csv(f"{save_dir}/noiseless_data.csv", index=False)

    print(f"Finished {experiment_name}")
    return top_equations, results_df
