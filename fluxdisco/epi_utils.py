"""Utility functions for running epidemic experiments."""

import os

import numpy as np
import pandas as pd

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
