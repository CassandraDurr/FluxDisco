"""Run SIR experiments given incidence data, prevalence data, or both."""

import argparse
import os
import random
import sys

import numpy as np
import sympy
from joblib import Parallel, delayed, parallel_config
from scipy.integrate import odeint

# Add utils directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import run_single_experiment, sir_odes  # noqa: E402

# -----------------------------
# System configurations
# -----------------------------

# Sympy state symbols
s_sym, i_sym, r_sym = sympy.symbols("s i r")
total_population = 100  # Total population for SIR model

configurations = {
    "sir": {
        "states": ["s", "i", "r"],
        "state_variables": [s_sym, i_sym, r_sym],
        "num_fluxes": 2,
        "system_structure": {"s": [-1, 0], "i": [1, -1], "r": [0, 1]},
        "grammar": {
            "M": [f"M -> {state}" for state in ["s", "i", "r"]]
            + [
                "M -> C",
                "M -> sqrt(M)",
                "M -> M + M",
                "M -> M - M",
                "M -> M * M",
            ],
        },
        "flux_priors": {
            0: {
                "M -> s": 1,
                "M -> i": 1,
                "M -> r": 0,
                "M -> C": 1,
                "M -> M + M": 1,
                "M -> M - M": 1,
                "M -> M * M": 1,
                "M -> sqrt(M)": 1,
            },
            1: {
                "M -> s": 0,
                "M -> i": 1,
                "M -> r": 1,
                "M -> C": 1,
                "M -> M + M": 1,
                "M -> M - M": 1,
                "M -> M * M": 1,
                "M -> sqrt(M)": 1,
            },
        },
        "ode_func": sir_odes,
        "regimes": {
            "Standard": (0.4, 0.1),
            "Squared": (1.3, 0.08),
            "Sqrt": (0.15, 0.06),
        },
        "initial_conditions": {
            "Standard": [98 / total_population, 2 / total_population, 0],
            "Squared": [88 / total_population, 12 / total_population, 0],
            "Sqrt": [98 / total_population, 2 / total_population, 0],
        },
        "times": np.linspace(0, 80, 80),
        "noise_levels": {"No Noise": 0, "Low Noise": 0.02, "High Noise": 0.05},
        "search_params": {"checkpoint_saving": False},
    },
}

# Assert sum of initial conditions equals 1 for each regime
for regime, initial_conditions in configurations["sir"]["initial_conditions"].items():
    assert np.isclose(
        sum(initial_conditions), 1.0
    ), f"Initial conditions for regime '{regime}' do not sum to 1."

# -----------------------------
# Separate out configurations for:
# - num_realisations = [1, 5, 10]
# - data_type = ["prevalence_only", "incidence_only", "both"]
# -----------------------------
num_realisations_list = [1, 5, 10]
data_types = ["prevalence_only", "incidence_only", "both"]
for num_realisations in num_realisations_list:
    for data_type in data_types:
        configurations[f"sir_{data_type}_{num_realisations}"] = {
            **configurations["sir"],
            "num_realisations": num_realisations,
            "data_type": data_type,
        }

# Remove base "sir" configuration to avoid duplication
del configurations["sir"]

# -----------------------------
# Generate data
# -----------------------------
# General data parameters
data_seed = 1234


def generate_latent_sir_data(
    system_name: str,
    regime: str,
    noise_level: str,
    data_seed: int,
    total_population: int,
    configurations: dict,
) -> tuple[list[dict], list[dict], list[list]]:
    """Generate data for a given system (model) x regime x noise level.

    Args:
        system_name (str): ODE system name.
        regime (str): Regime name.
        noise_level (str): Noise level name.
        data_seed (int): Data seed.
        total_population (int): Total population size for the SIR model.
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
    base_initial_conditions = model_config["initial_conditions"][regime]
    num_realisations = model_config.get("num_realisations", 1)
    data_type = model_config.get("data_type", "both")

    for _ in range(num_realisations):
        # Set initial conditions
        if num_realisations > 1:
            realisation_init_conditions = [
                base_initial_conditions[0] * random.uniform(0.75, 1.25),  # s
                base_initial_conditions[1] * random.uniform(0.75, 1.25),  # i
                base_initial_conditions[2],  # r
            ]
            # Normalise
            total = sum(realisation_init_conditions)
            realisation_init_conditions = [
                init_real / total for init_real in realisation_init_conditions
            ]
        else:
            # Only use base initial conditions for single realisation
            realisation_init_conditions = base_initial_conditions

        all_initial_conditions.append(realisation_init_conditions)

        # Noiseless solution
        constants = model_config["regimes"][regime]
        noiseless_solution = odeint(
            model_config["ode_func"],
            realisation_init_conditions,
            model_config["times"],
            args=(constants, regime),
        )

        # Extract prevalence and incidence
        clean_susceptibles = noiseless_solution[:, 0]
        noise_val = model_config["noise_levels"][noise_level]

        if noise_val > 0:
            # Get noisy prevalence
            solution_range = np.ptp(noiseless_solution, axis=0)
            noisy_solution = (
                noiseless_solution
                + noise_val
                * solution_range
                * np.random.normal(size=noiseless_solution.shape)
            )

            # Ensure positivity & proportions sum to 1
            noisy_solution = np.maximum(noisy_solution, 0.0)
            row_sums = noisy_solution.sum(axis=1, keepdims=True)
            noisy_solution = noisy_solution / row_sums
            infection_prevalence = noisy_solution[:, 1]
            # Get noisy infection incidence
            new_infections = -np.diff(clean_susceptibles) * total_population
            incidence = np.random.poisson(lam=new_infections)
            infection_incidence = incidence / total_population
        else:
            infection_prevalence = noiseless_solution[:, 1]
            infection_incidence = -np.diff(clean_susceptibles)
            # Noisy solution is just the clean solution
            noisy_solution = noiseless_solution.copy()

        # Model data
        if data_type == "incidence_only":
            data_X.append(
                {
                    "time": model_config["times"],
                    "initial_conditions": realisation_init_conditions,
                    "states": {
                        "incidence": infection_incidence,
                    },
                }
            )
        elif data_type == "prevalence_only":
            data_X.append(
                {
                    "time": model_config["times"],
                    "initial_conditions": realisation_init_conditions,
                    "states": {
                        "prevalence": infection_prevalence,
                    },
                }
            )
        elif data_type == "both":
            data_X.append(
                {
                    "time": model_config["times"],
                    "initial_conditions": realisation_init_conditions,
                    "states": {
                        "prevalence": infection_prevalence,
                        "incidence": infection_incidence,
                    },
                }
            )
        else:
            raise ValueError(
                f"Invalid data_type {data_type}: prevalence_only, incidence_only, or both."
            )

        # Plotting data
        data_plot.append(
            {
                "time": model_config["times"],
                **{state: noisy_solution[:, idx] for idx, state in enumerate(states)},
            }
        )

    return data_X, data_plot, all_initial_conditions, noiseless_solution


# Generate data and add to configurations
for system_name in configurations.keys():
    # Initialise the system's data dictionary
    configurations[system_name]["data"] = {}
    for regime_name in configurations[system_name]["regimes"].keys():
        for noise_level in configurations[system_name]["noise_levels"].keys():
            # Generate data for each combination of model, regime, and noise level
            data_X, data_plot, all_initial_conditions, noiseless_solution = (
                generate_latent_sir_data(
                    system_name=system_name,
                    regime=regime_name,
                    noise_level=noise_level,
                    data_seed=data_seed,
                    configurations=configurations,
                    total_population=total_population,
                )
            )

            # Add to configurations
            configurations[system_name]["data"][(regime_name, noise_level)] = {
                "data_X": data_X,
                "data_plot": data_plot,
                "all_initial_conditions": all_initial_conditions,
                "noiseless_solution": noiseless_solution,
            }

if __name__ == "__main__":
    # -----------------------------
    # Run experiments
    # -----------------------------

    parser = argparse.ArgumentParser(description="Run FluxDisco experiments.")
    parser.add_argument(
        "--system",
        nargs="+",  # Allow multiple systems to be specified
        type=str,
        required=True,
        help="System(s) to run or 'all'",
    )
    parser.add_argument("--results_dir", type=str, default="results")
    # Select specific regime, otherwise all regimes will be run
    parser.add_argument("--regime", type=str, default="all")
    args = parser.parse_args()

    systems_to_run = (
        list(configurations.keys()) if "all" in args.system else args.system
    )

    print("EXPERIMENT CONFIGURATION")
    print(f"Regime:  {args.regime}")
    print("Systems: ")
    for sys_run in systems_to_run:
        print(f"  - {sys_run}")

    tasks = []
    for system_name in systems_to_run:
        config = configurations[system_name]
        regimes_to_run = (
            list(config["regimes"].keys()) if args.regime == "all" else [args.regime]
        )
        for regime in regimes_to_run:
            for noise_level in config["noise_levels"].keys():
                tasks.append(
                    (
                        system_name,
                        config,
                        noise_level,
                        regime,
                        args.results_dir,
                    )
                )

    print(f"Running {len(tasks)} experiments in parallel...")
    with parallel_config(backend="loky", prefer="processes", n_jobs=len(tasks)):
        results = Parallel()(delayed(run_single_experiment)(task) for task in tasks)
