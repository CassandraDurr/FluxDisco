"""Run base FluxDisco experiments from the paper."""

import argparse

import numpy as np
import sympy
from joblib import Parallel, delayed, parallel_config
from utils import (
    brusselator_odes,
    fairen_velarde_odes,
    generate_data,
    lotka_volterra_odes,
    run_single_experiment,
    sir_odes,
)

# -----------------------------
# System configurations
# -----------------------------

# Sympy state symbols
s_sym, i_sym, r_sym, x_sym, y_sym = sympy.symbols("s i r x y")

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
            "Standard": [98 / 100, 2 / 100, 0],
            "Squared": [88 / 100, 12 / 100, 0],
            "Sqrt": [98 / 100, 2 / 100, 0],
        },
        "times": np.linspace(0, 80, 80),
        "noise_levels": {"No Noise": 0, "Low Noise": 0.02, "High Noise": 0.05},
        "search_params": {"checkpoint_saving": True},
    },
    "lotka-volterra": {
        "states": ["x", "y"],
        "state_variables": [x_sym, y_sym],
        "num_fluxes": 3,
        "system_structure": {"x": [1, -1, 0], "y": [0, 1, -1]},
        "grammar": {
            "M": [f"M -> {state}" for state in ["x", "y"]]
            + [
                "M -> C",
                "M -> M + M",
                "M -> M - M",
                "M -> M * M",
            ],
        },
        "flux_priors": {
            0: {  # ax
                "M -> x": 1,
                "M -> y": 0,
                "M -> C": 1,
                "M -> M + M": 1,
                "M -> M - M": 1,
                "M -> M * M": 1,
            },
            1: {  # bxy
                "M -> x": 1,
                "M -> y": 1,
                "M -> C": 1,
                "M -> M + M": 1,
                "M -> M - M": 1,
                "M -> M * M": 1,
            },
            2: {  # cy
                "M -> x": 0,
                "M -> y": 1,
                "M -> C": 1,
                "M -> M + M": 1,
                "M -> M - M": 1,
                "M -> M * M": 1,
            },
        },
        "ode_func": lotka_volterra_odes,
        "regimes": {
            "Default": (1.0, 0.1, 1.5),
        },
        "initial_conditions": {
            "Default": [10.0, 5.0],
        },
        "times": np.linspace(0, 30, 150),
        "noise_levels": {"No Noise": 0, "Low Noise": 0.03, "High Noise": 0.07},
        "search_params": {"checkpoint_saving": True},
    },
    "brusselator": {
        "states": ["x", "y"],
        "state_variables": [x_sym, y_sym],
        "num_fluxes": 3,
        "system_structure": {"x": [1, -1, 1], "y": [-1, 1, 0]},
        "grammar": {
            "M": [f"M -> {state}" for state in ["x", "y"]]
            + [
                "M -> C",
                "M -> M + M",
                "M -> M - M",
                "M -> M * M",
            ],
        },
        "flux_priors": {
            0: {  # x^2y
                "M -> x": 1,
                "M -> y": 1,
                "M -> C": 1,
                "M -> M + M": 1,
                "M -> M - M": 1,
                "M -> M * M": 1,
            },
            1: {  # bx
                "M -> x": 1,
                "M -> y": 0,
                "M -> C": 1,
                "M -> M + M": 1,
                "M -> M - M": 1,
                "M -> M * M": 1,
            },
            2: {  # a-x
                "M -> x": 1,
                "M -> y": 0,
                "M -> C": 1,
                "M -> M + M": 1,
                "M -> M - M": 1,
                "M -> M * M": 1,
            },
        },
        "ode_func": brusselator_odes,
        "regimes": {"Stable": (1.0, 1.5), "Unstable": (1.0, 3.0)},
        "initial_conditions": {
            "Stable": [0.5, 2.0],
            "Unstable": [0.5, 2.0],
        },
        "times": np.linspace(0, 30, 150),
        "noise_levels": {"No Noise": 0, "Low Noise": 0.03, "High Noise": 0.07},
        "search_params": {"checkpoint_saving": True},
    },
    "fairen-velarde": {
        "states": ["x", "y"],
        "state_variables": [x_sym, y_sym],
        "num_fluxes": 3,
        "system_structure": {"x": [-1, 1, 0], "y": [-1, 0, 1]},
        # Give correct flux for oxy and nut
        "initial_state": (
            [[], ["M -> M - M", "M -> C", "M -> x"], ["M -> C"]],
            [["M"], [], []],
            [0, 3, 1],
        ),
        "grammar": {
            "M": [f"M -> {state}" for state in ["x", "y"]]
            + [
                "M -> C",
                "M -> M + M",
                "M -> M - M",
                "M -> M * M",
                "M -> M / M",
            ],
        },
        "flux_priors": {
            0: {  # Consumption: xy/(1+qx^2)
                "M -> x": 1,
                "M -> y": 1,
                "M -> C": 1,
                "M -> M + M": 1,
                "M -> M - M": 1,
                "M -> M * M": 1,
                "M -> M / M": 1,
            },
            1: {  # Oxygen dynamics: B - x
                "M -> x": 1,
                "M -> y": 0,
                "M -> C": 1,
                "M -> M + M": 1,
                "M -> M - M": 1,
                "M -> M * M": 1,
                "M -> M / M": 1,
            },
            2: {  # Nutrient dynamics: A
                "M -> x": 0,
                "M -> y": 1,
                "M -> C": 1,
                "M -> M + M": 1,
                "M -> M - M": 1,
                "M -> M * M": 1,
                "M -> M / M": 1,
            },
        },
        "ode_func": fairen_velarde_odes,
        "regimes": {"Unstable": (15.0, 10.0, 0.5)},
        "initial_conditions": {"Unstable": [10.0, 25.0]},
        "times": np.linspace(0, 60, 150),
        "noise_levels": {"No Noise": 0, "Low Noise": 0.03, "High Noise": 0.07},
        "search_params": {
            "kappa": 12,  # Increase kappa (true max flux depth = 11)
            "eta": 0.995,  # Increase eta to allow for more complex expressions
            "temperature": 0.05,  # Increase temperature to increase reward values
            "episodes": 40,  # Decrease to keep time manageable with increased kappa
            "steps": 2 * (12 + 12 + 1),
            "checkpoint_saving": True,
            "constant_only_flux_allowed": True,  # Allow constant-only fluxes (nutrient dynamics)
        },
    },
}

# -----------------------------
# Generate data
# -----------------------------
# General data parameters
data_seed = 1234
num_realisations = 1


# Generate data and add to configurations
for system_name in configurations.keys():
    # Initialise the system's data dictionary
    configurations[system_name]["data"] = {}
    for regime_name in configurations[system_name]["regimes"].keys():
        for noise_level in configurations[system_name]["noise_levels"].keys():
            # Generate data for each combination of model, regime, and noise level
            data_X, data_plot, all_initial_conditions, noiseless_solution = (
                generate_data(
                    system_name,
                    regime_name,
                    noise_level,
                    data_seed,
                    num_realisations,
                    configurations,
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
        type=str,
        choices=list(configurations.keys()) + ["all"],
        required=True,
    )
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    systems_to_run = (
        list(configurations.keys()) if args.system == "all" else [args.system]
    )

    tasks = []
    for system_name in systems_to_run:
        config = configurations[system_name]
        for regime in config["regimes"].keys():
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
