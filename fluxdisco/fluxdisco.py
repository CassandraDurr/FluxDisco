"""FluxDisco run script."""

import os
import random
import time
from typing import Any, Optional

import numpy as np
import pandas as pd

from .utils.data_saving import estimation_saving, time_saving
from .utils.mcgs import MCGS


def run_fluxdisco(
    data: list[dict[str, Any]],
    system_config: dict[str, Any],
    save_dir: str = "results",
    experiment_name: str = "experiment",
    search_params: Optional[dict[str, Any]] = None,
    seed: int = 1234,
) -> tuple[list, pd.DataFrame]:
    """Run FluxDisco on a given system with specified parameters.

    Args:
        data (list[dict[str, Any]]): List of dictionaries containing the input data for the system.
        system_config (dict[str, Any]): Configuration dictionary for the system.
        save_dir (str, optional): Directory to save results. Defaults to "results".
        experiment_name (str, optional): Name for the specific run folder. Defaults to "experiment".
        search_params (Optional[dict[str, Any]], optional): Search hyperparameters. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to 1234.

    Returns:
        tuple[list, pd.DataFrame]: Top search results (list) and all candidate systems (pd.DataFrame).
    """  # noqa: E501
    # Get search parameters
    if search_params is None:
        search_params = {}

    # Default steps per episode
    kappa = search_params.get("kappa", 6)
    steps = search_params.get("steps", 2 * (kappa + kappa + 1))

    print(f"Starting FluxDisco search: {experiment_name}")
    experiment_start_time = time.time()

    # Setup save and results directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    run_dir = os.path.join(save_dir, experiment_name)
    os.makedirs(run_dir, exist_ok=True)

    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Create initial state
    num_fluxes = system_config["num_fluxes"]
    if system_config.get("initial_state") is not None:
        initial_state = system_config["initial_state"]
    else:
        initial_state = (
            [[] for _ in range(num_fluxes)],
            [["M"] for _ in range(num_fluxes)],
            [0 for _ in range(num_fluxes)],
        )

    # Create terminal grammar rules if not provided
    if system_config.get("terminal_rules_for_M") is not None:
        terminal_rules_for_M = system_config["terminal_rules_for_M"]
    else:
        terminal_rules_for_M = [
            f"M -> {state}" for state in system_config["states"]
        ] + ["M -> C"]

    # Initialise the search graph
    search_graph = MCGS(
        data_X=data,
        state_keys=system_config["states"],
        system_structure=system_config["system_structure"],
        state_variables=system_config["state_variables"],
        state_to_variable=system_config["state_to_variable"],
        num_fluxes=num_fluxes,
        terminal_rules_for_M=terminal_rules_for_M,
        grammar=system_config["grammar"],
        flux_priors=system_config["flux_priors"],
        initial_state=initial_state,
        kappa=search_params.get("kappa", 6),
        eta=search_params.get("eta", 0.99),
        gamma=search_params.get("gamma", 0.9),
        epsilon=search_params.get("epsilon", 0.01),
        rollouts_per_leaf=search_params.get("rollouts_per_leaf", 2),
        warm_start_rollouts=search_params.get("warm_start_rollouts", 1),
        temperature=search_params.get("temperature", 0.005),
    )

    # Run search
    top_results = search_graph.run_search(
        episodes=search_params.get("episodes", 100),
        steps=steps,
        checkpoint_saving=search_params.get("checkpoint_saving", False),
        print_epi=search_params.get("print_epi", 10),
        save_dir=run_dir,
        save_freq=search_params.get("save_freq", 5),
    )

    # Save graph node statistics and explored candidates
    search_graph.export_graph_bounds(os.path.join(run_dir, "nodes_and_bounds.csv"))

    reward_cache = search_graph.reward_calculator.export_reward_cache()
    reward_cache_df = pd.DataFrame(reward_cache)
    if not reward_cache_df.empty:
        reward_cache_df = reward_cache_df[
            ["rank", "reward", "equations", "equations_with_constants"]
        ]
        reward_cache_df.to_csv(os.path.join(run_dir, "full_results.csv"), index=False)

    # Save trajectories for the top estimated system
    if top_results:
        top_str_fluxes = top_results[0][2]
        try:
            base_times = data[0]["time"]
            base_initial_conditions = data[0]["initial_conditions"]

            estimation_saving(
                states=system_config["states"],
                state_variables=system_config["state_variables"],
                top_str_fluxes=top_str_fluxes,
                initial_conditions_base=base_initial_conditions,
                times=base_times,
                system_structure=system_config["system_structure"],
                save_dir=run_dir,
            )
        except Exception as e:
            print(f"Error during estimation saving: {e}")
    else:
        print("Search failed. Skipping estimation.")

    # Save time taken for the experiment
    time_saving(
        experiment_end_time=time.time(),
        experiment_start_time=experiment_start_time,
        save_dir=run_dir,
    )

    print(f"Finished {experiment_name}. Results saved to {run_dir}")
    return top_results, reward_cache_df
