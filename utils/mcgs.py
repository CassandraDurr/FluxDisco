"""Core Monte Carlo Graph Search (MCGS)/ FluxDisco class."""

from __future__ import annotations

import csv
import re
import time
import uuid

import numpy as np
import pandas as pd

from .const_folding import constant_folding, constant_folding_partial
from .data_saving import time_saving
from .expression_builder import sympy_expression_builder
from .grammar_probabilities import initialise_grammar_probabilities_util
from .reward_calculator import RewardCalculator
from .symbolic_graph_node import SymbolicNode


class MCGS:
    """MCGS class for dynamical symbolic regression of coupled ODE systems."""

    def __init__(
        self,
        data_X: list[dict],  # time, initial conditions, and observations
        system_structure: dict,  # Stoichiometry
        state_variables: list,  # State variables (sympy symbols)
        state_to_variable: dict,  # Map from state names to sympy symbols
        state_keys: list,  # State keys (strings)
        num_fluxes: int,
        grammar: dict,
        terminal_rules_for_M: list[str],
        flux_priors: dict,
        initial_state: tuple,
        kappa: int,
        eta: float,
        gamma: float,
        epsilon: float,
        rollouts_per_leaf: int,
        warm_start_rollouts: int,
        top_N: int = 10,
        temperature: float = 1.0,
        normalise_reward: bool = False,
        constant_only_flux_allowed: bool = False,
        prevent_merging: bool = False,
    ):
        """Initialise graph instance."""
        self.data_X = data_X
        self.grammar = grammar
        self.flux_priors = flux_priors
        self.system_structure = system_structure
        self.num_fluxes = num_fluxes
        self.state_keys = state_keys
        self.state_var_list = state_variables
        self.state_var_set = set(state_variables)
        self.state_to_variable = state_to_variable
        self.constant_only_flux_allowed = constant_only_flux_allowed
        self.prevent_merging = prevent_merging
        self.total_samples = 0

        # Hyperparameters
        self.kappa = kappa
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon
        self.top_N = top_N
        self.rollouts_per_leaf = rollouts_per_leaf
        self.warm_start_rollouts = warm_start_rollouts

        # Update threshold
        self.update_threshold = max(
            ((1 - self.gamma) / self.gamma) * self.epsilon, 0.01
        )

        # Grammar helpers
        self.non_terminals = set(grammar.keys())
        self.non_terminals_regex = re.compile(f"({'|'.join(self.non_terminals)})")
        self.terminal_rules_for_M = terminal_rules_for_M

        # Prevent recalculating expesive reward function
        # Cache: {tuple(str_exprs): (reward, details)}
        self.reward_cache = {}

        # Prevent refolding expressions for hashing
        self.folding_cache = {}

        # Intialise flux prior probabilities and rules
        self.initialise_grammar_probabilities()

        # Get time, data, and intial conditions (vectorised for replications)
        self.vectorise_data(data_X=data_X)

        # Initialise reward calculator
        self.reward_calculator = RewardCalculator(
            eta=self.eta,
            system_structure=self.system_structure,
            state_keys=self.state_keys,
            state_variables=self.state_var_list,
            state_to_variable=self.state_to_variable,
            time_array=self.time_array,
            initial_conditions_matrix=self.initial_conditions_matrix,
            observations=self.observations,
            obs_keys=self.obs_keys,
            temperature=temperature,
            normalise_reward=normalise_reward,
        )

        # Transposition table: maps state hash -> SymbolicNode
        self.graph_nodes = {}

        # Initialise root (and add to graph)
        self.root = SymbolicNode(state=initial_state, graph=self)
        root_key = self.hash_state(self.root.state)
        self.graph_nodes[root_key] = self.root

    def vectorise_data(self, data_X: list[dict]) -> None:
        """Vectorise time, intial conditions, and data for multiple replications."""
        # Get keys of observational data
        obs_keys = list(data_X[0]["states"].keys())
        self.obs_keys = obs_keys

        # Time vector: assumption = realisations share the same time steps.
        self.time_array = data_X[0]["time"]

        # Initial conditions matrix: shape (num_states, num_realisations)
        # NOTE: We assume the initial conditions are provided for all state variables
        # in the same order as self.state_keys.
        initial_conditions = [np.array(data["initial_conditions"]) for data in data_X]
        self.initial_conditions_matrix = np.vstack(initial_conditions).T

        # Store observations in a dict of arrays: {obs_key: np.array}
        self.observations = {}
        for key in self.obs_keys:
            self.observations[key] = np.stack(
                [data["states"][key] for data in self.data_X], axis=-1
            )

    def initialise_grammar_probabilities(self):
        """
        Calculate normalised probabilities for the flux grammar rules based on priors.

        Valid M rules used in rollout & getting untried actions, valid terminal M rules used in force-completion.
        Valid M terminal state rules used in rollout to ensure branches have at least one state.

        Define:
            - self.valid_M_rules: rules allowed per flux
            - self.M_rule_probs: probabilities of rules allowed per flux
            - self.valid_terminal_M_rules: terminal rules allowed per flux
            - self.terminal_M_rule_probs: probabilities of terminal rules allowed per flux
            - self.valid_state_terminal_M_rules: terminal rules allowed per flux containing state values
            - self.terminal_M_rule_probs: probabilities of terminal rules allowed per flux containing state values
        """  # noqa: E501
        (
            self.valid_M_rules,
            self.M_rule_probs,
            self.valid_terminal_M_rules,
            self.terminal_M_rule_probs,
            self.valid_state_terminal_M_rules,
            self.valid_state_terminal_M_rule_probs,
        ) = initialise_grammar_probabilities_util(
            grammar=self.grammar,
            flux_priors=self.flux_priors,
            terminal_rules_for_M=self.terminal_rules_for_M,
            num_fluxes=self.num_fluxes,
            state_keys=self.state_keys,
        )

    def hash_state(self, state) -> tuple:
        """
        Create a unique key for a graph node using its state representation.

        Simplify both partial (internal nodes) and complete expressions (terminal nodes).
        """
        action_lists, stacks, _ = state

        # Check if terminal (all stacks empty)
        is_terminal = all(not stack for stack in stacks)

        if is_terminal:
            # Simplify full expression (fold constants, etc)
            try:
                canonical_form = self.get_simplified_exprs(
                    action_lists, partial_mode=False
                )
                return ("TERMINAL", canonical_form)
            except Exception as e:
                print(f"Hash exception for terminal node: {e}")
                # Fallback
                return ("TERMINAL", tuple(tuple(x) for x in action_lists))
        else:
            # Internal node: Simplify partial expression
            try:
                partial_structure = self.get_simplified_exprs(
                    action_lists, partial_mode=True
                )
                return ("INTERNAL", partial_structure)
            except Exception as e:
                print(f"Hash exception for internal node: {e}")
                # Fallback
                return ("INTERNAL", tuple(tuple(x) for x in action_lists))

    def get_simplified_exprs(
        self, flux_action_lists: list[list[str]], partial_mode: bool = False
    ) -> tuple:
        """Convert actions to a canonical, simplified string representations.

        Args:
            flux_action_lists (list[list[str]]): List of action lists per flux.
            partial_mode (bool, optional): Whether to simplify partial ODEs. Defaults to False.

        Returns:
            tuple: Tuple of simplified expression strings per flux.
        """
        folded_strs = []
        k_index = 0  # constant offset for clean, folded expressions

        for action_list in flux_action_lists:
            # Build cache key for this flux
            cache_key = (tuple(action_list), partial_mode)

            if cache_key not in self.folding_cache:
                # Build the messy expression
                # Start with 0 index for constants - reindex after folding
                eq_expr, _ = sympy_expression_builder(
                    action_list,
                    const_offset=0,
                    partial_mode=partial_mode,
                    state_mapping=self.state_to_variable,
                )

                if partial_mode:
                    # Simplify and fold constants for partial expressions
                    folded_expr, flux_const_idx, _, _ = constant_folding_partial(
                        expr=eq_expr,
                        state_vars_base=self.state_var_set,
                        start_index=0,
                        eval_rules=False,
                    )
                else:
                    # Simplify and fold constants for terminal expressions
                    folded_expr, flux_const_idx, _, _ = constant_folding(
                        expr=eq_expr,
                        state_vars=self.state_var_set,
                        start_index=0,
                        eval_rules=False,
                    )

                # Store the folded expression and the number of constants for this flux
                self.folding_cache[cache_key] = (str(folded_expr), flux_const_idx)

            # Retrieve from cache
            folded_expr_str, flux_const_idx = self.folding_cache[cache_key]

            # Shift constant symbols from flux-specific to system-wide
            # We don't want to share constants across fluxes
            # (i.e. each flux having {B0, B1, ...} as its constants)
            if k_index > 0 and flux_const_idx > 0:
                final_str = re.sub(
                    r"\bB(\d+)\b",  # B followed by digits
                    lambda m, off=k_index: f"B{int(m.group(1)) + off}",  # Shift digits by k_index
                    folded_expr_str,
                )
            else:
                # No constants yet in prior fluxes (k_index still = 0)
                final_str = folded_expr_str

            # Store final folded and reindexed string for this flux
            folded_strs.append(final_str)

            # Update k_index for the next flux
            k_index += flux_const_idx

        return tuple(folded_strs)

    def get_or_create_node(self, state) -> SymbolicNode:
        """Check transposition table before creating a new node (unless merging is prevented)."""
        state_key = self.hash_state(state)

        if self.prevent_merging:
            # CREATE: Return new node without checking for existing nodes
            new_node = SymbolicNode(state=state, graph=self)

            # Store the new node in the nodes dictionary with a unique key
            unique_key = (state_key, uuid.uuid4())
            self.graph_nodes[unique_key] = new_node

            return new_node

        else:
            if state_key in self.graph_nodes:
                # MERGE: Return existing node
                return self.graph_nodes[state_key]
            else:
                # CREATE: Return new node
                new_node = SymbolicNode(state=state, graph=self)
                self.graph_nodes[state_key] = new_node
                return new_node

    def propagate_upwards(self, start_node: SymbolicNode, max_iterations: int = 10000):
        """
        Efficient queue-based propagation update.

        Use Algorithm 4 from the supplementary material of MCGS paper.

        Add max iterations to avoid infinite loops (should not happen in practice).
        """
        # Queue q initialised with s_n (q <- [s_n])
        processing_queue = [start_node]

        # Set of nodes in queue to avoid duplicates
        queue_set = {start_node}

        iterations = 0
        while processing_queue and iterations < max_iterations:
            iterations += 1
            if iterations == max_iterations:
                print("Warning: Max iterations reached in propagate_upwards.")
                break

            # Pop the first node s'
            node = processing_queue.pop(0)
            queue_set.remove(node)

            # Bounds proposed by Bellman operator
            new_U_candidate, new_L_candidate = node.calculate_bellman_values()

            # Enforce monotonicity of bounds: U is non-increasing; L is non-decreasing
            new_U = min(node.U, new_U_candidate)
            new_L = max(node.L, new_L_candidate)

            # Check stopping condition for the upper bound U
            # If |new U - old U| > threshold, we update and propagate.
            # We also always propagate if it's the start_node (expanded node),
            # or if the lower bound improves.
            should_propagate = False
            update_condition = (
                (abs(new_U - node.U) > self.update_threshold)
                or (new_L > node.L)
                or (node == start_node)
            )
            if update_condition:
                node.U = new_U
                node.L = new_L
                should_propagate = True

            # If updated, push predecessors s to queue q
            if should_propagate:
                for parent in node.parents:
                    if parent not in queue_set:
                        processing_queue.append(parent)
                        queue_set.add(parent)

    def run_search(
        self,
        episodes: int = 100,
        steps: int = 100,
        print_epi: int = 10,
        checkpoint_saving: bool = False,
        save_dir: str | None = None,
        save_freq: int = 5,
    ) -> list:
        """Run the FluxDisco search algorithm."""
        experiment_start_time = time.time()
        for episode in range(episodes):
            # Restart at root node per epsisode
            node = self.root
            # Track nodes visited in this trajectory for a single final propagation
            trajectory_nodes = [node]

            # Print progress
            if episode % print_epi == 0:
                print(f"Episode {episode}/{episodes}...")

            for step in range(steps):
                # --- 0A. CHECK IF TERMINAL ---
                if node.is_terminal_flag:
                    # No children possible for selection
                    print(f"   Reached terminal node at step {step + 1}")
                    break

                # --- 0B. MAKE CHILDREN ---
                # Check if the node actually has any children to select from
                # If there are no children nodes get - create them so you can select them
                if not node.children:
                    # Expand all untried actions to create all children
                    children = node.expand_all_children()
                    # Rollout children (warm start bounds and confidence intervals)
                    if children:
                        for child in children:
                            for _ in range(self.warm_start_rollouts):
                                reward, data = child.rollout()
                                child.update_stats(reward, result_data=data)

                        # Propagate from parent (bellman update uses newly expanded children)
                        self.propagate_upwards(node)

                # --- 1. SELECTION ---
                node = node.best_child()
                trajectory_nodes.append(node)

                # --- 2. ROLLOUT ---
                for _ in range(self.rollouts_per_leaf):
                    reward, data = node.rollout()
                    # --- 3. UPDATE STATS ---
                    node.update_stats(reward, result_data=data)

            # --- 4. GRAPH PROPAGATION ---
            # Propagate updates from the last node in the trajectory
            # (should include all ancestor nodes)
            self.propagate_upwards(trajectory_nodes[-1])

            # --- 5. CHECKPOINT SAVING ---
            if checkpoint_saving and save_dir and ((episode + 1) % save_freq == 0):
                print(f"Saving checkpoint at episode {episode + 1}...")
                # Save nodes and bounds
                self.export_graph_bounds(
                    f"{save_dir}/nodes_and_bounds_{episode + 1}.csv"
                )
                # Save top results
                reward_cache = self.reward_calculator.export_reward_cache()
                if reward_cache:  # Ensure there is data to avoid pandas errors
                    reward_cache_df = pd.DataFrame(reward_cache)
                    reward_cache_df = reward_cache_df[
                        ["rank", "reward", "equations", "equations_with_constants"]
                    ]
                    reward_cache_df.to_csv(
                        f"{save_dir}/full_results_{episode + 1}.csv", index=False
                    )

                # Save timing
                time_saving(
                    experiment_end_time=time.time(),
                    experiment_start_time=experiment_start_time,
                    save_dir=f"{save_dir}/timing_{episode + 1}",
                )

        return self.get_top_results_from_graph()

    def traverse_graph_and_collect_results(self):
        """
        Traverse the graph and collect all stored results.

        Returns:
            list[tuple]: A list of tuples containing (reward, result_data).
        """
        all_results = []
        visited = set()  # Required for graph merges (multiple parents)
        stack = [self.root]

        while stack:
            node = stack.pop()

            # Skip if we have already processed this node
            if node in visited:
                continue
            visited.add(node)

            # Collect result
            if node.best_result_data is not None:
                # node.best_result_data format: (flux_exprs, final_flux_strs)
                flux_exprs, final_flux_strs = node.best_result_data
                all_results.append((node.best_reward, flux_exprs, final_flux_strs))

            # Add children
            stack.extend(node.children.values())

        # Sort by reward descending
        all_results.sort(key=lambda x: x[0], reverse=True)

        return all_results

    def get_top_results_from_graph(self):
        """Get the top N results from the nodes."""
        all_results = self.traverse_graph_and_collect_results()
        return all_results[: self.top_N]

    def export_graph_bounds(self, filepath: str) -> None:
        """Export the bounds (L, U) and hash of every node in the graph."""
        # Order by visit counts and then best_reward
        sorted_nodes = sorted(
            self.graph_nodes.items(),
            key=lambda item: (item[1].visit_counts, item[1].best_reward),
            reverse=True,
        )
        rows = []

        # Iterate over the transposition table (all registered nodes)
        for state_hash, node in sorted_nodes:
            hash_str = str(state_hash)
            rows.append(
                (
                    hash_str,
                    node.best_reward,
                    node.mean_reward,
                    node.var_rewards if node.var_rewards is not None else "",
                    node.visit_counts,
                    node.L,
                    node.U,
                    node.l_CI if node.l_CI is not None else 0.0,
                    node.u_CI if node.u_CI is not None else 1.0,
                    node.is_terminal_flag,
                    node.invalid_constant_condition,
                    len(node.children),
                    len(node.parents),
                )
            )

        with open(filepath, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "node_hash",
                    "best_reward_seen",
                    "average_reward",
                    "variance_rewards",
                    "visit_count",
                    "L_bound",
                    "U_bound",
                    "lower_confidence_interval",
                    "upper_confidence_interval",
                    "is_terminal",
                    "is_invalid_constant",
                    "num_children",
                    "num_parents",
                ]
            )
            writer.writerows(rows)
