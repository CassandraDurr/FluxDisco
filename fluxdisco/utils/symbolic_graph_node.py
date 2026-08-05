"""Symbolic Node class for each graph node in FluxDisco."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

from .conf_intervals_rewards import confidence_intervals_rewards

if TYPE_CHECKING:
    from .mcgs import MCGS

# Method for confidence intervals
CONF_INTERVAL_METHOD = "beta"


class SymbolicNode:
    """Class representing a state, or node, in the search graph."""

    def __init__(
        self,
        state: tuple[list[list[str]], list[list[str]], list[int]],
        graph: MCGS,
    ):
        """
        Initialise graph node.

        Args:
            state: The unique identifier state (flux_actions, stacks, ts)
            graph: Reference to the graph instance
        """
        self.state = state
        self.graph = graph

        # Calculate V_max
        v_max = self.graph.eta / (1 - self.graph.gamma)

        # Node value bounds
        self.L = 0.0
        self.U = v_max  # V_max; Max reward per step is 1.0

        # Confidence interval (or credible interval) for rewards
        self.l_CI, self.u_CI = None, None

        # Graph connectivity
        self.parents: list[SymbolicNode] = []
        self.children: dict[str, SymbolicNode] = {}  # Map action -> Node

        # Storing best results
        self.best_result_data = None
        self.best_reward = 0.0

        # Visits
        self.visit_counts = 0

        # Online mean/variance (Welford)
        self.mean_reward = 0.0  # Mean of rewards
        self.m2 = 0.0  # Sum of squares of differences from mean
        self.var_rewards = None  # Sample variance of rewards

        # Allow constant-only fluxes or not
        self.constant_only_flux_allowed = self.graph.constant_only_flux_allowed

        # Rollout retries for constant-only fluxes
        self.rollout_retries = 10

        # Terminal nodes: complete fluxes, or invalid constant-only fluxes (get zero reward)
        self.terminal_condition = self.check_is_terminal()
        if self.constant_only_flux_allowed:
            # Allow constant-only fluxes, so only complete fluxes are terminal
            self.invalid_constant_condition = False
        else:
            self.invalid_constant_condition = self.check_has_invalid_constant_flux()
        self.is_terminal_flag = (
            self.terminal_condition or self.invalid_constant_condition
        )

        # Possible actions
        if self.is_terminal_flag:
            self.untried_actions = []
        else:
            self.untried_actions = self.get_actions()

    def check_is_terminal(self) -> bool:
        """Check if all fluxes are complete/ terminal."""
        _, flux_stacks, _ = self.state
        for flux in range(len(flux_stacks)):
            if flux_stacks[flux]:  # Stack not empty = not terminal
                return False
        return True  # All fluxes are terminal (all stacks empty)

    def check_has_invalid_constant_flux(self) -> bool:
        """
        Check if any flux is complete and constant-only (no state variables).

        Returns:
            bool: True if any flux is an invalid constant flux
        """
        flux_actions, flux_stacks, _ = self.state

        # Early exit if no fluxes are complete (no possibility of invalid constant flux)
        if all(stack for stack in flux_stacks):
            return False

        for flux_idx, stack in enumerate(flux_stacks):
            # If stack is empty, the flux is complete
            if not stack:
                # Check if this flux has any state variables
                has_state_var = self.has_state_variable(flux_idx, flux_actions)

                # If complete and has no state variables, it's an invalid constant flux.
                if not has_state_var:
                    return True

        return False

    def has_state_variable(self, flux_idx: int, flux_actions: list[str]) -> bool:
        """Check if a flux has state variables.

        Args:
            flux_idx (int): Flux index to check.
            flux_actions (list[str]): Flux action list.

        Returns:
            bool: True if the flux has state variables, False otherwise.
        """
        # Check if this flux has any state variables
        state_rules = set(self.graph.valid_state_terminal_M_rules[flux_idx])
        has_state_var = bool(state_rules.intersection(flux_actions[flux_idx]))
        return has_state_var

    def add_parent(self, parent_node: SymbolicNode):
        """Register a new parent for this node (graph merging)."""
        if parent_node not in self.parents:
            self.parents.append(parent_node)

    def get_actions(self) -> list[tuple[int, str, float]]:
        """
        Return a list of actions with their flux index and prior probabilities.

        Fluxes with empty stacks/ no non-terminal M tokens are skipped.
        For fluxes that have reached kappa, only terminal rules are allowed (force-completion).

        Format: [(flux_idx, action_str, probability), ...]
        """
        all_actions = []
        _, flux_stacks, flux_ts = self.state

        for flux_idx, stack in enumerate(flux_stacks):
            if not stack:
                continue  # This flux is complete (stack empty)

            # Check if flux has reached kappa
            if flux_ts[flux_idx] >= self.graph.kappa:
                # Force-completion: only allow terminal rules
                rules = self.graph.valid_terminal_M_rules[flux_idx]
                probs = self.graph.terminal_M_rule_probs[flux_idx]

            else:
                # Normal expansion: use all valid M rules
                rules = self.graph.valid_M_rules[flux_idx]
                probs = self.graph.M_rule_probs[flux_idx]

            # Add actions and probs for this flux
            for rule, prob in zip(rules, probs):
                all_actions.append((flux_idx, rule, prob))

        return all_actions

    def apply_action(self, state: tuple, flux_idx: int, action: str):
        """Apply a grammar rule to one specific flux."""
        current_flux_actions, current_flux_stacks, current_flux_ts = state
        new_flux_actions = [list(flux_action) for flux_action in current_flux_actions]
        new_flux_stacks = [list(flux_stack) for flux_stack in current_flux_stacks]
        new_flux_ts = list(current_flux_ts)

        # Get the specific stack/actions/t for the flux we're expanding
        current_actions = new_flux_actions[flux_idx]
        current_stack = new_flux_stacks[flux_idx]
        ts = new_flux_ts[flux_idx]

        # Add action
        new_actions = current_actions + [action]

        # New stack is current stack without the last element
        new_stack = current_stack[:-1]

        # Get the action "LHS -> RHS"
        _, rhs = action.split(" -> ")

        # Get all non-terminals in the RHS
        new_non_terminals = self.graph.non_terminals_regex.findall(rhs)

        # Add new non-terminals to the stack in reverse (LIFO)
        new_stack.extend(new_non_terminals[::-1])

        # Increase number of applied grammar rules
        new_ts = ts + 1

        # Put the modified lists back into the new state tuple
        new_flux_actions[flux_idx] = new_actions
        new_flux_stacks[flux_idx] = new_stack
        new_flux_ts[flux_idx] = new_ts

        return (new_flux_actions, new_flux_stacks, new_flux_ts)

    def expand(self, random_action_selection: bool = False) -> SymbolicNode:
        """
        Graph expansion.

        Take a random action if you intend to expand one node at a time,
        else take actions in order (doesn't matter if all are expanded at once)

        1. Select action.
        2. Generate state.
        3. Check transposition table.
        4. Merge or create.
        """
        # --- 1. Select action ---
        if random_action_selection:
            # Get flux prior probabilities associated with actions
            weights = [item[2] for item in self.untried_actions]
            # Normalise weights to sum to 1
            sum_weights = sum(weights)
            probs = [weight / sum_weights for weight in weights]
            # Sample an index based on the priors
            action_idx = np.random.choice(len(self.untried_actions), p=probs)
            # Pop the selected action (removes it from untried so we don't expand it again)
            flux_idx, action, _ = self.untried_actions.pop(action_idx)

        else:
            # Pop the first action in the list
            flux_idx, action, _ = self.untried_actions.pop(0)

        # --- 2. Generate new state ---
        new_state = self.apply_action(self.state, flux_idx, action)

        # --- 3. Check transposition table ---
        # Key MCGS step: Merge if state exists, else create new state
        child_node = self.graph.get_or_create_node(new_state)

        # --- 4. Merge ---
        # The action needs to be flux x action string to be unique
        action_key = f"flux{flux_idx}:{action}"
        self.children[action_key] = child_node
        child_node.add_parent(self)

        return child_node

    def expand_all_children(self) -> list[SymbolicNode]:
        """Expand all untried actions to create all children."""
        # While there are untried actions, perform node expansion
        children = []
        while self.untried_actions:
            child = self.expand(random_action_selection=False)
            children.append(child)
        return children

    def selection_score(self, child: SymbolicNode) -> float:
        """
        Implement the selection rule.

        Rule: b_t = argmax (max(r_star, u_t(s,a))) + gamma * U(s')
        """
        # Compute upper confidence bound for reward (u_t(s,a))
        _, u_reward = confidence_intervals_rewards(
            r_hat=child.mean_reward,
            r_var=child.var_rewards,
            n_sa=child.visit_counts,
            n_total=self.graph.total_samples,
            method=CONF_INTERVAL_METHOD,
        )

        # Get the maximum of the upper confidence bound and the best reward found so far
        u_reward_max = max(u_reward, child.best_reward)

        return u_reward_max + self.graph.gamma * child.U

    def best_child(self) -> SymbolicNode:
        """Select the child with highest selection score."""
        # Iterate over values() because children is a dict {action: Node}
        # If scores are equal, compare random.random() (random tie breaker)
        return max(
            self.children.values(),
            key=lambda node: (self.selection_score(node), random.random()),
        )

    def calculate_bellman_values(self) -> tuple[float, float]:
        """
        Calculate the stochastic Bellman operators.

        B_t+(U)(s) = max_a [max(r_star, u_t(s,a)) + gamma * U(s')]
        B_t-(L)(s) = max_a [l_t(s,a) + gamma * L(s')]

        Stochastic rewards, but deterministic transitions between states.

        Returns:
            tuple[float, float]: New upper and lower bounds (new_U, new_L)
        """
        # If invalid constant-only flux, bounds are zero.
        if (not self.constant_only_flux_allowed) and self.invalid_constant_condition:
            return 0.0, 0.0

        # If terminal, bounds are fixed to the reward.
        if self.terminal_condition:
            return self.best_reward, self.best_reward

        # If the node is unexpanded, keep current values.
        if not self.children:
            return self.U, self.L

        # We track values across all actions to find the maximum
        action_upper_values = []
        action_lower_values = []

        for child in self.children.values():
            # Calculate confidence intervals for rewards
            try:
                l_reward, u_reward = confidence_intervals_rewards(
                    r_hat=child.mean_reward,
                    r_var=child.var_rewards,
                    n_sa=child.visit_counts,
                    n_total=self.graph.total_samples,
                    method=CONF_INTERVAL_METHOD,
                )
                # Update child confidence intervals for rewards
                child.l_CI, child.u_CI = l_reward, u_reward
            except Exception as e:
                raise Exception(f"Error computing KL bounds: {e}")

            # Get the maximum of the confidence interval bounds and the best reward found so far
            # Only consider upper bound
            u_reward_max = max(u_reward, child.best_reward)

            # Get Bellman values for this action
            action_upper_values.append(u_reward_max + self.graph.gamma * child.U)
            action_lower_values.append(l_reward + self.graph.gamma * child.L)

        # The value of the state is the max over all available actions
        new_U = max(action_upper_values)
        new_L = max(action_lower_values)

        return new_U, new_L

    def update_stats(self, result: float = None, result_data: tuple = None):
        """Post-rollout, update the local node's stats and best results."""
        # Update visit counts
        self.visit_counts += 1

        # Update online mean and variance (Welford's algorithm)
        delta = result - self.mean_reward  # (x_n - mean_{n-1})
        self.mean_reward += delta / self.visit_counts  # Update mean -> mean_n
        delta2 = result - self.mean_reward  # (x_n - mean_n)
        self.m2 += delta * delta2  # Update M2
        # Sample variance of rewards
        if self.visit_counts > 1:
            self.var_rewards = self.m2 / (self.visit_counts - 1)
        else:
            self.var_rewards = None

        # Update graph-wide total samples
        self.graph.total_samples += 1

        # Track best specific equation seen
        if result > self.best_reward:
            self.best_reward = result  # actual reward
            self.best_result_data = result_data  # equations

    def rollout(self):
        """
        Randomly select actions until a terminal state is reached.

        If kappa is reached, force-complete the equation with terminals.
        """
        # Check if this is a terminal node that has already been calculated
        # Terminal nodes are deterministic, so no need to re-rollout
        if self.is_terminal_flag and self.best_result_data is not None:
            return self.best_reward, self.best_result_data

        if (not self.constant_only_flux_allowed) and self.invalid_constant_condition:
            # Skip rollout
            return 0.0, ([], [])

        # Setup the retry logic for constant-only fluxes
        for _ in range(self.rollout_retries):
            # Get the current state
            orig_flux_actions, orig_flux_stacks, orig_flux_ts = self.state
            flux_action_lists = [list(flux_action) for flux_action in orig_flux_actions]
            flux_stacks = [list(flux_stack) for flux_stack in orig_flux_stacks]
            flux_ts = list(orig_flux_ts)

            # Loop while any stack is not empty and t < kappa
            while any(
                stack and t_val < self.graph.kappa
                for stack, t_val in zip(flux_stacks, flux_ts)
            ):
                # Find all fluxes that can be expanded
                expandable_fluxes = [
                    idx
                    for idx, (stack_flux, t_val_flux) in enumerate(
                        zip(flux_stacks, flux_ts)
                    )
                    if stack_flux and t_val_flux < self.graph.kappa
                ]

                if not expandable_fluxes:
                    # No fluxes can be expanded
                    break

                # Pick one of the expandable fluxes at random
                flux_to_expand_idx = random.choice(expandable_fluxes)

                # Standard rollout for the chosen flux
                flux_action_lists, flux_stacks, flux_ts = self.rollout_take_action(
                    flux_action_lists, flux_stacks, flux_ts, flux_to_expand_idx
                )

            # Force-complete all fluxes that hit kappa
            for flux in range(len(flux_stacks)):
                while flux_stacks[flux]:  # If a fluxes stack is not empty
                    flux_action_lists, flux_stacks, flux_ts = (
                        self.force_complete_action(
                            flux_action_lists, flux_stacks, flux_ts, flux
                        )
                    )

            # Get flux expressions (pre-folding) and check for invalid constant-only condition
            original_flux_exprs = self.graph.reward_calculator.build_expressions(
                flux_action_lists
            )

            if not self.constant_only_flux_allowed:
                is_invalid_constant = (
                    self.graph.reward_calculator.check_invalid_constant_flux(
                        original_flux_exprs
                    )
                )
            else:
                is_invalid_constant = False

            if is_invalid_constant:
                # Retry the rollout if:
                # We don't allow constant-only fluxes, and
                # We have an invalid constant flux
                continue

            # Evaluate rollout
            reward, (flux_exprs, final_flux_strs) = (
                self.graph.reward_calculator.calculate_reward(
                    original_flux_exprs=original_flux_exprs, return_details=True
                )
            )

            # Reward could still be zero with a valid flux. For now, allow this result.
            # It is a valid solution - just not a good one (large error).
            # It is expensive to run reward calc, so rather just accept poor solution.
            # Only rerun for constant-only solutions (when disallowed).

            return reward, (flux_exprs, final_flux_strs)

        # All retries used up with constant-only fluxes
        return 0.0, ([], [])

    def rollout_take_action(
        self,
        flux_action_lists: list,
        flux_stacks: list,
        flux_ts: list,
        flux_to_expand_idx: int,
    ) -> tuple:
        """Take a valid action in the rollout phase."""
        # Standard rollout for the chosen flux
        stack = flux_stacks[flux_to_expand_idx]

        # Use the pre-calculated normalised probabilities
        possible_rules = self.graph.valid_M_rules[flux_to_expand_idx]
        probs = self.graph.M_rule_probs[flux_to_expand_idx]

        if not self.constant_only_flux_allowed:
            # Check if we only have one item left and if the equation has no states in it,
            # that we don't use "M <- C".
            if len(stack) == 1:
                # Check if the flux expression has a state in it
                has_state_var = self.has_state_variable(
                    flux_idx=flux_to_expand_idx, flux_actions=flux_action_lists
                )

                if not has_state_var:
                    # Filter to exclude "C" action
                    filtered_data = [
                        (rule, prob)
                        for rule, prob in zip(possible_rules, probs)
                        if "C" not in rule
                    ]

                    # Re-normalise
                    if filtered_data:
                        possible_rules, filtered_probs = zip(*filtered_data)
                        possible_rules = list(possible_rules)
                        sum_p = sum(filtered_probs)
                        probs = [p / sum_p for p in filtered_probs]

        # If there is only one stack left,
        # and there are no states used at all in the expression,
        # make sure not to use "M <- C"
        action = np.random.choice(possible_rules, p=probs)

        # Apply action and get full new state
        new_state = self.apply_action(
            (flux_action_lists, flux_stacks, flux_ts), flux_to_expand_idx, action
        )

        return new_state

    def force_complete_action(
        self, flux_action_lists: list, flux_stacks: list, flux_ts: list, flux: int
    ) -> tuple:
        """Take a valid terminal action in force completion."""
        if not self.constant_only_flux_allowed:
            # Check if the flux expression has a state in it
            has_state_var = self.has_state_variable(
                flux_idx=flux, flux_actions=flux_action_lists
            )

            if has_state_var:
                # We have a state variable, so we can use any terminal rules.
                term_rules = self.graph.valid_terminal_M_rules[flux]
                term_probs = self.graph.terminal_M_rule_probs[flux]
            else:
                # Pick a state variable, or the equation will be invalid.
                term_rules = self.graph.valid_state_terminal_M_rules[flux]
                term_probs = self.graph.valid_state_terminal_M_rule_probs[flux]

        else:
            # We allow constant-only fluxes, so we can use any terminal rules.
            term_rules = self.graph.valid_terminal_M_rules[flux]
            term_probs = self.graph.terminal_M_rule_probs[flux]

        # Sample weighted terminal action
        action = np.random.choice(term_rules, p=term_probs)

        new_state = self.apply_action(
            (flux_action_lists, flux_stacks, flux_ts), flux, action
        )

        return new_state
