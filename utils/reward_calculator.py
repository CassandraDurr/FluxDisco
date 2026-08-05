"""Reward calculation utilities for FluxDisco."""

import math
import typing

import numpy as np
import sympy
from joblib import Parallel, delayed, parallel_config
from scipy.optimize import least_squares
from scipy.stats import qmc

from .const_folding import constant_folding
from .expression_builder import sympy_expression_builder
from .ode_solver import euler_method, rk4_method


def sympy_ode(
    t_point: float,
    y_val: np.ndarray,
    callable_func: typing.Callable,
    c_args_tuple: tuple,
) -> list[np.ndarray]:
    """Calculate derivatives using a callable function.

    Args:
        t_point (float): Time
        y_val (np.ndarray): Current state values for all realisations
            Shape:(num_states, num_realisations)
            Order of states should match the order of state_keys & state_variables.
        callable_func (callable): Function to evaluate the ODE
        c_args_tuple (tuple): Constants to pass to the function

    Returns:
        list[np.ndarray]: List of derivative arrays for each state.
            Shape of each array: (num_realisations,)
    """
    # Call derivative function with current state and constants
    results = callable_func(*y_val, *c_args_tuple)

    # Broadcast scalars to match the realisations
    return [
        (
            np.full_like(y_val[0], result, dtype=float)
            if np.ndim(result) == 0  # Scalar
            else np.asarray(result, dtype=float)  # Already an array
        )
        for result in results
    ]


class RewardCalculator:
    """Reward calculator for FluxDisco."""

    def __init__(
        self,
        eta: float,
        system_structure: dict,
        state_variables: list[sympy.Symbol],
        state_to_variable: dict[str, sympy.Symbol],
        state_keys: list[str],
        time_array: np.ndarray,
        initial_conditions_matrix: np.ndarray,
        observations: dict[str, np.ndarray],
        obs_keys: list[str],
        n_term: str = "max",
        temperature: float = 1.0,
        solver_method: str = "rk4",  # Choice: "euler" or "rk4"
        normalise_reward: bool = False,
        max_inner_workers: int = 8,
    ):
        """Initialise reward calculator.

        Args:
            eta: Parsimony penalty factor
            system_structure: Stoichiometry matrix
            state_variables: List of sympy symbols representing state variables
            state_to_variable: Map from state names to sympy symbols
            state_keys: List of state variable names (e.g., ['s', 'i', 'r'] or ['x', 'y'])
            time_array: Shape (num_times,)
            initial_conditions_matrix: Shape (num_states, num_realisations)
            observations: Dictionary of observation arrays
            obs_keys: List of keys in observations to use for reward calculation
            n_term: Choose between "max" and "avg" for parsimony calculation
            temperature: Temperature parameter for reward calculation
            solver_method: Choice of ODE solver ("euler" or "rk4")
            normalise_reward: Whether to normalise the reward
            max_inner_workers: Number of allowed inner workers for parallel constant opt.
        """
        self.eta = eta
        self.n_term = n_term
        self.temperature = temperature
        self.solver_method = solver_method
        self.max_inner_workers = max_inner_workers

        # System structure
        self.system_structure = system_structure
        self.state_keys = state_keys
        self.state_var_list = state_variables
        self.state_var_set = set(state_variables)
        self.state_to_variable = state_to_variable

        # Data for ODE solving
        self.observations = observations
        self.obs_keys = obs_keys
        self.time_array = time_array
        self.initial_conditions_matrix = initial_conditions_matrix

        # Reward cache to save time
        self.reward_cache = {}

        # Normalisation of reward
        self.normalise_reward = normalise_reward
        self.weights = None
        if self.normalise_reward:
            # Get max of observations
            self.weights = {
                key: np.max(self.observations[key]) for key in self.obs_keys
            }
            # Avoid division by zero
            for key in self.obs_keys:
                if self.weights[key] == 0:
                    self.weights[key] = 1.0

    def build_expressions(self, flux_action_lists: list[list[str]]) -> list[sympy.Expr]:
        """Build expressions for each flux in a list of action lists."""
        flux_exprs = []
        const_count = 0

        # Build SymPy expressions
        for action_list in flux_action_lists:
            # Pass the current constant count as the offset
            eq_expr, consts = sympy_expression_builder(
                action_list,
                const_offset=const_count,
                partial_mode=False,
                state_mapping=self.state_to_variable,
            )
            flux_exprs.append(eq_expr)
            const_count += len(consts)

        return flux_exprs

    def check_invalid_constant_flux(self, flux_exprs: list[sympy.Expr]) -> bool:
        """Check if any flux expression is a constant with no state variables."""
        for expr in flux_exprs:
            if not expr.free_symbols.intersection(self.state_var_set):
                # Invalid flux -> should use at least one state variable.
                return True
        return False

    def calculate_reward(
        self,
        original_flux_exprs: list[sympy.Expr],
        return_details: bool = False,
    ) -> float:
        """Calculate the reward for a terminal state, including constant optimisation.

        Reward function: r = (eta^n) * exp(- MSE/ temperature)

        Args:
            original_flux_exprs (list[sympy.Expr]): Flux expressions (pre-folding) for each of the fluxes.
            return_details (bool, optional): Whether to return details of the computed best equation. Defaults to False.

        Raises:
            NotImplementedError: if self.n_term is not max or avg.

        Returns:
            float: reward, r
        """  # noqa: E501
        try:
            # Constant folding
            folded_flux_exprs = []
            flux_rule_count = []
            all_consts = []
            k_index = 0

            # Apply folding to each flux individually
            for expr in original_flux_exprs:
                # Pass k_index to ensure fluxes have unique constants
                folded_expr, k_index, k_consts, flux_rules = constant_folding(
                    expr=expr,
                    state_vars=self.state_var_set,
                    start_index=k_index,
                    eval_rules=True,
                )
                all_consts.extend(k_consts)
                folded_flux_exprs.append(folded_expr)
                flux_rule_count.append(flux_rules)

            # Create a unique key for this system of equations
            cache_key = tuple(str(expr) for expr in folded_flux_exprs)

            # Check if we have already evaluated these set of equations
            if cache_key in self.reward_cache:
                c_reward, (c_exprs, c_strs) = self.reward_cache[cache_key]
                if return_details:
                    return c_reward, (c_exprs, c_strs)
                return c_reward

            # Build the full system equations using system structure (stoichiometry)
            system_eqs = {}
            for state_key in self.state_keys:
                state_eq = sympy.Float(0.0)
                for idx, flux_coeff in enumerate(self.system_structure[state_key]):
                    if flux_coeff != 0:
                        state_eq += flux_coeff * folded_flux_exprs[idx]
                system_eqs[state_key] = state_eq

            # Flux expressions individually
            flux_eqs = {f"J{idx+1}": expr for idx, expr in enumerate(folded_flux_exprs)}

            # Find the optimal constants and the resulting RMSE
            best_total_error, optimal_const_values = self.optimise_constants(
                system_eqs=system_eqs,
                flux_exprs=flux_eqs,
                consts=all_consts,
            )

            # Calculate parsimony-penalised reward
            if self.n_term == "max":
                parsimony_n = max(flux_rule_count)
            elif self.n_term == "avg":
                parsimony_n = sum(flux_rule_count) / len(flux_rule_count)
            else:
                raise NotImplementedError()
            reward = (self.eta**parsimony_n) * math.exp(-best_total_error)

            # Store the top result (as a system)
            if optimal_const_values.size > 0:
                subs_dict = dict(zip(all_consts, optimal_const_values))
                final_flux_strs = [
                    str(expr.subs(subs_dict)) for expr in folded_flux_exprs
                ]
            else:
                final_flux_strs = [str(expr) for expr in folded_flux_exprs]

            # Cache the result
            self.reward_cache[cache_key] = (
                reward,
                (folded_flux_exprs, final_flux_strs),
            )

            if return_details:
                # Also return details of the equation form
                return reward, (folded_flux_exprs, final_flux_strs)
            return reward

        except Exception as e:
            print(f"Reward exception: {e}")
            if return_details:
                return 0.0, ([], [])
            return 0.0

    def optimise_constants(
        self,
        system_eqs: dict[str, sympy.Expr],
        flux_exprs: dict[str, sympy.Expr],
        consts: list[sympy.Symbol],
    ) -> tuple[float, np.ndarray]:
        """
        Find the optimal values for constants to minimise error.

        Use least squares method with trust regions and multiple starting points.

        Returns:
            tuple[float, np.ndarray]: (best_rmse, optimal_const_values)
        """
        num_consts = len(consts)

        # Ordered state equations
        ordered_eqs = [system_eqs[key] for key in self.state_keys]

        try:
            # Lambdify the whole system as a vectorised function
            vectorised_ode_func = sympy.lambdify(
                self.state_var_list + consts,
                ordered_eqs,
                modules=[
                    {
                        "Abs": np.abs,
                        "sqrt": np.sqrt,
                        "sign": np.sign,
                        "zoo": lambda: 1e12,
                    },
                    "numpy",
                ],
            )

            # Lambdify flux expressions (for predictions)
            lambdified_fluxes = {}
            for flux_key, flux_expr in flux_exprs.items():
                lambdified_fluxes[flux_key] = sympy.lambdify(
                    self.state_var_list + consts,
                    flux_expr,
                    modules=[
                        {
                            "Abs": np.abs,
                            "sqrt": np.sqrt,
                            "sign": np.sign,
                            "zoo": lambda: 1e12,
                        },
                        "numpy",
                    ],
                )
        except Exception as e:
            raise Exception(f"Error in lambdification: {e}")

        # Arguments for the objective function
        args_tuple = (vectorised_ode_func, lambdified_fluxes)

        # Handle case with no constants
        if num_consts == 0:
            # Just evaluate the expression directly
            error = self.objective_function(np.array([]), *args_tuple)
            return error, np.array([])

        # Find the optimal constants
        best_total_error = math.inf
        optimal_const_values = np.array([])

        # Sobol sampler to cover search space more efficiently
        sampler = qmc.Sobol(
            d=num_consts, scramble=True
        )  # Scramble adds a bit of randomness
        sample = sampler.random(n=self.max_inner_workers)
        # Scale and transform to log-space
        log_samples = qmc.scale(sample, -4, 2)
        starting_guesses = 10**log_samples

        def const_opt(
            starting_guesses: np.ndarray,
        ) -> tuple[float, np.ndarray | None]:
            try:
                # Trust region least squares optimisation from starting point
                res = least_squares(
                    fun=self.get_residuals,
                    x0=starting_guesses,
                    args=args_tuple,
                    method="trf",
                    ftol=1e-5,
                )

                if not res.success:
                    # No solution found from this initialisation
                    return math.inf, None

                # Get MSE from residuals
                err = self.objective_function(
                    c_values=res.x,
                    vectorised_ode_func=vectorised_ode_func,
                    lambdified_fluxes=lambdified_fluxes,
                    residuals=res.fun,
                )

                # Check for numerical issues
                if not np.isfinite(err):
                    return math.inf, None

                return err, res.x

            except Exception:
                print(f"Error occurred while fitting restart {starting_guesses}")
                return math.inf, None

        with parallel_config(
            backend="loky", prefer="processes", n_jobs=self.max_inner_workers
        ):
            scored_results = Parallel()(
                delayed(const_opt)(start_val) for start_val in starting_guesses
            )

        for err, const_vals in scored_results:
            if const_vals is not None and err < best_total_error:
                best_total_error = err
                optimal_const_values = const_vals

        # If all attempts failed, return high penalty
        if best_total_error == math.inf:
            print("Constant optimisation failed, returning high error")
            return 1e6, np.array([0.01] * num_consts)

        return best_total_error, optimal_const_values

    def extract_predictions(
        self,
        observations: dict[str, np.ndarray],
        pred_traj_tensor: np.ndarray,
        lambdified_fluxes: dict[str, callable],
        c_values: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Extract predictions from ODE solver output.

        Args:
            observations (dict[str, np.ndarray]): Observed data, {"name": data array}.
            pred_traj_tensor (np.ndarray): Output from the ODE solver.
            lambdified_fluxes (dict[str, callable]): Lambdified flux expressions.
            c_values (np.ndarray): Constant values to pass to lambdified functions.

        Raises:
            ValueError: Unsupported observation keys.

        Returns:
            dict[str, np.ndarray]: Dictionary of predictions in the same form as observations.
        """
        obs_keys = set(observations.keys())

        # === Options specific to the SIR epidemic system ===
        if obs_keys == {"prevalence", "incidence"}:
            predicted_prevalence = pred_traj_tensor[:, 1, :]
            # shape (time, num_realisations)
            predicted_susceptibles = pred_traj_tensor[:, 0, :]
            # shape (time, num_realisations)
            pred_incidence = -np.diff(predicted_susceptibles, axis=0)
            # shape (time-1, num_realisations)
            return {"prevalence": predicted_prevalence, "incidence": pred_incidence}

        elif obs_keys == {
            "prevalence",
            "incidence",
            "hospitalisations",
            "hosp_recoveries",
            "deaths",
        }:
            # Include hospitalisations and deaths in the system
            pred_s, pred_i, pred_r, pred_h, pred_d = [
                pred_traj_tensor[:, i, :] for i in range(5)
            ]
            pred_incidence = -np.diff(pred_s, axis=0)
            pred_deaths = np.diff(pred_d, axis=0)
            # Adjust states for lambdified flux calculations
            s_start, i_start, r_start, h_start, d_start = (
                pred_s[:-1],
                pred_i[:-1],
                pred_r[:-1],
                pred_h[:-1],
                pred_d[:-1],
            )
            # Hospitalisations corresponds to flux J3 (I->H)
            pred_hospitalisations = lambdified_fluxes["J3"](
                s_start, i_start, r_start, h_start, d_start, *c_values
            )
            # Hospital recoveries corresponds to flux J4 (H->R)
            pred_hosp_recoveries = lambdified_fluxes["J4"](
                s_start, i_start, r_start, h_start, d_start, *c_values
            )

            return {
                "prevalence": pred_i,
                "incidence": pred_incidence,
                "deaths": pred_deaths,
                "hospitalisations": pred_hospitalisations,
                "hosp_recoveries": pred_hosp_recoveries,
            }

        # === General case: Observation keys match state keys ===
        # This is the case for our FluxDisco paper
        elif obs_keys == set(self.state_keys):
            return {
                key: pred_traj_tensor[:, idx, :]
                for idx, key in enumerate(self.state_keys)
            }

        else:
            raise ValueError("Prediction extraction not setup.")

    def get_model_predictions(
        self,
        c_values: np.ndarray,
        vectorised_ode_func: callable,
        lambdified_fluxes: dict,
    ) -> dict[str, np.ndarray] | None:
        """Get model predictions by combining constants and candidate expressions, then solve ODEs.

        Args:
            c_values (np.ndarray): Constant values to evaluate the ODEs with.
            vectorised_ode_func (callable): Lambdified ODE system function.
            lambdified_fluxes (dict): Lambdified flux functions.

        Raises:
            ValueError: Invalid solver method specified.

        Returns:
            dict[str, np.ndarray] | None: None if there was an error, otherwise return predictions.
        """
        c_args = tuple(c_values)

        # Create a callable function for the ODE solver
        ode_func = lambda t_val, y_val, *args: sympy_ode(  # noqa: E731
            t_point=t_val,
            y_val=y_val,
            callable_func=vectorised_ode_func,
            c_args_tuple=c_args,
        )

        try:
            # Solve ODEs for all realisations at once
            # Returns shape: (num_times, num_states, num_realisations)
            if self.solver_method == "euler":
                pred_traj_tensor = euler_method(
                    func=ode_func,
                    initial_conditions_matrix=self.initial_conditions_matrix,
                    times=self.time_array,
                    args=c_args,
                )
            elif self.solver_method == "rk4":
                pred_traj_tensor = rk4_method(
                    func=ode_func,
                    initial_conditions_matrix=self.initial_conditions_matrix,
                    times=self.time_array,
                    args=c_args,
                )
            else:
                raise ValueError(f"Invalid solver method: {self.solver_method}")

            # Check for numerical explosion (NaN/Inf)
            if not np.all(np.isfinite(pred_traj_tensor)):
                print("Numerical explosion in prediction")
                return None

            # Extract predictions
            predictions = self.extract_predictions(
                observations=self.observations,
                pred_traj_tensor=pred_traj_tensor,
                lambdified_fluxes=lambdified_fluxes,
                c_values=c_values,
            )

        except Exception as e:
            print(f"Fail in getting model predictions: {e}")
            return None

        return predictions

    def objective_function(
        self,
        c_values: np.ndarray,
        vectorised_ode_func: callable,
        lambdified_fluxes: dict,
        residuals: np.ndarray | None = None,
    ) -> float:
        """
        MSE calculation using observed and predicted trajectories.

        Returns the average MSE over realisations, divided by temperature.
        """
        failure_penalty = 1e6

        # If we have residuals (from least squares), calculate MSE from these
        # SSE is related to error function in reward
        if residuals is not None:
            num_realisations = self.initial_conditions_matrix.shape[1]
            mse_per_real = np.zeros(num_realisations)

            current_idx = 0
            for obs_key in self.obs_keys:
                # Expected array length for this observation key
                num_times_feat = self.observations[obs_key].shape[0]
                total_elements = num_times_feat * num_realisations

                # Slice out the portion of residuals from this observation key
                # If normalising, these residuals are already scaled
                feat_residuals = residuals[
                    current_idx : current_idx + total_elements  # noqa: E203
                ]
                # Reshape to (num_times, num_realisations)
                feat_residuals = feat_residuals.reshape(
                    num_times_feat, num_realisations
                )

                # Calculate MSE for this feature
                mse_feature = np.mean(
                    feat_residuals**2, axis=0
                )  # Shape: (num_realisations,)
                mse_per_real += mse_feature

                # Increase the index for the next observation key
                current_idx += total_elements

            return np.mean(mse_per_real) / self.temperature

        # Extract predictions
        predictions = self.get_model_predictions(
            c_values=c_values,
            vectorised_ode_func=vectorised_ode_func,
            lambdified_fluxes=lambdified_fluxes,
        )

        if predictions is None:
            # If there was an error in getting predictions, return failure penalty
            return failure_penalty

        # MSE over time per realisation
        mse_per_real = np.zeros(predictions[self.obs_keys[0]].shape[1])
        for obs_key in self.obs_keys:
            # Residuals
            obs_residuals = predictions[obs_key] - self.observations[obs_key]

            # Normalise residuals if required
            if self.normalise_reward and self.weights is not None:
                scaled_residuals = obs_residuals / self.weights[obs_key]
                obs_residuals = scaled_residuals

            # Calculate squared error: Shape (num_times, num_realisations)
            squared_error = obs_residuals**2

            # MSE/ average over time (axis 0): Shape (num_realisations,)
            mse_feature = np.mean(squared_error, axis=0)

            # Increment total MSE
            mse_per_real += mse_feature

        # Total error = average over realisations, scaled by temperature
        # Shape: scalar
        avg_total_mse = np.mean(mse_per_real) / self.temperature

        if np.isnan(avg_total_mse) or np.isinf(avg_total_mse):
            print("Numerical explosion in MSE")
            return failure_penalty

        return avg_total_mse

    def get_residuals(
        self, c_values, vectorised_ode_func, lambdified_fluxes
    ) -> np.ndarray:
        """Get residuals for least squares optimisation.

        Args:
            c_values (np.ndarray): Constant values to evaluate the ODEs with.
            vectorised_ode_func (callable): Lambdified ODE system function.
            lambdified_fluxes (dict): Lambdified flux functions.

        Raises:
            ValueError: Error in ODE solving or prediction extraction.

        Returns:
            np.ndarray: Flattened residuals vector.
        """
        # Extract predictions
        predictions = self.get_model_predictions(
            c_values=c_values,
            vectorised_ode_func=vectorised_ode_func,
            lambdified_fluxes=lambdified_fluxes,
        )

        if predictions is None:
            raise ValueError("Failed to get predictions for residual calculation.")

        # Get flattened residuals
        residuals = []
        for obs_key in self.obs_keys:
            # Shape of residuals: (num_times, num_realisations)
            obs_residuals = predictions[obs_key] - self.observations[obs_key]
            if self.normalise_reward and self.weights is not None:
                scaled_residuals = obs_residuals / self.weights[obs_key]
                obs_residuals = scaled_residuals
            # Flatten to shape (num_times * num_realisations,)
            flatten_residuals = obs_residuals.flatten()
            residuals.append(flatten_residuals)

        # Final shape: (num_obs_keys * num_times * num_realisations,)
        return np.concatenate(residuals)

    def export_reward_cache(self) -> dict:
        """Export the reward cache."""
        export_data = []
        for _, (reward, (exprs, expr_strs)) in self.reward_cache.items():
            folded_strs = [str(e) for e in exprs] if exprs else []
            export_data.append(
                {
                    "reward": reward,
                    "equations": " ; ".join(folded_strs) if folded_strs else "",
                    "equations_with_constants": (
                        " ; ".join(expr_strs) if expr_strs else ""
                    ),
                }
            )
        # Sort by reward descending
        export_data.sort(key=lambda x: x["reward"], reverse=True)
        # Add rank
        for idx, item in enumerate(export_data):
            item["rank"] = idx + 1
        return export_data
