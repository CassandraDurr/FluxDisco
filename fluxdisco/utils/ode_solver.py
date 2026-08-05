"""Methods for ODE solving with prediction boundaries."""

import typing

import numpy as np


def euler_method(
    func: typing.Callable,
    initial_conditions_matrix: np.ndarray,
    times: np.ndarray,
    args=(),
    prediction_boundary: float = 1e2,
) -> np.ndarray:
    """
    Solve a system of ODEs for multiple realisations simultaneously using Euler's method.

    Added prediction boundary clipping for stability of constant optimisation.

    Args:
        func (typing.Callable): Function defining ODEs. Must accept y of shape (num_states, num_realisations) & return dy/dt of the same shape.
        initial_conditions_matrix (np.ndarray): Initial conditions matrix. Shape: (num_states, num_realisations).
        times (np.ndarray): Time points vector. Assumes all realisations share the same time grid. Shape (time,).
        args (tuple, optional): Extra arguments for func (constants). Defaults to ().
        prediction_boundary (float, optional): Boundary to clip predicted trajectories for stability. Defaults to 1e2.

    Returns:
        np.ndarray: Predicted trajectories. Shape: (num_times, num_states, num_realisations).
    """  # noqa: E501
    # Get dimensions
    num_times = len(times)
    num_states, num_realisations = initial_conditions_matrix.shape

    # Initialise the predicted trajectory matrix
    # Shape: (time, num_states, num_realisations)
    pred_traj = np.zeros((num_times, num_states, num_realisations))

    # Set initial conditions
    pred_traj[0] = initial_conditions_matrix

    # Calculate time steps
    dts = times[1:] - times[:-1]

    # Iterate through time steps
    for idx in range(num_times - 1):
        dt = dts[idx]
        current_state = pred_traj[idx]  # Shape: (num_states, num_realisations)

        # Calculate derivative for all realisations at once
        dy_dt = func(times[idx], current_state, *args)

        # Euler step
        next_state = current_state + dt * np.asarray(dy_dt)

        # Clip stability boundary
        pred_traj[idx + 1] = np.clip(
            next_state, -prediction_boundary, prediction_boundary
        )

    return pred_traj


def rk4_method(
    func: typing.Callable,
    initial_conditions_matrix: np.ndarray,
    times: np.ndarray,
    args=(),
    prediction_boundary: float = 1e2,
) -> np.ndarray:
    """
    Solve a system of ODEs for multiple realisations simultaneously using 4th Order Runge-Kutta solver.

    Added prediction boundary clipping for stability of constant optimisation.

    Args:
        func (typing.Callable): Function defining ODEs. Must accept y of shape (num_states, num_realisations) & return dy/dt of the same shape.
        initial_conditions_matrix (np.ndarray): Initial conditions matrix. Shape: (num_states, num_realisations).
        times (np.ndarray): Time points vector. Assumes all realisations share the same time grid. Shape (time,).
        args (tuple, optional): Extra arguments for func (constants). Defaults to ().
        prediction_boundary (float, optional): Boundary to clip predicted trajectories for stability. Defaults to 1e2.

    Returns:
        np.ndarray: Predicted trajectories. Shape: (num_times, num_states, num_realisations).
    """  # noqa: E501
    # Get dimensions
    num_times = len(times)
    num_states, num_realisations = initial_conditions_matrix.shape

    # Initialise the predicted trajectory matrix
    # Shape: (time, num_states, num_realisations)
    pred_traj = np.zeros((num_times, num_states, num_realisations))

    # Set initial conditions
    pred_traj[0] = initial_conditions_matrix

    # Calculate time steps
    dts = times[1:] - times[:-1]

    # Iterate through time steps
    for idx in range(num_times - 1):
        time = times[idx]  # Time
        h = dts[idx]  # Time step
        y = pred_traj[idx]  # Current state (num_states, num_realisations)

        # k1
        k1 = np.asarray(func(time, y, *args))

        # k2
        y_mid1 = np.clip(y + (h / 2.0) * k1, -prediction_boundary, prediction_boundary)
        k2 = np.asarray(func(time + (h / 2.0), y_mid1, *args))

        # k3
        y_mid2 = np.clip(y + (h / 2.0) * k2, -prediction_boundary, prediction_boundary)
        k3 = np.asarray(func(time + (h / 2.0), y_mid2, *args))

        # k4
        y_end = np.clip(y + h * k3, -prediction_boundary, prediction_boundary)
        k4 = np.asarray(func(time + h, y_end, *args))

        # Combine
        update = (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        next_state = y + np.nan_to_num(
            update,
            nan=prediction_boundary,
            posinf=prediction_boundary,
            neginf=-prediction_boundary,
        )
        pred_traj[idx + 1] = np.clip(
            next_state, -prediction_boundary, prediction_boundary
        )

    return pred_traj
