"""Module to compute confidence intervals for stochastic rewards using various methods."""

import numpy as np
import scipy.stats as stats
from scipy.optimize import root_scalar


def binary_kl_divergence(u, v, numerical_error=1e-12):
    """
    Binary KL divergence between Bernoulli(u) and Bernoulli(v) distributions.

    In the context of the MCGS paper, u represents the empirical mean reward (r_hat),
    and v is the variable we are optimising over to find confidence bounds.
    """
    # Convert to float
    u, v = float(u), float(v)
    # Numerical safety
    u = np.clip(u, numerical_error, 1 - numerical_error)
    v = np.clip(v, numerical_error, 1 - numerical_error)
    return u * np.log(u / v) + (1 - u) * np.log((1 - u) / (1 - v))


def exploration_function(n_total: int, n_sa: int) -> float:
    """Exploration function for confidence intervals.

    Args:
        n_total (int): Total number of samples.
        n_sa (int): Number of times action a was taken in state s.

    Returns:
        float: Exploration bonus.
    """
    # MCGS paper: np.log(n_total) / n_sa
    # - takes very long for confidence intervals to tighten
    return np.sqrt(np.log(n_total)) / n_sa


def kl_ci(r_hat: float, n_sa: int, n_total: int) -> tuple[float, float]:
    """Compute [l_t, u_t] BKL confidence interval for rewards using Brent's method.

    Formula: kl(r_hat, v) <= log(n_total) / n_sa (MCGS paper)

    Args:
        r_hat (float): Empirical mean reward.
        n_sa (int): Number of times action a was taken in state s.
        n_total (int): Total number of samples.

    Returns:
        tuple: Lower and upper confidence bounds (l_t, u_t).
    """
    # Exploration function
    # -> beta_r is log(n) based on implementation details in the supplementary material
    divergence_budget = exploration_function(n_total, n_sa)

    # Define the root function for optimisation
    def root_function(v):
        return binary_kl_divergence(r_hat, v) - divergence_budget

    # Tolerance for root finding
    xtol, rtol = 1e-5, 1e-5

    # --- Calculate upper confidence bound u_t ---
    # If the max possible divergence (v=1) is within budget, u_t=1.
    if binary_kl_divergence(r_hat, 1.0) <= divergence_budget:
        u_t = 1.0
    else:
        try:
            # Max plausible reward >= empirical reward
            u_t = root_scalar(
                root_function,
                bracket=[r_hat, 1.0],
                method="brentq",
                xtol=xtol,
                rtol=rtol,
            ).root
        except Exception as e:
            raise Exception(f"Error computing upper confidence bound u_t: {e}")

    # --- Calculate lower confidence bound l_t ---
    # If the max possible divergence (v=0) is within budget, l_t=0.
    if binary_kl_divergence(r_hat, 0.0) <= divergence_budget:
        l_t = 0.0
    else:
        try:
            # Min plausible reward <= empirical reward
            l_t = root_scalar(
                root_function,
                bracket=[0.0, r_hat],
                method="brentq",
                xtol=xtol,
                rtol=rtol,
            ).root
        except Exception as e:
            raise Exception(f"Error computing lower confidence bound l_t: {e}")

    return l_t, u_t


def ucb1_ci(r_hat: float, n_sa: int, n_total: int) -> tuple[float, float]:
    """Compute [l_t, u_t] confidence interval for rewards using UCB1 formula.

    Args:
        r_hat (float): Empirical mean reward.
        n_sa (int): Number of times action a was taken in state s.
        n_total (int): Total number of samples.

    Returns:
        tuple: Lower and upper confidence bounds (l_t, u_t).
    """
    # Exploration bonus based on UCB1
    exploration_bonus = np.sqrt(2 * np.log(n_total) / n_sa)

    # Calculate confidence bounds
    l_t = max(0.0, r_hat - exploration_bonus)
    u_t = min(1.0, r_hat + exploration_bonus)

    return l_t, u_t


def ubc1_tuned_ci(
    r_hat: float, r_var: float, n_sa: int, n_total: int
) -> tuple[float, float]:
    """Compute [l_t, u_t] confidence interval for rewards using UCB1-Tuned formula.

    Args:
        r_hat (float): Empirical mean reward.
        r_var (float): Empirical variance of reward.
        n_sa (int): Number of times action a was taken in state s.
        n_total (int): Total number of samples.

    Returns:
        tuple: Lower and upper confidence bounds (l_t, u_t).
    """
    if n_sa < 2:
        # If not enough samples, return maximum uncertainty
        return 0.0, 1.0

    # Empirical variance of rewards
    empirical_var = r_var + np.sqrt((2 * np.log(n_total)) / n_sa)

    # Exploration bonus based on UCB1-Tuned
    exploration_bonus = np.sqrt((np.log(n_total) / n_sa) * min(0.25, empirical_var))

    # Calculate confidence bounds
    l_t = max(0.0, r_hat - exploration_bonus)
    u_t = min(1.0, r_hat + exploration_bonus)

    return l_t, u_t


def t_dist_ci(r_hat: float, r_var: float, n_sa: int, confidence_level: float = 0.95):
    """Compute [l_t, u_t] confidence interval for rewards using t-distribution.

    Args:
        r_hat (float): Empirical mean reward.
        r_var (float): Empirical variance of reward.
        n_sa (int): Number of times action a was taken in state s.
        confidence_level (float): Confidence level (default 0.95).

    Returns:
        tuple: Lower and upper confidence bounds (l_t, u_t).
    """
    if n_sa < 2:
        # If not enough samples, return maximum uncertainty
        return 0.0, 1.0

    # Standard error of the mean
    std_error = np.sqrt(r_var / n_sa)

    # Critical value from t-distribution
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha / 2, n_sa - 1)

    # Compute confidence bounds
    margin = t_critical * std_error

    l_t = max(0.0, r_hat - margin)
    u_t = min(1.0, r_hat + margin)

    return l_t, u_t


def beta_ci(
    r_hat: float, n_sa: int, confidence_level: float = 0.95
) -> tuple[float, float]:
    """Compute [l_t, u_t] Bayesian credible interval using the Beta distribution.

    The parameters alpha and beta represent successes and failures.
    For scalar rewards in [0, 1], alpha = 1 + sum(rewards) and beta = 1 + sum(1 - rewards).

    Args:
        r_hat (float): Empirical mean reward.
        n_sa (int): Number of times action a was taken in state s.
        confidence_level (float): Confidence level (default 0.95).

    Returns:
        tuple: Lower and upper confidence bounds (l_t, u_t).
    """
    # Number of successes (reward) + 1 for prior
    alpha_param = 1.0 + (n_sa * r_hat)
    # Number of failures (1 - reward) + 1 for prior
    beta_param = 1.0 + (n_sa * (1.0 - r_hat))

    # Calculate bounds based on the inverse cdf of the Beta distribution
    tail_prob = (1.0 - confidence_level) / 2.0
    l_t = stats.beta.ppf(tail_prob, alpha_param, beta_param)
    u_t = stats.beta.ppf(1.0 - tail_prob, alpha_param, beta_param)

    return l_t, u_t


def confidence_intervals_rewards(
    r_hat: float,
    r_var: float,
    n_sa: int,
    n_total: int,
    method: str = "beta",  # Default method is "beta" for Bayesian credible intervals
) -> tuple[float, float]:
    """Compute [l_t, u_t] confidence interval for rewards.

    Args:
        r_hat (float): Empirical mean reward.
        r_var (float): Empirical variance of reward.
        n_sa (int): Number of times action a was taken in state s.
        n_total (int): Total number of samples.
        method (str): Method for intervals ("kl", "ucb1", "ucb1_tuned", "t_dist", "beta").

    Returns:
        tuple: Lower and upper confidence bounds (l_t, u_t).
    """
    if n_sa < 1:
        # No rollouts performed yet; return maximum uncertainty
        return 0.0, 1.0

    # Convert r_hat to float to avoid SymPy issues
    r_hat = float(r_hat)
    if method in ["ucb1_tuned", "t_dist"]:
        r_var = float(r_var)

    # Choose method for CIs (KL divergence, UCB1, UCB1-Tuned, t-distribution, or Beta)
    if method == "kl":
        return kl_ci(r_hat=r_hat, n_sa=n_sa, n_total=n_total)
    elif method == "ucb1":
        return ucb1_ci(r_hat=r_hat, n_sa=n_sa, n_total=n_total)
    elif method == "ucb1_tuned":
        return ubc1_tuned_ci(r_hat=r_hat, r_var=r_var, n_sa=n_sa, n_total=n_total)
    elif method == "t_dist":
        return t_dist_ci(r_hat=r_hat, r_var=r_var, n_sa=n_sa, confidence_level=0.95)
    elif method == "beta":
        return beta_ci(r_hat=r_hat, n_sa=n_sa, confidence_level=0.95)
    else:
        raise ValueError(f"Invalid method for confidence intervals: {method}")
