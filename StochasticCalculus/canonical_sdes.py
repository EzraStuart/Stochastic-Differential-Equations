"""Canonical SDE Problems with Known Solutions"""

import numpy as np
from StochasticCalculus.problems import SDEProblem, DiffusionType


def create_geometric_brownian_motion(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    t0: float = 0.0
) -> SDEProblem:
    """
    Geometric Brownian Motion (GBM)

    SDE: dS = μ S dt + σ S dW

    Analytical solution:
        S(t) = S(0) exp((μ - σ²/2)t + σ W(t))

    Used for stock price modeling in Black-Scholes framework.

    Args:
        S0: Initial value
        mu: Drift coefficient (expected return)
        sigma: Volatility coefficient
        T: Terminal time
        t0: Initial time (default 0)

    Returns:
        SDEProblem instance
    """

    def drift(t: float, x: np.ndarray) -> np.ndarray:
        """Drift: μ S"""
        return mu * x

    def diffusion(t: float, x: np.ndarray) -> np.ndarray:
        """Diffusion: σ S"""
        return sigma * x

    def analytical_solution(t: float, x0: np.ndarray) -> np.ndarray:
        """
        Exact solution: S(t) = S(0) exp((μ - σ²/2)t + σ W(t))

        For Monte Carlo comparison, we need the distribution at time t.
        Since W(t) ~ N(0, t), we have:
            log(S(t)/S(0)) ~ N((μ - σ²/2)t, σ²t)

        For vectorized x0 (n_paths, 1), returns (n_paths, 1)
        """
        # Generate independent samples for each path
        if x0.ndim == 1:
            n_paths = 1
            x0_use = x0.reshape(1, -1)
        else:
            n_paths = x0.shape[0]
            x0_use = x0

        # W(t) ~ N(0, t) for each path
        W_t = np.random.normal(0, np.sqrt(t), size=(n_paths, 1))

        result = x0_use * np.exp((mu - 0.5 * sigma**2) * t + sigma * W_t)

        return result.reshape(x0.shape)

    def theoretical_mean(t: float, x0: np.ndarray) -> np.ndarray:
        """Theoretical mean: E[S(t)] = S(0) * exp(mu * t)"""
        return x0 * np.exp(mu * t)

    def theoretical_variance(t: float, x0: np.ndarray) -> np.ndarray:
        """Theoretical variance: Var[S(t)] = S(0)^2 * exp(2*mu*t) * (exp(sigma^2*t) - 1)"""
        return (x0 ** 2) * np.exp(2 * mu * t) * (np.exp(sigma**2 * t) - 1)

    return SDEProblem(
        name="Geometric Brownian Motion",
        dim=1,
        x0=np.array([S0]),
        t_span=(t0, T),
        diffusion_type=DiffusionType.DIAGONAL,
        drift_fn=drift,
        diffusion_fn=diffusion,
        analytical_solution=analytical_solution,
        theoretical_mean=theoretical_mean,
        theoretical_variance=theoretical_variance
    )


def create_ornstein_uhlenbeck(
    x0: float,
    theta: float,
    mu: float,
    sigma: float,
    T: float,
    t0: float = 0.0
) -> SDEProblem:
    """
    Ornstein-Uhlenbeck (OU) Process

    SDE: dX = θ(μ - X) dt + σ dW

    Analytical solution:
        X(t) = x(0) exp(-θt) + μ(1 - exp(-θt)) + σ ∫₀ᵗ exp(-θ(t-s)) dW(s)

    Mean-reverting process used for interest rates, volatility modeling.

    Args:
        x0: Initial value
        theta: Mean reversion speed (θ > 0)
        mu: Long-term mean
        sigma: Volatility
        T: Terminal time
        t0: Initial time

    Returns:
        SDEProblem instance
    """

    def drift(t: float, x: np.ndarray) -> np.ndarray:
        """Drift: θ(μ - X)"""
        return theta * (mu - x)

    def diffusion(t: float, x: np.ndarray) -> np.ndarray:
        """Diffusion: σ (constant)"""
        return sigma * np.ones_like(x)

    def analytical_solution(t: float, x0_val: np.ndarray) -> np.ndarray:
        """
        Exact solution:
            X(t) = x0 exp(-θt) + μ(1 - exp(-θt)) + σ√((1-exp(-2θt))/(2θ)) Z

        where Z ~ N(0,1)
        """
        if x0_val.ndim == 1:
            n_paths = 1
            x0_use = x0_val.reshape(1, -1)
        else:
            n_paths = x0_val.shape[0]
            x0_use = x0_val

        # Mean at time t
        mean = x0_use * np.exp(-theta * t) + mu * (1 - np.exp(-theta * t))

        # Variance at time t
        variance = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * t))

        # Generate samples
        Z = np.random.normal(0, 1, size=(n_paths, 1))
        result = mean + np.sqrt(variance) * Z

        return result.reshape(x0_val.shape)

    return SDEProblem(
        name="Ornstein-Uhlenbeck",
        dim=1,
        x0=np.array([x0]),
        t_span=(t0, T),
        diffusion_type=DiffusionType.DIAGONAL,
        drift_fn=drift,
        diffusion_fn=diffusion,
        analytical_solution=analytical_solution
    )


def create_cir(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    t0: float = 0.0
) -> SDEProblem:
    """
    Cox-Ingersoll-Ross (CIR) Process

    SDE: dr = κ(θ - r) dt + σ√r dW

    Used for interest rate modeling. Ensures positivity if 2κθ ≥ σ² (Feller condition).

    No closed-form solution for arbitrary times, but transition density is known
    (non-central chi-squared). For simplicity, no analytical_solution provided here.

    Args:
        r0: Initial rate
        kappa: Mean reversion speed
        theta: Long-term mean
        sigma: Volatility
        T: Terminal time
        t0: Initial time

    Returns:
        SDEProblem instance
    """

    def drift(t: float, x: np.ndarray) -> np.ndarray:
        """Drift: κ(θ - r)"""
        return kappa * (theta - x)

    def diffusion(t: float, x: np.ndarray) -> np.ndarray:
        """Diffusion: σ√r (state-dependent)"""
        return sigma * np.sqrt(np.maximum(x, 0))  # Ensure non-negative under sqrt

    return SDEProblem(
        name="Cox-Ingersoll-Ross",
        dim=1,
        x0=np.array([r0]),
        t_span=(t0, T),
        diffusion_type=DiffusionType.DIAGONAL,
        drift_fn=drift,
        diffusion_fn=diffusion,
        analytical_solution=None  # No simple closed form
    )
