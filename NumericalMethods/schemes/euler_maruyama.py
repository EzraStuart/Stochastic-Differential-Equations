"""Euler-Maruyama Scheme"""

import numpy as np

from NumericalMethods.base import NumericalScheme
from StochasticCalculus.problems import SDEProblem, DiffusionType


class EulerMaruyama(NumericalScheme):
    """
    Euler-Maruyama Scheme

    The simplest explicit method for SDEs:
        X_{n+1} = X_n + μ(t_n, X_n)Δt + σ(t_n, X_n)ΔW_n

    Convergence:
    - Strong order: 0.5 (error ~ O(√Δt))
    - Weak order: 1.0 (error ~ O(Δt))

    Works for both Itô and Stratonovich formulations (via drift correction).
    """

    name: str = "Euler-Maruyama"
    order_strong: float = 0.5
    order_weak: float = 1.0

    def step(
        self,
        problem: SDEProblem,
        t: float,
        x: np.ndarray,
        dt: float,
        dW: np.ndarray
    ) -> np.ndarray:
        """
        Single Euler-Maruyama step

        Args:
            problem: SDE problem
            t: Current time
            x: Current state (n_paths, dim)
            dt: Time step
            dW: Brownian increment (n_paths, dim)

        Returns:
            Next state (n_paths, dim)
        """
        # Compute drift and diffusion at current state
        drift = problem.drift(t, x)
        diffusion = problem.diffusion(t, x)

        # Euler-Maruyama update
        if problem.diffusion_type == DiffusionType.DIAGONAL:
            # Element-wise multiplication for diagonal diffusion
            # drift: (n_paths, dim), diffusion: (n_paths, dim), dW: (n_paths, dim)
            x_next = x + drift * dt + diffusion * dW

        elif problem.diffusion_type == DiffusionType.FULL_MATRIX:
            # Matrix-vector multiplication for full diffusion matrix
            # diffusion: (n_paths, dim, dim), dW: (n_paths, dim)
            # Need to compute Σ @ dW for each path
            if diffusion.ndim == 3:
                # (n_paths, dim, dim) @ (n_paths, dim) -> (n_paths, dim)
                stochastic_term = np.einsum('ijk,ik->ij', diffusion, dW)
            else:
                # Single matrix (dim, dim) @ (n_paths, dim)
                stochastic_term = (diffusion @ dW.T).T

            x_next = x + drift * dt + stochastic_term

        else:
            raise ValueError(f"Unknown diffusion type: {problem.diffusion_type}")

        return x_next
