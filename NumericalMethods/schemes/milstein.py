"""Milstein Scheme"""

import numpy as np

from NumericalMethods.base import NumericalScheme
from StochasticCalculus.problems import SDEProblem, DiffusionType


class Milstein(NumericalScheme):
    """
    Milstein Scheme

    Higher-order explicit method that includes derivative of diffusion:
        X_{n+1} = X_n + μ(t_n, X_n)Δt + σ(t_n, X_n)ΔW_n
                  + (1/2)σ(t_n, X_n)σ'(t_n, X_n)(ΔW_n^2 - Δt)

    where σ'(t,x) = ∂σ/∂x

    Convergence:
    - Strong order: 1.0 (error ~ O(Δt))
    - Weak order: 1.0 (error ~ O(Δt))

    More accurate than Euler-Maruyama for same dt, but requires computing
    derivatives of diffusion coefficient.

    Note: For problems with state-independent diffusion (σ' = 0),
          Milstein reduces to Euler-Maruyama.
    """

    name: str = "Milstein"
    order_strong: float = 1.0
    order_weak: float = 1.0

    def _diffusion_derivative(
        self,
        problem: SDEProblem,
        t: float,
        x: np.ndarray,
        h: float = 1e-5
    ) -> np.ndarray:
        """
        Compute derivative of diffusion with respect to x using finite differences

        For diagonal diffusion: ∂σ_i/∂x_i for each component i
        For full matrix: ∂Σ_{ij}/∂x_k (more complex)

        Args:
            problem: SDE problem
            t: Time
            x: State (n_paths, dim)
            h: Finite difference step size

        Returns:
            Derivative array (same shape as diffusion output)
        """
        if problem.diffusion_type == DiffusionType.DIAGONAL:
            # For diagonal: compute ∂σ_i/∂x_i for each i
            # sigma: (n_paths, dim), derivative: (n_paths, dim)

            sigma = problem.diffusion(t, x)
            sigma_deriv = np.zeros_like(sigma)

            # Finite difference for each dimension
            for i in range(problem.dim):
                # Perturb x in dimension i
                x_plus = x.copy()
                x_plus[:, i] += h

                x_minus = x.copy()
                x_minus[:, i] -= h

                # Central difference
                sigma_plus = problem.diffusion(t, x_plus)
                sigma_minus = problem.diffusion(t, x_minus)

                sigma_deriv[:, i] = (sigma_plus[:, i] - sigma_minus[:, i]) / (2 * h)

            return sigma_deriv

        elif problem.diffusion_type == DiffusionType.FULL_MATRIX:
            # For full matrix: need tensor of derivatives
            # This is more complex - for now, return zeros (reduces to EM)
            # TODO: Implement full matrix derivative
            sigma = problem.diffusion(t, x)
            return np.zeros_like(sigma)

        else:
            raise ValueError(f"Unknown diffusion type: {problem.diffusion_type}")

    def step(
        self,
        problem: SDEProblem,
        t: float,
        x: np.ndarray,
        dt: float,
        dW: np.ndarray
    ) -> np.ndarray:
        """
        Single Milstein step

        Args:
            problem: SDE problem
            t: Current time
            x: Current state (n_paths, dim)
            dt: Time step
            dW: Brownian increment (n_paths, dim)

        Returns:
            Next state (n_paths, dim)
        """
        # Compute drift and diffusion
        drift = problem.drift(t, x)
        diffusion = problem.diffusion(t, x)

        # Compute diffusion derivative
        diffusion_deriv = self._diffusion_derivative(problem, t, x)

        if problem.diffusion_type == DiffusionType.DIAGONAL:
            # Diagonal case: element-wise operations
            # X_{n+1} = X_n + μ*dt + σ*dW + (1/2)*σ*σ'*(dW^2 - dt)

            # Euler-Maruyama terms
            x_next = x + drift * dt + diffusion * dW

            # Milstein correction term: (1/2) * σ * σ' * (dW^2 - dt)
            milstein_term = 0.5 * diffusion * diffusion_deriv * (dW**2 - dt)
            x_next += milstein_term

        elif problem.diffusion_type == DiffusionType.FULL_MATRIX:
            # Full matrix case (not fully implemented yet)
            # For now, fall back to Euler-Maruyama
            if diffusion.ndim == 3:
                stochastic_term = np.einsum('ijk,ik->ij', diffusion, dW)
            else:
                stochastic_term = (diffusion @ dW.T).T

            x_next = x + drift * dt + stochastic_term

        else:
            raise ValueError(f"Unknown diffusion type: {problem.diffusion_type}")

        return x_next
