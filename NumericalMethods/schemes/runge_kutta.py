"""Runge-Kutta Schemes for SDEs"""

import numpy as np

from NumericalMethods.base import NumericalScheme
from StochasticCalculus.problems import SDEProblem, DiffusionType


class RungeKutta2(NumericalScheme):
    """
    Stochastic Runge-Kutta Order 2 (SRK2)

    Also known as Heun's method for SDEs. Derivative-free method that
    achieves order 1.0 strong convergence without computing derivatives.

    Algorithm:
        1. Predictor step:
           X_bar = X_n + μ(t_n, X_n)Δt + σ(t_n, X_n)ΔW_n

        2. Corrector step:
           X_{n+1} = X_n + (1/2)[μ(t_n, X_n) + μ(t_{n+1}, X_bar)]Δt
                     + (1/2)[σ(t_n, X_n) + σ(t_{n+1}, X_bar)]ΔW_n

    Convergence:
    - Strong order: 1.0 (error ~ O(Δt))
    - Weak order: 1.0 (error ~ O(Δt))

    Advantages:
    - No derivatives needed (vs Milstein)
    - Better stability than Euler-Maruyama
    - Same strong order as Milstein without derivative computation

    Disadvantages:
    - Requires 2 drift evaluations per step
    - Requires 2 diffusion evaluations per step
    - ~2x cost of Euler-Maruyama
    """

    name: str = "Runge-Kutta 2 (SRK2)"
    order_strong: float = 1.0
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
        Single SRK2 step

        Args:
            problem: SDE problem
            t: Current time
            x: Current state (n_paths, dim)
            dt: Time step
            dW: Brownian increment (n_paths, dim)

        Returns:
            Next state (n_paths, dim)
        """
        # Evaluate at current point
        drift_n = problem.drift(t, x)
        diffusion_n = problem.diffusion(t, x)

        # Predictor step (Euler step)
        if problem.diffusion_type == DiffusionType.DIAGONAL:
            x_bar = x + drift_n * dt + diffusion_n * dW
        elif problem.diffusion_type == DiffusionType.FULL_MATRIX:
            if diffusion_n.ndim == 3:
                stochastic_term = np.einsum('ijk,ik->ij', diffusion_n, dW)
            else:
                stochastic_term = (diffusion_n @ dW.T).T
            x_bar = x + drift_n * dt + stochastic_term
        else:
            raise ValueError(f"Unknown diffusion type: {problem.diffusion_type}")

        # Evaluate at predicted point
        drift_bar = problem.drift(t + dt, x_bar)
        diffusion_bar = problem.diffusion(t + dt, x_bar)

        # Corrector step (average of values at t_n and t_{n+1})
        drift_avg = 0.5 * (drift_n + drift_bar)
        diffusion_avg = 0.5 * (diffusion_n + diffusion_bar)

        if problem.diffusion_type == DiffusionType.DIAGONAL:
            x_next = x + drift_avg * dt + diffusion_avg * dW
        elif problem.diffusion_type == DiffusionType.FULL_MATRIX:
            if diffusion_avg.ndim == 3:
                stochastic_term = np.einsum('ijk,ik->ij', diffusion_avg, dW)
            else:
                stochastic_term = (diffusion_avg @ dW.T).T
            x_next = x + drift_avg * dt + stochastic_term
        else:
            raise ValueError(f"Unknown diffusion type: {problem.diffusion_type}")

        return x_next


class RungeKutta4(NumericalScheme):
    """
    Stochastic Runge-Kutta Order 4 (SRK4)

    Adaptation of classical RK4 to SDEs. Uses 4 stage evaluations
    with specific combinations for stochastic terms.

    Note: For SDEs, achieving order > 1.5 strong is difficult without
    multiple Brownian increments. This implementation provides improved
    accuracy over RK2 but may not achieve full order 2.0 strong.

    Algorithm (simplified):
        k1 = μ(t_n, X_n)
        l1 = σ(t_n, X_n)

        k2 = μ(t_n + Δt/2, X_n + k1*Δt/2 + l1*ΔW/2)
        l2 = σ(t_n + Δt/2, X_n + k1*Δt/2 + l1*ΔW/2)

        k3 = μ(t_n + Δt/2, X_n + k2*Δt/2 + l2*ΔW/2)
        l3 = σ(t_n + Δt/2, X_n + k2*Δt/2 + l2*ΔW/2)

        k4 = μ(t_n + Δt, X_n + k3*Δt + l3*ΔW)
        l4 = σ(t_n + Δt, X_n + k3*Δt + l3*ΔW)

        X_{n+1} = X_n + (k1 + 2*k2 + 2*k3 + k4)*Δt/6
                      + (l1 + 2*l2 + 2*l3 + l4)*ΔW/6

    Convergence:
    - Strong order: ~1.0-1.5 (depends on problem structure)
    - Weak order: ~1.5-2.0

    Advantages:
    - High accuracy for smooth problems
    - Good stability properties
    - Derivative-free

    Disadvantages:
    - 4 drift/diffusion evaluations per step
    - ~4x cost of Euler-Maruyama
    - Doesn't achieve full order 2.0 strong without multiple dW terms
    """

    name: str = "Runge-Kutta 4 (SRK4)"
    order_strong: float = 1.0  # Conservative estimate
    order_weak: float = 1.5

    def step(
        self,
        problem: SDEProblem,
        t: float,
        x: np.ndarray,
        dt: float,
        dW: np.ndarray
    ) -> np.ndarray:
        """
        Single SRK4 step

        Args:
            problem: SDE problem
            t: Current time
            x: Current state (n_paths, dim)
            dt: Time step
            dW: Brownian increment (n_paths, dim)

        Returns:
            Next state (n_paths, dim)
        """
        # Helper to compute stochastic term
        def compute_stochastic(diffusion, dW_term):
            if problem.diffusion_type == DiffusionType.DIAGONAL:
                return diffusion * dW_term
            elif problem.diffusion_type == DiffusionType.FULL_MATRIX:
                if diffusion.ndim == 3:
                    return np.einsum('ijk,ik->ij', diffusion, dW_term)
                else:
                    return (diffusion @ dW_term.T).T
            else:
                raise ValueError(f"Unknown diffusion type: {problem.diffusion_type}")

        # Stage 1: Evaluate at current point
        k1 = problem.drift(t, x)
        l1 = problem.diffusion(t, x)
        stoch1 = compute_stochastic(l1, dW)

        # Stage 2: Evaluate at midpoint using k1, l1
        x2 = x + k1 * (dt / 2) + stoch1 / 2
        k2 = problem.drift(t + dt/2, x2)
        l2 = problem.diffusion(t + dt/2, x2)
        stoch2 = compute_stochastic(l2, dW)

        # Stage 3: Evaluate at midpoint using k2, l2
        x3 = x + k2 * (dt / 2) + stoch2 / 2
        k3 = problem.drift(t + dt/2, x3)
        l3 = problem.diffusion(t + dt/2, x3)
        stoch3 = compute_stochastic(l3, dW)

        # Stage 4: Evaluate at endpoint using k3, l3
        x4 = x + k3 * dt + stoch3
        k4 = problem.drift(t + dt, x4)
        l4 = problem.diffusion(t + dt, x4)
        stoch4 = compute_stochastic(l4, dW)

        # Combine stages with RK4 weights
        x_next = x + (k1 + 2*k2 + 2*k3 + k4) * (dt / 6) \
                   + (stoch1 + 2*stoch2 + 2*stoch3 + stoch4) / 6

        return x_next
