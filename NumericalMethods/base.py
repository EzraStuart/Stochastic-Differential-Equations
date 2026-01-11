"""Base Classes for Numerical SDE Solvers"""

from pydantic import BaseModel, Field
import numpy as np
import xarray as xr
from abc import ABC, abstractmethod
from typing import Optional

from StochasticCalculus.problems import SDEProblem


class SchemeConfig(BaseModel):
    """Configuration for numerical scheme execution"""

    dt: float = Field(gt=0, description="Time step size")
    n_paths: int = Field(gt=0, description="Number of Monte Carlo paths")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    enforce_positivity: bool = Field(
        default=False,
        description="Enforce non-negativity via absorption (max with 0)"
    )


class NumericalScheme(ABC, BaseModel):
    """
    Abstract base class for SDE numerical schemes

    Implements the solve() method that handles:
    - Time discretization
    - Brownian motion generation
    - Path storage in xarray format

    Subclasses only need to implement step() for single timestep update.
    """

    name: str = Field(description="Scheme name")
    order_strong: float = Field(description="Strong convergence order")
    order_weak: float = Field(description="Weak convergence order")

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def step(
        self,
        problem: SDEProblem,
        t: float,
        x: np.ndarray,
        dt: float,
        dW: np.ndarray
    ) -> np.ndarray:
        """
        Single timestep update: X_{n+1} = f(t, X_n, dW)

        Args:
            problem: SDE problem definition
            t: Current time
            x: Current state (shape: (n_paths, dim))
            dt: Time step
            dW: Brownian increment (shape: (n_paths, dim))

        Returns:
            Next state X_{n+1} (shape: (n_paths, dim))
        """
        pass

    def solve(
        self,
        problem: SDEProblem,
        config: SchemeConfig
    ) -> xr.DataArray:
        """
        Solve SDE over full time span

        Args:
            problem: SDE problem to solve
            config: Scheme configuration

        Returns:
            xarray.DataArray with dimensions:
            - 'path': Monte Carlo sample index (0 to n_paths-1)
            - 'time': Discretization times
            - 'variable': SDE components (X0, X1, ...)

            Attributes:
            - scheme: Name of numerical scheme
            - problem: Problem name
            - dt: Time step size
        """
        t0, T = problem.t_span
        n_steps = int((T - t0) / config.dt)
        times = np.linspace(t0, T, n_steps + 1)

        # Initialize paths: (n_paths, n_steps+1, dim)
        paths = np.zeros((config.n_paths, n_steps + 1, problem.dim))
        paths[:, 0, :] = problem.x0

        # Generate Brownian increments
        rng = np.random.default_rng(config.seed)
        dW = rng.normal(
            loc=0.0,
            scale=np.sqrt(config.dt),
            size=(config.n_paths, n_steps, problem.dim)
        )

        # Time integration loop
        for i in range(n_steps):
            paths[:, i+1, :] = self.step(
                problem=problem,
                t=times[i],
                x=paths[:, i, :],
                dt=config.dt,
                dW=dW[:, i, :]
            )

            # Enforce positivity if requested
            if config.enforce_positivity:
                paths[:, i+1, :] = np.maximum(paths[:, i+1, :], 0.0)

        # Package as xarray with labeled dimensions
        return xr.DataArray(
            data=paths,
            dims=['path', 'time', 'variable'],
            coords={
                'path': np.arange(config.n_paths),
                'time': times,
                'variable': [f'X{j}' for j in range(problem.dim)]
            },
            attrs={
                'scheme': self.name,
                'problem': problem.name,
                'dt': config.dt,
                'order_strong': self.order_strong,
                'order_weak': self.order_weak
            }
        )
