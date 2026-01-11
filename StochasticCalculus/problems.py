"""SDE Problem Definitions"""

from pydantic import BaseModel, Field, field_validator
import numpy as np
from typing import Callable, Optional, Tuple
from enum import Enum


class DiffusionType(str, Enum):
    """Type of diffusion coefficient"""
    DIAGONAL = "diagonal"  # Vector σ(t,x), independent noise per dimension
    FULL_MATRIX = "full_matrix"  # Matrix Σ(t,x), correlated noise


class SDEProblem(BaseModel):
    """
    Stochastic Differential Equation Problem

    Defines SDE of the form:
        dX = drift(t, X)dt + diffusion(t, X)dW

    where:
    - drift: R x R^d -> R^d (drift coefficient μ)
    - diffusion: R x R^d -> R^d (diagonal) or R^(d×d) (full matrix)
    - dW: d-dimensional Brownian motion
    """

    name: str
    dim: int = Field(gt=0, description="Problem dimension")
    x0: np.ndarray = Field(description="Initial condition")
    t_span: Tuple[float, float] = Field(description="Time span (t0, T)")
    diffusion_type: DiffusionType = Field(default=DiffusionType.DIAGONAL)

    # Callables - not validated by Pydantic
    drift_fn: Callable[[float, np.ndarray], np.ndarray]
    diffusion_fn: Callable[[float, np.ndarray], np.ndarray]
    analytical_solution: Optional[Callable[[float, np.ndarray], np.ndarray]] = None

    # Optional theoretical moments for weak convergence testing
    theoretical_mean: Optional[Callable[[float, np.ndarray], np.ndarray]] = None
    theoretical_variance: Optional[Callable[[float, np.ndarray], np.ndarray]] = None

    class Config:
        arbitrary_types_allowed = True

    @field_validator('x0')
    @classmethod
    def validate_x0(cls, v: np.ndarray, info) -> np.ndarray:
        """Validate initial condition matches dimension"""
        if not isinstance(v, np.ndarray):
            v = np.array(v)

        # Check dimension consistency if dim is already set
        if 'dim' in info.data and v.shape[0] != info.data['dim']:
            raise ValueError(
                f"x0 must have length {info.data['dim']}, got {v.shape[0]}"
            )

        return v

    @field_validator('t_span')
    @classmethod
    def validate_t_span(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        """Validate time span is positive"""
        t0, T = v
        if T <= t0:
            raise ValueError(f"t_span must satisfy T > t0, got t0={t0}, T={T}")
        return v

    def drift(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Drift coefficient μ(t, X)

        Args:
            t: Time
            x: State (shape: (n_paths, dim) or (dim,))

        Returns:
            Drift vector (same shape as x)
        """
        return self.drift_fn(t, x)

    def diffusion(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Diffusion coefficient σ(t, X)

        Args:
            t: Time
            x: State (shape: (n_paths, dim) or (dim,))

        Returns:
            - DIAGONAL: Vector of shape (n_paths, dim) or (dim,)
            - FULL_MATRIX: Matrix of shape (n_paths, dim, dim) or (dim, dim)
        """
        return self.diffusion_fn(t, x)

    def exact_solution(self, t: float, x0: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Analytical solution at time t (if available)

        Args:
            t: Time
            x0: Initial condition (uses self.x0 if None)

        Returns:
            Exact solution X(t)

        Raises:
            NotImplementedError: If no analytical solution available
        """
        if self.analytical_solution is None:
            raise NotImplementedError(
                f"No analytical solution available for {self.name}"
            )

        if x0 is None:
            x0 = self.x0

        return self.analytical_solution(t, x0)
