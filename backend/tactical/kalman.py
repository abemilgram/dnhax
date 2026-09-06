"""Float64 constant-velocity Kalman primitives."""

import math

import numpy as np

from .schema import Observation


MEASUREMENT = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64
)


def sanitize_covariance(
    covariance: np.ndarray,
    minimum_eigenvalue: float = 1e-9,
    maximum_eigenvalue: float = 1e9,
) -> np.ndarray:
    """Make covariance finite, symmetric, and positive definite."""

    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    if not np.isfinite(covariance).all():
        raise ValueError("covariance must be finite")
    symmetric = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.clip(eigenvalues, minimum_eigenvalue, maximum_eigenvalue)
    result = (eigenvectors * eigenvalues) @ eigenvectors.T
    return (result + result.T) * 0.5


def transition(dt: float) -> np.ndarray:
    dt = float(dt)
    if not math.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and non-negative")
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def acceleration_noise(dt: float, acceleration_variance: float) -> np.ndarray:
    """Piecewise-constant acceleration process noise for x and z."""

    if not math.isfinite(acceleration_variance) or acceleration_variance < 0.0:
        raise ValueError("acceleration_variance must be finite and non-negative")
    transition(dt)
    dt2, dt3, dt4 = dt * dt, dt**3, dt**4
    return acceleration_variance * np.array(
        [
            [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
            [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
            [dt3 / 2.0, 0.0, dt2, 0.0],
            [0.0, dt3 / 2.0, 0.0, dt2],
        ],
        dtype=np.float64,
    )


def observation_noise(
    observation: Observation,
    base_variance: float,
    confidence_floor: float = 0.05,
) -> np.ndarray:
    """Build xz measurement noise, weakening low-confidence observations."""

    if not math.isfinite(base_variance) or base_variance <= 0.0:
        raise ValueError("base_variance must be finite and positive")
    if not 0.0 < confidence_floor <= 1.0:
        raise ValueError("confidence_floor must be in (0, 1]")
    confidence = max(observation.conf, confidence_floor)
    noise = np.eye(2, dtype=np.float64) * (base_variance / confidence)
    if observation.covariance is not None:
        xyz = np.asarray(observation.covariance, dtype=np.float64)
        noise += xyz[np.ix_((0, 2), (0, 2))]
    return sanitize_covariance(noise)


def predict(
    state: np.ndarray,
    covariance: np.ndarray,
    dt: float,
    acceleration_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (4,) or not np.isfinite(state).all():
        raise ValueError("state must be a finite [x,z,vx,vz] vector")
    covariance = sanitize_covariance(covariance)
    matrix = transition(dt)
    predicted_state = matrix @ state
    predicted_covariance = (
        matrix @ covariance @ matrix.T
        + acceleration_noise(dt, acceleration_variance)
    )
    return predicted_state, sanitize_covariance(predicted_covariance)


def innovation(
    state: np.ndarray,
    covariance: np.ndarray,
    observation: Observation,
    base_variance: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    measurement = np.array(
        [observation.xyz[0], observation.xyz[2]], dtype=np.float64
    )
    residual = measurement - MEASUREMENT @ state
    residual_covariance = sanitize_covariance(
        MEASUREMENT @ covariance @ MEASUREMENT.T
        + observation_noise(observation, base_variance)
    )
    solved = np.linalg.solve(residual_covariance, residual)
    mahalanobis_squared = max(0.0, float(residual @ solved))
    return residual, residual_covariance, mahalanobis_squared


def update(
    state: np.ndarray,
    covariance: np.ndarray,
    observation: Observation,
    base_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Update with a position observation using Joseph covariance form."""

    covariance = sanitize_covariance(covariance)
    residual, residual_covariance, _ = innovation(
        state, covariance, observation, base_variance
    )
    gain = np.linalg.solve(
        residual_covariance, MEASUREMENT @ covariance
    ).T
    updated_state = np.asarray(state, dtype=np.float64) + gain @ residual
    identity = np.eye(4, dtype=np.float64)
    remainder = identity - gain @ MEASUREMENT
    noise = observation_noise(observation, base_variance)
    updated_covariance = (
        remainder @ covariance @ remainder.T + gain @ noise @ gain.T
    )
    return updated_state, sanitize_covariance(updated_covariance)
