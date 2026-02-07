"""Model II: stochastic reliability and CVaR optimization."""
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from model_i import ModelIParams, completion_time, compute_capacities


@dataclass(frozen=True)
class ModelIIParams:
    samples: int = 3000
    seed: int = 42
    se_sigma: float = 0.08
    tr_sigma: float = 0.12
    cvar_tail: float = 0.05


@dataclass(frozen=True)
class ModelIIResults:
    alpha_star: float
    alpha_rob: float
    alpha_grid: np.ndarray
    cvar_values: np.ndarray
    time_samples_star: np.ndarray
    time_samples_rob: np.ndarray


def _sample_capacities(
    rng: np.random.Generator,
    q_se: float,
    q_tr: float,
    params: ModelIIParams,
) -> Tuple[np.ndarray, np.ndarray]:
    se_multiplier = rng.normal(loc=1.0, scale=params.se_sigma, size=params.samples)
    tr_multiplier = rng.normal(loc=1.0, scale=params.tr_sigma, size=params.samples)
    se_multiplier = np.clip(se_multiplier, 0.6, None)
    tr_multiplier = np.clip(tr_multiplier, 0.6, None)
    return q_se * se_multiplier, q_tr * tr_multiplier


def _completion_samples(
    total_mass: float,
    alpha: float,
    q_se_samples: np.ndarray,
    q_tr_samples: np.ndarray,
) -> np.ndarray:
    se_time = alpha * total_mass / q_se_samples
    tr_time = (1.0 - alpha) * total_mass / q_tr_samples
    return np.maximum(se_time, tr_time)


def _cvar(values: np.ndarray, tail: float) -> float:
    threshold = np.quantile(values, 1.0 - tail)
    tail_values = values[values >= threshold]
    return float(tail_values.mean())


def run_model_ii(
    model_i_params: ModelIParams,
    alpha_star: float,
    params: ModelIIParams = ModelIIParams(),
) -> ModelIIResults:
    rng = np.random.default_rng(params.seed)
    caps = compute_capacities(model_i_params)
    q_se = caps["q_se"]
    q_tr = caps["q_tr"]
    q_se_samples, q_tr_samples = _sample_capacities(rng, q_se, q_tr, params)

    alpha_grid = np.linspace(0.0, 1.0, 101)
    cvar_values = []
    for alpha in alpha_grid:
        times = _completion_samples(model_i_params.total_mass, alpha, q_se_samples, q_tr_samples)
        cvar_values.append(_cvar(times, params.cvar_tail))
    cvar_values = np.array(cvar_values)
    alpha_rob = float(alpha_grid[np.argmin(cvar_values)])

    time_samples_star = _completion_samples(
        model_i_params.total_mass, alpha_star, q_se_samples, q_tr_samples
    )
    time_samples_rob = _completion_samples(
        model_i_params.total_mass, alpha_rob, q_se_samples, q_tr_samples
    )

    return ModelIIResults(
        alpha_star=alpha_star,
        alpha_rob=alpha_rob,
        alpha_grid=alpha_grid,
        cvar_values=cvar_values,
        time_samples_star=time_samples_star,
        time_samples_rob=time_samples_rob,
    )


def build_cdf(samples: np.ndarray) -> Dict[str, np.ndarray]:
    sorted_samples = np.sort(samples)
    cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
    return {"x": sorted_samples, "cdf": cdf}
