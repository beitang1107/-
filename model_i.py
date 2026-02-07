"""Model I: deterministic logistics optimization."""
from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class ModelIParams:
    total_mass: float = 1e8
    n_se: int = 3
    k_se: float = 179_000.0
    n_tr: int = 10
    r_tr: float = 1.0
    m_tr: float = 125.0
    c_se: float = 750.0
    c_tr: float = 1200.0


@dataclass(frozen=True)
class ModelIResults:
    alpha_star: float
    q_se: float
    q_tr: float
    time_se_only: float
    time_tr_only: float
    time_hybrid: float
    cost_se_only: float
    cost_tr_only: float
    cost_hybrid: float


def compute_capacities(params: ModelIParams) -> Dict[str, float]:
    q_se = params.n_se * params.k_se
    q_tr = 365.0 * params.n_tr * params.r_tr * params.m_tr
    return {"q_se": q_se, "q_tr": q_tr}


def compute_alpha_star(q_se: float, q_tr: float) -> float:
    return q_se / (q_se + q_tr)


def completion_time(total_mass: float, alpha: float, q_se: float, q_tr: float) -> float:
    se_time = alpha * total_mass / q_se
    tr_time = (1.0 - alpha) * total_mass / q_tr
    return max(se_time, tr_time)


def compute_cost(total_mass: float, alpha: float, c_se: float, c_tr: float) -> float:
    return alpha * total_mass * c_se + (1.0 - alpha) * total_mass * c_tr


def run_model_i(params: ModelIParams = ModelIParams()) -> ModelIResults:
    caps = compute_capacities(params)
    q_se = caps["q_se"]
    q_tr = caps["q_tr"]
    alpha_star = compute_alpha_star(q_se, q_tr)

    time_se_only = completion_time(params.total_mass, 1.0, q_se, q_tr)
    time_tr_only = completion_time(params.total_mass, 0.0, q_se, q_tr)
    time_hybrid = completion_time(params.total_mass, alpha_star, q_se, q_tr)

    cost_se_only = compute_cost(params.total_mass, 1.0, params.c_se, params.c_tr)
    cost_tr_only = compute_cost(params.total_mass, 0.0, params.c_se, params.c_tr)
    cost_hybrid = compute_cost(params.total_mass, alpha_star, params.c_se, params.c_tr)

    return ModelIResults(
        alpha_star=alpha_star,
        q_se=q_se,
        q_tr=q_tr,
        time_se_only=time_se_only,
        time_tr_only=time_tr_only,
        time_hybrid=time_hybrid,
        cost_se_only=cost_se_only,
        cost_tr_only=cost_tr_only,
        cost_hybrid=cost_hybrid,
    )


def build_time_cost_curve(
    params: ModelIParams, alpha_grid: np.ndarray
) -> Dict[str, np.ndarray]:
    caps = compute_capacities(params)
    q_se = caps["q_se"]
    q_tr = caps["q_tr"]
    times = np.array(
        [completion_time(params.total_mass, alpha, q_se, q_tr) for alpha in alpha_grid]
    )
    costs = np.array(
        [compute_cost(params.total_mass, alpha, params.c_se, params.c_tr) for alpha in alpha_grid]
    )
    return {"alpha": alpha_grid, "time": times, "cost": costs}
