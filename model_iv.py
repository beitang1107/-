"""Model IV: environmental impact and multi-objective trade-offs."""
from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class ModelIVParams:
    co2_se_only_mt: float = 834.0
    co2_tr_only_mt: float = 1037.0
    grid_intensity_g_per_kwh: float = 445.0


@dataclass(frozen=True)
class ModelIVResults:
    alpha: np.ndarray
    co2_mt: np.ndarray


def compute_co2_curve(params: ModelIVParams, alpha_grid: np.ndarray) -> ModelIVResults:
    co2_mt = alpha_grid * params.co2_se_only_mt + (1.0 - alpha_grid) * params.co2_tr_only_mt
    return ModelIVResults(alpha=alpha_grid, co2_mt=co2_mt)


def grid_intensity_sensitivity(
    params: ModelIVParams, alpha: float, intensity_factors: np.ndarray
) -> Dict[str, np.ndarray]:
    co2_base = alpha * params.co2_se_only_mt + (1.0 - alpha) * params.co2_tr_only_mt
    adjusted = co2_base * intensity_factors
    return {"factor": intensity_factors, "co2_mt": adjusted}
