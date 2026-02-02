"""Model III: water demand and logistics coupling."""
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ModelIIIParams:
    population: int = 100_000
    per_capita_liters: float = 50.0
    reserve_days: int = 30
    reserve_tonnes: float = 50_000.0
    recycle_rate: float = 0.94
    transport_days: int = 96
    transport_cost_trillion: float = 0.248


@dataclass(frozen=True)
class ModelIIIResults:
    daily_demand_tonnes: float
    annual_demand_tonnes: float
    net_supply_tonnes: float
    transport_days: int
    transport_cost_trillion: float


def compute_water_demand(params: ModelIIIParams) -> ModelIIIResults:
    daily_demand_tonnes = params.population * params.per_capita_liters / 1000.0
    annual_demand = daily_demand_tonnes * 365.0 + params.reserve_tonnes
    net_supply = (1.0 - params.recycle_rate) * annual_demand
    return ModelIIIResults(
        daily_demand_tonnes=daily_demand_tonnes,
        annual_demand_tonnes=annual_demand,
        net_supply_tonnes=net_supply,
        transport_days=params.transport_days,
        transport_cost_trillion=params.transport_cost_trillion,
    )


def recycle_sensitivity(
    annual_demand: float, recycle_grid: list[float]
) -> Dict[str, list[float]]:
    net_supply = [(1.0 - rate) * annual_demand for rate in recycle_grid]
    return {"recycle_rate": recycle_grid, "net_supply": net_supply}
