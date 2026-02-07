"""Run models I-IV, generate plots, and write a comparison log."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from model_i import ModelIParams, build_time_cost_curve, run_model_i
from model_ii import ModelIIParams, build_cdf, run_model_ii
from model_iii import ModelIIIParams, compute_water_demand, recycle_sensitivity
from model_iv import ModelIVParams, compute_co2_curve, grid_intensity_sensitivity


OUTPUT_DIR = Path("output")
FIG_DIR = OUTPUT_DIR / "figs"
LOG_PATH = OUTPUT_DIR / "summary_log.txt"


def _setup_output() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_time_cost_curve(curve: dict, alpha_star: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve["alpha"], curve["time"], label="Completion time")
    ax2 = ax.twinx()
    ax2.plot(curve["alpha"], curve["cost"] / 1e12, color="tab:orange", label="Cost (T$)")
    ax.axvline(alpha_star, color="tab:green", linestyle="--", label=f"alpha*={alpha_star:.3f}")
    ax.set_xlabel("Hybrid share alpha")
    ax.set_ylabel("Time (years)")
    ax2.set_ylabel("Cost (T$)")
    ax.set_title("Fig. 3 Time-Cost vs Hybrid Share (Caption: SE blue, TR orange)")
    fig.legend(loc="upper center", ncol=3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_time_cost.png", dpi=200)
    plt.close(fig)


def plot_cumulative_mass(params: ModelIParams, result) -> None:
    q_se = result.q_se
    q_tr = result.q_tr
    total = params.total_mass
    years = np.linspace(0, max(result.time_tr_only, result.time_se_only), 200)
    se_mass = np.minimum(years * q_se, total)
    tr_mass = np.minimum(years * q_tr, total)
    hybrid_mass = np.minimum(years * (result.alpha_star * q_se + (1 - result.alpha_star) * q_tr), total)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(years, se_mass / 1e6, label="SE-only", color="tab:blue")
    ax.plot(years, tr_mass / 1e6, label="TR-only", color="tab:orange")
    ax.plot(years, hybrid_mass / 1e6, label="Hybrid", color="tab:green")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Delivered mass (Mt)")
    ax.set_title("Fig. 3 Construction Mass Delivered vs Time (Caption: SE/ TR/ Hybrid)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_mass_time.png", dpi=200)
    plt.close(fig)


def plot_stochastic_results(model_ii_results) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(model_ii_results.alpha_grid, model_ii_results.cvar_values, color="tab:red")
    ax.axvline(model_ii_results.alpha_rob, color="tab:green", linestyle="--")
    ax.set_xlabel("Hybrid share alpha")
    ax.set_ylabel("CVaR (years)")
    ax.set_title("Fig. 6 CVaR vs Hybrid Share (Caption: robust alpha)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_cvar_alpha.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(model_ii_results.time_samples_star, bins=30, alpha=0.6, label="alpha*")
    ax.hist(model_ii_results.time_samples_rob, bins=30, alpha=0.6, label="alpha_rob")
    ax.set_xlabel("Completion time (years)")
    ax.set_ylabel("Frequency")
    ax.set_title("Fig. 7 Monte Carlo Completion Time Histogram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_time_hist.png", dpi=200)
    plt.close(fig)

    cdf_star = build_cdf(model_ii_results.time_samples_star)
    cdf_rob = build_cdf(model_ii_results.time_samples_rob)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cdf_star["x"], cdf_star["cdf"], label="alpha*")
    ax.plot(cdf_rob["x"], cdf_rob["cdf"], label="alpha_rob")
    ax.set_xlabel("Completion time (years)")
    ax.set_ylabel("CDF")
    ax.set_title("Fig. 8 Completion Time CDF (Caption: alpha* vs alpha_rob)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_time_cdf.png", dpi=200)
    plt.close(fig)


def plot_water(results, sensitivity) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sensitivity["recycle_rate"], np.array(sensitivity["net_supply"]) / 1e6)
    ax.set_xlabel("Recycle rate")
    ax.set_ylabel("Net annual water (Mt)")
    ax.set_title("Fig. 9 Net Water vs Recycle Rate (Caption: 10^5 population)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_water_recycle.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Annual", "Net"], [results.annual_demand_tonnes / 1e6, results.net_supply_tonnes / 1e6])
    ax.set_ylabel("Water (Mt)")
    ax.set_title("Fig. 10 Annual vs Net Water Demand (Caption: recycle applied)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_water_summary.png", dpi=200)
    plt.close(fig)


def plot_co2(curve, sensitivity) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve.alpha, curve.co2_mt)
    ax.set_xlabel("Hybrid share alpha")
    ax.set_ylabel("CO2 (Mt)")
    ax.set_title("Fig. 11 CO2 vs Hybrid Share (Caption: SE vs TR intensity)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_co2_alpha.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sensitivity["factor"], sensitivity["co2_mt"], marker="o")
    ax.set_xlabel("Grid intensity factor")
    ax.set_ylabel("CO2 (Mt)")
    ax.set_title("Fig. 12 Grid Intensity Sensitivity (Caption: alpha_rob)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_co2_sensitivity.png", dpi=200)
    plt.close(fig)


def plot_tradeoff_surface(time_curve, co2_curve) -> None:
    alpha = time_curve["alpha"]
    time = time_curve["time"]
    cost = time_curve["cost"] / 1e12
    co2 = co2_curve.co2_mt
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(alpha, time, cost, label="Time-Cost")
    ax.plot(alpha, time, co2, label="Time-CO2")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Time (years)")
    ax.set_zlabel("Cost (T$) / CO2 (Mt)")
    ax.set_title("Fig. 13 Multi-Objective Tradeoff Surface")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_tradeoff_surface.png", dpi=200)
    plt.close(fig)


def write_log(model_i, model_ii, model_iii, model_iv) -> None:
    lines = [
        "Model I Results:",
        f"  Q_SE={model_i.q_se:.0f} t/yr, Q_TR={model_i.q_tr:.0f} t/yr",
        f"  alpha*={model_i.alpha_star:.3f}",
        f"  Time SE-only={model_i.time_se_only:.1f} yr, TR-only={model_i.time_tr_only:.1f} yr, Hybrid={model_i.time_hybrid:.1f} yr",
        f"  Cost SE-only={model_i.cost_se_only/1e12:.2f} T$, TR-only={model_i.cost_tr_only/1e12:.2f} T$, Hybrid={model_i.cost_hybrid/1e12:.2f} T$",
        "",
        "Model II Results:",
        f"  alpha_rob={model_ii.alpha_rob:.3f} (CVaR optimized)",
        f"  CVaR min={model_ii.cvar_values.min():.2f} yr",
        "",
        "Model III Results:",
        f"  Daily demand={model_iii.daily_demand_tonnes:.0f} t/day",
        f"  Annual demand={model_iii.annual_demand_tonnes:.0f} t/yr",
        f"  Net supply={model_iii.net_supply_tonnes:.0f} t/yr",
        f"  Transport={model_iii.transport_days} days, cost={model_iii.transport_cost_trillion:.3f} T$",
        "",
        "Model IV Results:",
        f"  CO2 SE-only={model_iv.co2_mt[-1]:.1f} Mt, TR-only={model_iv.co2_mt[0]:.1f} Mt",
    ]
    LOG_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _setup_output()

    model_i_params = ModelIParams()
    model_i_results = run_model_i(model_i_params)

    curve = build_time_cost_curve(model_i_params, np.linspace(0, 1, 101))
    plot_time_cost_curve(curve, model_i_results.alpha_star)
    plot_cumulative_mass(model_i_params, model_i_results)

    model_ii_results = run_model_ii(model_i_params, model_i_results.alpha_star, ModelIIParams())
    plot_stochastic_results(model_ii_results)

    model_iii_results = compute_water_demand(ModelIIIParams())
    recycle_grid = np.linspace(0.8, 0.99, 20)
    water_sensitivity = recycle_sensitivity(model_iii_results.annual_demand_tonnes, recycle_grid.tolist())
    plot_water(model_iii_results, water_sensitivity)

    model_iv_params = ModelIVParams()
    alpha_grid = np.linspace(0, 1, 101)
    co2_curve = compute_co2_curve(model_iv_params, alpha_grid)
    sensitivity = grid_intensity_sensitivity(model_iv_params, model_ii_results.alpha_rob, np.linspace(0.8, 1.2, 9))
    plot_co2(co2_curve, sensitivity)
    plot_tradeoff_surface(curve, co2_curve)

    write_log(model_i_results, model_ii_results, model_iii_results, co2_curve)


if __name__ == "__main__":
    main()
