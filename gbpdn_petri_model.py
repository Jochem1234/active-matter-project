"""
Python ODE version of the Snoopy continuous Petri net for GbPdn/Pdn action.

Model idea:
- Inflammation induces Gba2.
- Gba2 catalyzes conversion of inactive GbPdn_local into active Pdn_local.
- Pdn_local suppresses neutrophil migration.
- Pdn_local can leak to the systemic compartment as Pdn_systemic.
- Pdn_systemic increases glucose production.
- Glucose is cleared back toward baseline.

This file is meant as a starting point for:
1. validation runs,
2. sensitivity analysis,
3. parameter sweeps.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


STATE_NAMES = [
    "Inflammation_signal",
    "Gba2",
    "GbPdn_local",
    "Pdn_local",
    "Pdn_systemic",
    "Neutrophils_wound",
    "Glucose",
]


@dataclass
class Parameters:
    # Inflammation and Gba2
    k_inf_decay: float = 0.12
    k_gba2_prod: float = 0.60
    k_gba2_deg: float = 0.25

    # GbPdn -> Pdn conversion
    # Use Michaelis-Menten-like conversion:
    # conversion = kcat_conv * Gba2 * GbPdn_local / (Km_GbPdn + GbPdn_local)
    kcat_conv: float = 0.80
    Km_GbPdn: float = 0.50

    # Drug degradation and leakage
    k_deg_GbPdn: float = 0.03
    k_leak: float = 0.05
    k_deg_local: float = 0.10
    k_deg_sys: float = 0.25

    # Neutrophil dynamics
    k_mig: float = 1.20
    alpha_Pdn: float = 5.00
    k_neut_clear: float = 0.35

    # Glucose dynamics
    # Glucose is modelled around a baseline of 1.0.
    # k_glucose_clear returns glucose toward baseline.
    glucose_baseline: float = 1.00
    k_glucose_prod: float = 0.80
    k_glucose_clear: float = 0.50


def default_initial_conditions(
    inflammation: float = 1.0,
    gba2: float = 0.0,
    gbpdn_local: float = 1.0,
    pdn_local: float = 0.0,
    pdn_systemic: float = 0.0,
    neutrophils_wound: float = 0.0,
    glucose: float = 1.0,
) -> np.ndarray:
    """Return initial state vector in the order defined by STATE_NAMES."""
    return np.array(
        [
            inflammation,
            gba2,
            gbpdn_local,
            pdn_local,
            pdn_systemic,
            neutrophils_wound,
            glucose,
        ],
        dtype=float,
    )


def rates(y: np.ndarray, p: Parameters) -> Dict[str, float]:
    """Calculate transition rates corresponding to the Petri-net transitions."""
    (
        Inflammation_signal,
        Gba2,
        GbPdn_local,
        Pdn_local,
        Pdn_systemic,
        Neutrophils_wound,
        Glucose,
    ) = y

    # Avoid division by zero if Km is accidentally set to zero.
    denom = max(p.Km_GbPdn + GbPdn_local, 1e-12)

    return {
        "Reduce_inflammation": p.k_inf_decay * Inflammation_signal,
        "Gba2_production": p.k_gba2_prod * Inflammation_signal,
        "Gba2_degradation": p.k_gba2_deg * Gba2,
        "Conversion": p.kcat_conv * Gba2 * GbPdn_local / denom,
        "GbPdn_degradation_local": p.k_deg_GbPdn * GbPdn_local,
        "Pdn_leakage": p.k_leak * Pdn_local,
        "Pdn_degradation_local": p.k_deg_local * Pdn_local,
        "Pdn_degradation_systemic": p.k_deg_sys * Pdn_systemic,
        "Neutrophil_migration": (
            p.k_mig * Inflammation_signal / (1.0 + p.alpha_Pdn * Pdn_local)
        ),
        "Neutrophil_removal": p.k_neut_clear * Neutrophils_wound,
        "Glucose_production": p.k_glucose_prod * Pdn_systemic,
        "Glucose_clearance": p.k_glucose_clear * (Glucose - p.glucose_baseline),
    }


def ode_system(t: float, y: np.ndarray, p: Parameters) -> List[float]:
    """ODE system derived from the continuous Petri net."""
    r = rates(y, p)

    dInflammation_signal = -r["Reduce_inflammation"]

    dGba2 = r["Gba2_production"] - r["Gba2_degradation"]

    dGbPdn_local = -r["Conversion"] - r["GbPdn_degradation_local"]

    dPdn_local = (
        r["Conversion"]
        - r["Pdn_leakage"]
        - r["Pdn_degradation_local"]
    )

    dPdn_systemic = (
        r["Pdn_leakage"]
        - r["Pdn_degradation_systemic"]
    )

    dNeutrophils_wound = (
        r["Neutrophil_migration"]
        - r["Neutrophil_removal"]
    )

    dGlucose = (
        r["Glucose_production"]
        - r["Glucose_clearance"]
    )

    return [
        dInflammation_signal,
        dGba2,
        dGbPdn_local,
        dPdn_local,
        dPdn_systemic,
        dNeutrophils_wound,
        dGlucose,
    ]


def simulate(
    p: Parameters,
    y0: np.ndarray,
    t_end: float = 48.0,
    n_points: int = 500,
) -> pd.DataFrame:
    """Run a single simulation and return a dataframe."""
    t_eval = np.linspace(0, t_end, n_points)

    sol = solve_ivp(
        fun=lambda t, y: ode_system(t, y, p),
        t_span=(0.0, t_end),
        y0=y0,
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    df = pd.DataFrame(sol.y.T, columns=STATE_NAMES)
    df.insert(0, "time", sol.t)
    return df


def calculate_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate output metrics useful for sensitivity analysis and sweeps."""
    t = df["time"].values

    metrics = {
        "peak_neutrophils": df["Neutrophils_wound"].max(),
        "auc_neutrophils": np.trapz(df["Neutrophils_wound"], t),
        "peak_glucose": df["Glucose"].max(),
        "auc_glucose_above_baseline": np.trapz(np.maximum(df["Glucose"] - 1.0, 0.0), t),
        "peak_pdn_local": df["Pdn_local"].max(),
        "peak_pdn_systemic": df["Pdn_systemic"].max(),
        "final_inflammation": df["Inflammation_signal"].iloc[-1],
    }
    return metrics


def run_validation_controls(p: Parameters) -> Dict[str, pd.DataFrame]:
    """
    Four qualitative validation cases:
    1. no drug
    2. direct local Pdn
    3. GbPdn with Gba2/inflammation-driven conversion
    4. GbPdn without conversion, mimicking Gba2 knockout/inhibition
    """
    runs = {}

    # 1. No drug control
    y0 = default_initial_conditions(gbpdn_local=0.0, pdn_local=0.0, glucose=p.glucose_baseline)
    runs["No drug"] = simulate(p, y0)

    # 2. Direct active Pdn treatment
    y0 = default_initial_conditions(gbpdn_local=0.0, pdn_local=1.0, glucose=p.glucose_baseline)
    runs["Direct Pdn"] = simulate(p, y0)

    # 3. GbPdn treatment
    y0 = default_initial_conditions(gbpdn_local=1.0, pdn_local=0.0, glucose=p.glucose_baseline)
    runs["GbPdn"] = simulate(p, y0)

    # 4. GbPdn without Gba2 conversion
    p_no_conversion = Parameters(**asdict(p))
    p_no_conversion.kcat_conv = 0.0
    y0 = default_initial_conditions(gbpdn_local=1.0, pdn_local=0.0, glucose=p.glucose_baseline)
    runs["GbPdn, no conversion"] = simulate(p_no_conversion, y0)

    return runs


def plot_validation_runs(runs: Dict[str, pd.DataFrame]) -> None:
    """Make separate plots for the main readouts."""
    for variable in ["Neutrophils_wound", "Glucose", "Pdn_local", "Pdn_systemic", "Gba2"]:
        plt.figure(figsize=(7, 4))
        for label, df in runs.items():
            plt.plot(df["time"], df[variable], label=label)
        plt.xlabel("Time")
        plt.ylabel(variable)
        plt.title(variable)
        plt.legend()
        plt.tight_layout()
        plt.show()


def local_sensitivity_analysis(
    p: Parameters,
    y0: np.ndarray,
    parameter_names: List[str],
    factor_low: float = 0.5,
    factor_high: float = 2.0,
) -> pd.DataFrame:
    """
    Simple one-at-a-time sensitivity analysis.
    Each parameter is multiplied by factor_low and factor_high.
    """
    baseline_df = simulate(p, y0)
    baseline_metrics = calculate_metrics(baseline_df)

    rows = []
    for par in parameter_names:
        for factor in [factor_low, factor_high]:
            p_new = Parameters(**asdict(p))
            setattr(p_new, par, getattr(p_new, par) * factor)

            df = simulate(p_new, y0)
            metrics = calculate_metrics(df)

            row = {
                "parameter": par,
                "factor": factor,
            }

            for metric_name, metric_value in metrics.items():
                base_value = baseline_metrics[metric_name]
                if abs(base_value) > 1e-12:
                    row[f"{metric_name}_relative_change"] = (
                        (metric_value - base_value) / base_value
                    )
                else:
                    row[f"{metric_name}_relative_change"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def sweep_two_parameters(
    p: Parameters,
    y0: np.ndarray,
    par_x: str,
    values_x: np.ndarray,
    par_y: str,
    values_y: np.ndarray,
    neutrophil_threshold_fraction: float = 0.60,
    glucose_threshold_fraction: float = 1.20,
    no_drug_metrics: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Two-dimensional parameter sweep.

    Classification:
    - therapeutic = AUC neutrophils below threshold relative to no-drug control
    - safe = peak glucose below threshold relative to no-drug control
    """
    if no_drug_metrics is None:
        p_no_drug = Parameters(**asdict(p))
        y0_no_drug = y0.copy()
        y0_no_drug[STATE_NAMES.index("GbPdn_local")] = 0.0
        y0_no_drug[STATE_NAMES.index("Pdn_local")] = 0.0
        y0_no_drug[STATE_NAMES.index("Pdn_systemic")] = 0.0
        no_drug_metrics = calculate_metrics(simulate(p_no_drug, y0_no_drug))

    neut_threshold = neutrophil_threshold_fraction * no_drug_metrics["auc_neutrophils"]
    glucose_threshold = glucose_threshold_fraction * no_drug_metrics["peak_glucose"]

    rows = []
    for x in values_x:
        for y in values_y:
            p_new = Parameters(**asdict(p))
            setattr(p_new, par_x, float(x))
            setattr(p_new, par_y, float(y))

            df = simulate(p_new, y0)
            m = calculate_metrics(df)

            therapeutic = m["auc_neutrophils"] <= neut_threshold
            safe = m["peak_glucose"] <= glucose_threshold

            if therapeutic and safe:
                classification = "good_window"
            elif therapeutic and not safe:
                classification = "side_effect_risk"
            elif not therapeutic and safe:
                classification = "ineffective"
            else:
                classification = "bad"

            rows.append(
                {
                    par_x: x,
                    par_y: y,
                    **m,
                    "therapeutic": therapeutic,
                    "safe": safe,
                    "classification": classification,
                }
            )

    return pd.DataFrame(rows)


def plot_sweep_heatmap(
    sweep_df: pd.DataFrame,
    par_x: str,
    par_y: str,
    value: str,
    title: str | None = None,
) -> None:
    """Plot a heatmap for a metric from a two-parameter sweep."""
    pivot = sweep_df.pivot(index=par_y, columns=par_x, values=value)

    plt.figure(figsize=(7, 5))
    plt.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        extent=[
            sweep_df[par_x].min(),
            sweep_df[par_x].max(),
            sweep_df[par_y].min(),
            sweep_df[par_y].max(),
        ],
    )
    plt.colorbar(label=value)
    plt.xlabel(par_x)
    plt.ylabel(par_y)
    plt.title(title or value)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Baseline parameters and initial condition
    p = Parameters()
    y0 = default_initial_conditions(
        inflammation=1.0,
        gba2=0.0,
        gbpdn_local=1.0,
        pdn_local=0.0,
        pdn_systemic=0.0,
        neutrophils_wound=0.0,
        glucose=p.glucose_baseline,
    )

    # 1. Validation runs
    runs = run_validation_controls(p)

    print("Validation metrics:")
    for label, df in runs.items():
        print(label, calculate_metrics(df))

    plot_validation_runs(runs)

    # 2. Local sensitivity analysis
    parameters_to_test = [
        "kcat_conv",
        "Km_GbPdn",
        "k_gba2_prod",
        "k_gba2_deg",
        "k_leak",
        "k_deg_local",
        "k_deg_sys",
        "alpha_Pdn",
        "k_glucose_prod",
        "k_glucose_clear",
    ]

    sens = local_sensitivity_analysis(p, y0, parameters_to_test)
    print("\nLocal sensitivity analysis:")
    print(sens)

    # 3. Example parameter sweep
    values_k_leak = np.linspace(0.005, 0.30, 30)
    values_kcat = np.linspace(0.05, 2.00, 30)

    sweep = sweep_two_parameters(
        p=p,
        y0=y0,
        par_x="k_leak",
        values_x=values_k_leak,
        par_y="kcat_conv",
        values_y=values_kcat,
    )

    plot_sweep_heatmap(
        sweep,
        par_x="k_leak",
        par_y="kcat_conv",
        value="auc_neutrophils",
        title="AUC neutrophils across leakage and conversion",
    )

    plot_sweep_heatmap(
        sweep,
        par_x="k_leak",
        par_y="kcat_conv",
        value="peak_glucose",
        title="Peak glucose across leakage and conversion",
    )
