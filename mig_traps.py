from dataclasses import dataclass
from pagn import Sirko as S
from pagn import Thompson as T
import pagn.constants as ct
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from pagn.opacities import electron_scattering_opacity
import pandas as pd

sirko = S.SirkoAGN()
thomp = T.ThompsonAGN()


@dataclass(frozen=True)
class MigrationParameters:
    """Numerical coefficients used by the torque model."""

    alpha: float = 0.01
    gamma_adiabat: float = 5.0 / 3.0
    hydrogen_fraction: float = 0.7
    kappa_t: float = 20.0
    kappa_norm: float = 25.0


def safe_divide(numerator, denominator):
    """Return ``numerator / denominator`` while avoiding warnings at zero."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.zeros_like(np.broadcast_arrays(numerator,
                                            denominator)[0],
                        dtype=float)
    return np.divide(numerator, denominator,
                     out=out, where=denominator != 0)


def radial_log_slope(radius, values, *, smoothing=0.005, k=3):
    """Estimate ``d log(values) / d log(radius)`` from a smooth spline."""
    radius = np.asarray(radius, dtype=float)
    values = np.asarray(values, dtype=float)

    radius_log = np.log10(radius)
    values_log = np.log10(values)

    spline = UnivariateSpline(radius_log, values_log, k=k, s=smoothing, ext=0)
    return spline.derivative()(radius_log)


def total_pressure(disk):
    """Return the gas + radiation pressure profile in SI units."""
    gas_pressure = disk.rho * disk.T * ct.Kb / ct.massU
    radiation_pressure = disk.tauV * ct.sigmaSB * disk.Teff4 / (2 * ct.c)
    return gas_pressure + radiation_pressure


def surface_density(disk):
    """Return the gas surface density profile in kg m^-2."""
    return 2.0 * disk.rho * disk.h


def comp_corot(dsigma_dlogr, dtemp_dlogr, gamma_adiabat):
    """Corotation torque coefficient."""
    return (0.46 + 0.96 * dsigma_dlogr - 1.8 * dtemp_dlogr) / gamma_adiabat


def thermal_diffusivity(disk, gamma_adiabat):
    """Thermal diffusivity used by the Lindblad and thermal torque terms."""
    return (
        16.0
        * gamma_adiabat
        * (gamma_adiabat - 1.0)
        * ct.sigmaSB
        * disk.T**4
        / (3.0 * disk.kappa * disk.rho**2 * disk.h**2 * disk.Omega**2)
    )


def comp_lind(dsigma_dlogr, dtemp_dlogr, disk, gamma_adiabat):
    """Lindblad torque coefficient with thermal-diffusion correction."""
    chi = thermal_diffusivity(disk, gamma_adiabat)
    chi_c = disk.h**2 * disk.Omega
    x_sq = np.sqrt(safe_divide(chi, 2.0 * chi_c))
    f_gamma = (x_sq + 1.0 / gamma_adiabat) / (x_sq + 1.0)

    return (-2.34 - 0.1 * dsigma_dlogr + 1.5 * dtemp_dlogr) * f_gamma


def comp_kappa(q, h_over_r, alpha):
    """Gap-opening parameter used to attenuate the Type-I torques."""
    return q**2 / (h_over_r**5 * alpha)


def comp_gamma_0(q, radius, scale_height, omega, rho):
    """Torque normalization Γ0."""
    sigma = 2.0 * rho * scale_height
    h_over_r = scale_height / radius
    return q**2 * radius**4 * omega**2 * sigma / (h_over_r**2)


def comp_gamma_th(dpressure_dlogr, disk, migrator_mass, params):
    """Thermal torque coefficient from the current backend implementation."""
    gamma = params.gamma_adiabat

    chi = thermal_diffusivity(disk, gamma)
    x_c = np.abs(dpressure_dlogr) * disk.cs**2 / (3.0 * gamma * disk.R * disk.Omega**2)

    k_es = electron_scattering_opacity(X=params.hydrogen_fraction)
    l_edd = 4.0 * np.pi * ct.G * ct.c * migrator_mass / k_es
    l_crit = 4.0 * np.pi * ct.G * migrator_mass * disk.rho * chi / gamma
    lambda_c = np.sqrt(2.0 * chi / (3.0 * gamma * disk.Omega))

    bondi_radius = ct.G * migrator_mass / disk.cs**2
    mu_th = chi / (disk.cs * np.minimum(disk.h, bondi_radius))

    unsaturated = (
        safe_divide(l_edd, l_crit) * (4.0 * mu_th / (1.0 + 4.0 * mu_th))
        - (2.0 * mu_th / (1.0 + 2.0 * mu_th))
    )
    saturated = safe_divide(l_edd, l_crit) - 1.0

    prefactor = 1.61 * (gamma - 1.0) * safe_divide(x_c, gamma * lambda_c) * (disk.R / disk.h)
    return np.where(mu_th < 1.0, prefactor * unsaturated, prefactor * saturated)


def compute_torque_profile(disk, *, migrator_mass,
                           z=0.0, smoothing=0.005,
                           params=None):
    """Compute torque components and damping timescales across a disk profile."""
    if params is None:
        params = MigrationParameters()

    sigma = surface_density(disk)
    pressure = total_pressure(disk)

    dsigma_dlogr = radial_log_slope(disk.R, sigma, smoothing=smoothing)
    dtemp_dlogr = radial_log_slope(disk.R, disk.T, smoothing=smoothing)
    dpressure_dlogr = radial_log_slope(disk.R, pressure, smoothing=smoothing)

    q = migrator_mass / disk.Mbh
    h_over_r = disk.h / disk.R
    dep_factor = np.exp(-0.5 * (z / disk.h) ** 2)

    corot = comp_corot(dsigma_dlogr, dtemp_dlogr, params.gamma_adiabat)
    lind = comp_lind(dsigma_dlogr, dtemp_dlogr, disk, params.gamma_adiabat)
    kappa_gap = comp_kappa(q, h_over_r, params.alpha)
    gamma_0 = comp_gamma_0(q, disk.R, disk.h, disk.Omega, disk.rho)
    gamma_th_coeff = comp_gamma_th(dpressure_dlogr, disk, migrator_mass, params)

    gamma_i = (
        (corot * np.exp(-kappa_gap / params.kappa_t) + lind)
        / (1.0 + kappa_gap / params.kappa_norm)
        * gamma_0
    )
    gamma_th = gamma_0 * gamma_th_coeff
    gamma_tot = gamma_i + np.exp(-kappa_gap / params.kappa_t) * gamma_th

    orbital_angular_momentum = migrator_mass * disk.Omega * disk.R**2
    tau_mig = np.full_like(gamma_tot, np.inf, dtype=float)

    finite_mask = (gamma_tot != 0.0) & np.isfinite(gamma_tot) & (dep_factor > 0.0)
    tau_mig[finite_mask] = (
        orbital_angular_momentum[finite_mask]
        / (2.0 * gamma_tot[finite_mask] * dep_factor[finite_mask])
    )

    tau_e = np.abs(tau_mig) * h_over_r**2 / 0.780
    tau_i = np.abs(tau_mig) * h_over_r**2 / 0.544

    return {
        "radius": disk.R,
        "radius_rs": disk.R / disk.Rs,
        "surface_density": sigma,
        "dsigma_dlogr": dsigma_dlogr,
        "dtemp_dlogr": dtemp_dlogr,
        "dpressure_dlogr": dpressure_dlogr,
        "corotation_coeff": corot,
        "lindblad_coeff": lind,
        "thermal_coeff": gamma_th_coeff,
        "gap_parameter": kappa_gap,
        "gamma_0": gamma_0,
        "gamma_i": gamma_i,
        "gamma_th": gamma_th,
        "gamma_tot": gamma_tot,
        "tau_mig": tau_mig,
        "tau_e": tau_e,
        "tau_i": tau_i,
    }


def find_zero_crossings(radius, values):
    """Locate approximate zero crossings by linear interpolation."""
    radius = np.asarray(radius, dtype=float)
    values = np.asarray(values, dtype=float)

    finite = np.isfinite(values)
    radius = radius[finite]
    values = values[finite]

    nonzero = values != 0.0
    radius = radius[nonzero]
    values = values[nonzero]

    if values.size < 2:
        return np.array([])

    sign_changes = np.where(np.sign(values[1:]) != np.sign(values[:-1]))[0]
    roots = []

    for index in sign_changes:
        r0, r1 = radius[index], radius[index + 1]
        y0, y1 = values[index], values[index + 1]
        root = r0 - y0 * (r1 - r0) / (y1 - y0)
        roots.append(root)

    return np.asarray(roots)


def plot_signed_magnitude(ax, radius_rs, values, *, label_positive=None, label_negative=None):
    """Plot positive and negative segments separately on a log y-axis."""
    values = np.asarray(values, dtype=float)
    positive = np.where(values > 0.0, values, np.nan)
    negative = np.where(values < 0.0, -values, np.nan)

    ax.plot(radius_rs, positive, color="C0", label=label_positive)
    ax.plot(radius_rs, negative, color="C1", label=label_negative)


# Sirko & Goodman (2003)
sirko_mbh = 1e8 * ct.MSun           # 10^8 solar-mass SMBH
sirko_le = 0.5                      # Eddington ratio
sirko_mdot = None                   # let the model infer Mdot from le
sirko_alpha = 0.01                  # Shakura-Sunyaev alpha
sirko_X = 0.7                       # hydrogen mass fraction
sirko_b = 0                         # alpha-disk case
sirko_opacity = "combined"          # combined opacity tables

# Thompson et al. (2005)
sigma = 300e3                       # stellar velocity dispersion [m/s]
thompson_mbh = (2e8 * ct.MSun) * (sigma / 200e3) ** 4
thompson_epsilon = 1e-3             # star-formation radiative efficiency
thompson_m = 0.2                    # angular momentum transport efficiency
thompson_xi = 1.0                   # supernova energy conversion efficiency
thompson_mdot_out = None            # let the model use its default scaling
thompson_rout = None                # default outer boundary
thompson_rin = None                 # default inner boundary
thompson_opacity = "combined"

n_resolution = 10_000
mmass = 70
migrator_mass = mmass * ct.MSun
migration_params = MigrationParameters()
smoothing = 0.005

sirko_disk = S.SirkoAGN(
    Mbh=sirko_mbh,
    le=sirko_le,
    Mdot=sirko_mdot,
    alpha=sirko_alpha,
    X=sirko_X,
    b=sirko_b,
    opacity=sirko_opacity,
)
sirko_disk.solve_disk(N=n_resolution)

thompson_disk = T.ThompsonAGN(
    Mbh=thompson_mbh,
    sigma=sigma,
    epsilon=thompson_epsilon,
    m=thompson_m,
    xi=thompson_xi,
    Mdot_out=thompson_mdot_out,
    Rout=thompson_rout,
    Rin=thompson_rin,
    opacity=thompson_opacity,
)
thompson_disk.solve_disk(N=n_resolution)

sirko_torques = compute_torque_profile(
    sirko_disk,
    migrator_mass=migrator_mass,
    smoothing=smoothing,
    params=migration_params,
)

thompson_torques = compute_torque_profile(
    thompson_disk,
    migrator_mass=migrator_mass,
    smoothing=smoothing,
    params=migration_params,
)

trap_records = []

for model_name, torques in [
    ("Sirko & Goodman", sirko_torques),
    ("Thompson et al.", thompson_torques),
]:
    trap_radii = find_zero_crossings(torques["radius_rs"], torques["gamma_tot"])
    for trap_radius in trap_radii:
        trap_records.append(
            {
                "Model": model_name,
                r"Trap radius [$R_S$]": trap_radius,
            }
        )

trap_table = pd.DataFrame(trap_records)
print(trap_table)
trap_table = trap_table.to_dict()

import json

with open(f"{mmass}.json", 'w') as f:
    json.dump(trap_table, f, indent=4)
