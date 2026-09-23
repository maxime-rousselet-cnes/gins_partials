"""
Dependencies: numpy scipy matplotlib
Solid curves: original Burgers models.
Dashed curves: fitted attenuation-band models.
Fit: normalized recoverable creep, uniformly sampled in log(time).
Plot: full complex compliance mu0 / mu*, including Maxwell flow.
Only alpha is fitted; Delta = sum(mu0/mu_K) and S = Delta/(1+Delta).
Formal sigma_alpha assumes independent, equal-variance creep residuals.
It measures local fit uncertainty, not uncertainty in the source parameters.
Convention: exp(+i*omega*t), giving positive Im(mu*).
Plot period T = 2*pi/omega.
All relaxation times entering tau**alpha are numerical seconds.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.pyplot import tight_layout
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh
from scipy.optimize import least_squares

# ------------------------ User settings ------------------------
FIT_YEARS = (1.0, 1e2)  # Elapsed times used for creep fitting
PLOT_YEARS = (0.01, 1e4)  # Oscillation periods used for plotting
OUTPUT = Path("post_seismic_equivalence")
MU0 = 64e9  # Pa; common reference for comparison
ETA_M = 3e19  # Pa s; common Maxwell viscosity
Q_MU = 168.0
TAU_L = 1.0 / 3.09e-4  # Seconds
YEAR = 365.25 * 86400.0
N_FIT = 101
N_QUAD = 96  # Gauss-Legendre spectrum discretization
ALPHA_BOUNDS = (1e-4, 1.0)

# Each model: (name, Kelvin viscosities, ratios mu0/mu_K).
# Boulze reference plus four bounding scenarios.
# Klein parameters follow its Figure 12 two-Kelvin model.
MODELS = [
    ("Reference", (2.7e18,), (5.0,)),
    ("B1", (3.0e18,), (5.0,)),
    ("B2", (3.0e18,), (15.0,)),
    ("B3", (4.5e18,), (3.4,)),
    ("B4", (4.5e18,), (15.0,)),
    ("Two-Kelvin-Voigt reference", (4.8e18, 7.4e17), (5.0, 1.0 / 3.9)),
]

# ---------------------- Constitutive models --------------------


def upper_cutoff(alpha, S):
    """
    Derive tau_h from the integrated modulus strength:
        S = (tau_h**alpha - tau_l**alpha)/(alpha*Q_mu).
    (alpha, S) is a reparameterization of (alpha, tau_h).
    S is NOT an additional free parameter.
    """
    if abs(alpha) < 1e-12:
        return TAU_L * np.exp(Q_MU * S)
    width = np.log1p(alpha * Q_MU * S / TAU_L**alpha) / alpha
    return TAU_L * np.exp(width)


GL_X, GL_W = leggauss(N_QUAD)


def band_spectrum(alpha, S):
    """Quadrature nodes tau_j and weights w_j for the full integral."""
    H = upper_cutoff(alpha, S)
    width = np.log(H / TAU_L)
    log_tau = np.log(TAU_L) + 0.5 * width * (GL_X + 1)
    tau = np.exp(log_tau)
    weights = 0.5 * width * GL_W * tau**alpha / Q_MU
    return tau, weights


def burgers_creep(time, model):
    """Recoverable Burgers creep, normalized by 1/mu0."""
    _, eta_k, ratios = model
    ratios = np.asarray(ratios)
    tau_k = np.asarray(eta_k) * ratios / MU0
    return 1 + np.sum(
        ratios * (-np.expm1(-time[:, None] / tau_k)),
        axis=1,
    )


def band_creep(time, alpha, S):
    """
    Recoverable attenuation-band creep, normalized by 1/mu0.
    The discretized recoverable modulus is
        M(s) = 1 - sum_j w_j*d_j/(s+d_j),  d_j = 1/tau_j.
    Set v_j = sqrt(w_j*d_j), A = diag(d_j) - v*v.T.
    Then
        1/M(s) = 1 + v.T @ inv(s*I + A) @ v.
    Diagonalizing A gives the creep as a sum of exponentials.
    This is a numerical evaluation of the supplied finite-band law.
    """
    tau, weights = band_spectrum(alpha, S)
    rates = 1.0 / tau
    v = np.sqrt(weights * rates)
    A = np.diag(rates) - np.outer(v, v)
    eigenvalues, eigenvectors = eigh(A, check_finite=False)
    if np.any(eigenvalues <= 0):
        raise FloatingPointError("Non-positive creep-mode rate.")
    amplitudes = (eigenvectors.T @ v) ** 2 / eigenvalues
    return 1 + (-np.expm1(-time[:, None] * eigenvalues)) @ amplitudes


def burgers_modulus(omega, model):
    """Full complex Burgers modulus / mu0."""
    _, eta_k, ratios = model
    ratios = np.asarray(ratios)
    tau_k = np.asarray(eta_k) * ratios / MU0
    denominator = (
        1
        + np.sum(
            ratios / (1 + 1j * omega[:, None] * tau_k),
            axis=1,
        )
        + MU0 / (1j * omega * ETA_M)
    )
    return 1 / denominator


def band_modulus(omega, alpha, S):
    """Full complex attenuation-band modulus / mu0."""
    tau, weights = band_spectrum(alpha, S)
    # Recoverable modulus from the full finite-band integral.
    M_A = 1 - np.sum(
        weights / (1 + 1j * omega[:, None] * tau),
        axis=1,
    )
    # Same Maxwell dashpot as in the original Burgers model.
    return M_A / (1 + M_A * MU0 / (1j * omega * ETA_M))


# ------------------------------ Fit ----------------------------


def fit_model(model, time):
    time = np.asarray(time, dtype=float)
    if time.ndim != 1 or time.size < 2 or not np.all(np.isfinite(time)) or np.any(time <= 0):
        raise ValueError("Fit times must contain at least two finite positive values.")
    delta = float(sum(model[2]))
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("The total modulus ratio must be finite and positive.")
    S = delta / (1.0 + delta)
    target = burgers_creep(time, model)

    def residuals(parameters):
        return band_creep(time, float(parameters[0]), S) - target

    # A single fitted parameter. Multiple starts reduce sensitivity to local minima.
    solutions = []
    for start in (0.08, 0.14, 0.17, 0.20, 0.30, 0.60, 0.90):
        try:
            sol = least_squares(
                residuals,
                x0=[start],
                bounds=ALPHA_BOUNDS,
                jac="3-point",
                diff_step=1e-5,
                ftol=1e-11,
                xtol=1e-11,
                gtol=1e-11,
                max_nfev=2000,
            )
        except (FloatingPointError, np.linalg.LinAlgError):
            continue
        if sol.success and np.all(np.isfinite(sol.fun)):
            solutions.append(sol)
    if not solutions:
        raise RuntimeError(f"No fit converged for {model[0]}.")
    best = min(solutions, key=lambda sol: sol.cost)
    alpha = float(best.x[0])
    if min(alpha - ALPHA_BOUNDS[0], ALPHA_BOUNDS[1] - alpha) < 1e-5:
        raise RuntimeError(
            f"{model[0]} reached an alpha bound; a symmetric uncertainty is unreliable."
        )

    # Local linearized covariance with unknown, common residual variance:
    # Var(alpha) = [RSS / (N - 1)] / sum_i (d residual_i / d alpha)^2.
    # Delta, Q_mu, tau_l and all Burgers parameters are treated as exact.
    rss = float(best.fun @ best.fun)
    dof = time.size - 1
    information = float(best.jac[:, 0] @ best.jac[:, 0])
    if not np.isfinite(information) or information <= 0:
        raise RuntimeError(f"{model[0]} has no identifiable local alpha sensitivity.")
    sigma_alpha = float(np.sqrt((rss / dof) / information))

    return {
        "model": model[0],
        "alpha": float(alpha),
        "S": float(S),
        "Delta": delta,
        "sigma_alpha": sigma_alpha,
        "residual_degrees_of_freedom": int(dof),
        "tau_h_s": float(upper_cutoff(alpha, S)),
        "tau_h_years": float(upper_cutoff(alpha, S) / YEAR),
        "normalized_creep_RMS": float(np.sqrt(rss / time.size)),
        "Q_mu": Q_MU,
        "tau_l_s": TAU_L,
        "mu0_Pa": MU0,
        "eta_M_Pa_s": ETA_M,
        "eta_K_Pa_s": ";".join(map(str, model[1])),
        "mu0_over_mu_K": ";".join(map(str, model[2])),
        "fit_time_min_years": FIT_YEARS[0],
        "fit_time_max_years": FIT_YEARS[1],
    }


# -------------------------- Plot and save -----------------------


def plot_results(results):
    period = np.geomspace(*PLOT_YEARS, 1800)
    omega = 2 * np.pi / (period * YEAR)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = plt.get_cmap("tab10").colors[: len(MODELS)]
    left_handles, right_handles = [], []
    for model, result, color in zip(MODELS, results, colors):
        # The modulus functions return mu*/mu0.
        # Invert the COMPLEX quantity to obtain mu0/mu*.
        original = 1.0 / burgers_modulus(omega, model)
        fitted = 1.0 / band_modulus(omega, result["alpha"], result["S"])
        for ax, y_original, y_fitted in (
            (axes[0], original.real, fitted.real),
            (axes[1], -original.imag, -fitted.imag),
        ):
            ax.plot(period, y_original, color=color, lw=2, ls="-")
            ax.plot(period, y_fitted, color=color, lw=2, ls="--")
        eta_text = r"\ &\ ".join(f"{v / 1e18:g}" for v in model[1])
        ratio_text = r"\ &\ ".join(f"{v:.4g}" for v in model[2])
        left_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=2,
                label=(
                    f"{model[0]}\n"
                    rf"$\eta_K={eta_text}\cdot 10^{{18}} Pa.s,"
                    rf"\quad \mu_0/\mu_K={ratio_text}$"
                ),
            )
        )
        right_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=2,
                ls="--",
                label=(
                    f"{model[0]}\n"
                    rf"$\alpha={result['alpha']:.4f},"
                    rf"\quad\tau_h={result['tau_h_years']:.2f}$ yr"
                ),
            )
        )
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(*PLOT_YEARS)
        ax.set_xlabel(r"Years")
        ax.grid(which="major", alpha=0.22)
        ax.axvspan(
            FIT_YEARS[0],
            FIT_YEARS[1],
            facecolor="grey",
            alpha=0.15,
            edgecolor="none",
            zorder=0,
        )
    for ax in axes:
        ax.set_yscale("log")
    axes[0].set_ylabel(r"$\operatorname{Re}(\mu_0/\mu(\omega))$")
    axes[1].set_ylabel(r"$\operatorname{Im}(\mu_0/\mu(\omega))$")
    axes[1].invert_yaxis()
    axes[1].yaxis.set_major_formatter(lambda y, _: f"{-y:g}")
    axes[0].legend(
        handles=left_handles,
        title=("Burgers parameters"),
        loc="upper left",
        bbox_to_anchor=(0, -1.27),
        frameon=False,
        fontsize=9,
        title_fontsize=10,
        labelspacing=0.8,
    )
    axes[1].legend(
        handles=right_handles,
        title=("Fitted attenuation-band parameters"),
        loc="upper left",
        bbox_to_anchor=(0.6, -0.1),
        frameon=False,
        fontsize=9,
        title_fontsize=10,
        labelspacing=0.8,
    )
    fig.suptitle(
        r"$\alpha$-power law values retrieved from post-seismic deformation studies: $\mu_0/\mu(\omega)$",
        fontsize=12,
    )
    return fig


def main():
    if not 0 < FIT_YEARS[0] < FIT_YEARS[1]:
        raise ValueError("Invalid FIT_YEARS.")
    if not 0 < PLOT_YEARS[0] < PLOT_YEARS[1]:
        raise ValueError("Invalid PLOT_YEARS.")
    time = np.geomspace(*FIT_YEARS, N_FIT) * YEAR
    results = []
    for model in MODELS:
        print(f"Fitting {model[0]}...", flush=True)
        results.append(fit_model(model, time))
    print(
        f"\n{'Model':24s} {'alpha':>9s} {'sigma_alpha':>12s} {'Delta':>10s}"
        f" {'tau_h/yr':>11s} {'creep RMS':>11s}"
    )
    for result in results:
        print(
            f"{result['model']:24s}"
            f" {result['alpha']:9.5f}"
            f" {result['sigma_alpha']:12.6f}"
            f" {result['Delta']:10.4f}"
            f" {result['tau_h_years']:11.4f}"
            f" {result['normalized_creep_RMS']:11.5f}"
        )
    fig = plot_results(results)
    tight_layout()
    fig.savefig(str(OUTPUT) + ".pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
