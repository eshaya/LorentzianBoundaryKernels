"""Minimal dynamical per-k Lorentzian boundary-kernel implementation.

This script implements the workflow described in Appendix D of the LBK paper:
for each k it constructs a coupled canonical frequency matrix, initializes a
regular positive-frequency fundamental matrix, evolves the matrix mode equation,
forms the late-time boundary kernel, checks positivity of its Hermitian
imaginary part, integrates out two hidden modes by a Schur complement, and
builds a pivot-normalized primordial spectrum.

The background and turn profile are deliberately illustrative.  The script is
a reproducible implementation test, not a microscopic UECKK prediction.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

A_S = 2.10e-9
N_S = 0.9649
SIGMA_LNA = 0.014
SIGMA_NS = 0.0042
K_STAR = 0.05  # Mpc^{-1}

# Canonical homogeneous-modulus mass-squared benchmarks divided by H^2=10.
MODELS = {
    r"$(3,3)$": np.array([12.26, 25.56]),
    r"$(1,5)$": np.array([3.41, 7.21]),
    r"$S^3\times S^3$": np.array([1.21, 3.03]),
}

@dataclass(frozen=True)
class EvolutionConfig:
    x_initial: float = 60.0
    x_final: float = 0.05
    turn_eta_abs: float = 1.0 / K_STAR
    turn_width: float = 0.55
    mixing: tuple[float, float] = (0.05, 0.025)
    rtol: float = 2.0e-8
    atol: float = 2.0e-10

CFG = EvolutionConfig()


def planck_power(k: np.ndarray) -> np.ndarray:
    return A_S * (k / K_STAR) ** (N_S - 1.0)


def planck_envelope(k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p1 = np.exp(np.log(A_S) - SIGMA_LNA) * (k / K_STAR) ** ((N_S - SIGMA_NS) - 1.0)
    p2 = np.exp(np.log(A_S) + SIGMA_LNA) * (k / K_STAR) ** ((N_S + SIGMA_NS) - 1.0)
    return np.minimum(p1, p2), np.maximum(p1, p2)


def frequency_correction(x: float, k: float, mu2: np.ndarray, cfg: EvolutionConfig) -> np.ndarray:
    """Return C in F_xx + [I + C/x^2]F=0.

    The curvature mode has the de Sitter coefficient -2.  Hidden canonical
    modes have coefficients mu_a^2-2.  A localized turn in physical conformal
    time produces k-dependent curvature-hidden mixing.
    """
    n = 1 + len(mu2)
    c = np.zeros((n, n), dtype=float)
    c[0, 0] = -2.0
    c[1:, 1:] = np.diag(mu2 - 2.0)

    eta_abs = x / k
    log_distance = np.log(eta_abs / cfg.turn_eta_abs)
    turn = np.exp(-0.5 * (log_distance / cfg.turn_width) ** 2)
    for a, coupling in enumerate(cfg.mixing, start=1):
        c[0, a] = c[a, 0] = coupling * turn
    return c


def evolve_kernel(k: float, mu2: np.ndarray, cfg: EvolutionConfig = CFG) -> tuple[np.ndarray, np.ndarray]:
    """Evolve the fundamental matrix and return K and Im_H K.

    x=-k eta.  Positive frequency is exp(i x).  With the paper's Gaussian
    convention the boundary kernel is K=k F_x F^{-1}, whose Hermitian
    imaginary part is positive for the initial vacuum.
    """
    n = 1 + len(mu2)
    xi, xf = cfg.x_initial, cfg.x_final
    f0 = np.eye(n, dtype=complex) * np.exp(1j * xi) / np.sqrt(2.0 * k)
    p0 = 1j * f0
    y0 = np.concatenate((f0.ravel(), p0.ravel()))

    def rhs(x: float, y: np.ndarray) -> np.ndarray:
        f = y[: n * n].reshape(n, n)
        p = y[n * n :].reshape(n, n)
        omega2 = np.eye(n) + frequency_correction(x, k, mu2, cfg) / (x * x)
        return np.concatenate((p.ravel(), (-omega2 @ f).ravel()))

    sol = solve_ivp(
        rhs,
        (xi, xf),
        y0,
        method="DOP853",
        rtol=cfg.rtol,
        atol=cfg.atol,
    )
    if not sol.success:
        raise RuntimeError(f"Mode evolution failed for k={k}: {sol.message}")

    f = sol.y[: n * n, -1].reshape(n, n)
    p = sol.y[n * n :, -1].reshape(n, n)
    kernel = k * (p @ np.linalg.inv(f))
    kernel_i = (kernel - kernel.conj().T) / (2j)
    return kernel, kernel_i


def schur_effective(kernel: np.ndarray) -> complex:
    a = kernel[0, 0]
    b = kernel[0, 1:]
    c = kernel[1:, 1:]
    return a - b @ np.linalg.solve(c, kernel[1:, 0])


def response_for_model(k_grid: np.ndarray, mu2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    response = np.empty_like(k_grid)
    min_eig = np.empty_like(k_grid)
    zero_mix = EvolutionConfig(**{**CFG.__dict__, "mixing": (0.0, 0.0)})

    for i, k in enumerate(k_grid):
        kernel, kernel_i = evolve_kernel(float(k), mu2, CFG)
        kernel0, _ = evolve_kernel(float(k), mu2, zero_mix)
        eff = schur_effective(kernel)
        eff0 = schur_effective(kernel0)
        if eff.imag <= 0.0 or eff0.imag <= 0.0:
            raise RuntimeError(f"Non-positive effective imaginary kernel at k={k}")
        response[i] = eff0.imag / eff.imag
        min_eig[i] = np.linalg.eigvalsh(kernel_i).min()
    return response, min_eig


def effective_tilt(k: np.ndarray, spectrum: np.ndarray) -> float:
    mask = (k >= 0.01) & (k <= 0.1)
    slope = np.polyfit(np.log(k[mask] / K_STAR), np.log(spectrum[mask]), 1)[0]
    return 1.0 + slope


def main() -> None:
    k = np.logspace(-4, np.log10(0.2), 90)
    baseline = planck_power(k)
    lower, upper = planck_envelope(k)

    rows: list[dict[str, float | str]] = []
    curves: dict[str, np.ndarray] = {}
    for name, mu2 in MODELS.items():
        response, min_eig = response_for_model(k, mu2)
        response /= np.interp(K_STAR, k, response)
        spectrum = baseline * response
        curves[name] = spectrum
        inside = (spectrum >= lower) & (spectrum <= upper)
        rows.append(
            {
                "model": name.replace("$", ""),
                "inside_percent": 100.0 * float(inside.mean()),
                "min_kernel_eigenvalue": float(min_eig.min()),
                "effective_ns": effective_tilt(k, spectrum),
                "max_abs_fractional_residual": float(np.max(np.abs(response - 1.0))),
            }
        )

    with open("matrix_per_k_kernel_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open("matrix_per_k_kernel_values.tex", "w") as f:
        f.write("% Automatically generated by matrix_per_k_kernel.py\n")
        labels = ["ThirtyThree", "OneFive", "Sthree"]
        for label, row in zip(labels, rows):
            f.write(r"\newcommand{\Matrix%sInside}{%.1f}" "\n" % (label, row["inside_percent"]))
            f.write(r"\newcommand{\Matrix%sMinEig}{%.2e}" "\n" % (label, row["min_kernel_eigenvalue"]))
            f.write(r"\newcommand{\Matrix%sNs}{%.4f}" "\n" % (label, row["effective_ns"]))
            f.write(r"\newcommand{\Matrix%sMaxResidual}{%.3f}" "\n" % (label, row["max_abs_fractional_residual"]))

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "mathtext.fontset": "cm",
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.fill_between(k, 100.0 * (lower / baseline - 1.0), 100.0 * (upper / baseline - 1.0), alpha=0.22,
                    label=r"Planck 2018 $1\sigma$ parameter envelope")
    ax.axhline(0.0, lw=1.1, ls="--")
    for name, spectrum in curves.items():
        ax.semilogx(k, 100.0 * (spectrum / baseline - 1.0), lw=2.0, label=name)
    ax.axvline(K_STAR, ls=":", lw=1.2)
    ax.set_xlabel(r"$k\, [{\rm Mpc}^{-1}]$")
    ax.set_ylabel("residual from Planck power law [percent]")
    ax.set_title(r"Matrix-evolved per-$k$ LBK screen")
    ax.legend(frameon=False, loc="best")
    ax.margins(x=0.02)
    fig.savefig(f"{OUT}/matrix_per_k_screen.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/matrix_per_k_screen.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(rows)


if __name__ == "__main__":
    main()
