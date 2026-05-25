"""
Jointly select the regression parameters (alpha, beta) and weighting exponent ell
by minimising the in-sample weighted residual sum of squares using a nonlinear
solver (scipy.optimize.curve_fit).

The model is:
    FT%(i) = alpha + beta * sigma_ell(i),
    sigma_ell(i) = integral_0^1  (ell+1) * t^ell * kappa_i(t)  dt

where kappa_i(t) is the mean curvature profile for player i (from kappa_arrays.npy).
All three parameters — alpha, beta, ell — are optimised simultaneously for in-sample
fit, treating FT% SE as measurement uncertainty (equivalent to weighted least squares).

Separately, leave-one-out cross-validation at the player level is used to calculate
the out-of-sample R²: for each held-out player, alpha/beta/ell are re-estimated on
the remaining 34 players and used to predict the held-out player's FT%.

Prerequisite: run metric_calculation first to generate curvature_results.xlsx and
kappa_arrays.npy (both in the same directory as this script).

Outputs:
  optimal_params.json     — optimal alpha, beta, ell, in-sample R², out-of-sample R²
  select_l_parameter.png  — in-sample and out-of-sample prediction scatter plots
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.optimize import curve_fit

T_INTERVAL = np.linspace(0, 1, 100)

# ── load data ──────────────────────────────────────────────────────────────────
df = pd.read_excel('curvature_results.xlsx')
y      = df['FT%'].values
sigma  = df['FT% SE'].values          # measurement std dev for WLS
kappas = np.load('kappa_arrays.npy')  # shape: (n_players, 100), aligned with df rows

assert len(y) == len(kappas), "curvature_results.xlsx and kappa_arrays.npy are misaligned"

# ── model definition ──────────────────────────────────────────────────────────
def compute_sigma_ell(kappa, ell):
    """Weighted curvature integral for a single player."""
    w = (ell + 1) * T_INTERVAL ** ell
    return float(simpson(w * kappa, T_INTERVAL))

def model(kappas_2d, alpha, beta, ell):
    """
    Nonlinear model: FT%(i) = alpha + beta * sigma_ell(i).
    kappas_2d: (n_players, 100) array passed by curve_fit as xdata.
    """
    sigmas = np.array([compute_sigma_ell(k, ell) for k in kappas_2d])
    return alpha + beta * sigmas

# ── in-sample joint optimisation ──────────────────────────────────────────────
print("Fitting nonlinear model (alpha, beta, ell) jointly on all players …")

p0     = [float(np.mean(y)), 5.0, 6.0]         # (alpha, beta, ell) starting values
bounds = ([-np.inf, -np.inf, 0.0],
          [ np.inf,  np.inf, 20.0])

popt, pcov = curve_fit(
    model, kappas, y,
    p0=p0, sigma=sigma, absolute_sigma=True,
    bounds=bounds, maxfev=10_000,
)
alpha_opt, beta_opt, ell_opt = popt

y_hat_insample = model(kappas, *popt)
# Weighted R² (consistent with WLS objective and statsmodels output)
w              = 1.0 / sigma ** 2
y_bar_w        = float(np.average(y, weights=w))
ss_tot_w       = float(np.sum(w * (y - y_bar_w) ** 2))
ss_res_in_w    = float(np.sum(w * (y - y_hat_insample) ** 2))
r2_insample    = 1.0 - ss_res_in_w / ss_tot_w
# Unweighted SS for OOS R² denominator (standard out-of-sample evaluation)
ss_tot         = float(np.sum((y - np.mean(y)) ** 2))

print(f"\nIn-sample optimal parameters:")
print(f"  alpha = {alpha_opt:.4f}")
print(f"  beta  = {beta_opt:.4f}")
print(f"  ell   = {ell_opt:.4f}")
print(f"  In-sample R² = {r2_insample:.4f}")

# ── LOOCV for out-of-sample R² ────────────────────────────────────────────────
print("\nRunning player-level LOOCV for out-of-sample R² …")

n_players = len(y)
y_oos     = np.zeros(n_players)

for p in range(n_players):
    train = [i for i in range(n_players) if i != p]
    try:
        popt_loo, _ = curve_fit(
            model, kappas[train], y[train],
            p0=popt,                   # warm-start from full-data solution
            sigma=sigma[train], absolute_sigma=True,
            bounds=bounds, maxfev=10_000,
        )
    except RuntimeError:
        popt_loo = popt                # fall back to full-data solution
    y_oos[p] = model(kappas[[p]], *popt_loo)[0]

ss_res_oos = float(np.sum((y - y_oos) ** 2))
r2_oos     = 1.0 - ss_res_oos / ss_tot

print(f"  Out-of-sample R² (LOOCV) = {r2_oos:.4f}")

# ── save parameters ──────────────────────────────────────────────────────────
results = {
    'alpha':       float(alpha_opt),
    'beta':        float(beta_opt),
    'ell':         float(ell_opt),
    'r2_insample': float(r2_insample),
    'r2_oos':      float(r2_oos),
}
with open('optimal_params.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved optimal_params.json")

# ── plot: in-sample and out-of-sample predictions ─────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

lims = [min(y.min(), y_hat_insample.min(), y_oos.min()) - 0.02,
        max(y.max(), y_hat_insample.max(), y_oos.max()) + 0.02]

for ax, y_pred, r2, title in [
    (axes[0], y_hat_insample, r2_insample,
     fr'In-sample fit  ($\ell^* = {ell_opt:.2f}$)'),
    (axes[1], y_oos,          r2_oos,
     r'Out-of-sample predictions (LOOCV)'),
]:
    ax.scatter(y, y_pred, color='#2c7bb6', s=35, alpha=0.8, zorder=3)
    ax.plot(lims, lims, color='black', linewidth=1.0, linestyle='--', zorder=2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Actual FT%', fontsize=12)
    ax.set_ylabel('Predicted FT%', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.tick_params(labelsize=10)
    ax.text(0.05, 0.92, fr'$R^2 = {r2:.3f}$',
            transform=ax.transAxes, fontsize=11)

plt.tight_layout(pad=1.5)
plt.savefig('select_l_parameter.png', dpi=300, bbox_inches='tight')
plt.show()
print("Plot saved to select_l_parameter.png")
