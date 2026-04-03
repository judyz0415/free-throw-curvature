"""
Select the optimal weighting parameter l 

Uses leave-one-out (LOO) cross-validation at the *player* level:
  for each candidate l and each held-out player p,
    fit WLS on the remaining 34 players → predict FT% for player p.
The l minimising LOO prediction MSE is selected as optimal.

Prerequisite: run metric_calculation first to generate curvature_results.xlsx,
which must contain sigma_l_<value> columns for all candidate l values).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── load data ──────────────────────────────────────────────────────────────────
df = pd.read_excel('curvature_results.xlsx')

y = df['FT%'].values
weights = 1.0 / (df['FT% SE'].values ** 2)

# candidate l values: every column named sigma_l_<value>
sigma_cols = sorted(
    [c for c in df.columns if c.startswith('sigma_l_')],
    key=lambda c: float(c.split('_')[-1])
)
l_values = np.array([float(c.split('_')[-1]) for c in sigma_cols])

# ── LOO-CV WLS ────────────────────────────────────────────────────────────────
def loo_cv_wls(sigma_vec, y, weights):
    """
    LOO-CV for WLS: y ~ alpha + beta * sigma_vec.
    Returns (mean LOO MSE, array of per-player squared errors).
    """
    n = len(y)
    sq_errors = np.zeros(n)
    for p in range(n):
        train = [i for i in range(n) if i != p]
        X_tr  = np.column_stack([np.ones(n - 1), sigma_vec[train]])
        W     = np.diag(weights[train])
        XtW   = X_tr.T @ W
        b     = np.linalg.solve(XtW @ X_tr, XtW @ y[train])
        y_hat = b[0] + b[1] * sigma_vec[p]
        sq_errors[p] = (y[p] - y_hat) ** 2
    return float(np.mean(sq_errors)), sq_errors

# ── also compute in-sample R² for context ─────────────────────────────────────
import statsmodels.api as sm

def wls_r2(sigma_vec, y, weights):
    try:
        X = sm.add_constant(sigma_vec)
        res = sm.WLS(y, X, weights=weights).fit()
        return res.rsquared, res.pvalues[1]
    except Exception:
        return np.nan, np.nan

# ── run across all l values ───────────────────────────────────────────────────
print("Computing LOO-CV MSE and in-sample R² for each l …")
loo_mse = np.zeros(len(l_values))
r2_vals = np.zeros(len(l_values))
pval_vals = np.zeros(len(l_values))

for j, (col, l) in enumerate(zip(sigma_cols, l_values)):
    sigma_vec = df[col].values
    loo_mse[j], _ = loo_cv_wls(sigma_vec, y, weights)
    r2_vals[j], pval_vals[j] = wls_r2(sigma_vec, y, weights)

# ── identify optimal l ────────────────────────────────────────────────────────
optimal_idx = int(np.argmin(loo_mse))
optimal_l   = l_values[optimal_idx]

print(f"\nOptimal l (min LOO-CV MSE): l = {optimal_l:.1f}")
print(f"  In-sample R² at optimal l:  {r2_vals[optimal_idx]:.4f}")
print(f"  p-value at optimal l:        {pval_vals[optimal_idx]:.6f}")

# print summary for integer l values
print("\nSummary for integer l values:")
print(f"  {'l':>5}  {'LOO-CV MSE':>12}  {'R²':>8}  {'p-value':>10}")
for j, l in enumerate(l_values):
    if l == int(l):
        marker = " ←" if j == optimal_idx else ""
        print(f"  {l:5.0f}  {loo_mse[j]:12.6f}  {r2_vals[j]:8.4f}  {pval_vals[j]:10.6f}{marker}")

# ── plot ───────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# panel 1: LOO-CV MSE vs l
axes[0].plot(l_values, loo_mse, color='#2c7bb6', linewidth=1.8)
axes[0].scatter(l_values, loo_mse, color='#2c7bb6', s=25, zorder=4)
axes[0].axvline(optimal_l, color='#d7191c', linestyle='--', linewidth=1.2,
                label=fr'Optimal $\ell = {optimal_l:.1f}$')
axes[0].set_xlabel(r'Weighting parameter $\ell$', fontsize=12)
axes[0].set_ylabel('LOO-CV MSE', fontsize=12)
axes[0].set_title(r'LOO-CV MSE vs. $\ell$', fontsize=13)
axes[0].tick_params(labelsize=10)
axes[0].legend(fontsize=10, frameon=False)

# panel 2: in-sample R² vs l
axes[1].plot(l_values, r2_vals, color='#2c7bb6', linewidth=1.8)
axes[1].scatter(l_values, r2_vals, color='#2c7bb6', s=25, zorder=4)
axes[1].axvline(optimal_l, color='#d7191c', linestyle='--', linewidth=1.2,
                label=fr'Optimal $\ell = {optimal_l:.1f}$')
axes[1].set_xlabel(r'Weighting parameter $\ell$', fontsize=12)
axes[1].set_ylabel(r'In-sample $R^2$', fontsize=12)
axes[1].set_title(r'In-sample $R^2$ vs. $\ell$', fontsize=13)
axes[1].tick_params(labelsize=10)
axes[1].legend(fontsize=10, frameon=False)

plt.tight_layout(pad=1.5)
plt.savefig('select_l_parameter.png', dpi=300, bbox_inches='tight')
plt.show()
print("Plot saved to select_l_parameter.png")
