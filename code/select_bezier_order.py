"""
Select the optimal Bézier curve order (n) using leave-one-out cross-validation
at the trajectory-point level.

For each shot and each candidate order n ∈ {3, …, 12}, every trajectory point is
left out in turn: a Bézier curve of order n is fitted to the remaining points, and
the squared prediction error at the left-out point is recorded.  The per-shot LOO-CV
error is the mean squared error across all left-out points.  The overall LOO-CV error
for a given n is the mean across all shots.  The optimal n minimises this quantity.

Produces a plot saved as select_bezier_order.png.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
from bezier_curve_functions import bernstein_poly, find_end

# ── data ──────────────────────────────────────────────────────────────────────
player_names = [
    'Klay Thompson', 'Steph Curry', 'Bogdan Bogdanovic', 'Damian Lillard',
    'Anfernee Simons', 'Paul George', 'Derrick White', 'Tobias Harris',
    'Shai GilgeousAlexander', 'Karl Anthony Towns', 'Tim Hardaway Jr',
    'Luka Doncic', 'Dante Exum', 'Jaden Hardy', 'Alec Burks',
    'Paolo Banchero', 'Ivica Zubac', 'Derrick Jones Jr', 'Dwight Powell',
    'Domantas Sabonis', 'Maxi Kleber', 'Zion Williamson',
    'Russell Westbrook', 'Josh Green', 'Omax Prosper',
    'Giannis Antetokounmpo', 'Aaron Gordon', 'Rudy Gobert', 'Clint Capela',
    'PJ Washington', 'Daniel Gafford', 'Andre Drummond', 'Jakob Poeltl',
    'Nic Claxton', 'Dereck Lively',
]
file_temp = '/Users/ruoqianzhu/Desktop/undergrad/freethrows/{} FTs.xlsx'

# candidate Bézier orders
n_values = list(range(3, 13))   # 3 … 12

# ── point-level LOO-CV error for one shot ─────────────────────────────────────
def loo_cv_mse(xy, n):
    """
    For each trajectory point, leave it out, fit Bézier(n) to the remaining
    points (keeping their original arc-length parameter values), and compute
    the squared Euclidean error at the left-out point.
    Returns the mean squared error across all left-out points.
    """
    m = len(xy)
    t_all = np.linspace(0, 1, m)
    sq_errors = np.zeros(m)
    for k in range(m):
        idx = [i for i in range(m) if i != k]
        t_tr = t_all[idx]
        # Bernstein basis matrix for training points
        T_tr = np.column_stack([bernstein_poly(i, n, t_tr) for i in range(n + 1)])
        p_star, _, _, _ = lstsq(T_tr, xy[idx], rcond=None)
        # evaluate at the left-out point's original parameter value
        T_k = np.array([bernstein_poly(i, n, t_all[[k]])[0] for i in range(n + 1)])
        sq_errors[k] = np.sum((xy[k] - T_k @ p_star) ** 2)
    return float(np.mean(sq_errors))


# ── collect LOO-CV errors across all shots ────────────────────────────────────
shot_mses = {n: [] for n in n_values}

print("Computing point-level LOO-CV MSE by Bézier order …")

for player in player_names:
    print(f"  {player}")
    file_path = file_temp.format(player)
    sheets_dict = pd.read_excel(file_path, sheet_name=None)

    for _, df in sheets_dict.items():
        df['E_Distance'] = df['E_Distance'].abs()

        path_end = find_end(df)
        df_copy = df[['E_Distance', 'G_Height', 'F_Offset']].iloc[:path_end - 5]
        path_start = df_copy['E_Distance'].idxmax()
        df_shot = df[['E_Distance', 'G_Height']].iloc[path_start:path_end]

        xy = df_shot.values
        if len(xy) < n_values[-1] + 2:
            continue  # skip very short shots

        for n in n_values:
            shot_mses[n].append(loo_cv_mse(xy, n))

# ── aggregate ─────────────────────────────────────────────────────────────────
mean_loo_mse = np.array([np.mean(shot_mses[n]) for n in n_values])

# ── select optimal n ──────────────────────────────────────────────────────────
optimal_n = n_values[int(np.argmin(mean_loo_mse))]

# ── report ────────────────────────────────────────────────────────────────────
print("\nMean point-level LOO-CV MSE by Bézier order:")
for n, mse in zip(n_values, mean_loo_mse):
    marker = "  ← selected (min LOO-CV MSE)" if n == optimal_n else ""
    print(f"  n = {n:2d}:  {mse:.8f}{marker}")
print(f"\nOptimal Bézier order: n = {optimal_n}")

# ── plot ───────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# left: LOO-CV MSE vs n
axes[0].plot(n_values, mean_loo_mse, color='#2c7bb6', linewidth=1.8)
axes[0].scatter(n_values, mean_loo_mse, color='#2c7bb6', s=35, zorder=4)
axes[0].axvline(optimal_n, color='#d7191c', linestyle='--', linewidth=1.2,
                label=f'Optimal $n = {optimal_n}$')
axes[0].set_xlabel('Bézier order $n$', fontsize=12)
axes[0].set_ylabel('Mean LOO-CV MSE', fontsize=12)
axes[0].set_title('LOO-CV error vs.\ Bézier order', fontsize=13)
axes[0].set_xticks(n_values)
axes[0].tick_params(labelsize=10)
axes[0].legend(fontsize=10, frameon=False)

# right: marginal LOO-CV improvement (first difference)
marginal = -np.diff(mean_loo_mse)   # positive = improvement gained going from n to n+1
axes[1].bar(n_values[1:], marginal, color='#2c7bb6', edgecolor='white', linewidth=0.5)
axes[1].axvline(optimal_n, color='#d7191c', linestyle='--', linewidth=1.2,
                label=f'Optimal $n = {optimal_n}$')
axes[1].set_xlabel('Bézier order $n$', fontsize=12)
axes[1].set_ylabel('Marginal LOO-CV MSE reduction', fontsize=12)
axes[1].set_title('Diminishing marginal returns (LOO-CV)', fontsize=13)
axes[1].set_xticks(n_values[1:])
axes[1].tick_params(labelsize=10)
axes[1].legend(fontsize=10, frameon=False)

plt.tight_layout(pad=1.5)
plt.savefig('select_bezier_order.png', dpi=300, bbox_inches='tight')
plt.show()
print("Plot saved to select_bezier_order.png")
