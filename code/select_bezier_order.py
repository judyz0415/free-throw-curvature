"""
Select the optimal Bézier curve order (n) by minimising the mean
in-sample fitting gap between the Bézier curve and the actual trajectory.

As n increases, the fitting error decreases monotonically, but the gains
eventually plateau (diminishing marginal returns).  We select the order at
the "knee" of the error-vs-n curve — i.e., the n just before the marginal
improvement becomes negligible — using the second-difference (curvature)
method: the knee is the n where the curve bends most sharply from steep to
flat, identified as the maximum of the second differences of the MSE series.

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

# ── per-shot fitting MSE ──────────────────────────────────────────────────────
def fitting_mse(xy, n):
    """
    Fit a Bézier curve of order n to trajectory xy (shape: m+1, 2)
    and return the mean squared residual.
    """
    m = len(xy) - 1
    t = np.linspace(0, 1, m + 1)
    T = np.zeros((m + 1, n + 1))
    for i in range(n + 1):
        T[:, i] = bernstein_poly(i, n, t)
    p_star, _, _, _ = lstsq(T, xy, rcond=None)
    y_hat = T @ p_star
    return float(np.mean(np.sum((xy - y_hat) ** 2, axis=1)))


# ── collect MSEs across all shots ─────────────────────────────────────────────
shot_mses = {n: [] for n in n_values}

print("Computing in-sample fitting MSE by Bézier order …")

for player in player_names:
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
            shot_mses[n].append(fitting_mse(xy, n))

# ── aggregate ─────────────────────────────────────────────────────────────────
mean_mse = np.array([np.mean(shot_mses[n]) for n in n_values])

# ── knee detection ────────────────────────────────────────────────────────────
# Use the relative drop in marginal improvement: ratio = marginal[k+1]/marginal[k].
# The knee is the n where the marginal gain shrank the most *relative* to the
# previous step — i.e. the first n after which adding a control point stops
# paying off proportionally.  argmin of the ratio gives that n.
marginal = -np.diff(mean_mse)              # improvement from n → n+1; shape len-1
ratio = marginal[1:] / marginal[:-1]       # relative drop; shape len-2
# ratio[k] = gain(n_values[k+2]) / gain(n_values[k+1])
# The knee is n_values[k+1] where ratio[k] is smallest.
knee_idx = int(np.argmin(ratio)) + 1       # index in n_values
optimal_n = n_values[knee_idx]

# ── report ─────────────────────────────────────────────────────────────────────
print("\nMean in-sample fitting MSE by Bézier order:")
for n, mse in zip(n_values, mean_mse):
    marker = "  ← knee (selected)" if n == optimal_n else ""
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

# left: MSE vs n
axes[0].plot(n_values, mean_mse, color='#2c7bb6', linewidth=1.8)
axes[0].scatter(n_values, mean_mse, color='#2c7bb6', s=35, zorder=4)
axes[0].axvline(optimal_n, color='#d7191c', linestyle='--', linewidth=1.2,
                label=f'Knee at $n = {optimal_n}$')
axes[0].set_xlabel('Bézier order $n$', fontsize=12)
axes[0].set_ylabel('Mean fitting MSE', fontsize=12)
axes[0].set_title('Fitting gap vs.\ Bézier order', fontsize=13)
axes[0].set_xticks(n_values)
axes[0].tick_params(labelsize=10)
axes[0].legend(fontsize=10, frameon=False)

# right: marginal improvement (first difference of MSE)
marginal = -np.diff(mean_mse)   # positive = improvement gained by going from n-1 to n
axes[1].bar(n_values[1:], marginal, color='#2c7bb6', edgecolor='white', linewidth=0.5)
axes[1].axvline(optimal_n, color='#d7191c', linestyle='--', linewidth=1.2,
                label=f'Knee at $n = {optimal_n}$')
axes[1].set_xlabel('Bézier order $n$', fontsize=12)
axes[1].set_ylabel('Marginal MSE reduction', fontsize=12)
axes[1].set_title('Diminishing marginal returns', fontsize=13)
axes[1].set_xticks(n_values[1:])
axes[1].tick_params(labelsize=10)
axes[1].legend(fontsize=10, frameon=False)

plt.tight_layout(pad=1.5)
plt.savefig('select_bezier_order.png', dpi=300, bbox_inches='tight')
plt.show()
print("Plot saved to select_bezier_order.png")
