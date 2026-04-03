#!/usr/bin/env python3
"""
consistency_analysis.py

Analyses shot-to-shot curvature consistency for each player and answers:
  • Do better free-throw shooters have more consistent shot mechanics?
  • Can a player with sub-optimal curvature still succeed through consistency?

The core visualisation is a circular "sandbox":
  • Distance from centre  = performance rank by FT%
      (best shooter sits at the centre; worst at the perimeter)
  • Angular position       = curvature metric rank (σ_l at optimal ℓ)
      (players with similar shot shape cluster together angularly)
  • Circle radius          = SD of the curvature metric across shots
      (larger circle = more variable shot mechanics = less consistent)

Outputs
-------
  consistency_sandbox.png   — the circular sandbox figure
  consistency_scatter.png   — SD of curvature vs FT%, colour-coded by quadrant
  consistency_quadrant.png  — 2-D quadrant map (performance × consistency)
  consistency_results.xlsx  — per-player summary table
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.integrate import simpson
from numpy.linalg import lstsq

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bezier_curve_functions import find_end, bernstein_poly, bezier_curvature

# ── configuration ──────────────────────────────────────────────────────────────
FILE_TEMP  = '/Users/ruoqianzhu/Desktop/undergrad/freethrows/{} FTs.xlsx'
CODE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATS_FILE = os.path.join(CODE_DIR, 'free_throw_stats.xlsx')

N_BEZIER   = 8       # must match the order used in metric_calculation
L_OPTIMAL  = 6.0    # ℓ chosen by LOO-CV in select_l_parameter.py
T_INTERVAL = np.linspace(0, 1, 100)
CURV_WIN   = (0.40, 0.91)  # trimmed window (fraction of T_INTERVAL) for max curvature

plt.rcParams.update({
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

PLAYER_NAMES = [
    'Klay Thompson', 'Steph Curry', 'Bogdan Bogdanovic', 'Damian Lillard',
    'Anfernee Simons', 'Paul George', 'Derrick White', 'Tobias Harris',
    'Shai GilgeousAlexander', 'Karl Anthony Towns', 'Tim Hardaway Jr',
    'Luka Doncic', 'Dante Exum', 'Jaden Hardy', 'Alec Burks', 'Paolo Banchero',
    'Ivica Zubac', 'Derrick Jones Jr', 'Dwight Powell', 'Domantas Sabonis',
    'Maxi Kleber', 'Zion Williamson', 'Russell Westbrook', 'Josh Green',
    'Omax Prosper', 'Giannis Antetokounmpo', 'Aaron Gordon', 'Rudy Gobert',
    'Clint Capela', 'PJ Washington', 'Daniel Gafford', 'Andre Drummond',
    'Jakob Poeltl', 'Nic Claxton', 'Dereck Lively',
]

QUAD_COLORS = {
    'Elite Consistent':      '#1a9641',
    'Good but Variable':     '#a6d96a',
    'Consistent but Limited': '#fdae61',
    'Inconsistent':          '#d7191c',
}

# ── per-shot metric helpers ────────────────────────────────────────────────────

def _curvatures_for_shot(df: pd.DataFrame) -> np.ndarray:
    """Fit Bézier to one shot's trajectory and return 100-point curvature array."""
    df = df.copy()
    df['E_Distance'] = df['E_Distance'].abs()
    path_end   = find_end(df)
    df_trim    = df[['E_Distance', 'G_Height', 'F_Offset']].iloc[:path_end - 5]
    path_start = df_trim['E_Distance'].idxmax()
    seg        = df[['E_Distance', 'G_Height']].iloc[path_start:path_end]

    n  = N_BEZIER
    m  = len(seg)
    t  = np.linspace(0, 1, m)
    Tn = np.column_stack([bernstein_poly(i, n, t) for i in range(n + 1)])
    p, *_ = lstsq(Tn, seg.values, rcond=None)
    cps   = list(map(tuple, p))
    return np.array([bezier_curvature(cps, ti) for ti in T_INTERVAL])


def _sigma(curv: np.ndarray, l: float = L_OPTIMAL) -> float:
    """Weighted curvature integral σ_l for a single shot."""
    w = (l + 1) * T_INTERVAL ** l
    return float(simpson(w * curv, T_INTERVAL))


def _max_curv(curv: np.ndarray) -> float:
    s = int(CURV_WIN[0] * len(curv))
    e = int(CURV_WIN[1] * len(curv))
    return float(np.max(curv[s:e]))


def _classify(ft_pct: float, std: float, ft_med: float, sd_med: float) -> str:
    hi = ft_pct >= ft_med
    lo = std     <= sd_med
    if hi  and lo:  return 'Elite Consistent'
    if hi  and not lo: return 'Good but Variable'
    if not hi and lo:  return 'Consistent but Limited'
    return 'Inconsistent'


# ── Step 1: collect per-shot metrics ──────────────────────────────────────────
print("Computing per-shot curvature metrics …")
records = []
for player in PLAYER_NAMES:
    fpath = FILE_TEMP.format(player)
    if not os.path.exists(fpath):
        print(f"  [skip] {player} — file not found")
        continue
    sheets = pd.read_excel(fpath, sheet_name=None)
    shot_sigma, shot_mc = [], []
    for _, df_shot in sheets.items():
        try:
            c = _curvatures_for_shot(df_shot)
            shot_sigma.append(_sigma(c))
            shot_mc.append(_max_curv(c))
        except Exception:
            continue
    n = len(shot_sigma)
    if n < 2:
        print(f"  [skip] {player} — fewer than 2 valid shots")
        continue
    records.append({
        'Player':      player,
        'n_shots':     n,
        'mean_sigma':  np.mean(shot_sigma),
        'std_sigma':   np.std(shot_sigma, ddof=1),
        'cv_sigma':    np.std(shot_sigma, ddof=1) / np.mean(shot_sigma),
        'mean_mc':     np.mean(shot_mc),
        'std_mc':      np.std(shot_mc, ddof=1),
    })
    print(f"  {player}: {n} shots")

df = pd.DataFrame(records)

# merge game-performance stats
stats = pd.read_excel(STATS_FILE)
stats['Player'] = stats['Player'].str.strip().str.replace('-', ' ', regex=False)
df = df.merge(stats[['Player', 'FT%', 'FTA', 'FT% SE']], on='Player', how='left')
df = df.dropna(subset=['FT%', 'std_sigma']).reset_index(drop=True)
print(f"\n{len(df)} players available for analysis.\n")

# ── Step 2: quadrant labels ────────────────────────────────────────────────────
ft_med = df['FT%'].median()
sd_med = df['std_sigma'].median()
df['Quadrant'] = df.apply(
    lambda r: _classify(r['FT%'], r['std_sigma'], ft_med, sd_med), axis=1
)

# save summary table
out_xlsx = os.path.join(CODE_DIR, 'consistency_results.xlsx')
df.to_excel(out_xlsx, index=False)
print(f"Saved {out_xlsx}")

# ── Step 3: two single-metric sandboxes ───────────────────────────────────────
# Left  → distance from centre = FT% rank   (best FT% at centre)
# Right → distance from centre = σ₆ rank    (optimal σ₆ at centre)
# Both  → circle radius = SD of σ₆  (shot-to-shot consistency)
#          colour      = FT%
#          angle       = other metric (evenly spaced for visual separation only)
#
# Name rule: players whose surname token is 'Jr' use the preceding token instead
#   (e.g. "Derrick Jones Jr" → "Jones", "Tim Hardaway Jr" → "Hardaway").

def _repel_circles(x_arr, y_arr, r_arr, n_iter=700, alpha=0.45):
    """Iteratively resolve circle overlaps while preserving general positions."""
    x, y = x_arr.astype(float).copy(), y_arr.astype(float).copy()
    for _ in range(n_iter):
        moved = False
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                dx, dy = x[j] - x[i], y[j] - y[i]
                d = np.hypot(dx, dy)
                need = r_arr[i] + r_arr[j] + 0.015
                if 1e-9 < d < need:
                    push = (need - d) * alpha
                    ux, uy = dx / d, dy / d
                    x[i] -= ux * push * 0.5
                    y[i] -= uy * push * 0.5
                    x[j] += ux * push * 0.5
                    y[j] += uy * push * 0.5
                    moved = True
        if not moved:
            break
    return x, y


def _text_color(rgba):
    """White on dark fills, dark on light fills (WCAG relative luminance)."""
    lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return 'white' if lum < 0.52 else '#111111'


def _last_name(player: str) -> str:
    """Return displayable surname. Players ending in 'Jr' use the token before it."""
    parts = player.split()
    return parts[-2] if len(parts) > 1 and parts[-1] == 'Jr' else parts[-1]


def _draw_sandbox(ax, x_rep, y_rep, circle_r_ser, colors, df_local,
                  ring_specs, centre_label, title, std_max, MAX_CIRCLE_R,
                  PAD=1.20):
    """Render one sandbox panel."""
    ax.set_aspect('equal')
    ax.set_xlim(-PAD, PAD)
    ax.set_ylim(-PAD, PAD)
    ax.set_facecolor('#f7f7f4')

    # arena rings
    for rc, lbl in ring_specs:
        ax.add_patch(mpatches.Circle(
            (0, 0), rc, fill=False,
            color='#c8c8c8', linewidth=0.9, linestyle='--', zorder=1,
        ))
        ax.text(0, rc + 0.022, lbl,
                ha='center', va='bottom', fontsize=7, color='#aaaaaa', zorder=2)

    # centre marker
    ax.plot(0, 0, '*', markersize=13, color='gold', markeredgecolor='#888', zorder=6)
    ax.text(0, -0.07, centre_label,
            ha='center', va='top', fontsize=7, color='#555',
            fontstyle='italic', zorder=6)

    MIN_INSIDE_R = 0.046

    for i in range(len(df_local)):
        xp, yp = float(x_rep[i]), float(y_rep[i])
        rp     = float(circle_r_ser.iloc[i])
        col    = colors[i]
        name   = _last_name(df_local.loc[i, 'Player'])
        tc     = _text_color(col)

        ax.add_patch(mpatches.Circle(
            (xp, yp), rp, facecolor=col, alpha=0.68,
            linewidth=1.0, edgecolor='k', zorder=4,
        ))

        if rp >= MIN_INSIDE_R:
            fs = max(5.0, min(9.0, rp * 108))
            ax.text(xp, yp, name,
                    ha='center', va='center', fontsize=fs,
                    fontweight='bold', color=tc, zorder=5)
        else:
            dist = np.hypot(xp, yp) + 1e-9
            ux, uy = xp / dist, yp / dist
            ox = xp + (rp + 0.030) * ux
            oy = yp + (rp + 0.030) * uy
            ax.plot([xp + rp * ux, ox - 0.008 * ux],
                    [yp + rp * uy, oy - 0.008 * uy],
                    '-', color='#999', lw=0.5, zorder=3)
            ax.text(ox, oy, name,
                    ha='center', va='center', fontsize=5.0, color='#222', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.10', fc='white', alpha=0.82, lw=0))

    # size legend (bottom-left)
    lx, ly = -PAD + 0.04, -PAD + 0.48
    ax.text(lx, ly + 0.02, 'Circle radius = SD of σ₆',
            fontsize=6.5, color='#555', va='bottom', fontweight='bold')
    for k, frac in enumerate([0.25, 0.50, 1.00]):
        lv = std_max * frac
        lr = (lv / std_max) * MAX_CIRCLE_R
        cx = lx + 0.13 + k * 0.27
        cy = ly - lr
        ax.add_patch(mpatches.Circle((cx, cy), lr, facecolor='#aaaaaa',
                                      alpha=0.50, lw=0.7, edgecolor='k', zorder=2))
        ax.text(cx, cy - lr - 0.025, f'SD={lv:.4f}',
                ha='center', va='top', fontsize=5.5, color='#555')

    ax.set_title(title, fontsize=10.5, pad=10)
    ax.axis('off')


# ── shared parameters ──────────────────────────────────────────────────────────
ft_min, ft_max   = df['FT%'].min(), df['FT%'].max()
sig_min, sig_max = df['mean_sigma'].min(), df['mean_sigma'].max()
std_max          = df['std_sigma'].max()
ARENA_R          = 1.0
MAX_CIRCLE_R     = 0.13
N                = len(df)

cmap   = plt.cm.RdYlGn
norm   = Normalize(vmin=ft_min, vmax=ft_max)
colors = cmap(norm(df['FT%'].values))
circle_r = (df['std_sigma'] / std_max) * MAX_CIRCLE_R

# ── FT% sandbox ────────────────────────────────────────────────────────────────
# r = 1 − norm(FT%) → highest FT% sits at centre
# θ = σ₆ rank, evenly spaced (spreads players around the ring for readability)
r_ft    = ARENA_R * (1.0 - (df['FT%'] - ft_min) / (ft_max - ft_min))
theta_ft = (df['mean_sigma'].rank(method='first').values - 1) / N * 2 * np.pi + np.pi / 2
x_ft_rep, y_ft_rep = _repel_circles(
    (r_ft * np.cos(theta_ft)).values,
    (r_ft * np.sin(theta_ft)).values,
    circle_r.values,
)

ring_specs_ft = [
    (0.25 * ARENA_R, f'Top 25%   (≥ {ft_min + 0.75*(ft_max-ft_min):.1f}%)'),
    (0.50 * ARENA_R, f'Median    (≥ {ft_med:.1f}%)'),
    (0.75 * ARENA_R, 'Bottom 25%'),
    (1.00 * ARENA_R, f'Worst  ({ft_min:.1f}%)'),
]

# ── σ₆ sandbox ─────────────────────────────────────────────────────────────────
# Determine which end of σ₆ predicts better FT% (negative corr → lower is better)
sig_ft_corr = float(np.corrcoef(df['FT%'], df['mean_sigma'])[0, 1])
if sig_ft_corr < 0:
    # lower σ₆ → better FT% → put lowest σ₆ at centre
    r_sig        = ARENA_R * (df['mean_sigma'] - sig_min) / (sig_max - sig_min)
    centre_sig   = f'Low σ₆  ({sig_min:.4f})'
    sig_q_labels = [
        (0.25, f'≤ {df["mean_sigma"].quantile(0.25):.4f}'),
        (0.50, f'median {df["mean_sigma"].median():.4f}'),
        (0.75, 'Bottom 25%'),
        (1.00, f'High σ₆  ({sig_max:.4f})'),
    ]
else:
    # higher σ₆ → better FT% → put highest σ₆ at centre
    r_sig        = ARENA_R * (1.0 - (df['mean_sigma'] - sig_min) / (sig_max - sig_min))
    centre_sig   = f'High σ₆  ({sig_max:.4f})'
    sig_q_labels = [
        (0.25, f'≥ {df["mean_sigma"].quantile(0.75):.4f}'),
        (0.50, f'median {df["mean_sigma"].median():.4f}'),
        (0.75, 'Bottom 25%'),
        (1.00, f'Low σ₆  ({sig_min:.4f})'),
    ]
ring_specs_sig = [(rc * ARENA_R, lbl) for rc, lbl in sig_q_labels]

theta_sig = (df['FT%'].rank(method='first').values - 1) / N * 2 * np.pi + np.pi / 2
x_sig_rep, y_sig_rep = _repel_circles(
    (r_sig * np.cos(theta_sig)).values,
    (r_sig * np.sin(theta_sig)).values,
    circle_r.values,
)

# ── figure ─────────────────────────────────────────────────────────────────────
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(24, 13))
fig.patch.set_facecolor('#f7f7f4')
gs = GridSpec(1, 2, figure=fig, wspace=0.06,
              left=0.02, right=0.95, top=0.90, bottom=0.04)
ax_ft  = fig.add_subplot(gs[0])
ax_sig = fig.add_subplot(gs[1])

_draw_sandbox(
    ax_ft, x_ft_rep, y_ft_rep, circle_r, colors, df,
    ring_specs_ft,
    centre_label='Best FT%',
    title='FT% Performance Sandbox\n'
          'Distance from centre = FT% rank  •  Circle radius = SD of σ₆',
    std_max=std_max, MAX_CIRCLE_R=MAX_CIRCLE_R,
)

_draw_sandbox(
    ax_sig, x_sig_rep, y_sig_rep, circle_r, colors, df,
    ring_specs_sig,
    centre_label=centre_sig,
    title='Curvature Metric (σ₆) Sandbox\n'
          'Distance from centre = σ₆ rank  •  Circle radius = SD of σ₆',
    std_max=std_max, MAX_CIRCLE_R=MAX_CIRCLE_R,
)

# shared colourbar
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = fig.colorbar(sm, ax=[ax_ft, ax_sig], fraction=0.015, pad=0.01, shrink=0.55)
cb.set_label('FT%', fontsize=11)

corr_sign = 'negative' if sig_ft_corr < 0 else 'positive'
plt.suptitle(
    f'Player Consistency Sandboxes  '
    f'(σ₆ ↔ FT% correlation: r = {sig_ft_corr:.3f}, {corr_sign})',
    fontsize=13, fontweight='bold', y=0.97,
)

sandbox_path = os.path.join(CODE_DIR, 'consistency_sandbox.png')
plt.savefig(sandbox_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved {sandbox_path}")

# ── Step 4: scatter — consistency vs performance ───────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6.5))

scatter_specs = [
    ('std_sigma', f'SD of σ₆  (absolute shot-to-shot spread)',  'Absolute Consistency vs Performance'),
    ('cv_sigma',  'CV of σ₆  (SD / mean, scale-free)',          'Relative Consistency vs Performance'),
]

for ax2, (ycol, ylabel, title) in zip(axes2, scatter_specs):
    for q, grp in df.groupby('Quadrant'):
        ax2.scatter(
            grp['FT%'], grp[ycol],
            s=55, label=q,
            color=QUAD_COLORS[q], edgecolors='k', linewidths=0.5, zorder=3,
        )
    for _, row in df.iterrows():
        ax2.annotate(
            row['Player'], (row['FT%'], row[ycol]),
            xytext=(3, 0), textcoords='offset points',
            fontsize=5.5, va='center', color='#333',
        )
    # median grid lines
    ax2.axvline(ft_med, color='gray', lw=0.9, ls='--', alpha=0.55, label=f'Median FT% ({ft_med:.1f}%)')
    ax2.axhline(df[ycol].median(), color='steelblue', lw=0.9, ls='--', alpha=0.55,
                label=f'Median {ycol.split("_")[0].upper()}')
    # OLS trend
    coef = np.polyfit(df['FT%'], df[ycol], 1)
    xfit = np.linspace(df['FT%'].min(), df['FT%'].max(), 200)
    ax2.plot(xfit, np.polyval(coef, xfit), 'k-', lw=1.5, alpha=0.50, label='OLS trend')
    corr = np.corrcoef(df['FT%'], df[ycol])[0, 1]
    ax2.text(0.97, 0.97, f'r = {corr:.3f}',
             transform=ax2.transAxes, ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, lw=0.5))
    ax2.set_xlabel('FT%', fontsize=12)
    ax2.set_ylabel(ylabel, fontsize=11)
    ax2.set_title(title, fontsize=12)
    ax2.legend(fontsize=7, frameon=False, loc='upper left')

plt.tight_layout(pad=2.0)
scatter_path = os.path.join(CODE_DIR, 'consistency_scatter.png')
plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved {scatter_path}")

# ── Step 5: quadrant map ───────────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(11, 8.5))

for q, grp in df.groupby('Quadrant'):
    ax3.scatter(
        grp['FT%'], grp['std_sigma'],
        s=75, label=q, color=QUAD_COLORS[q],
        edgecolors='k', linewidths=0.5, zorder=3,
    )
    for _, row in grp.iterrows():
        ax3.annotate(
            row['Player'], (row['FT%'], row['std_sigma']),
            xytext=(4, 0), textcoords='offset points',
            fontsize=6.5, va='center',
        )

# quadrant dividers
ax3.axvline(ft_med, color='gray', lw=1.0, ls='--', alpha=0.7)
ax3.axhline(sd_med, color='gray', lw=1.0, ls='--', alpha=0.7)

# quadrant background shading
xlim = ax3.get_xlim() if ax3.get_xlim()[0] != 0 else (df['FT%'].min() - 2, df['FT%'].max() + 2)
ylim = ax3.get_ylim() if ax3.get_ylim()[0] != 0 else (0, df['std_sigma'].max() * 1.15)
ax3.set_xlim(df['FT%'].min() - 2, df['FT%'].max() + 2)
ax3.set_ylim(0, df['std_sigma'].max() * 1.15)
xl, xr = ax3.get_xlim()
yb, yt = ax3.get_ylim()
quad_patches = [
    (xl, sd_med, ft_med - xl, yt - sd_med, '#1a9641', 'Elite Consistent'),
    (ft_med, sd_med, xr - ft_med, yt - sd_med, '#a6d96a', 'Good but Variable'),
    (xl, yb, ft_med - xl, sd_med - yb, '#fdae61', 'Consistent but Limited'),
    (ft_med, yb, xr - ft_med, sd_med - yb, '#d7191c', 'Inconsistent'),
]
for qx, qy, qw, qh, qc, _ in quad_patches:
    ax3.add_patch(mpatches.FancyBboxPatch(
        (qx, qy), qw, qh, boxstyle='square,pad=0',
        color=qc, alpha=0.08, zorder=0,
    ))

# quadrant annotation labels
quad_text = [
    (xl + (ft_med - xl) * 0.5, sd_med + (yt - sd_med) * 0.92, 'Elite Consistent',      '#1a9641'),
    (ft_med + (xr - ft_med) * 0.5, sd_med + (yt - sd_med) * 0.92, 'Good but Variable', '#5a8f28'),
    (xl + (ft_med - xl) * 0.5, yb + (sd_med - yb) * 0.05, 'Consistent\nbut Limited',   '#c57b00'),
    (ft_med + (xr - ft_med) * 0.5, yb + (sd_med - yb) * 0.05, 'Inconsistent',          '#a00000'),
]
for tx, ty, tlbl, tc in quad_text:
    ax3.text(tx, ty, tlbl, ha='center', va='bottom', fontsize=9,
             color=tc, fontstyle='italic', alpha=0.8)

ax3.set_xlabel('FT%', fontsize=13)
ax3.set_ylabel(f'SD of σ₆  (shot-to-shot consistency radius)', fontsize=12)
ax3.set_title(
    'Performance × Consistency Quadrant Map\n'
    f'Median FT% = {ft_med:.1f}%  •  Median SD of σ₆ = {sd_med:.4f}',
    fontsize=13, pad=12,
)
ax3.legend(fontsize=8, frameon=False, loc='upper right')
plt.tight_layout()
quadrant_path = os.path.join(CODE_DIR, 'consistency_quadrant.png')
plt.savefig(quadrant_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved {quadrant_path}")

# ── Step 6: print summary table ───────────────────────────────────────────────
print("\n── Consistency Summary ─────────────────────────────────────────────────────")
display_cols = ['Player', 'FT%', 'n_shots', 'mean_sigma', 'std_sigma', 'cv_sigma', 'Quadrant']
print(
    df.sort_values('FT%', ascending=False)[display_cols]
    .to_string(index=False, float_format='{:.4f}'.format)
)

print("\n── Quadrant counts ──────────────────────────────────────────────────────────")
print(df['Quadrant'].value_counts().to_string())

print("\n── Correlation: FT%  vs  SD of σ₆ ──────────────────────────────────────────")
r_val = np.corrcoef(df['FT%'], df['std_sigma'])[0, 1]
print(f"  Pearson r = {r_val:.4f}")
if r_val < -0.3:
    print("  → Better shooters tend to have MORE consistent mechanics (lower SD).")
elif r_val > 0.3:
    print("  → Surprisingly, better shooters show MORE variable mechanics (higher SD).")
else:
    print("  → No strong linear relationship between FT% and shot-to-shot consistency.")
