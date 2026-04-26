"""
FT_Visual.py
Premium Free Throw Curvature Dashboard.

Computation pipeline mirrors metric_calculation.py exactly:
  · bernstein_poly / bezier_curve / bezier_curvature  from bezier_curve_functions.py
  · path_start = df_copy['E_Distance'].idxmax()
  · path_end   = find_end(df)  (peak_velocity_index + 2)
  · N_BEZIER   = 8
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib import rcParams
from numpy.linalg import lstsq

from bezier_curve_functions import bernstein_poly, bezier_curve, bezier_curvature, find_end

# ── Palette ────────────────────────────────────────────────────────────────────
NAVY    = '#0B1F3A'
GOLD    = '#C9A84C'
WHITE   = '#FFFFFF'
OFF_WHT = '#F0F4F8'    
GRAY_LT = '#E2E8F0'
GRAY_MD = '#94A3B8'
GRAY_DK = '#334155'
BLUE    = '#1D4ED8'
RED     = '#B91C1C'
AMBER   = '#B45309'
BG      = '#0D1827'   

rcParams.update({'font.family': 'DejaVu Sans'})

# ── Config ─────────────────────────────────────────────────────────────────────
FILE_PATH = '/Users/ruoqianzhu/Desktop/undergrad/freethrows/Klay Thompson FTs.xlsx'
N_BEZIER  = 8          # must match metric_calculation.py
TIME_INT  = 1 / 25    # seconds per frame
N_CURVE   = 600        # points for smooth bezier evaluation

# ── Load data ──────────────────────────────────────────────────────────────────
player_name = (os.path.basename(FILE_PATH)
               .replace(' FTs.xlsx', '').replace('FTs.xlsx', ''))

all_sheets = pd.read_excel(FILE_PATH, sheet_name=None)
df_raw, sheet_label = None, ''
for sname, sdf in all_sheets.items():
    if 'E_Distance' in sdf.columns:
        df_raw, sheet_label = sdf.copy(), str(sname)
        break
if df_raw is None:
    raise ValueError("No sheet with 'E_Distance' column found.")

df_raw['E_Distance'] = df_raw['E_Distance'].abs()

# ── Crop path — identical to metric_calculation.py ─────────────────────────────
path_end   = find_end(df_raw)
df_copy    = df_raw[['E_Distance', 'G_Height', 'F_Offset']].iloc[:max(1, path_end - 5)]
path_start = int(df_copy['E_Distance'].idxmax())
df_shot    = (df_raw[['E_Distance', 'G_Height', 'F_Offset']]
              .iloc[path_start:path_end]
              .reset_index(drop=True))
m = len(df_shot)

# ── 3-D velocity ────────────────────────────────────────────────────────────────
def vel3d(dframe):
    v = [np.nan]
    for i in range(1, len(dframe)):
        d = np.sqrt(
            (dframe.iloc[i]['E_Distance'] - dframe.iloc[i-1]['E_Distance'])**2 +
            (dframe.iloc[i]['F_Offset']   - dframe.iloc[i-1]['F_Offset'])**2 +
            (dframe.iloc[i]['G_Height']   - dframe.iloc[i-1]['G_Height'])**2
        )
        v.append(d / TIME_INT)
    return np.array(v)

# ── Velocity ──────────────────────────────────────────────────────────────────
# Frame-by-frame 3-D speed (ft/s). First entry is always NaN (no prior frame).
vel_raw = vel3d(df_raw)

# ── Release frame detection ───────────────────────────────────────────────────
# Each clip in df_raw contains a pre-release windup phase (slow arm motion,
# typically < 12 ft/s) followed by an abrupt jump to free-flight speed (~ 20 ft/s)
# at the moment the ball leaves the hand.  We detect the release as the FIRST
# frame whose speed exceeds 65 % of the global peak speed in the recording.
#
# The 65 % threshold comfortably separates arm-motion speed from flight speed
# for any valid free-throw attempt while remaining robust to sensor noise.
# Falls back to path_start if no frame clears the threshold (degenerate clips).
max_vel_full  = float(np.nanmax(vel_raw))
_rel_thresh   = 0.65 * max_vel_full
_rel_cands    = [i for i in range(len(vel_raw))
                 if not np.isnan(vel_raw[i]) and vel_raw[i] >= _rel_thresh]
release_frame = _rel_cands[0] if _rel_cands else path_start

# Release velocity: 3-frame forward mean starting at the release frame.
# Forward-only averaging avoids including slow windup frames just before release.
_rf_hi      = min(len(vel_raw), release_frame + 3)
release_vel = float(np.nanmean(vel_raw[release_frame:_rf_hi]))

# Release height: G_Height at the identified release frame.
release_h = float(df_raw['G_Height'].iloc[release_frame])
apex_h    = float(df_shot['G_Height'].max())

# ── Shot distance ─────────────────────────────────────────────────────────────
# 2-D Euclidean horizontal displacement from the release frame to the last
# tracked frame in the recording (the closest available proxy for basket entry,
# since df_raw covers the full clip including post-release flight).
# Translation-invariant: does not rely on the absolute court-coordinate origin.
shot_dist = float(np.sqrt(
    (df_raw['E_Distance'].iloc[-1] - df_raw['E_Distance'].iloc[release_frame])**2 +
    (df_raw['F_Offset'].iloc[-1]   - df_raw['F_Offset'].iloc[release_frame])**2
))

# ── Release angle ─────────────────────────────────────────────────────────────
# df_shot = frames [path_start, path_end), which for these clips spans the
# windup through the release instant (path_end ≈ peak-velocity frame + 1).
# The LAST 3 frames of df_shot therefore correspond to the actual release moment
# (ball just leaving the hand), giving the true launch direction.
# Raw tracking frames are used to avoid Bézier Runge artefacts at t = 1.
if m >= 3:
    de = float(df_shot['E_Distance'].iloc[-1] - df_shot['E_Distance'].iloc[-3])
    dh = float(df_shot['G_Height'].iloc[-1]   - df_shot['G_Height'].iloc[-3])
elif m >= 2:
    de = float(df_shot['E_Distance'].iloc[-1] - df_shot['E_Distance'].iloc[-2])
    dh = float(df_shot['G_Height'].iloc[-1]   - df_shot['G_Height'].iloc[-2])
else:
    de, dh = 1.0, 1.0
rel_ang = float(np.degrees(np.arctan2(dh, de)))

# ── Bézier fit: side (E_Distance × G_Height) and lateral (F_Offset × G_Height) ─
n   = N_BEZIER
t_  = np.linspace(0, 1, m)
T_n = np.column_stack([bernstein_poly(i, n, t_) for i in range(n + 1)])

p_s, *_ = lstsq(T_n, df_shot[['E_Distance', 'G_Height']].values, rcond=None)
ctrl_s   = list(map(tuple, p_s))

p_l, *_ = lstsq(T_n, df_shot[['F_Offset', 'G_Height']].values, rcond=None)
ctrl_l   = list(map(tuple, p_l))

# ── Evaluate curves & curvature ─────────────────────────────────────────────────
t_fine = np.linspace(0, 1, N_CURVE)
sx, sy = bezier_curve(ctrl_s, num_points=N_CURVE)
lx, ly = bezier_curve(ctrl_l, num_points=N_CURVE)
curv_s = np.array([bezier_curvature(ctrl_s, t) for t in t_fine])
curv_l = np.array([bezier_curvature(ctrl_l, t) for t in t_fine])

# Peak curvature in trimmed window [0.4, 0.91] — matches metric_calculation.py
trim  = (t_fine >= 0.4) & (t_fine <= 0.91)
pt_s  = float(t_fine[trim][np.argmax(curv_s[trim])])
pk_s  = float(curv_s[trim].max())
pt_l  = float(t_fine[trim][np.argmax(curv_l[trim])])
pk_l  = float(curv_l[trim].max())

def at(xv, yv, tv):
    return float(np.interp(tv, t_fine, xv)), float(np.interp(tv, t_fine, yv))

pk_s_xy = at(sx, sy, pt_s)
pk_l_xy = at(lx, ly, pt_l)

# ── LineCollection: Bézier curve colored by curvature (plasma) ─────────────────
def curvature_lc(xv, yv, curvs, lw=3.8):
    pts  = np.array([xv, yv]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    vmax = float(max(np.percentile(curvs, 95), 1e-6))
    norm = plt.Normalize(vmin=0, vmax=vmax)
    lc   = LineCollection(segs, cmap='plasma', norm=norm,
                          linewidth=lw, zorder=4, capstyle='round')
    lc.set_array(0.5 * (curvs[:-1] + curvs[1:]))
    return lc

# ── Trajectory axis styling ────────────────────────────────────────────────────
def traj_style(ax, title, xlabel, ylabel):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=13, fontweight='bold', color=GOLD, pad=16)
    ax.set_xlabel(xlabel, fontsize=10, color=BG, labelpad=5)
    ax.set_ylabel(ylabel, fontsize=10, color=BG, labelpad=5)
    ax.tick_params(colors='#5A7A99', labelsize=9, length=3)
    ax.grid(True, color='#172540', linewidth=0.8, linestyle='--', zorder=0)
    for sp in ax.spines.values():
        sp.set_edgecolor('#1A3050')
        sp.set_linewidth(0.8)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 11), facecolor=WHITE, dpi=150)

# Header
hdr = fig.add_axes([0.0, 0.910, 1.0, 0.090])
hdr.set_facecolor(NAVY); hdr.axis('off')
fig.add_axes([0.0, 0.906, 1.0, 0.006]).set(facecolor=GOLD)
plt.gca().axis('off')

hdr.text(0.015, 0.78, 'FREE THROW CURVATURE ANALYSIS',
         transform=hdr.transAxes, color=GOLD, fontsize=40, fontweight='bold', va='center')
hdr.text(0.015, 0.27, player_name.upper(),
         transform=hdr.transAxes, color=WHITE, fontsize=40, fontweight='bold', va='center')

# Footer
ftr = fig.add_axes([0.0, 0.0, 1.0, 0.040])
ftr.set_facecolor(OFF_WHT); ftr.axis('off')
ftr.axhline(0.88, color=GRAY_LT, linewidth=1.1)
ftr.text(0.5, 0.35,
         r'Free Throw Curvature Analysis  ·  '
         r'Bézier Trajectory Modeling  ·  '
         r'Curvature  κ = |x′y″ − y′x″| / (x′² + y′²)^{3/2}  ·  '
         r'github.com/judyz0415/free-throw-curvature',
         transform=ftr.transAxes, color=BG, fontsize=8, ha='center', va='center')

# ── 3-column GridSpec: trajectory | card | trajectory ─────────────────────────
gs = GridSpec(1, 3, figure=fig,
              left=0.03, right=0.97, bottom=0.055, top=0.903,
              wspace=0.08, width_ratios=[5, 3, 5])

ax_side = fig.add_subplot(gs[0, 0])
ax_card = fig.add_subplot(gs[0, 1])
ax_lat  = fig.add_subplot(gs[0, 2])

# ══════════════════════════════════════════════════════════════════════════════
# Panel 1 — Side Trajectory
# ══════════════════════════════════════════════════════════════════════════════
traj_style(ax_side, 'Trajectory  ·  Side View', 'Distance (ft)', 'Height (ft)')

ax_side.scatter(df_shot['E_Distance'], df_shot['G_Height'],
                color='#A0C4E8', s=32, zorder=3, alpha=0.75, label='Tracking Points')

# Control points only (no connecting polygon lines)
cx_s = [p[0] for p in ctrl_s]; cy_s = [p[1] for p in ctrl_s]
ax_side.scatter(cx_s, cy_s, color=GOLD, s=55, zorder=5,
                edgecolors='#1A0D00', lw=0.8, alpha=0.85, label='Control Points')

# Bézier curve colored by curvature
lc_s = curvature_lc(sx, sy, curv_s, lw=4.0)
ax_side.add_collection(lc_s)

# Axis limits
xp, yp = 0.35, 0.6
# x_lo = df_shot['E_Distance'].min() - xp
# x_hi = df_shot['E_Distance'].max() + xp
y_lo = df_shot['G_Height'].min() - yp
y_hi = df_shot['G_Height'].max() + yp
ax_side.set_xlim(26, 31)
ax_side.set_ylim(y_lo, y_hi)


# Inset colorbar — upper-right corner, above the descending curve
cax_s = ax_side.inset_axes([0.80, 0.10, 0.04, 0.33])
cb_s  = fig.colorbar(lc_s, cax=cax_s)
cb_s.set_label('κ  (1/ft)', fontsize=7.5, color=GRAY_MD, labelpad=4)
cb_s.ax.tick_params(labelsize=7, colors=GRAY_MD, length=2)
cb_s.ax.yaxis.set_label_position('left')
cb_s.ax.yaxis.tick_left()
cb_s.outline.set_edgecolor('#1A3050')

ax_side.legend(fontsize=11, framealpha=0.15, edgecolor='#1A3050',
               facecolor=BG, loc='upper left', labelcolor='#A8CCEC',
               handlelength=1.4, borderpad=0.7)

# ══════════════════════════════════════════════════════════════════════════════
# Panel 2 — Shot Profile Card  (light background, dark text)
# ══════════════════════════════════════════════════════════════════════════════
ax_card.set_facecolor(OFF_WHT)
ax_card.set_xlim(0, 1); ax_card.set_ylim(0, 1); ax_card.axis('off')
for sp in ax_card.spines.values(): sp.set_visible(False)

# Gold top border
ax_card.plot([0, 1], [1.0, 1.0], color=GOLD, lw=3.5,
             transform=ax_card.transAxes, zorder=3, clip_on=False)

# Navy header strip
ax_card.add_patch(mpatches.FancyBboxPatch(
    (0.0, 0.878), 1.0, 0.122, boxstyle='square,pad=0', lw=0,
    facecolor=NAVY, transform=ax_card.transAxes, zorder=2))

ax_card.text(0.5, 0.966, 'SHOT  PROFILE',
             transform=ax_card.transAxes, fontsize=13, fontweight='bold',
             color=GOLD, ha='center', va='center', zorder=3)
ax_card.text(0.5, 0.910, 'Key Shooting Mechanics',
             transform=ax_card.transAxes, fontsize=8,
             color=GRAY_MD, ha='center', va='center', zorder=3)

# Thin gold separator below header
ax_card.plot([0.05, 0.95], [0.876, 0.876], color=GOLD, lw=1.0,
             transform=ax_card.transAxes, alpha=0.50, zorder=3)

# ── Metric rows ──────────────────────────────────────────────────────────────
metrics = [
    ('Release Velocity',  f'{release_vel:.1f}',   'ft/s',  BLUE),
    ('Release Height',    f'{release_h:.2f}',  'ft',    NAVY),
    ('Release Angle',     f'{rel_ang:.1f}°',   '',      NAVY),
    ('Apex Height',       f'{apex_h:.2f}',     'ft',    NAVY),
    ('Shot Distance',     f'{shot_dist:.2f}',  'ft',    NAVY),
    ('Peak Side  κ',      f'{pk_s:.3f}',       '1/ft',  RED),
    ('Peak Lateral  κ',   f'{pk_l:.3f}',       '1/ft',  AMBER),
    ('Bézier Order',      f'{N_BEZIER}',       'n',     '#1A56A0'),
]

y0, step = 0.845, 0.103
for i, (lbl, val, unit, col) in enumerate(metrics):
    y = y0 - i * step

    # Alternating row tint
    if i % 2 == 0:
        ax_card.add_patch(mpatches.FancyBboxPatch(
            (0.0, y - 0.078), 1.0, 0.098,
            boxstyle='square,pad=0', lw=0, facecolor='#E2EAF4', alpha=0.55,
            transform=ax_card.transAxes, zorder=1))

    # Separator line
    ax_card.plot([0.05, 0.95], [y - 0.078, y - 0.078],
                 color=GRAY_LT, lw=0.8, zorder=2, transform=ax_card.transAxes)

    # Label
    ax_card.text(0.08, y - 0.004, lbl,
                 transform=ax_card.transAxes, fontsize=8,
                 color=GRAY_MD, va='top', fontweight='bold')

    # Value (large, color-coded)
    ax_card.text(0.08, y - 0.040, val,
                 transform=ax_card.transAxes, fontsize=19,
                 color=col, va='top', fontweight='bold')

    # Unit
    if unit:
        ax_card.text(0.82, y - 0.025, unit,
                     transform=ax_card.transAxes, fontsize=9.5,
                     color=GRAY_MD, va='top')

# Applications footer
ax_card.plot([0.05, 0.95], [0.025, 0.025], color=GRAY_LT, lw=0.8,
             transform=ax_card.transAxes)
ax_card.text(0.5, 0.013,
             'Player Dev  ·  Scouting  ·  Draft Modeling',
             transform=ax_card.transAxes, fontsize=7.5, color=GRAY_MD,
             ha='center', va='center', style='italic')
# ══════════════════════════════════════════════════════════════════════════════
# Panel 3 — Lateral Trajectory
# ══════════════════════════════════════════════════════════════════════════════
traj_style(ax_lat, 'Trajectory  ·  Lateral View', 'Lateral Offset (ft)', 'Height (ft)')

# ── Tracking points ───────────────────────────────────────────────────────────
ax_lat.scatter(df_shot['F_Offset'], df_shot['G_Height'],
               color='#A0C4E8', s=32, zorder=3, alpha=0.75,
               label='Tracking Points')

# ── Control points (no connecting lines) ──────────────────────────────────────
cx_l = [p[0] for p in ctrl_l]
cy_l = [p[1] for p in ctrl_l]

ax_lat.scatter(cx_l, cy_l,
               color=GOLD, s=55, zorder=5,
               edgecolors='#1A0D00', lw=0.8, alpha=0.85,
               label='Control Points')

# ── Bézier curvature line ─────────────────────────────────────────────────────
lc_l = curvature_lc(lx, ly, curv_l, lw=4.0)
ax_lat.add_collection(lc_l)

# ── Zoom based on ACTUAL curve (not control points) ───────────────────────────
x_min, x_max = lx.min(), lx.max()
y_min, y_max = ly.min(), ly.max()

x_pad = (x_max - x_min) * 0.18
y_pad = (y_max - y_min) * 0.20

ax_lat.set_xlim(-2, 1)
ax_lat.set_ylim(max(3, y_min - y_pad), min(12, y_max + y_pad))

# ── Inset colorbar ────────────────────────────────────────────────────────────
cax_l = ax_lat.inset_axes([0.80, 0.10, 0.04, 0.33])
cb_l = fig.colorbar(lc_l, cax=cax_l)

cb_l.set_label('κ  (1/ft)', fontsize=7.5, color=GRAY_MD, labelpad=4)
cb_l.ax.tick_params(labelsize=7, colors=GRAY_MD, length=2)
cb_l.ax.yaxis.set_label_position('left')
cb_l.ax.yaxis.tick_left()
cb_l.outline.set_edgecolor('#1A3050')

# ── Legend ─────────────────────────────────────────────────────────────────────
ax_lat.legend(fontsize=11, framealpha=0.15, edgecolor='#1A3050',
              facecolor=BG, loc='upper right',
              labelcolor='#A8CCEC',
              handlelength=1.4, borderpad=0.7)

# ── Save ───────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
out_dir  = BASE_DIR
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f'{player_name}_FT.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=WHITE)
print(f'Saved → {out_path}')

repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
repo_path = os.path.join(repo_root, 'image.png')
plt.savefig(repo_path, dpi=150, bbox_inches='tight', facecolor=WHITE)
print(f'Saved → {os.path.abspath(repo_path)}')
plt.close()
