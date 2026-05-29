# Ball-Path Curvature and Free-Throw Shooting Proficiency in the NBA

> **One mechanical metric — terminal ball-path curvature — explains 33% of the between-player variance in NBA free-throw percentage.** This study turns coaching intuition into a scalable, objective signal derived directly from Second Spectrum ball tracking data.

![Free Throw Curvature](code/Klay%20Thompson_FT.png)

**Co-authored with Dave Love (NBA Shooting Coach) and Scott Powers (Rice University). Submitted to a peer-reviewed sports analytics journal.**

---

## The Key Finding

Shooting coaches have long said "smoother shots are better" — but smoother *where*? This study provides a data-driven answer:

**Late-path smoothness (terminal curvature) is what matters most.** The curvature of a player's ball path in the final moments before release — not the peak curvature earlier in the motion — is the strongest mechanical predictor of free-throw accuracy.

| Metric | Direction | R² | p-value |
|---|---|---|---|
| Max curvature | Negative | 0.12 | < 0.01 |
| Curvature integral (uniform weight) | Negative | 0.21 | < 0.01 |
| Time-weighted curvature integral (ℓ=3) | Negative | 0.29 | < 0.01 |
| **Time-weighted curvature integral (ℓ=6, CV-selected)** | **Negative** | **0.33** | **< 0.001** |

All four curvature metrics are statistically significant predictors of FT%. The cross-validated, release-weighted integral achieves the highest explanatory power — one mechanical feature accounting for a third of all between-player shooting variance in the NBA.

---

## Why This Matters for Basketball Operations

### Player Development
- Gives coaches a **quantitative target**: reduce terminal curvature, not peak curvature
- Challenges a widely held coaching belief: smoothing out the highest-curvature part of the motion (the initial forward push) is the wrong place to focus
- Enables objective before/after measurement of mechanical changes — no more relying on eye tests alone

### Scouting & Roster Construction
- Identifies mechanically sound shooters whose FT% may be suppressed by small-sample variance
- Flags players with poor terminal curvature before committing to long-term contracts
- Adds a physics-grounded dimension to shooting evaluation that standard stats miss entirely

### Draft Modeling
- Ball-path curvature metrics are measurable at any level with tracking data — college, G League, international
- Projects shooting translation by evaluating the *underlying mechanics*, not just surface stats from different competition levels

---

## Data & Scope

- **515 free throws** from **35 NBA players** in the 2023–24 regular season
- Source: **Second Spectrum** optical tracking system (25 frames/second, XYZ ball position)
- Season FT% drawn from Basketball Reference for regression targets
- Each player contributes ~15 tracked attempts; regression targets are full-season FT% to maximize statistical power

This study validates findings from controlled lab settings in real **in-game NBA conditions** — a meaningful step toward deploying curvature metrics operationally.

---

## Methods (Technical Summary)

1. **Path standardization.** Ball paths are trimmed to the shooting motion window: from the frame where the ball is furthest from the body (start of forward motion) to two frames after peak ball speed (release).
2. **Bézier curve fitting.** Each trajectory is modeled with an 8th-order Bézier curve, balancing fidelity to noisy tracking data against smoothness. Order selected via leave-one-out cross-validation.
3. **Curvature metrics.** Four metrics are computed per player: max curvature, uniform curvature integral, and time-weighted curvature integrals at ℓ=3 and ℓ=6 (ℓ controls the emphasis on late-path curvature). The ℓ=6 parameter was selected by cross-validation.
4. **Weighted least squares regression.** Each player's curvature metric is regressed against season FT% using WLS, with weights proportional to free-throw attempts to account for reliability differences.
5. **Consistency analysis.** A supplementary module measures shot-to-shot curvature consistency per player, testing whether mechanical repeatability — independent of average curvature — is also associated with accuracy.

---

## Repository Structure

```
free-throw-curvature/
├── articles/
│   └── san/                        # LaTeX source for journal submission
│       └── main.tex
├── articles/figures/               # Publication-ready figures
├── code/
│   ├── bezier_curve_functions.py   # Bézier fitting + curvature calculation
│   ├── metric_calculation.py       # Per-player curvature metrics pipeline
│   ├── result_analysis.py          # Regression + visualization
│   ├── consistency_analysis.py     # Shot-to-shot consistency module
│   ├── select_bezier_order.py      # LOO-CV for curve order selection
│   ├── select_l_parameter.py       # LOO-CV for ℓ (weighting) selection
│   ├── curvature_results.xlsx      # Output: per-player curvature metrics
│   ├── consistency_results.xlsx    # Output: per-player consistency metrics
│   └── free_throw_stats.xlsx       # 2023-24 NBA FT stats (ESPN/BBRef)
└── README.md
```

---

## Reproducing the Analysis

```bash
git clone https://github.com/judyz0415/free-throw-curvature.git
cd free-throw-curvature
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scipy matplotlib openpyxl

cd code
python metric_calculation.py      # regenerates curvature_results.xlsx
python result_analysis.py         # regressions + figures
python consistency_analysis.py    # consistency sandbox + scatter
```

---

## Paper

Zhu, J., Love, D., & Powers, S. (2025). *Ball path curvature and in-game free throw shooting proficiency in the National Basketball Association.* Manuscript submitted for publication.

```bibtex
@unpublished{zhu2025freethrowcurvature,
  title  = {Ball path curvature and in-game free throw shooting proficiency in the National Basketball Association},
  author = {Zhu, Judy and Love, Dave and Powers, Scott},
  year   = {2025},
  note   = {Manuscript submitted for publication. Code: https://github.com/judyz0415/free-throw-curvature}
}
```

---

## Contact

Judy Zhu — judy.zhu6052@gmail.com — [github.com/judyz0415](https://github.com/judyz0415)
