# Free-Throw Curvature

> Does the arc of the ball matter? A quantitative study of ball-path curvature and free-throw shooting proficiency in the NBA.

## TL;DR

We reconstruct NBA free-throw trajectories using Bézier curves, engineer curvature-based shooting metrics, and test how they relate to 2023–24 free-throw performance. End-to-end pipeline included: cleaned data, modeling, validation, and publication-ready visuals.

## Why this matters for basketball operations

Shooting mechanics aren’t just visual. They’re quantifiable.
Curvature-based metrics offer a new, objective lens on shot quality that complements film and scouting intuition.

**Applications:**

**Player development**: diagnose and refine shooting form
**Scouting & acquisition**: identify mechanically sound shooters beyond surface stats
**Draft modeling**: project shooting translation with physics-informed features

## Repository structure

```
free-throw-curvature/
├── articles/           # LaTeX source + compiled PDF of the research paper
├── code/               # Python modules and analysis scripts
│   ├── __init__.py
│   ├── bezier_curve_functions.py   # Bézier curve fitting + curvature calculation
│   ├── metric_calculation.py       # Computes curvature metrics for all players
│   ├── result_analysis.py          # Merging, statistical analysis, visualization
│   ├── curvature_results.xlsx      # Output: per-player curvature metrics
│   ├── free_throw_stats.xlsx       # 2023-24 NBA FT stats (source: ESPN.com)
│   ├── distributions.png           # Histograms of curvature metrics
│   └── regressionlines.png         # Regressions vs. FT performance
├── image.png           # Hero figure
├── .gitignore
├── LICENSE
└── README.md
```

## Methods

1. **Trajectory capture.** Ball paths for free-throw attempts are parameterized from Second Spectrum ball tracking data.
2. **Curve fitting.** Each path is modeled with a Bézier curve (`bezier_curve_functions.py`), trading off fidelity and smoothness.
3. **Metric calculation.** Per-player curvature statistics are aggregated in `metric_calculation.py` and written to `curvature_results.xlsx`.
4. **Statistical analysis.** `result_analysis.py` merges curvature metrics with season free-throw percentages from `free_throw_stats.xlsx`, then runs distributional comparisons and regressions. Key plots are exported to `distributions.png` and `regressionlines.png`.

## Data

| Source | File | Notes |
| --- | --- | --- |
| ESPN.com — 2023-24 NBA season | `code/free_throw_stats.xlsx` | Player-level free-throw attempts, makes, and FT%. |
| Derived (this project) | `code/curvature_results.xlsx` | Curvature metrics computed per player from ball-path fits. |

## Reproducing the analysis

```bash
# 1. Clone
git clone https://github.com/judyz0415/free-throw-curvature.git
cd free-throw-curvature

# 2. (Recommended) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt   # see note below

# 4. Run the pipeline
cd code
python metric_calculation.py     # regenerates curvature_results.xlsx
python result_analysis.py        # regenerates figures + runs regressions
```

> **Note:** if a `requirements.txt` is not yet present, the minimum dependencies are: `numpy`, `pandas`, `scipy`, `matplotlib`, `openpyxl`. A pinned `requirements.txt` will make the repo reproducible for reviewers — see `SETUP_NOTES.md` in this folder for the recommended contents.

## Paper

The LaTeX source and compiled PDF live in `articles/`. If you cite this work, please use the entry below (update once the paper is formally posted or published):

```bibtex
@misc{zhu2025freethrowcurvature,
  title  = {Ball-Path Curvature and Free-Throw Shooting Proficiency in the NBA},
  author = {Zhu, Judy},
  year   = {2025},
  note   = {Working paper. Code and data: https://github.com/judyz0415/free-throw-curvature}
}
```

## License

Released under the [MIT License](LICENSE). Free-throw stats are derived from publicly available ESPN.com data and are included here for research reproducibility.

## Contact

Judy Zhu — judy.zhu6052@gmail.com — [@judyz0415](https://github.com/judyz0415)
