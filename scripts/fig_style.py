"""Shared matplotlib style for the paper-style figures (scripts/fig_rounds.py, scripts/fig_cmp_official.py).

Uses the TrueType fonts found in --font-dir (e.g. Arimo, metric-compatible with Arial, from Google Fonts) and
falls back to matplotlib's DejaVu Sans when there are none. PDFs embed TrueType (fonttype 42) so they stay editable.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

BLUE, GREY = "#1f5fae", "#6b7280"  # v6a / reference method


def setup(font_dir=None):
    fonts = sorted(Path(font_dir).glob("*.ttf")) if font_dir and Path(font_dir).is_dir() else []
    for f in fonts:
        fm.fontManager.addfont(str(f))
    family = fm.FontProperties(fname=str(fonts[0])).get_name() if fonts else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family, "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 7.5, "axes.linewidth": 0.8, "xtick.major.width": 0.8,
        "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })
    return family
