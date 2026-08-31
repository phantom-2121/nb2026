from __future__ import annotations
import platform
from datetime import datetime
from itertools import combinations
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.stats import (f_oneway, kruskal, mannwhitneyu, pearsonr, shapiro, spearmanr, ttest_ind)

# ═══════════════════════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════════════════════

# analysis parameters 
SG_WINDOW = 11             # savitzky-golay smoothing window
SG_POLY = 3                # savitzky-golay poly order
PEAK_PROM_FRAC = 0.08      # peak prominence threshold
PEAK_MIN_DISTANCE = 5      # min peak separation
SIGMA_BOUND_NM = 300.0     # upper bound on fitted component width (sigma)
MU_BOUND_NM = 200.0        # allowed drift of fitted center from peak
FINE_CLASS_NM = 100.0      # upper edge of the fine size class
COARSE_CLASS_NM = 200.0    # lower edge of the coarse size class
ULTRAFINE_NM = 50.0        # threshold for the "sub-50 nm mode" census
ALPHA = 0.05

# fragmentation balance: a parent of PARENT_NM splitting into daughters of
# DAUGHTER_NM conserves gas volume, giving (PARENT/DAUGHTER)^3 daughters each
PARENT_NM = 300.0
DAUGHTER_NM = 150.0

# figure behaviour 
PANEL_SMOOTH_NM = 15.0     # display smoothing width for Fig. 4, in nm (0 = off).
                           # Specified in nm, not bins, so the visual degree of
                           # smoothing does not change with the export bin width.
PANEL_SMOOTH_POLY = 3
PANEL_XLIM_NM: float | None = None   # None -> auto: 99.5th percentile of the
                                     # pooled signal, rounded up to 50 nm
PANEL_MARK_ULTRAFINE = True          # dotted reference line at ULTRAFINE_NM
ANNOTATE_MIN_FRAC = 0.05             # label modes carrying >= this fitted-area fraction

SAVE_TIFF = True
FIG_DPI = 400

# data paths from my files
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

BASE = ROOT / "RD" / "ULTRASONIC"
OUT_DIR = ROOT / "NB_Analysis"

GROUPS: dict[str, list[Path]] = {
    "1 min": [BASE / "1min" / f"1minrep{i}.csv" for i in (1, 2, 3)],
    "3 min": [BASE / "3min" / f"3minrep{i}.csv" for i in (1, 2, 3)],
    "5 min": [BASE / "5min" / f"5minrep{i}.csv" for i in (1, 2, 3)],
    "1 h":   [BASE / "1hr" / f"1hrep{i}.csv" for i in (1, 2, 3)],
    "3 h":   [BASE / "3hr" / f"3hrep{i}.csv" for i in (1, 2, 3)],
}

CONTROL_FILES: list[Path] = [BASE / "control" / f"controlrep{i}.csv" for i in (1, 2, 3)]

TIME_LABELS = ["1 min", "3 min", "5 min", "1 h", "3 h"]
TIME_VALS = [1, 3, 5, 60, 180]
EARLY_PAIRS = [("1 min", "3 min"), ("3 min", "5 min")]

# -- plotting style ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 12,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

# one colour per condition, used consistently in Fig. 4
PANEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
CLASS_COLORS = {"coarse": "#c0392b", "mid": "#3d9a35", "fine": "#3b6ea5"}
DIA_COLOR = "#1f3b5c"       # left axis, mean diameter (deep navy)
CONC_COLOR = "#e08214"      # right axis, total concentration (ochre). A blue/orange
                            # pair stays legible in greyscale and to colour-blind readers.
GRID_COLOR = "#dcdcdc"


# ═══════════════════════════════════════════════════════════════════════════════
#  Logging
# ═══════════════════════════════════════════════════════════════════════════════
LOG: list[str] = []


def log(msg: str = "") -> None:
    """Copy for NB_Analysis/run_log.txt."""
    print(msg)
    LOG.append(str(msg))
def ensure_dirs() -> None:
    """Create the tables / data / figures subfolders under OUT_DIR."""
    for sub in ("tables", "data", "figures"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)
def write_table(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Save a manuscript-facing table and return it unchanged."""
    df.to_csv(OUT_DIR / "tables" / filename, index=False)
    return df
def write_data(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Save an intermediate/underlying data file and return it unchanged."""
    df.to_csv(OUT_DIR / "data" / filename, index=False)
    return df
def write_run_log() -> None:
    """Flush the accumulated log to disk."""
    (OUT_DIR / "run_log.txt").write_text("\n".join(LOG) + "\n")
def summary_value(df_summary: pd.DataFrame, label: str, column: str) -> float:
    """Look up one scalar from the per-condition summary table."""
    return float(df_summary.loc[df_summary["condition"] == label, column].iloc[0])

# ═══════════════════════════════════════════════════════════════════════════════
#  Data and Grid Alignment
# ═══════════════════════════════════════════════════════════════════════════════
SIZE_COL_CANDIDATES = ["Bin centre (nm)", "Bin Centre (nm)", "Bin center (nm)", "Size (nm)"]
CONC_COL_CANDIDATES = ["Concentration average", "Concentration Average",
                       "Concentration (particles / ml)", "Concentration"]
def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one NanoSight export and return (size_nm, concentration)."""
    df = pd.read_csv(path, header=0)

    size_col = next((c for c in SIZE_COL_CANDIDATES if c in df.columns), None)
    if size_col is None:
        raise KeyError(f"{path.name}: no size column found; looked for {SIZE_COL_CANDIDATES}")

    conc_col = next((c for c in CONC_COL_CANDIDATES if c in df.columns), None)
    if conc_col is None:
        conc_col = df.columns[-2]
        log(f"  ! {path.name}: concentration column not recognised, using '{conc_col}'")

    size = df[size_col].to_numpy(dtype=float)
    conc = np.nan_to_num(df[conc_col].to_numpy(dtype=float), nan=0.0)
    conc = np.clip(conc, 0.0, None)

    order = np.argsort(size)
    return size[order], conc[order]
def to_common_grid(reps: list[tuple[np.ndarray, np.ndarray]],
                   label: str) -> tuple[np.ndarray, list[np.ndarray]]:
    """Put every replicate of a condition on one bin grid (finest grid = reference)."""
    grids = [s for s, _ in reps]
    ref = max(grids, key=len)
    out = []
    for k, (s, c) in enumerate(reps, start=1):
        if s.shape == ref.shape and np.allclose(s, ref, rtol=1e-6, atol=1e-6):
            out.append(c)
        else:
            log(f"  ! [{label}] replicate {k} on a different bin grid; interpolated onto reference")
            out.append(np.interp(ref, s, c, left=0.0, right=0.0))
    return ref, out
def mean_curve_for(raw_store: dict, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (grid, replicate-mean PSD) for one condition."""
    grid = raw_store[label][0][0]
    stack = [np.interp(grid, np.asarray(sz), np.asarray(cc), left=0.0, right=0.0)
             for sz, cc in raw_store[label]]
    return grid, np.mean(stack, axis=0)
# ═══════════════════════════════════════════════════════════════════════════════
#  Curve Smoothing
# ═══════════════════════════════════════════════════════════════════════════════
def bins_for_nm(size: np.ndarray, width_nm: float) -> int:
    """Convert a smoothing width in nm to an odd number of bins for this grid."""
    bin_nm = float(np.median(np.diff(size)))
    w = int(round(width_nm / max(bin_nm, 1e-9)))
    if w % 2 == 0:
        w += 1
    return max(w, 5)
def smooth(conc: np.ndarray, window: int = SG_WINDOW, poly: int = SG_POLY) -> np.ndarray:
    """Savitzky-Golay smoothing (peak detection, and Fig. 4 display only)."""
    w = int(window)
    if w % 2 == 0:
        w += 1
    if w > 2 and len(conc) > w:
        return np.clip(np.asarray(savgol_filter(conc, w, poly), dtype=float), 0, None)
    return np.clip(np.asarray(conc, dtype=float), 0, None)
# ═══════════════════════════════════════════════════════════════════════════════
#  Raw Dist Metrics
# ═══════════════════════════════════════════════════════════════════════════════
def whole_features(size: np.ndarray, conc: np.ndarray) -> dict:
    """Moments, spread, entropy and size-class split of one replicate PSD."""
    total = float(np.sum(conc))
    p = conc / (total + 1e-30)

    mean = float(np.dot(size, p))
    std = float(np.sqrt(np.dot((size - mean) ** 2, p)))
    skew = float(np.dot(((size - mean) / (std + 1e-12)) ** 3, p))
    entropy = float(-np.sum(p * np.log(p + 1e-30)))

    cum = np.cumsum(p)
    i10 = min(int(np.searchsorted(cum, 0.10)), len(size) - 1)
    i90 = min(int(np.searchsorted(cum, 0.90)), len(size) - 1)

    fine_m = size < FINE_CLASS_NM
    mid_m = (size >= FINE_CLASS_NM) & (size <= COARSE_CLASS_NM)
    coarse_m = size > COARSE_CLASS_NM

    return {
        "total_conc": total,
        "mean_nm": mean,
        "std_nm": std,
        "skew": skew,
        "entropy": entropy,
        "width_80_nm": float(size[i90] - size[i10]),
        "frac_fine": float(np.sum(p[fine_m])),
        "frac_mid": float(np.sum(p[mid_m])),
        "frac_coarse": float(np.sum(p[coarse_m])),
        "conc_fine": float(np.sum(conc[fine_m])),
        "conc_mid": float(np.sum(conc[mid_m])),
        "conc_coarse": float(np.sum(conc[coarse_m])),
        "frac_sub50": float(np.sum(p[size < ULTRAFINE_NM])),
    }
def summarise_condition(label: str, rep_features: pd.DataFrame) -> dict:
    """Collapse the replicates of one condition into a single summary row."""
    n = len(rep_features)
    sd = (lambda col: float(rep_features[col].std(ddof=1)) if n > 1 else np.nan)
    mean_conc = float(rep_features["total_conc"].mean())
    sd_conc = sd("total_conc")

    return {
        "condition": label,
        "n_replicates": n,
        "mean_total_conc": mean_conc,
        "sd_total_conc": sd_conc,
        "conc_cv_pct": 100.0 * sd_conc / (mean_conc + 1e-30),
        "mean_size_nm": float(rep_features["mean_nm"].mean()),
        "sd_size_nm": sd("mean_nm"),
        "mean_width_80_nm": float(rep_features["width_80_nm"].mean()),
        "sd_width_80_nm": sd("width_80_nm"),
        "mean_entropy": float(rep_features["entropy"].mean()),
        "mean_skew": float(rep_features["skew"].mean()),
        "frac_fine": float(rep_features["frac_fine"].mean()),
        "frac_mid": float(rep_features["frac_mid"].mean()),
        "frac_coarse": float(rep_features["frac_coarse"].mean()),
    }
# ═══════════════════════════════════════════════════════════════════════════════
#  Gaussian mode decomp
# ═══════════════════════════════════════════════════════════════════════════════
def gaussian(x, amp, mu, sig):
    """Single Gaussian component."""
    return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)
def multi_gaussian(x, *params):
    """Sum of Gaussians, parameters flattened as (amp, mu, sigma) triplets."""
    y = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 3):
        y += gaussian(x, params[i], params[i + 1], params[i + 2])
    return y
def _seed_from_peaks(size: np.ndarray, cs: np.ndarray,
                     peaks: np.ndarray) -> tuple[list, list, list]:
    """Build (p0, lower, upper) for curve_fit from detected peaks and their widths."""
    bin_nm = float(np.median(np.diff(size)))
    fwhm_bins, _, _, _ = peak_widths(cs, peaks, rel_height=0.5)

    p0, lo, hi = [], [], []
    for k, pk in enumerate(peaks):
        amp = float(cs[pk])
        mu = float(size[pk])
        fwhm_nm = max(float(fwhm_bins[k]) * bin_nm, 2.0 * bin_nm)
        sig = float(np.clip(fwhm_nm / 2.355, 1.0, SIGMA_BOUND_NM))
        p0 += [amp, mu, sig]
        lo += [0.0, max(0.0, mu - MU_BOUND_NM), 1.0]
        hi += [amp * 3.0 + 1e-30, mu + MU_BOUND_NM, SIGMA_BOUND_NM]

    p0 = [float(np.clip(v, lo[i] + 1e-9, hi[i] - 1e-9)) for i, v in enumerate(p0)]
    return p0, lo, hi
def decompose_modes(size: np.ndarray, conc: np.ndarray, tag: str = "") -> list[dict]:
    """Peak-initialised sum-of-Gaussians decomposition."""
    cs = smooth(conc)
    if float(np.max(cs)) <= 0.0:
        return [{"mode_nm": np.nan, "amplitude": 0.0, "sigma_nm": np.nan,
                 "conc_fraction": np.nan, "fit_converged": False, "fit_r2": np.nan,
                 "n_components": 0}]

    peaks, _ = find_peaks(cs, prominence=float(np.max(cs)) * PEAK_PROM_FRAC,
                          distance=PEAK_MIN_DISTANCE)

    if len(peaks) == 0:
        total = float(np.sum(conc))
        p = conc / (total + 1e-30)
        mean = float(np.dot(size, p))
        log(f"  ! {tag}: no peaks passed the prominence threshold; weighted mean "
            f"({mean:.1f} nm) reported as a single unfitted mode")
        return [{"mode_nm": round(mean, 1), "amplitude": float(np.max(conc)),
                 "sigma_nm": np.nan, "conc_fraction": 1.0,
                 "fit_converged": False, "fit_r2": np.nan, "n_components": 1}]

    p0, lo, hi = _seed_from_peaks(size, cs, peaks)

    try:
        popt, _ = curve_fit(multi_gaussian, size, cs, p0=p0, bounds=(lo, hi), maxfev=50000)
        converged = True
    except Exception as exc:                                   # noqa: BLE001
        popt, converged = np.array(p0, dtype=float), False
        log(f"  ! {tag}: Gaussian fit did NOT converge ({type(exc).__name__}); "
            f"unfitted peak seeds reported")

    fitted_total = multi_gaussian(size, *popt)
    resid = cs - fitted_total
    ss_tot = float(np.sum((cs - cs.mean()) ** 2))
    fit_r2 = float(1.0 - np.sum(resid ** 2) / (ss_tot + 1e-30))
    total_fitted = float(np.sum(fitted_total))

    modes = []
    for i in range(0, len(popt), 3):
        amp_i, mu_i, sig_i = float(popt[i]), float(popt[i + 1]), abs(float(popt[i + 2]))
        curve = gaussian(size, amp_i, mu_i, sig_i)
        modes.append({
            "mode_nm": round(mu_i, 1),
            "amplitude": amp_i,
            "sigma_nm": round(sig_i, 1),
            "fwhm_nm": round(2.355 * sig_i, 1),
            "conc_fraction": round(float(np.sum(curve)) / (total_fitted + 1e-30), 3),
            "fit_converged": converged,
            "fit_r2": round(fit_r2, 4),
            "n_components": len(popt) // 3,
        })

    return sorted(modes, key=lambda m: m["mode_nm"])
# ═══════════════════════════════════════════════════════════════════════════════
#  Stats
# ═══════════════════════════════════════════════════════════════════════════════
def holm_adjust(pvals) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj
def omnibus_and_posthoc(df_whole: pd.DataFrame, present: list[str],
                        metric: str, pretty: str, scale: float = 1.0,
                        unit: str = "", log_transform: bool = False) -> dict:
    """
    Omnibus ANOVA and Kruskal-Wallis, then pairwise Welch's t-test and Mann-Whitney U
    under Holm-Bonferroni. Returns a dict with the results.
    """
    if log_transform and np.any(df_whole[metric].to_numpy(float) <= 0):
        log(f"  ! {pretty}: non-positive values present, log transform skipped")
        log_transform = False

    groups = []
    for lb in present:
        g = df_whole.loc[df_whole["condition"] == lb, metric].to_numpy(float) / scale
        groups.append(np.log10(g) if log_transform else g)
    n_per = [len(g) for g in groups]

    tag = f"{pretty} (log10)" if log_transform else pretty
    log("\n" + "-" * 78)
    log(f"{tag}  (n per condition: {n_per})")
    log("-" * 78)
    if not log_transform:
        for lb, g in zip(present, groups):
            vs = ", ".join(f"{v:.2f}" for v in g)
            log(f"  {lb:>6}: {g.mean():8.2f} +/- {g.std(ddof=1):6.2f} {unit}   values: {vs}")

    F, p_anova = f_oneway(*groups)
    H, p_kw = kruskal(*groups)
    k, N = len(groups), int(sum(n_per))
    eta2 = (F * (k - 1)) / (F * (k - 1) + (N - k)) if np.isfinite(F) else np.nan

    sw = []
    for lb, g in zip(present, groups):
        try:
            sw.append(f"{lb} {float(shapiro(g).pvalue):.3f}")
        except Exception:                                      # noqa: BLE001
            sw.append(f"{lb} n/a")

    log(f"  ANOVA F({k - 1},{N - k}) = {F:.3f}, p = {p_anova:.4f}, eta2 = {eta2:.3f}")
    log(f"  Kruskal-Wallis H({k - 1}) = {H:.3f}, p = {p_kw:.4f}")
    log("  Shapiro-Wilk p: " + "; ".join(sw))

    pairs, t_stat, t_df, p_t, p_u, diffs = [], [], [], [], [], []
    for (a, ga), (b, gb) in combinations(list(zip(present, groups)), 2):
        pairs.append(f"{a} vs {b}")
        diffs.append(float(gb.mean() - ga.mean()))
        res = ttest_ind(ga, gb, equal_var=False)
        t_stat.append(float(res.statistic))                    # type: ignore[union-attr]
        t_df.append(float(getattr(res, "df", np.nan)))
        p_t.append(float(res.pvalue))                          # type: ignore[union-attr]
        try:
            p_u.append(float(mannwhitneyu(ga, gb, alternative="two-sided").pvalue))
        except ValueError:
            p_u.append(np.nan)

    adj_t = holm_adjust(p_t)
    adj_u = holm_adjust([x if np.isfinite(x) else 1.0 for x in p_u])

    log(f"  {'contrast':<18}{'diff':>9}{'t':>8}{'df':>7}{'t p':>9}{'t p_adj':>9}{'U p':>8}{'U p_adj':>9}")
    for i, pr in enumerate(pairs):
        log(f"  {pr:<18}{diffs[i]:>9.2f}{t_stat[i]:>8.2f}{t_df[i]:>7.1f}"
            f"{p_t[i]:>9.4f}{adj_t[i]:>9.4f}{p_u[i]:>8.4f}{adj_u[i]:>9.4f}")

    suffix = f"{metric}_log" if log_transform else metric
    write_table(pd.DataFrame({"contrast": pairs, "difference": diffs,
                              "welch_t": t_stat, "welch_df": t_df,
                              "welch_p": p_t, "welch_p_holm": adj_t,
                              "mannwhitney_p": p_u, "mannwhitney_p_holm": adj_u}),
                f"stats_posthoc_{suffix}.csv")

    best = int(np.argmin(p_t))
    log(f"  Strongest: {pairs[best]}, t({t_df[best]:.1f}) = {t_stat[best]:.2f}, "
        f"p = {p_t[best]:.3f} raw, {adj_t[best]:.3f} adj")

    return {"metric": tag, "F": float(F), "p_anova": float(p_anova),
            "H": float(H), "p_kw": float(p_kw), "eta2": float(eta2),
            "df_between": k - 1, "df_within": N - k, "n_per": str(n_per),
            "strongest_contrast": pairs[best], "strongest_t": t_stat[best],
            "strongest_df": t_df[best], "strongest_p": p_t[best],
            "strongest_p_holm": float(adj_t[best])}
def run_condition_statistics(df_whole: pd.DataFrame, present: list[str]) -> dict:
    """Mean diameter and total concentration (linear and log10) across conditions."""
    log("\nSTATISTICAL TESTS ACROSS CONDITIONS")
    s_size = omnibus_and_posthoc(df_whole, present, "mean_nm", "Mean diameter", 1.0, "nm")
    s_conc = omnibus_and_posthoc(df_whole, present, "total_conc", "Total concentration",
                                 1e7, "x10^7 mL-1")
    s_conc_log = omnibus_and_posthoc(df_whole, present, "total_conc",
                                     "Total concentration", 1e7, "x10^7 mL-1",
                                     log_transform=True)
    write_table(pd.DataFrame([s_size, s_conc, s_conc_log]), "stats_omnibus.csv")
    return {"size": s_size, "conc": s_conc, "conc_log": s_conc_log}
def diameter_concentration_correlation(df_whole: pd.DataFrame, df_summary: pd.DataFrame,
                                       present: list[str]) -> pd.DataFrame:
    """Pearson and Spearman between mean diameter and total concentration."""
    cm_d = np.array([summary_value(df_summary, lb, "mean_size_nm") for lb in present])
    cm_c = np.array([summary_value(df_summary, lb, "mean_total_conc") for lb in present])
    rep_d = df_whole["mean_nm"].to_numpy(float)
    rep_c = df_whole["total_conc"].to_numpy(float)

    r_cond, p_cond = pearsonr(cm_d, cm_c)
    rs_cond, ps_cond = spearmanr(cm_d, cm_c)
    r_rep, p_rep = pearsonr(rep_d, rep_c)
    rs_rep, ps_rep = spearmanr(rep_d, rep_c)

    log("\nDIAMETER-CONCENTRATION CORRELATION")
    log(f"  condition means: r = {r_cond:.3f}, p = {p_cond:.4f}, n = {len(cm_d)}; "
        f"rho = {rs_cond:.3f}, p = {ps_cond:.4f}")
    log(f"  sample level:    r = {r_rep:.3f}, p = {p_rep:.4f}, n = {len(df_whole)}; "
        f"rho = {rs_rep:.3f}, p = {ps_rep:.4f}")

    return write_table(pd.DataFrame([
        {"level": "condition_means", "pearson_r": r_cond, "pearson_p": p_cond,
         "spearman_rho": rs_cond, "spearman_p": ps_cond, "n": len(cm_d)},
        {"level": "replicate", "pearson_r": r_rep, "pearson_p": p_rep,
         "spearman_rho": rs_rep, "spearman_p": ps_rep, "n": len(df_whole)},
    ]), "correlation.csv")

# ═══════════════════════════════════════════════════════════════════════════════
#  8. FIGURES
# ═══════════════════════════════════════════════════════════════════════════════
def save_fig(fig, stem: str) -> None:
    """Write a figure as PNG (and TIFF if enabled), then close it."""
    png = OUT_DIR / "figures" / f"{stem}.png"
    fig.savefig(png, dpi=FIG_DPI, bbox_inches="tight")
    if SAVE_TIFF:
        fig.savefig(OUT_DIR / "figures" / f"{stem}.tiff", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved: {png.name}" + (" (+ .tiff)" if SAVE_TIFF else ""))
def _time_axis(ax, x: np.ndarray, present: list[str]) -> None:
    """Shared log-scaled sonication-time x-axis with condition labels as ticks."""
    ax.set_xscale("log")
    ax.set_xlim(x.min() * 0.75, x.max() * 1.35)
    ax.set_xticks(x)
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(present))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_xlabel("Sonication time (log scale)")
def combined_trajectory_figure(df_whole: pd.DataFrame, present: list[str]) -> pd.DataFrame:
    """
    Fig. 2 - mean diameter and total concentration against sonication time.

    Twin y-axes on one shared log x-axis; condition means +/- SD only.
    """
    tmap = dict(zip(TIME_LABELS, TIME_VALS))
    x = np.array([tmap[lb] for lb in present], dtype=float)

    def agg(metric: str, scale: float) -> tuple[np.ndarray, np.ndarray]:
        m = np.array([df_whole.loc[df_whole["condition"] == lb, metric].mean() / scale
                      for lb in present], float)
        sd = np.array([df_whole.loc[df_whole["condition"] == lb, metric].std(ddof=1) / scale
                       for lb in present], float)
        return m, sd

    d_mean, d_sd = agg("mean_nm", 1.0)
    c_mean, c_sd = agg("total_conc", 1e7)

    fig, ax1 = plt.subplots(figsize=(8.4, 4.8))
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)

    ax1.set_axisbelow(True)
    ax1.grid(axis="y", color=GRID_COLOR, lw=0.7, ls="-", alpha=0.9, zorder=0)

    # slight multiplicative offset so the two error bars never overlap
    xl, xr = x * 0.972, x * 1.028

    h1 = ax1.errorbar(xl, d_mean, yerr=d_sd, color=DIA_COLOR, lw=2.0, marker="o", ms=7,
                      markerfacecolor="white", markeredgewidth=1.8,
                      capsize=3, capthick=1.0, elinewidth=1.0, alpha=0.95, zorder=5,
                      label="Mean diameter")
    h2 = ax2.errorbar(xr, c_mean, yerr=c_sd, color=CONC_COLOR, lw=2.0, ls=(0, (5, 2)),
                      marker="s", ms=7, markerfacecolor="white", markeredgewidth=1.8,
                      capsize=3, capthick=1.0, elinewidth=1.0, alpha=0.95, zorder=4,
                      label="Total concentration")

    _time_axis(ax1, x, present)
    ax1.set_ylabel("Mean diameter (nm)", color=DIA_COLOR, labelpad=8)
    ax1.tick_params(axis="y", colors=DIA_COLOR, length=3)
    ax1.spines["left"].set_color(DIA_COLOR)
    ax2.set_ylabel("Total concentration ($\\times 10^{7}$ particles/mL)",
                   color="black", labelpad=10)
    ax2.tick_params(axis="y", colors="black", length=3)
    ax2.spines["right"].set_color("black")

    # generous headroom on both axes: the point of the figure is the shape of the
    # two trajectories, so neither should run close to its frame
    d_lo, d_hi = float(np.nanmin(d_mean - d_sd)), float(np.nanmax(d_mean + d_sd))
    d_pad = 0.45 * (d_hi - d_lo)
    ax1.set_ylim(max(0.0, d_lo - d_pad), d_hi + d_pad)
    c_hi = float(np.nanmax(c_mean + c_sd))
    ax2.set_ylim(0.0, c_hi * 1.55)

    ax1.legend(handles=[h1, h2], loc="upper center", bbox_to_anchor=(0.5, 1.13),
               ncol=2, frameon=False, handlelength=2.8, columnspacing=2.4)

    fig.tight_layout()
    save_fig(fig, "fig2_diameter_concentration")

    return pd.DataFrame({"condition": present, "time_min": x,
                         "mean_diameter_nm": d_mean, "sd_diameter_nm": d_sd,
                         "mean_conc_x1e7": c_mean, "sd_conc_x1e7": c_sd})
def composition_figure(df_summary: pd.DataFrame, present: list[str]) -> pd.DataFrame:
    """Fig. 3 - stacked mean size-class fractions per condition."""
    fine = np.array([summary_value(df_summary, lb, "frac_fine") for lb in present], float)
    mid = np.array([summary_value(df_summary, lb, "frac_mid") for lb in present], float)
    coarse = np.array([summary_value(df_summary, lb, "frac_coarse") for lb in present], float)
    tot = fine + mid + coarse
    fine, mid, coarse = fine / tot, mid / tot, coarse / tot

    xpos = np.arange(len(present), dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.bar(xpos, fine, 0.55, color=CLASS_COLORS["fine"], edgecolor="black", lw=0.8,
           label=f"< {FINE_CLASS_NM:.0f} nm")
    ax.bar(xpos, mid, 0.55, bottom=fine, color=CLASS_COLORS["mid"], edgecolor="black",
           lw=0.8, label=f"{FINE_CLASS_NM:.0f}-{COARSE_CLASS_NM:.0f} nm")
    ax.bar(xpos, coarse, 0.55, bottom=fine + mid, color=CLASS_COLORS["coarse"],
           edgecolor="black", lw=0.8, label=f"> {COARSE_CLASS_NM:.0f} nm")

    ax.set_xticks(xpos)
    ax.set_xticklabels(present)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Sonication time")
    ax.set_ylabel("Fraction of total concentration")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)

    fig.tight_layout()
    save_fig(fig, "fig3_size_classes")

    return pd.DataFrame({"condition": present, "frac_fine": fine,
                         "frac_mid": mid, "frac_coarse": coarse})
def _panel_xmax(raw_store: dict, present: list[str]) -> float:
    """Common display window: 99.5th percentile of the pooled signal, rounded to 50 nm."""
    if PANEL_XLIM_NM is not None:
        return float(PANEL_XLIM_NM)
    edges = []
    for label in present:
        grid, mc = mean_curve_for(raw_store, label)
        cum = np.cumsum(mc) / (np.sum(mc) + 1e-30)
        edges.append(float(grid[min(int(np.searchsorted(cum, 0.995)), len(grid) - 1)]))
    return float(np.ceil(max(edges) / 50.0) * 50.0)
def _panel_side_text(df_whole: pd.DataFrame, label: str, modes: list[dict]) -> str:
    """Concentration (mean +/- SE), mean diameter and dominant mode for one panel."""
    sub = df_whole[df_whole["condition"] == label]
    n = len(sub)
    c_mean = float(sub["total_conc"].mean()) / 1e7
    c_se = (float(sub["total_conc"].std(ddof=1)) / np.sqrt(n) / 1e7) if n > 1 else np.nan
    d_mean = float(sub["mean_nm"].mean())
    dom = max(modes, key=lambda m: (float(m["conc_fraction"])
                                    if np.isfinite(m["conc_fraction"]) else -1))
    return (f"$C$ = {c_mean:.2f} $\\pm$ {c_se:.2f} $\\times10^{{7}}$ mL$^{{-1}}$\n"
            f"$\\bar{{d}}$ = {d_mean:.0f} nm\n"
            f"dominant {float(dom['mode_nm']):.0f} nm "
            f"({100 * float(dom['conc_fraction']):.0f}%)")
def psd_panel_figure(raw_store: dict, df_whole: pd.DataFrame,
                     present: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fig. 4 - one panel per condition showing the replicate-mean PSD.
    IMPORTANT: Avg curve derived modes, not the per replicate raw data as seen in Table 3
    """
    xmax = _panel_xmax(raw_store, present)

    ncols = 3
    nrows = int(np.ceil(len(present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), squeeze=False)
    flat = axes.ravel()

    handles, curve_frames, mode_rows = [], [], []
    for i, (label, color) in enumerate(zip(present, PANEL_COLORS)):
        ax = flat[i]
        grid, mean_curve = mean_curve_for(raw_store, label)

        win = bins_for_nm(grid, PANEL_SMOOTH_NM) if PANEL_SMOOTH_NM else 0
        shown = smooth(mean_curve, win, PANEL_SMOOTH_POLY) if win else mean_curve
        y = shown / 1e6

        line, = ax.plot(grid, y, color=color, lw=1.9, zorder=4, label=label)
        ax.fill_between(grid, y, color=color, alpha=0.15, zorder=3)
        handles.append(line)

        if PANEL_MARK_ULTRAFINE:
            ax.axvline(ULTRAFINE_NM, color="0.5", lw=0.8, ls=":", alpha=0.8, zorder=1)

        # modes of the averaged curve
        modes = decompose_modes(grid, mean_curve, tag=f"{label} mean curve")
        for m in modes:
            mode_rows.append({"condition": label, "mode_nm": m["mode_nm"],
                              "sigma_nm": m["sigma_nm"], "conc_fraction": m["conc_fraction"],
                              "fit_converged": m["fit_converged"], "fit_r2": m["fit_r2"]})

        drawn = sorted((m for m in modes
                        if np.isfinite(m["mode_nm"]) and m["mode_nm"] <= xmax
                        and float(m["conc_fraction"]) >= ANNOTATE_MIN_FRAC),
                       key=lambda m: m["mode_nm"])
        for k, m in enumerate(drawn):
            xm = float(m["mode_nm"])
            ym = float(np.interp(xm, grid, y))
            ax.annotate(f"{xm:.0f} nm\n{100 * float(m['conc_fraction']):.0f}%",
                        xy=(xm, ym), xytext=(0, 7 if k % 2 == 0 else 20),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7.2, color="0.15", zorder=6)

        ax.text(0.97, 0.97, _panel_side_text(df_whole, label, modes),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.4, linespacing=1.5, zorder=7,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="0.85", linewidth=0.6, alpha=0.9))

        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1.55 * max(float(np.max(y)), 1e-12))
        ax.tick_params(labelsize=8)
        if i + ncols >= len(present):
            ax.set_xlabel("Size (nm)", fontsize=9)
        if i % ncols == 0:
            ax.set_ylabel("Concentration ($\\times 10^{6}$ particles/mL)", fontsize=9)

        curve_frames.append(pd.DataFrame({"condition": label,
                                          "size_nm": grid.astype(float),
                                          "mean_conc_mL-1": mean_curve.astype(float)}))

    for j in range(len(present), len(flat)):
        flat[j].axis("off")

    key_ax = flat[len(present)] if len(present) < len(flat) else flat[0]
    key_ax.legend(handles=handles, labels=list(present), loc="center", frameon=False,
                  fontsize=10, title="Sonication time", handlelength=2.4)

    fig.tight_layout()
    save_fig(fig, "fig4_psd_panels")

    df_modes_mean = write_table(pd.DataFrame(mode_rows), "mean_curve_mode_census.csv")
    return pd.concat(curve_frames, ignore_index=True), df_modes_mean
# ═══════════════════════════════════════════════════════════════════════════════
#  Tables
# ═══════════════════════════════════════════════════════════════════════════════
def build_table1(df_summary: pd.DataFrame, present: list[str]) -> pd.DataFrame:
    """Table 1 - per-condition size, concentration and size-class summary."""
    rows = []
    for lb in present:
        s = df_summary.loc[df_summary["condition"] == lb].iloc[0]
        rows.append({
            "Condition": lb,
            "Mean dia. (nm)": f"{s['mean_size_nm']:.0f} ± {s['sd_size_nm']:.0f}",
            "Conc. (x10^7 mL-1)": f"{s['mean_total_conc'] / 1e7:.2f}",
            "CV conc. (%)": f"{s['conc_cv_pct']:.1f}",
            ">200 nm (%)": f"{100 * s['frac_coarse']:.1f}",
            "100-200 nm (%)": f"{100 * s['frac_mid']:.1f}",
            "<100 nm (%)": f"{100 * s['frac_fine']:.1f}",
            "80% width (nm)": f"{s['mean_width_80_nm']:.0f}",
        })
    return write_table(pd.DataFrame(rows), "table1_metrics.csv")
def build_table2(df_whole: pd.DataFrame, present: list[str]) -> tuple[pd.DataFrame, dict]:
    """Table 2 - absolute size-class concentrations and interval changes, early phase."""
    early = [lb for lb in ("1 min", "3 min", "5 min") if lb in present]
    means = {lb: df_whole.loc[df_whole["condition"] == lb,
                              ["conc_fine", "conc_mid", "conc_coarse", "total_conc"]].mean()
             for lb in early}

    rows = []
    for lb in early:
        m = means[lb]
        rows.append({"Condition": lb,
                     "<100 nm (mL-1)": f"{m['conc_fine']:.2e}",
                     "100-200 nm (mL-1)": f"{m['conc_mid']:.2e}",
                     ">200 nm (mL-1)": f"{m['conc_coarse']:.2e}",
                     "Total (mL-1)": f"{m['total_conc']:.2e}"})

    for a, b in EARLY_PAIRS:
        if a in means and b in means:
            ma, mb = means[a], means[b]

            def pct(key: str, _ma=ma, _mb=mb) -> float:
                return 100.0 * (_mb[key] - _ma[key]) / (_ma[key] + 1e-30)

            rows.append({"Condition": f"Δ {a}→{b}",
                         "<100 nm (mL-1)": f"{pct('conc_fine'):+.0f}%",
                         "100-200 nm (mL-1)": f"{pct('conc_mid'):+.0f}%",
                         ">200 nm (mL-1)": f"{pct('conc_coarse'):+.0f}%",
                         "Total (mL-1)": f"{pct('total_conc'):+.0f}%"})

    return write_table(pd.DataFrame(rows), "table2_size_class_absolute.csv"), means
def build_table3(df_modes: pd.DataFrame, present: list[str]) -> pd.DataFrame:
    """Table 3 - per-replicate mode census and fit quality."""
    rows = []
    for lb in present:
        sub = df_modes[df_modes["condition"] == lb]
        reps_fine = sub.loc[sub["mode_nm"] < ULTRAFINE_NM, "replicate"].nunique()
        n_reps = sub["replicate"].nunique()
        rows.append({
            "Condition": lb,
            "Modes (n, 3 samples)": int(len(sub)),
            "Smallest (nm)": f"{sub['mode_nm'].min():.1f}",
            "Largest (nm)": f"{sub['mode_nm'].max():.1f}",
            f"Samples with a sub-{ULTRAFINE_NM:.0f} nm mode": f"{reps_fine} of {n_reps}",
            "All fits converged": bool(sub["fit_converged"].all()),
            "Min fit R2": f"{sub['fit_r2'].min():.3f}" if sub["fit_r2"].notna().any() else "-",
        })
    return write_table(pd.DataFrame(rows), "table3_mode_census.csv")
def dominant_mode_census(df_modes: pd.DataFrame, present: list[str]) -> pd.DataFrame:
    """Dominant fitted component in each replicate (feeds the 3 min unimodality claim)."""
    log("\nDOMINANT COMPONENT PER SAMPLE")
    rows = []
    for lb in present:
        sub = df_modes[df_modes["condition"] == lb]
        per_condition = []
        for rep, g in sub.groupby("replicate"):
            top = g.iloc[int(g["conc_fraction"].to_numpy().argmax())]
            per_condition.append({"condition": lb, "replicate": int(str(rep)),
                                  "n_modes": len(g),
                                  "dominant_nm": float(top["mode_nm"]),
                                  "dominant_fraction": float(top["conc_fraction"])})
        rows.extend(per_condition)
        d = pd.DataFrame(per_condition)
        log(f"  {lb:>6}: dominant {d['dominant_nm'].min():.0f}-{d['dominant_nm'].max():.0f} nm, "
            f"carrying {d['dominant_fraction'].min() * 100:.0f}-"
            f"{d['dominant_fraction'].max() * 100:.0f}%")
    return write_data(pd.DataFrame(rows), "dominant_mode_by_replicate.csv")
def fragmentation_balance(class_means: dict, df_summary: pd.DataFrame,
                          present: list[str]) -> pd.DataFrame:
    """Number and concentration checks against a fragmentation-dominated early phase."""
    rows = []
    a, b = "1 min", "3 min"
    if a in class_means and b in class_means:
        lost = float(class_means[a]["conc_coarse"] - class_means[b]["conc_coarse"])
        n_daughters = (PARENT_NM / DAUGHTER_NM) ** 3
        expected = lost * n_daughters
        observed = float((class_means[b]["conc_fine"] - class_means[a]["conc_fine"]) +
                         (class_means[b]["conc_mid"] - class_means[a]["conc_mid"]))
        rows += [
            {"quantity": "coarse concentration lost 1->3 min (mL-1)", "value": f"{lost:.3e}"},
            {"quantity": f"daughters per {PARENT_NM:.0f} nm parent at {DAUGHTER_NM:.0f} nm",
             "value": f"{n_daughters:.0f}"},
            {"quantity": "daughters predicted (mL-1)", "value": f"{expected:.3e}"},
            {"quantity": "observed gain in <200 nm classes (mL-1)", "value": f"{observed:.3e}"},
            {"quantity": "observed as % of predicted",
             "value": f"{100 * observed / (expected + 1e-30):.1f}%"},
        ]

    d = {lb: summary_value(df_summary, lb, "mean_size_nm") for lb in present}
    if "1 min" in d and "5 min" in d:
        ratio = (d["1 min"] / d["5 min"]) ** 3
        drop = 100 * (1 - d["5 min"] / d["1 min"])
        c1 = summary_value(df_summary, "1 min", "mean_total_conc")
        c5 = summary_value(df_summary, "5 min", "mean_total_conc")
        rows += [
            {"quantity": "mean diameter fall 1->5 min", "value": f"{drop:.0f}%"},
            {"quantity": "concentration rise required if volume conserved",
             "value": f"{ratio:.1f}-fold"},
            {"quantity": "concentration actually observed 1->5 min",
             "value": f"{c1:.2e} -> {c5:.2e} mL-1 ({100 * (c5 - c1) / c1:+.0f}%)"},
        ]

    return write_table(pd.DataFrame(rows), "fragmentation_balance.csv")
def analyse_controls() -> pd.DataFrame | None:
    """Unsonicated Milli-Q controls: integrated concentration and size composition."""
    sub50_col = f"frac_sub{ULTRAFINE_NM:.0f}nm_pct"
    rows = []
    for i, path in enumerate(CONTROL_FILES, start=1):
        if not path.exists():
            log(f"  ! control file not found, skipping: {path}")
            continue
        size, conc = load_csv(path)
        f = whole_features(size, conc)
        rows.append({"control": f"Tube {i}",
                     "total_conc_mL-1": f["total_conc"],
                     "mean_nm": f["mean_nm"],
                     sub50_col: 100 * f["frac_sub50"],
                     "frac_coarse_pct": 100 * f["frac_coarse"]})
    if not rows:
        log("  ! no control files found")
        return None

    t = pd.DataFrame(rows)
    mean_c = t["total_conc_mL-1"].mean()
    sd_c = t["total_conc_mL-1"].std(ddof=1) if len(t) > 1 else np.nan
    t.loc[len(t)] = {"control": "Mean", "total_conc_mL-1": mean_c, "mean_nm": np.nan,
                     sub50_col: t[sub50_col].mean(),
                     "frac_coarse_pct": t["frac_coarse_pct"].mean()}
    log(t.to_string(index=False))
    log(f"  mean {mean_c:.2e} +/- {sd_c:.2e} mL-1")
    return write_table(t, "controls.csv")
# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════
def log_environment() -> None:
    """Record the run timestamp and library versions at the top of the log."""
    log("=" * 78)
    log(f"NTA analysis run  {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"Python {platform.python_version()} | NumPy {np.__version__} | "
        f"pandas {pd.__version__} | SciPy {scipy.__version__} | "
        f"Matplotlib {matplotlib.__version__}")
    log("=" * 78)
def load_all_conditions() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read every condition and build the three core tables.

    Returns (raw_store, per-replicate metrics, per-mode decomposition,
    per-condition summary).
    """
    whole_rows, mode_rows, summary_rows = [], [], []
    raw_store: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

    for label, files in GROUPS.items():
        reps: list[tuple[np.ndarray, np.ndarray]] = []
        for path in files:
            if not path.exists():
                log(f"  x missing file, skipped: {path}")
                continue
            reps.append(load_csv(path))

        if not reps:
            log(f"  x no files loaded for condition '{label}', skipping")
            continue

        grid, concs = to_common_grid(reps, label)
        raw_store[label] = [(grid, c) for c in concs]

        rep_feats = []
        for r_idx, conc in enumerate(concs, start=1):
            feats = whole_features(grid, conc)
            feats.update({"condition": label, "replicate": r_idx})
            rep_feats.append(feats)
            whole_rows.append(feats)

            for m_idx, m in enumerate(decompose_modes(grid, conc,
                                                      tag=f"{label} rep {r_idx}"), start=1):
                mode_rows.append({"condition": label, "replicate": r_idx,
                                  "mode_index": m_idx, **m})

        summary_rows.append(summarise_condition(label, pd.DataFrame(rep_feats)))

    return (raw_store, pd.DataFrame(whole_rows),
            pd.DataFrame(mode_rows), pd.DataFrame(summary_rows))
def report_fit_diagnostics(df_modes: pd.DataFrame) -> None:
    """Flag any mode rows that came from a non-converged Gaussian fit."""
    n_bad = int((~df_modes["fit_converged"]).sum())
    log("\nFIT DIAGNOSTICS")
    if n_bad:
        log(f"  ! {n_bad} of {len(df_modes)} mode rows came from NON-CONVERGED fits; "
            f"those rows report unfitted peak seeds:")
        log(df_modes.loc[~df_modes["fit_converged"],
                         ["condition", "replicate", "mode_nm"]].to_string(index=False))
    else:
        log(f"  {len(df_modes)}/{len(df_modes)} fits converged, "
            f"R2 {df_modes['fit_r2'].min():.3f} to {df_modes['fit_r2'].max():.3f}")
def log_mean_curve_modes(modes_mean: pd.DataFrame, present: list[str]) -> None:
    """List the modes resolved on each averaged curve (Fig. 4, not Table 3)."""
    log("\nMODES OF THE AVERAGED CURVE (Fig. 4; not the Table 3 per-replicate census)")
    for lb in present:
        sub = modes_mean[modes_mean["condition"] == lb]
        peaks = ", ".join(
            f"{float(nm):.0f} nm ({100 * float(frac):.0f}%)"
            for nm, frac in zip(sub["mode_nm"].to_numpy(dtype=float),
                                sub["conc_fraction"].to_numpy(dtype=float))
        )
        log(f"  {lb:>6}: {len(sub)} modes | {peaks}")
def log_manuscript_numbers(df_modes: pd.DataFrame, df_summary: pd.DataFrame,
                           stats: dict, fb: pd.DataFrame) -> None:
    """Crib sheet of the values quoted in the manuscript text."""
    s_size, s_conc, s_conc_log = stats["size"], stats["conc"], stats["conc_log"]
    r2_min, r2_max = float(df_modes["fit_r2"].min()), float(df_modes["fit_r2"].max())
    cv_min = float(df_summary["conc_cv_pct"].min())
    cv_max = float(df_summary["conc_cv_pct"].max())

    log("\nMANUSCRIPT NUMBERS")
    log(f"  2.6  fits: all {len(df_modes)} converged, R2 {r2_min:.3f}-{r2_max:.3f}")
    log(f"  2.6  CV of concentration: {cv_min:.0f}-{cv_max:.0f}%")
    log(f"  3.1  ANOVA F({s_size['df_between']},{s_size['df_within']}) = {s_size['F']:.2f}, "
        f"p = {s_size['p_anova']:.3f}, eta2 = {s_size['eta2']:.2f}")
    log(f"  3.1  Kruskal-Wallis H({s_size['df_between']}) = {s_size['H']:.2f}, "
        f"p = {s_size['p_kw']:.3f}")
    log(f"  3.1  strongest contrast: {s_size['strongest_contrast']}, "
        f"t({s_size['strongest_df']:.1f}) = {s_size['strongest_t']:.2f}, "
        f"p = {s_size['strongest_p']:.3f} raw, {s_size['strongest_p_holm']:.3f} adj")
    log(f"  3.4  concentration ANOVA p = {s_conc['p_anova']:.3f}, "
        f"KW p = {s_conc['p_kw']:.3f}, log10 p = {s_conc_log['p_anova']:.3f}")
    frag = fb.loc[fb["quantity"] == "observed as % of predicted", "value"]
    if len(frag):
        log(f"  3.2  fragmentation recovery: {frag.iloc[0]} of predicted number")
def main() -> None:
    """Run the full analysis and write every table, figure and data export."""
    ensure_dirs()
    log_environment()

    raw_store, df_whole, df_modes, df_summary = load_all_conditions()
    if df_summary.empty:
        log("\nNo data loaded. Check BASE and the GROUPS file names, then rerun.")
        write_run_log()
        return

    write_data(df_whole, "per_replicate_metrics.csv")
    write_data(df_modes, "per_mode_decomposition.csv")
    write_data(df_summary, "condition_summary.csv")

    present = [lb for lb in TIME_LABELS if lb in set(df_summary["condition"])]

    # ---- fit diagnostics ----
    report_fit_diagnostics(df_modes)

    # ---- tables ----
    log("\nTABLE 1")
    log(build_table1(df_summary, present).to_string(index=False))

    t2, class_means = build_table2(df_whole, present)
    log("\nTABLE 2")
    log(t2.to_string(index=False))

    log("\nTABLE 3")
    log(build_table3(df_modes, present).to_string(index=False))

    dominant_mode_census(df_modes, present)

    # ---- statistics ----
    stats = run_condition_statistics(df_whole, present)
    diameter_concentration_correlation(df_whole, df_summary, present)

    # ---- balance checks ----
    fb = fragmentation_balance(class_means, df_summary, present)
    log("\nFRAGMENTATION BALANCE")
    log(fb.to_string(index=False))

    # ---- controls ----
    log("\nUNSONICATED CONTROLS")
    analyse_controls()

    # ---- figures ----
    log("\nFIGURES")
    write_data(combined_trajectory_figure(df_whole, present), "trajectory_values.csv")

    comp = write_data(composition_figure(df_summary, present), "size_class_fractions.csv")
    log("\nSIZE-CLASS COMPOSITION (%)")
    log((comp.set_index("condition") * 100).round(1).to_string())

    mean_psd, modes_mean = psd_panel_figure(raw_store, df_whole, present)
    write_data(mean_psd, "fig4_mean_psd.csv")
    log_mean_curve_modes(modes_mean, present)

    # ---- manuscript crib sheet ----
    log_manuscript_numbers(df_modes, df_summary, stats, fb)

    write_run_log()
    print(f"\nOutput written to: {OUT_DIR}")


if __name__ == "__main__":
    main()