import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# =============================================================================
# Paths
# =============================================================================
# NOTE: the original line was
#     Path(__file__).parent / "/Users/amaansiddiqi/.../Sheet6.csv"
# pathlib discards the left-hand side whenever the right-hand side is
# absolute, so `.parent` was doing nothing. Written plainly below.
DEFAULT_ZETA_CSV = Path(
    "/Users/amaansiddiqi/Documents/Nanobubble Analysis/ZS-NS SUMMARY - Sheet6.csv"
)

ZETA_CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ZETA_CSV
OUT_DIR  = Path(__file__).parent

TIME_LABELS = ["1 min", "3 min", "5 min", "1 h", "3 h"]
TIME_VALS   = [1, 3, 5, 60, 180]

N_RUNS = 3          # instrument runs per sample (R1, R2, R3 columns)
DDOF   = 1          # sample SD

plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# =============================================================================
# Functions
# =============================================================================
def _safe(val):
    """Blank string for missing cells, so downstream parsing never sees NaN."""
    return "" if pd.isna(val) else str(val)
def _nums(raw):
    """Parse a comma-separated cell into floats, dropping blanks and N/A."""
    raw = _safe(raw).strip()

    if raw.upper() in ("N/A", "", "NAN"):
        return []

    out = []
    for v in raw.split(","):
        v = v.strip()

        if not v or v.lower() in ("n/a", "nan"):
            continue

        try:
            out.append(float(v))
        except ValueError:
            pass

    return out
def _match(s):
    """Map a free-text sample label onto one of the five conditions."""
    s = str(s).lower()

    if "1 min" in s:
        return "1 min"
    if "3 min" in s:
        return "3 min"
    if "5 min" in s:
        return "5 min"
    if "1 h" in s:
        return "1 h"
    if "3 h" in s:
        return "3 h"

    return None
def dominant_negative_peak(peaks, areas):
    """
    Zeta potential of the negative peak carrying the largest area.

    Returns None when the row has no usable negative peak, so the caller can
    fall back to the summary ZP column.
    """
    if not peaks or not areas or len(peaks) != len(areas):
        return None

    negative = [(p, a) for p, a in zip(peaks, areas) if p < 0]

    if not negative:
        return None

    return max(negative, key=lambda pa: pa[1])[0]
# =============================================================================
# Data Loading
# =============================================================================
if not ZETA_CSV.exists():
    raise FileNotFoundError(f"Could not find:\n{ZETA_CSV}")

dfz = pd.read_csv(ZETA_CSV)
dfz.columns = [c.strip() for c in dfz.columns]

records = []
n_from_peaks, n_from_summary, n_dropped = 0, 0, 0

for _, row in dfz.iterrows():

    condition = _match(row.get("Sample Type", ""))

    if condition is None:
        continue

    sample_num = int(row.get("Sample #", 0))
    zp_vals = _nums(row.get("ZP (mV)", ""))

    for run in range(1, N_RUNS + 1):

        peaks = _nums(row.get(f"R{run} Peaks (mV)", "N/A"))
        areas = _nums(row.get(f"R{run} Area (%)", "N/A"))

        zp = dominant_negative_peak(peaks, areas)
        source = "peak"

        if zp is None:
            # fall back to the run's entry in the summary ZP column
            if len(zp_vals) >= run and zp_vals[run - 1] < 0:
                zp, source = zp_vals[run - 1], "summary"
            else:
                zp, source = np.nan, "none"

        if np.isnan(zp):
            n_dropped += 1
            continue

        n_from_peaks += (source == "peak")
        n_from_summary += (source == "summary")

        records.append({
            "condition": condition,
            "sample_num": sample_num,
            "run": run,
            "zp": zp,
            "source": source,
        })

df = pd.DataFrame(records)

if df.empty:
    raise ValueError(
        "No usable negative zeta potential values were parsed. Check that the "
        "column names in the CSV match 'Sample Type', 'Sample #', 'ZP (mV)' "
        "and 'R{n} Peaks (mV)' / 'R{n} Area (%)'."
    )

print(f"\nParsed {len(df)} run-level values "
      f"({n_from_peaks} from peak/area, {n_from_summary} from the ZP column); "
      f"{n_dropped} runs had no usable negative value.")


# =============================================================================
# Averaging runs
# =============================================================================
df_sample = (
    df.groupby(["condition", "sample_num"])["zp"]
      .agg(sample_mean="mean", n_runs="count")
      .reset_index()
)


# =============================================================================
# Condition Mean ± SD Summary
# =============================================================================
stats = []

for label in TIME_LABELS:

    values = (
        df_sample.loc[df_sample["condition"] == label, "sample_mean"]
        .dropna()
        .to_numpy(dtype=float)
    )

    n = len(values)

    stats.append({
        "condition": label,
        "mean_mV": np.mean(values) if n else np.nan,
        "sd_mV": np.std(values, ddof=DDOF) if n > DDOF else np.nan,
        "n_samples": n,
        "min_mV": np.min(values) if n else np.nan,
        "max_mV": np.max(values) if n else np.nan,
    })

df_stats = pd.DataFrame(stats)

print("\nZeta Potential Summary  (error term = sample SD, ddof=1)")
print(df_stats.round(2).to_string(index=False))

if (df_stats["n_samples"] <= DDOF).any():
    thin = df_stats.loc[df_stats["n_samples"] <= DDOF, "condition"].tolist()
    print(f"\n  !  SD undefined (n <= 1) for: {', '.join(thin)}. "
          "Those points are plotted without an error bar.")

df_sample.to_csv(OUT_DIR / "zeta_sample_means.csv", index=False)
df_stats.to_csv(OUT_DIR / "zeta_condition_stats.csv", index=False)
# =============================================================================
# Fig 6
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 5))

means = df_stats["mean_mV"].to_numpy(dtype=float)
sds   = np.nan_to_num(df_stats["sd_mV"].to_numpy(dtype=float), nan=0.0)

ax.errorbar(
    TIME_VALS,
    means,
    yerr=sds,
    fmt="o-",
    color="black",
    linewidth=2,
    markersize=7,
    capsize=5,
    capthick=1.2,
    elinewidth=1.2,
    label="Mean $\\pm$ SD",
)

ax.axhline(
    0,
    color="gray",
    linestyle=":",
    linewidth=0.8,
)

ax.set_xscale("log")
ax.set_xlim(min(TIME_VALS) * 0.75, max(TIME_VALS) * 1.35)
ax.set_xticks(TIME_VALS)
ax.xaxis.set_major_formatter(mticker.FixedFormatter(TIME_LABELS))
ax.xaxis.set_minor_locator(mticker.NullLocator())
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.tick_params(axis="x", which="minor", bottom=False)
ax.set_xlabel("Sonication time")
ax.set_ylabel("Zeta potential (mV)")
lower = float(np.nanmin(means - sds))
ax.set_ylim(min(-50.0, 1.12 * lower), 0.0)
ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
ax.legend(loc="lower right", frameon=True, framealpha=0.9)
plt.tight_layout()
# =============================================================================
# File saver
# =============================================================================
outfile = OUT_DIR / "zeta_potential_vs_time.png"
fig.savefig(outfile, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved figure to:\n{outfile}")
print(f"Saved tables to:\n{OUT_DIR / 'zeta_sample_means.csv'}"
      f"\n{OUT_DIR / 'zeta_condition_stats.csv'}")