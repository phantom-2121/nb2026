#!/usr/bin/env python3
"""
Particle-level NanoSight export processing and scattering-intensity comparison
=============================================================================

Reads the per-particle NanoSight exports (Particle ID / Size / Diffusion
coefficient / Frame / X / Y / Ln(Adjusted intensity) / Included in distribution),
collapses frame-level rows to one row per tracked particle, restores the
sample -> capture structure that the flat "Run 1..15" numbering discards, and
runs the scattering-intensity comparison against the unsonicated controls.

Why the intensity comparison matters: a gas body has a refractive index near
unity against about 1.33 for water, so at equal diameter it scatters more weakly
than a solid particle. Comparing Ln(Adjusted intensity) inside a size band where
both sonicated samples and controls carry signal is therefore a test of whether
the sonicated population is gas rather than shed solid material (cf. Ao et al.).

Outputs (written to OUT_DIR):
  combined/    <condition>_particle_data.csv     one row per tracked particle
  tables/      capture_inventory.csv             file -> sample/capture mapping
               validation_report.csv             every check and its result
               intensity_by_sample.csv           per-sample band statistics
               intensity_tests.csv               condition vs control tests
               session_check.csv                 acquisition-time grouping
  figures/     intensity_band.png/.tiff          distribution of ln intensity
  run_log.txt

Run:  python light_intensity_analysis.py
"""

from __future__ import annotations

import platform
import re
from datetime import datetime
from itertools import combinations
from typing import cast
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import mannwhitneyu

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

BASE = ROOT / "RD" / "light"
OUT_DIR = ROOT / "light_output"

GROUPS: dict[str, str] = {
    "1 min": "1min",
    "3 min": "3min",
    "5 min": "5min",
    "1 h": "1hr",
    "3 h": "3hr",
    "control": "control",
}
CONTROL_LABEL = "control"
CONDITION_ORDER = ["1 min", "3 min", "5 min", "1 h", "3 h", "control"]

# design: three independent samples per condition, five captures each
CAPTURES_PER_SAMPLE = 5
SAMPLES_PER_CONDITION = 3
EXPECTED_RUNS = CAPTURES_PER_SAMPLE * SAMPLES_PER_CONDITION

# If filenames encode the sample, give a regex with a group named "sample"
# (e.g. r"rep(?P<sample>\d)" for "1min_rep2_capture3.csv"). When None, captures
# are assigned in natural-sorted blocks of CAPTURES_PER_SAMPLE and the resulting
# mapping is printed so it can be checked against the acquisition notes.
# The exports name the tube in the filename: "... (redo)", "redo(1)", "redo(2)",
# and "tube 1/2/3 (milli-q control)". Both patterns are matched below, so samples
# are parsed rather than inferred from sort order. Set to None to fall back to
# natural-sorted blocks of CAPTURES_PER_SAMPLE.
SAMPLE_FROM_FILENAME: str | None = r"tube\s*(?P<sample>\d+)|redo\s*\(\s*(?P<sample_b>\d+)\s*\)"

# Filenames also carry the acquisition timestamp, e.g.
#   "1 min ultrasonic (redo) 2026-03-04 17-08-34_AllTracks.csv"
# Use it in preference to file mtime, which reflects when a file was last copied
# or re-exported and can place old data in a spurious recent session.
TIMESTAMP_FROM_FILENAME: str | None = r"(?P<ts>\d{4}-\d{2}-\d{2})[ _](?P<tm>\d{2}-\d{2}-\d{2})"

# scattering-intensity comparison
BAND_NM = (100.0, 300.0)     # size band where controls carry signal
INCLUDED_ONLY = True         # use only particles flagged as included in the PSD
RUN_INCLUDED_SENSITIVITY = True   # repeat the comparison with every tracked particle
MIN_PARTICLES_PER_SAMPLE = 20
ALPHA = 0.05

# treat files whose modification times fall within this many hours as one
# acquisition session (proxy for "same sitting on the instrument")
SESSION_GAP_HOURS = 6.0

FIG_DPI = 400
SAVE_TIFF = True

REQUIRED_COLUMNS = [
    "Particle ID",
    "Size/nm",
    "Diffusion coefficient/nm^2 s^-1",
    "Frame",
    "X/pixels",
    "Y/pixels",
    "Ln(Adjusted intensity)/AU",
    "Included in distribution?",
]
NUMERIC_COLUMNS = REQUIRED_COLUMNS[:-1]

plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

LOG: list[str] = []
CHECKS: list[dict] = []


def log(msg: str = "") -> None:
    print(msg)
    LOG.append(str(msg))


def check(name: str, passed: bool, detail: str = "") -> bool:
    CHECKS.append({"check": name, "result": "PASS" if passed else "FAIL", "detail": detail})
    if not passed:
        log(f"  ! CHECK FAILED — {name}: {detail}")
    return passed


def ensure_dirs() -> None:
    for sub in ("combined", "tables", "figures"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)


def natural_sort_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


# ═══════════════════════════════════════════════════════════════════════════════
#  READING
# ═══════════════════════════════════════════════════════════════════════════════
def read_nanosight_csv(path: Path) -> pd.DataFrame | None:
    """Read one particle-level export, with encoding fallback and column checking."""
    for enc in ("utf-8-sig", "latin1"):
        try:
            df = pd.read_csv(path, sep=",", encoding=enc)
            break
        except UnicodeDecodeError:
            continue
        except Exception as exc:                                   # noqa: BLE001
            log(f"  ERROR reading {path.name}: {type(exc).__name__}: {exc}")
            return None
    else:
        log(f"  ERROR reading {path.name}: no working encoding")
        return None

    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        log(f"  ! {path.name} missing columns: {missing}")
        log(f"    columns present: {list(df.columns)}")
        return None

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def assign_sample(run_number: int, filename: str) -> tuple[int, str]:
    """
    Sample index for a capture, parsed from the filename where possible.

    Returns (sample, source) so the inventory records whether the index was
    parsed or inferred. Block assignment is a fallback: it assumes the natural
    sort order groups captures by sample, which must be checked against the
    acquisition notes if it is ever used.
    """
    if SAMPLE_FROM_FILENAME:
        m = re.search(SAMPLE_FROM_FILENAME, filename, flags=re.IGNORECASE)
        if m:
            for key in ("sample", "sample_b"):
                if m.groupdict().get(key):
                    return int(m.group(key)), "filename"
        # a bare "(redo)" with no number is the first sample of its condition
        if re.search(r"redo\)", filename, flags=re.IGNORECASE) or \
           re.search(r"\((?:1|one)\)", filename, flags=re.IGNORECASE):
            return 1, "filename"
    return (run_number - 1) // CAPTURES_PER_SAMPLE + 1, "block"


def file_timestamp(path: Path) -> tuple[datetime, str]:
    """
    Acquisition time for a capture.

    The NanoSight export embeds the acquisition timestamp in the filename.
    That is preferred over st_mtime, which records when the file was last
    written and will place re-exported data in a spurious recent session.
    """
    if TIMESTAMP_FROM_FILENAME:
        m = re.search(TIMESTAMP_FROM_FILENAME, path.name)
        if m:
            try:
                return (datetime.strptime(f"{m.group('ts')} {m.group('tm')}",
                                          "%Y-%m-%d %H-%M-%S"), "filename")
            except ValueError:
                pass
    return datetime.fromtimestamp(path.stat().st_mtime), "mtime"


def normalise_included(series: pd.Series) -> pd.Series:
    """
    Map the 'Included in distribution?' column onto booleans.

    The original code compared against the literal string 'TRUE'. NanoSight
    exports also use 1/0, Yes/No and True/False, any of which silently produced
    'every particle excluded'. Unrecognised values are reported, not swallowed.
    """
    raw = series.astype(str).str.strip().str.upper()
    true_set = {"TRUE", "T", "YES", "Y", "1", "1.0", "INCLUDED"}
    false_set = {"FALSE", "F", "NO", "N", "0", "0.0", "EXCLUDED", "NAN", ""}
    unknown = sorted(set(raw.unique()) - true_set - false_set)
    if unknown:
        log(f"  ! unrecognised 'Included in distribution?' values, treated as excluded: {unknown}")
    return raw.isin(true_set)


# ═══════════════════════════════════════════════════════════════════════════════
#  PER-FILE PARTICLE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
def summarise_particles(df: pd.DataFrame, condition: str, sample: int,
                        capture: int, path: Path) -> pd.DataFrame:
    """Collapse frame-level rows to one row per tracked particle."""
    n_null_id = int(df["Particle ID"].isna().sum())
    if n_null_id:
        log(f"  ! {path.name}: {n_null_id} rows with no Particle ID, dropped")
        df = df.loc[df["Particle ID"].notna()].copy()

    df["Included_bool"] = normalise_included(df["Included in distribution?"])

    df["condition"] = condition
    df["sample"] = sample
    df["capture"] = capture
    df["source_file"] = path.name
    df["particle_uid"] = (f"{condition}|S{sample}|C{capture}|P"
                          + df["Particle ID"].astype("Int64").astype(str))

    out = (df.groupby(["condition", "sample", "capture", "source_file",
                       "Particle ID", "particle_uid"], dropna=False)
             .agg(size_nm=("Size/nm", "first"),
                  size_nm_sd=("Size/nm", "std"),          # should be 0/NaN per particle
                  diffusion_nm2_s=("Diffusion coefficient/nm^2 s^-1", "first"),
                  ln_intensity_mean=("Ln(Adjusted intensity)/AU", "mean"),
                  ln_intensity_median=("Ln(Adjusted intensity)/AU", "median"),
                  ln_intensity_sd=("Ln(Adjusted intensity)/AU", "std"),
                  start_frame=("Frame", "min"),
                  end_frame=("Frame", "max"),
                  track_length_frames=("Frame", "nunique"),   # frames, not rows
                  n_rows=("Frame", "size"),
                  included=("Included_bool", "first"))
             .reset_index())

    dup = int((out["n_rows"] - out["track_length_frames"]).gt(0).sum())
    if dup:
        log(f"  ! {path.name}: {dup} particles have repeated frames "
            f"(rows > distinct frames); track lengths use distinct frames")

    bad_size = int(out["size_nm_sd"].fillna(0).gt(1e-6).sum())
    if bad_size:
        log(f"  ! {path.name}: {bad_size} particles have a varying Size/nm across frames; "
            f"'first' was taken — inspect this file")

    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION CHECK
# ═══════════════════════════════════════════════════════════════════════════════
def session_check(inventory: pd.DataFrame) -> pd.DataFrame:
    """
    Group captures into acquisition sessions using file modification time.

    Section 4.4 of the manuscript reports the intensity comparison as
    inconclusive because the conditions that differed from the controls were
    acquired in different measurement sessions. This groups files by mtime gap
    so that claim can be checked rather than assumed.
    """
    inv = inventory.sort_values("mtime").reset_index(drop=True)
    session, last = 1, None
    ids = []
    for t in inv["mtime"]:
        if last is not None and (t - last).total_seconds() / 3600.0 > SESSION_GAP_HOURS:
            session += 1
        ids.append(session)
        last = t
    inv["session"] = ids

    tab = (inv.groupby(["session", "condition"])
              .agg(n_files=("source_file", "count"),
                   first=("mtime", "min"), last=("mtime", "max"))
              .reset_index())
    tab.to_csv(OUT_DIR / "tables" / "session_check.csv", index=False)

    ctrl_sessions = set(inv.loc[inv["condition"] == CONTROL_LABEL, "session"])
    log("\nAcquisition sessions (file mtime, gap > "
        f"{SESSION_GAP_HOURS:g} h starts a new session):")
    log(tab.to_string(index=False))
    if ctrl_sessions:
        for cond in [c for c in CONDITION_ORDER if c != CONTROL_LABEL]:
            cs = set(inv.loc[inv["condition"] == cond, "session"])
            shared = cs & ctrl_sessions
            log(f"  {cond:>7}: sessions {sorted(cs)}"
                f" | shares a session with controls: {'yes' if shared else 'NO'}")
        check("controls share at least one session with a sonicated condition",
              any(set(inv.loc[inv['condition'] == c, 'session']) & ctrl_sessions
                  for c in GROUPS if c != CONTROL_LABEL),
              "if NO for every condition, an intensity difference cannot be separated "
              "from a session-dependent instrumental difference")
    else:
        log("  ! no control files, so the session confound cannot be assessed")

    log("\n  NOTE: sessions use the acquisition timestamp embedded in each filename where"
        "\n  available. Any capture falling back to file mtime is flagged in "
        "capture_inventory.csv;\n  mtime records when a file was last written, not when it "
        "was acquired.")
    return inv


# ═══════════════════════════════════════════════════════════════════════════════
#  INTENSITY COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
def intensity_analysis(all_particles: pd.DataFrame, included_only: bool = True,
                       suffix: str = "") -> None:
    lo, hi = BAND_NM
    band = all_particles[(all_particles["size_nm"] >= lo) & (all_particles["size_nm"] <= hi)]
    if included_only:
        band = band[band["included"]]

    log("\n" + "=" * 78)
    log(f"SCATTERING INTENSITY IN THE {lo:.0f}-{hi:.0f} nm BAND"
        f"{' (included particles only)' if included_only else ' (ALL tracked particles)'}")
    log("=" * 78)

    if band.empty:
        log("  ! no particles in the band; check BAND_NM and the included flag")
        return

    per_sample = (band.groupby(["condition", "sample"])
                      .agg(n_particles=("ln_intensity_mean", "size"),
                           ln_intensity_mean=("ln_intensity_mean", "mean"),
                           ln_intensity_median=("ln_intensity_mean", "median"),
                           ln_intensity_sd=("ln_intensity_mean", "std"))
                      .reset_index())
    per_sample.to_csv(OUT_DIR / "tables" / f"intensity_by_sample{suffix}.csv", index=False)

    n_samp = per_sample.groupby("condition")["sample"].nunique()
    thin_cond = n_samp[n_samp < SAMPLES_PER_CONDITION]
    for cond, n in thin_cond.items():
        log(f"  ! {cond}: only {int(n)} samples available (design is "
            f"{SAMPLES_PER_CONDITION}); report this wherever the result is quoted")

    thin = per_sample[per_sample["n_particles"] < MIN_PARTICLES_PER_SAMPLE]
    for _, r in thin.iterrows():
        log(f"  ! {r['condition']} sample {int(r['sample'])}: only "
            f"{int(r['n_particles'])} particles in the band")

    log("\nPer-sample band statistics:")
    log(per_sample.to_string(index=False))

    if CONTROL_LABEL not in set(band["condition"]):
        log("\n  ! no control data, so no comparison was run")
        return

    ctrl = band.loc[band["condition"] == CONTROL_LABEL, "ln_intensity_mean"].to_numpy(float)
    ctrl_samples = per_sample.loc[per_sample["condition"] == CONTROL_LABEL,
                                  "ln_intensity_mean"].to_numpy(float)

    rows = []
    conds = [c for c in CONDITION_ORDER if c != CONTROL_LABEL and c in set(band["condition"])]
    for cond in conds:
        vals = band.loc[band["condition"] == cond, "ln_intensity_mean"].to_numpy(float)
        u_p = float(mannwhitneyu(vals, ctrl, alternative="two-sided").pvalue)

        samp = per_sample.loc[per_sample["condition"] == cond,
                              "ln_intensity_mean"].to_numpy(float)
        try:
            u_p_samp = float(mannwhitneyu(samp, ctrl_samples, alternative="two-sided").pvalue)
        except ValueError:
            u_p_samp = np.nan

        # rank-biserial effect size, direction: negative = weaker than control
        n1, n2 = len(vals), len(ctrl)
        u_stat = float(mannwhitneyu(vals, ctrl, alternative="two-sided").statistic)
        rbc = 2.0 * u_stat / (n1 * n2) - 1.0

        rows.append({"condition": cond,
                     "n_particles": n1, "n_control_particles": n2,
                     "median_ln_intensity": float(np.median(vals)),
                     "control_median": float(np.median(ctrl)),
                     "difference": float(np.median(vals) - np.median(ctrl)),
                     "rank_biserial": rbc,
                     "mannwhitney_p_particle_level": u_p,
                     "mannwhitney_p_sample_level": u_p_samp})

    tests = pd.DataFrame(rows)
    # Holm correction across the conditions tested
    p = tests["mannwhitney_p_particle_level"].to_numpy(float)
    order = np.argsort(p)
    adj, running = np.empty(len(p)), 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    tests["p_holm_particle_level"] = adj
    tests["n_samples"] = [int(per_sample[per_sample["condition"] == c]["sample"].nunique())
                          for c in tests["condition"]]
    tests.to_csv(OUT_DIR / "tables" / f"intensity_tests{suffix}.csv", index=False)

    log("\nCondition vs control (particle level, Holm-corrected):")
    log(tests.to_string(index=False))
    log("\n  Particle-level n is large, so small differences reach significance; the")
    log("  sample-level test (n = 3 vs 3) is the one that respects the design and")
    log("  cannot fall below p = 0.10. Read the two together, with the effect size.")

    sig = tests.loc[tests["p_holm_particle_level"] < ALPHA, "condition"].tolist()
    ns = tests.loc[tests["p_holm_particle_level"] >= ALPHA, "condition"].tolist()
    log(f"\n  Weaker than control after correction: {sig if sig else 'none'}")
    log(f"  Indistinguishable from control:        {ns if ns else 'none'}")

    if suffix == "":
        _intensity_figure(band, conds)


def _intensity_figure(band: pd.DataFrame, conds: list[str]) -> None:  # noqa: D103
    order = conds + [CONTROL_LABEL]
    data = [band.loc[band["condition"] == c, "ln_intensity_mean"].to_numpy(float)
            for c in order]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    parts = ax.violinplot(data, showmedians=True, widths=0.8)
    for pc in cast(list, parts["bodies"]):
        pc.set_facecolor("#3b6ea5")
        pc.set_alpha(0.35)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.0)

    for i, (c, vals) in enumerate(zip(order, data), start=1):
        x = np.random.default_rng(0).normal(i, 0.045, len(vals))
        ax.scatter(x, vals, s=4, alpha=0.15, color="#1f77b4", zorder=1, linewidths=0)

    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Ln(adjusted intensity) / AU")
    ax.text(0.99, 0.02,
            f"{BAND_NM[0]:.0f}-{BAND_NM[1]:.0f} nm band"
            + (", included particles" if INCLUDED_ONLY else ""),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5)

    fig.tight_layout()
    png = OUT_DIR / "figures" / "intensity_band.png"
    fig.savefig(png, dpi=FIG_DPI, bbox_inches="tight")
    if SAVE_TIFF:
        fig.savefig(OUT_DIR / "figures" / "intensity_band.tiff", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"\nSaved: {png.name}" + (" (+ .tiff)" if SAVE_TIFF else ""))


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ensure_dirs()
    log("=" * 78)
    log(f"Particle-level NanoSight processing  {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"Python {platform.python_version()} | NumPy {np.__version__} | "
        f"pandas {pd.__version__} | SciPy {scipy.__version__} | "
        f"Matplotlib {matplotlib.__version__}")
    log(f"Data root: {BASE}")
    log("=" * 78)

    inventory_rows, all_particles = [], []

    for condition, folder in GROUPS.items():
        log("\n" + "=" * 78)
        log(f"PROCESSING: {condition}   ({BASE / folder})")
        log("=" * 78)

        folder_path = BASE / folder
        if not folder_path.exists():
            log(f"  ! folder does not exist, skipped: {folder_path}")
            check(f"folder present: {condition}", False, str(folder_path))
            continue

        files = sorted(folder_path.glob("*.csv"), key=natural_sort_key)
        log(f"Found {len(files)} CSV file(s).")

        check(f"{condition}: file count is a whole number of samples",
              len(files) % CAPTURES_PER_SAMPLE == 0,
              f"{len(files)} files is not divisible by {CAPTURES_PER_SAMPLE} captures per sample")
        check(f"{condition}: expected {EXPECTED_RUNS} files",
              len(files) == EXPECTED_RUNS, f"found {len(files)}")

        tables = []
        seen_per_sample: dict[int, int] = {}
        for run_number, path in enumerate(files, start=1):
            sample, sample_src = assign_sample(run_number, path.name)
            seen_per_sample[sample] = seen_per_sample.get(sample, 0) + 1
            capture = seen_per_sample[sample]
            stamp, stamp_src = file_timestamp(path)

            log(f"  run {run_number:02d} -> sample {sample} ({sample_src}), "
                f"capture {capture} : {path.name}")
            inventory_rows.append({"condition": condition, "run_number": run_number,
                                   "sample": sample, "capture": capture,
                                   "source_file": path.name, "mtime": stamp,
                                   "sample_source": sample_src, "time_source": stamp_src})

            df = read_nanosight_csv(path)
            if df is None:
                log("    skipped")
                continue
            if len(df) == 0:
                log(f"    ! {path.name}: file parsed but contains no tracked particles")
                continue
            summ = summarise_particles(df, condition, sample, capture, path)
            if len(summ) == 0:
                log(f"    ! {path.name}: no particles survived summarisation")
                continue
            tables.append(summ)

        if not tables:
            log(f"  ! no usable files for {condition}")
            continue

        combined = (pd.concat(tables, ignore_index=True)
                      .sort_values(["sample", "capture", "Particle ID"])
                      .reset_index(drop=True))

        safe = condition.replace(" ", "_").replace("/", "_")
        combined.to_csv(OUT_DIR / "combined" / f"{safe}_particle_data.csv", index=False)

        n_inc = int(combined["included"].sum())
        log(f"\n  particles: {len(combined):,} | included: {n_inc:,} | "
            f"excluded: {len(combined) - n_inc:,}")
        log(f"  samples: {sorted(combined['sample'].unique())} | "
            f"captures per sample: "
            f"{combined.groupby('sample')['capture'].nunique().to_dict()}")
        check(f"{condition}: some particles flagged included", n_inc > 0,
              "every particle read as excluded — check the flag values above")
        check(f"{condition}: {SAMPLES_PER_CONDITION} samples resolved",
              combined["sample"].nunique() == SAMPLES_PER_CONDITION,
              f"resolved {combined['sample'].nunique()}")

        all_particles.append(combined)

    if not all_particles:
        log("\nNo data loaded. Check BASE and the folder names, then rerun.")
        (OUT_DIR / "run_log.txt").write_text("\n".join(LOG) + "\n")
        return

    particles = pd.concat(all_particles, ignore_index=True)
    inventory = pd.DataFrame(inventory_rows)
    inventory.to_csv(OUT_DIR / "tables" / "capture_inventory.csv", index=False)

    log("\n" + "=" * 78)
    log("SAMPLE ASSIGNMENT")
    log("=" * 78)
    n_parsed = int((inventory["sample_source"] == "filename").sum())
    n_block = int((inventory["sample_source"] == "block").sum())
    log(f"  Sample index parsed from the filename for {n_parsed} of {len(inventory)} captures; "
        f"{n_block} fell back to natural-sorted blocks.")
    if n_block:
        log("  CHECK capture_inventory.csv for the block-assigned captures: if the export")
        log("  order does not group captures by sample, that assignment is wrong.")
    n_ts = int((inventory["time_source"] == "filename").sum())
    log(f"  Acquisition time read from the filename for {n_ts} of {len(inventory)} captures; "
        f"{len(inventory) - n_ts} fell back to file mtime.")

    session_check(inventory)
    intensity_analysis(particles, included_only=INCLUDED_ONLY)

    if RUN_INCLUDED_SENSITIVITY:
        log("\n" + "=" * 78)
        log("SENSITIVITY: same comparison using EVERY tracked particle")
        log("=" * 78)
        log("  Only a small fraction of tracked particles is flagged as included in the")
        log("  PSD, so the filter itself could drive the result. If the two passes agree,")
        log("  the conclusion does not depend on the instrument's inclusion criterion.")
        intensity_analysis(particles, included_only=False, suffix="_all_particles")

    checks = pd.DataFrame(CHECKS)
    checks.to_csv(OUT_DIR / "tables" / "validation_report.csv", index=False)
    n_fail = int((checks["result"] == "FAIL").sum())
    log("\n" + "=" * 78)
    log(f"VALIDATION: {len(checks) - n_fail} passed, {n_fail} failed")
    log("=" * 78)
    if n_fail:
        log(checks[checks["result"] == "FAIL"].to_string(index=False))

    (OUT_DIR / "run_log.txt").write_text("\n".join(LOG) + "\n")
    print(f"\nAll output written to: {OUT_DIR}")


if __name__ == "__main__":
    main()