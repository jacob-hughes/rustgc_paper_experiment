#! /usr/bin/env python

import csv
import gc
import glob
import math
import os
import pprint
import random
import sys
import warnings
from os import listdir, stat
from pathlib import Path
from statistics import geometric_mean, stdev

warnings.simplefilter(action="ignore", category=FutureWarning)
import matplotlib
import matplotlib.lines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from scipy import stats

PLOT_DIR = None
DRYRUN = os.environ.get("DRYRUN", "false") in ["true", "on", "1", "yes"]
RESULTS_DIR = None
STATS_FILE = None
PEXECS = int(os.environ["PEXECS"])
EXPERIMENTS = os.environ.get("EXPERIMENTS", "").split()
BENCHMARKS = os.environ.get("BENCHMARKS", "").split()
PLOTS = os.environ.get("METRICS", "").split()
BOOTSTRAP = os.environ.get("BOOTSTRAP", "true") in ["true", "on", "1", "yes"]
Z = 2.576  # 99% interval

# ============== HELPERS ================


def print_success(message):
    print(f"\033[92m✓ {message}\033[0m")


def print_warning(message):
    """Print a warning message in yellow."""
    print(f"\033[93m⚠ {message}\033[0m")


def print_info(message):
    print(f"\033[94mℹ {message}\033[0m")


def print_error(message):
    print(f"\033[91m✗ {message}\033[0m")


def bytes_formatter(max_value):
    units = [
        ("B", 1),
        ("KiB", 1024),
        ("MiB", 1024 * 1024),
        ("GiB", 1024 * 1024 * 1024),
    ]

    for unit, factor in reversed(units):
        if max_value >= factor:
            break

    def format_func(x, pos):
        return f"{x/factor:.2f}"

    return FuncFormatter(format_func), unit


def format_number(number):
    suffixes = ["", "K", "M", "B", "T"]
    magnitude = 0
    while abs(number) >= 1000 and magnitude < len(suffixes) - 1:
        number /= 1000.0
        magnitude += 1
    return f"{number:.1f}{suffixes[magnitude]}".replace(".0", "")


def format_bytes(number):
    suffixes = ["B", "KiB", "MiB", "GiB", "TiB"]
    magnitude = 0
    while abs(number) >= 1000 and magnitude < len(suffixes) - 1:
        number /= 1000.0
        magnitude += 1
    return f"{number:.2f}{suffixes[magnitude]}".replace(".00", "")


def ltxify(s):
    s = s.replace("_", " ").replace("-", " ")
    return " ".join(word.capitalize() for word in s.split())


def fmt_value_ci(row):
    # lower = row['lower']:.2f
    # upper = row['upper']:.2f
    return f"{row['value']:.2f} ({row['lower']:.2f}-{row['upper']:.2f})"


def ltx_value_ci(row):
    s = f"{row['value']:.2f} \\footnotesize{{({row['lower']:.2f}-{row['upper']:.2f})}}"
    return s


def format_value_ci_asym(row):
    if math.isnan(row["ci"]):
        return "-"
    s = f"row['value']:.2f row['ci']"
    return s


def format_mem_ci(row):
    return f"{format_bytes(row['value'])} \\footnotesize{{± {format_bytes(row['ci'])}}}"


# ============== PLOT FORMATTING =================

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        # LaTeX and font settings
        "text.usetex": True,
        "svg.fonttype": "none",
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "sans-serif",
        # Basic graph axes styling
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.xmargin": 0,
        "axes.ymargin": 0,
        # Grid and line settings
        "lines.linewidth": 1,
        "grid.linewidth": 0.25,
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        # Tick settings
        "xtick.bottom": True,
        "ytick.left": True,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        # Legend and figure settings
        "legend.title_fontsize": 0,
        "errorbar.capsize": 2,
    }
)

SUITES = {
    "som-rs-ast": r"\somrsast",
    "som-rs-bc": r"\somrsbc",
    "alacritty": r"\alacritty",
    "yksom": r"\yksom",
    "grmtools": r"\grmtools",
    "binary-trees": r"\binarytrees",
    "binary-t": r"\binarytrees",  # it's a mem error
    "grmtool": r"\grmtools",  # it's a mem error
    "regex-redux": r"\regexredux",
    "ripgrep": r"\ripgrep",
    "fd": r"\fd",
}

CFGS = {
    "gcvs-gc": "Alloy",
    "gcvs-rc": "RC",
    "gcvs-arc": "ARC",
    "gcvs-typed-arena": "Typed Arena",
    "gcvs-typed_arena": "Typed Arena",
    "gcvs-rust-gc": "Rust-GC",
    "premopt-opt": "Barriers Opt",
    "premopt-naive": "Barriers Naive",
    "premopt-none": "Barriers None",
    "premopt-opt": "Barriers Opt",
    "elision-naive": "Elision Naive",
    "elision-opt": "Elision Opt",
}

# METRICS = {
#     "finalizers registered": "Finalizable Objects",
#     "finalizers completed": "Total Finalized",
#     "barriers visited": "Barrier Chokepoints",
#     "Gc allocated": "Allocations (Gc)",
#     "Box allocated": "Allocations (Box)",
#     "Rc alocated": "Allocations (Rc)",
#     "Arc allocated": "Allocations (Arc)",
#     "STW pauses": r"Gc Cycles",
# }

BASELINE = {
    "som-rs-ast": "gcvs-rc",
    "som-rs-bc": "gcvs-rc",
    "grmtool": "gcvs-rc",
    "grmtools": "gcvs-rc",
    "binary-t": "gcvs-arc",
    "binary-trees": "gcvs-arc",
    "regex-redux": "gcvs-arc",
    "alacritty": "gcvs-arc",
    "fd": "gcvs-arc",
    "ripgrep": "gcvs-arc",
    "premopt": "premopt-none",
    "elision": "elision-naive",
}

PERF_PLOT_WIDTHS = {
    "alacritty": 8,
    "binary-trees": 8,
    "regex-redux": 8,
    "ripgrep": 8,
    "static-web-server": 8,
    "som": 8,
    "grmtools": 8,
    "fd": 8,
}

PROFILE_PLOTS = {
    "grmtools": {"r": 1, "c": 4},
    "alacritty": {"r": 1, "c": 4},
    "som": {"r": 7, "c": 4},
    "som": {"r": 7, "c": 4},
}

# ============== STATISTICS =================


def pdiff(a, b):
    return (a / (a + b)) * 100


def ci(row, pexecs):
    return Z * (row / math.sqrt(pexecs))


def ci_inl(value, std, pexecs):
    ci = Z * (value / math.sqrt(pexecs))
    lower = value - ci
    upper = value + ci
    return pd.Series({"value": value, "ci": ci, "upper": upper, "lower": lower})


def bootstrap(
    values, kind, method, num_bootstraps=10000, confidence=0.99, symmetric=True
):

    # if DRYRUN:
    #     # This should never be used for real, but it's useful to prevent things
    #     # taking forever when trying to quickly debug the script
    #     num_bootstraps = 1000

    res = stats.bootstrap(
        (values,),
        statistic=kind,
        n_resamples=num_bootstraps,
        confidence_level=confidence,
        method=method,
        vectorized=True,
    )

    value = kind(values)
    ci_lower, ci_upper = res.confidence_interval
    if symmetric:
        margin = max(value - ci_lower, ci_upper - value)
        data = {
            "value": value,
            "ci": margin,
        }
    else:
        data = {
            "value": value,
            "lower": res.confidence_interval.low,
            "upper": res.confidence_interval.high,
        }
    return pd.Series(data)


def bootstrap_geomean_ci(means, num_bootstraps=10000, confidence=0.99, symmetric=True):
    # We use the BCa (bias-corrected and accelerated) bootstrap method. This
    # can provide more accurate CIs over the more straightforward percentile
    # method but it is more computationally expensive -- though this doesn't
    # matter so much when we run this using PyPy.
    #
    # This is generally better for smaller sample sizes such as ours (where the
    # number of pexecs < 100), and where the dataset is not known to be
    # normally distributed.
    #
    # We could also consider using the studentized bootstrap method which
    # libkalibera tends to prefer when deealing with larger sample sizes.
    # Though this is more computationally expensive and the maths looks a bit
    # tricky to get right!
    method = "Bca"
    return bootstrap(means, stats.gmean, method, num_bootstraps, confidence, symmetric)


def geomean_with_ci(row, confidence=0.99):
    """Calculate geometric mean and CI for a DataFrame row"""
    # Clean data and validate
    clean_vals = row.dropna()
    n = len(clean_vals)

    # Handle edge cases
    if n == 0 or (clean_vals <= 0).any():
        return pd.Series([np.nan] * 3, index=["value", "lower", "upper"])

    # Log-transform and calculate statistics
    log_vals = np.log(clean_vals)
    mean_log = np.mean(log_vals)
    std_log = np.std(log_vals, ddof=1)  # Sample standard deviation

    # Calculate confidence interval
    if n == 1:
        return pd.Series(
            [np.exp(mean_log), np.nan, np.nan],
            index=["value", "lower", "upper"],
        )

    sem_log = std_log / np.sqrt(n)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)

    ci_log = (mean_log - t_crit * sem_log, mean_log + t_crit * sem_log)

    # Convert back to original scale
    return pd.Series(
        [np.exp(mean_log), np.exp(ci_log[0]), np.exp(ci_log[1])],
        index=["value", "lower", "upper"],
    )


def arith_mean_ci(series):
    n = len(series)
    mean = series.mean()
    std_err = series.std(ddof=1) / (n**0.5)  # Standard error
    margin_of_error = stats.t.ppf((1 + 0.99) / 2, df=n - 1) * std_err  # t-score * SE
    return pd.Series(
        {
            "value": mean,
            "ci": margin_of_error,
            "lower": mean - margin_of_error,
            "upper": mean + margin_of_error,
        }
    )


def normalize_time(df):
    group["normalized_time"] = (group["timestamp"] - group["timestamp"].min()) / (
        group["time"].max() - group["time"].min()
    )
    return group


def aggregate(grouped, col, method, unstack=True):
    df = grouped[col].apply(method).unstack()
    if unstack:
        df = df.unstack()
    else:
        df = df.reset_index()
    return (df["value"], df["ci"])


def normalize(df, baseline_col):
    timecol = "normalized_time"
    df[timecol] = df[timecol].astype(float)

    normcols = [
        "mem",
        "mem_ci",
        "peak_heap_usage",
        "peak_heap_usage_ci",
        "mean_heap_usage",
        "mean_heap_usage_ci",
    ]
    cmps = df["configuration"][~(df["configuration"] == baseline_col)].unique()

    baseline = (
        df[df["configuration"] == baseline_col]
        .sort_values(timecol)
        .reset_index(drop=True)
        .set_index(timecol)
        .sort_index()
    )

    def find_nearest(time_value):
        idx = np.abs(baseline.index - time_value).argmin()
        return baseline.iloc[idx]

    def normalize_value(row, value_col, nearest):
        if baseline.empty:
            return np.nan
        # nearest = find_nearest(row[timecol])
        return row[value_col] / nearest[value_col]

    def normalize_ci(row, value_col, ci_col, timecol):
        if baseline.empty:
            return np.nan
        nearest = find_nearest(row[timecol])
        normalized_value = row[value_col] / nearest[value_col]
        return (
            np.sqrt(
                (row[ci_col] / row[value_col]) ** 2
                + (nearest[ci_col] / nearest[value_col]) ** 2
            )
            * normalized_value
            * Z
        )

    for value_col, ci_col in zip(normcols[::2], normcols[1::2]):
        df.loc[df["configuration"].isin(cmps), value_col] = df[
            df["configuration"].isin(cmps)
        ].apply(
            lambda row: normalize_value(row, value_col, find_nearest(row[timecol])),
            axis=1,
        )
        df.loc[df["configuration"].isin(cmps), ci_col] = df[
            df["configuration"].isin(cmps)
        ].apply(lambda row: normalize_ci(row, value_col, ci_col, timecol), axis=1)

    df = df.drop(df[df["configuration"] == baseline_col].index)
    return df


def normalize_time(df):
    for (c, b, p), group in df.groupby(["configuration", "benchmark", "pexec"]):
        min = group["time"].min()
        max = group["time"].max()
        idxs = group.index

        # Normalize time to 0-1 scale
        df.loc[idxs, "normalized_time"] = (df.loc[idxs, "time"] - min) / (max - min)
    return df


def interpolate(df, oversampling=1):
    interpolated = []

    for (c, b, p), group in df.groupby(["configuration", "benchmark", "pexec"]):
        samples = int(group["snapshot"].max() * oversampling)
        # print(f"Interpolating {c} with {samples} samples.")
        dist = np.linspace(0, 1, samples)
        # Aggregate duplicate normalized time values by calculating mean
        aggregated = (
            group.sort_values("normalized_time")
            .groupby("normalized_time")["mem"]
            .mean()
        )

        # Reindex to standard time points and interpolate
        series = (
            aggregated.reindex(index=np.union1d(aggregated.index, dist))
            .interpolate(method="linear")
            .loc[dist]
        )
        interpolated.append(
            pd.DataFrame(
                {
                    "configuration": c,
                    "benchmark": b,
                    "pexec": p,
                    "normalized_time": dist,
                    "mem": series.values,
                }
            )
        )

    # Concatenate all interpolated dataframes
    df = pd.concat(interpolated, ignore_index=True)

    df = (
        df.groupby(["configuration", "benchmark", "normalized_time"])["mem"]
        .agg(mem=("mean"), mem_ci=(lambda x: ci(x.std(), PEXECS)))
        .reset_index()
    )

    peak_memory, peak_ci = aggregate(
        df.groupby(["configuration", "benchmark"]),
        "mem",
        bootstrap_max_ci,
        unstack=True,
    )
    peak_memory = peak_memory.unstack().reset_index()
    peak_memory.rename(columns={0: "peak_heap_usage"}, inplace=True)
    peak_ci = peak_ci.unstack().reset_index()
    peak_ci.rename(columns={0: "peak_heap_usage_ci"}, inplace=True)
    df = df.merge(peak_memory, on=["configuration", "benchmark"])
    df = df.merge(peak_ci, on=["configuration", "benchmark"])

    mean_memory, mean_ci = aggregate(
        df.groupby(["configuration", "benchmark"]),
        "mem",
        arith_mean_ci,
        unstack=True,
    )
    mean_memory = mean_memory.unstack().reset_index()
    mean_memory.rename(columns={0: "mean_heap_usage"}, inplace=True)
    mean_ci = mean_ci.unstack().reset_index()
    mean_ci.rename(columns={0: "mean_heap_usage_ci"}, inplace=True)
    df = df.merge(mean_memory, on=["configuration", "benchmark"])
    df = df.merge(mean_ci, on=["configuration", "benchmark"])
    return df


# ============== GRAPHS =================


def write_stat(stat):
    with open(STATS_FILE, "a") as f:
        f.write(stat + "\n")


def write_stats(df, experiment, fmt, summary=False):
    def ltxcmd(kind):
        ltxmap = {
            "best": "best",
            "worst": "worst",
            "best_all": "bestsi",
            "worst_all": "worstsi",
        }
        return ltxmap[kind]

    ltxfmt = {"perf": lambda x: f"{x:0.2f}", "mem": format_bytes}

    df = df.fillna("")

    for idx, row in df.iterrows():
        if summary:
            write_stat(f"% Summary stats: {experiment}:{idx}:{fmt}")
            latex_name = experiment + fmt + idx.split("-")[1]
        else:
            latex_name = (
                experiment + fmt + idx[0].replace("-", "") + idx[1].split("-")[1]
            )
            write_stat(
                f"% Config stats: {experiment}:{idx[0]}:{idx[1].split('-')[1]}:{fmt}"
            )
        for (kind, name), value in row.items():
            if not value:
                continue
            if name == "diff_pct":
                write_stat(
                    f"\\newcommand\\{latex_name}{ltxcmd(kind)}pct{{{value:0.2f}\\%\\xspace}}"
                )
            elif name == "benchmark" and not summary:
                write_stat(
                    f"\\newcommand\\{latex_name}{ltxcmd(kind)}benchmark{{{ltxify(value)}\\xspace}}"
                )
            elif name == "suite" and summary:
                write_stat(
                    f"\\newcommand\\{latex_name}{ltxcmd(kind)}suite{{{SUITES[value]}\\xspace}}"
                )
            elif name == "value":
                write_stat(
                    f"\\newcommand\\{latex_name}{ltxcmd(kind)}value{{{ltxfmt[fmt](value)}\\xspace}}"
                )
        write_stat("")


def write_table(outfile, df, include_html=True):
    df["ltxval"] = df.apply(ltx_value_ci, axis=1)
    ltxtable = df.pivot(
        index="suite",
        columns="configuration",
        values="ltxval",
    ).fillna("-")

    latex_tabular = ltxtable.to_latex(
        index=True,
        escape=False,
        column_format="l" + "r" * len(df.columns),
        caption=None,
        label=None,
        header=True,
        position=None,
    )

    # Removes lines before \begin{tabular} and after \end{tabular}
    latex_tabular = "\n".join(
        line
        for line in latex_tabular.split("\n")
        if "begin{table}" not in line and "end{table}" not in line
    )

    with open(outfile, "w") as f:
        f.write(latex_tabular)

    print_success(f"Plotted table: {outfile.parts[-2]}:{outfile.stem.replace('_',':')}")

    if not include_html:
        return

    df["valci"] = df.apply(fmt_value_ci, axis=1)
    df = df.pivot(
        index="suite",
        columns="configuration",
        values="valci",
    )
    print(df)
    df = df.fillna("-")
    t = outfile.parts[-2].upper() + " Summary"
    html = f"""
        <html>
        <head>
            <title>{t}</title>
        </head>
        <body>
            <h2>{t}</h2>
            {df.to_html()}
        </body>
        </html>
    """

    with open(outfile.with_suffix(".html"), "w") as f:
        f.write(html)


def plot_perf(outfile, values, rows, cols):
    if DRYRUN:
        return
    values = values.copy().rename(columns=CFGS)
    # values["configuration"] = values["configuration"].replace(CFGS)
    fig, axes = plt.subplots(
        rows, cols, figsize=(PERF_PLOT_WIDTHS[outfile.parts[-2]], rows * 3)
    )
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    num_cfgs = values["configuration"].nunique()
    colours = [sns.color_palette("colorblind")[i] for i in range(num_cfgs)]

    formatter = ScalarFormatter()
    formatter.set_scientific(False)

    num_benchmarks = values["benchmark"].nunique()

    for i, (suite, results) in enumerate(values.groupby("suite")):
        ax = axes[i]
        ax.set_title(f"{suite}")

        df = results.pivot(
            index="benchmark", columns="configuration", values=["value", "ci"]
        )
        df.plot(kind="bar", y="value", yerr="ci", ax=axes[i], width=0.8)

        ax.legend().set_title(None)
        if i != 0:
            ax.legend().set_visible(False)

        ax.set_xticklabels(df.index, rotation=45, ha="right")
        ax.set_ylabel("Wall-clock time (ms)\n(lower is better)")
        ax.xaxis.label.set_visible(False)
        ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.savefig(outfile, format="svg", bbox_inches="tight")
    print_success(
        f"Plotted graph: {outfile.parts[-3]}:{outfile.parts[-2]}:{outfile.stem}:individual"
    )
    plt.close()


def plot_perf_aggregate(outfile, values, width):
    # values["configuration"] = values["configuration"].replace(CFGS)
    fig, ax = plt.subplots(figsize=(width, 4))

    num_cfgs = values["configuration"].nunique()
    colours = [sns.color_palette("colorblind")[i] for i in range(num_cfgs)]

    formatter = ScalarFormatter()
    formatter.set_scientific(False)

    num_benchmarks = values["benchmark"].nunique()

    for i, (suite, results) in enumerate(values.groupby("suite")):
        ax = axes[i]
        ax.set_title(f"{suite}")

        df = results.pivot(
            index="benchmark", columns="configuration", values=["value", "ci"]
        )
        df.plot(kind="bar", y="value", yerr="ci", ax=axes[i], width=0.8)

        ax.legend().set_title(None)
        if i != 0:
            ax.legend().set_visible(False)

        ax.set_xticklabels(df.index, rotation=45, ha="right")
        ax.set_ylabel("Wall-clock time (ms)\n(lower is better)")
        ax.xaxis.label.set_visible(False)
        ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.savefig(outfile, format="svg", bbox_inches="tight")
    print_success(
        f"Plotted graph: {outfile.parts[-3]}:{outfile.parts[-2]}:{outfile.stem}:individual"
    )


def plot_mem_time_series(outfile, benchmarks, rows, cols, cmp=False):
    if DRYRUN:
        return
    benchmarks["configuration"] = benchmarks["configuration"].replace(CFGS)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    num_cfgs = benchmarks["configuration"].nunique()
    num_benchmarks = benchmarks["benchmark"].nunique()
    colours = [sns.color_palette("colorblind")[i] for i in range(num_cfgs)]

    formatter, unit = bytes_formatter(np.max(benchmarks["mem"].max()))

    for i, (bench, results) in enumerate(benchmarks.groupby("benchmark")):
        ax = axes[i]
        ax.set_title(f"{bench}")
        if not cmp:
            ax.yaxis.set_major_formatter(FuncFormatter(formatter))
        mems = []
        for j, (cfg, samples) in enumerate(results.groupby("configuration")):
            samples = samples.sort_values("normalized_time")
            (real,) = ax.plot(
                samples["normalized_time"],
                samples["mem"],
                color=colours[j],
            )

            ax.fill_between(
                samples["normalized_time"],
                samples["mem"] - samples["mem_ci"],
                samples["mem"] + samples["mem_ci"],
                alpha=0.2,
                color=colours[j],
            )

            # Plot mean heap usage as line
            mean = samples["mean_heap_usage"].iloc[0]
            mean_ci = samples["mean_heap_usage_ci"].iloc[0]
            mean = ax.axhline(
                y=mean,
                color=colours[j],
                linestyle=":",
                alpha=0.7,
            )

            if cmp:
                ax.axhline(
                    y=1,
                    color="grey",
                    linestyle="-",
                    alpha=0.5,
                )

            # Plot peak heap usage as line
            peak = samples["peak_heap_usage"].iloc[0]
            peak_ci = samples["peak_heap_usage_ci"].iloc[0]
            peak = ax.axhline(
                y=peak,
                color=colours[j],
                linestyle="--",
                alpha=0.5,
            )

            mems.append(real)

    # Remove extra subplots
    for i in range(num_benchmarks, rows * cols):
        fig.delaxes(axes[i])

    if not cmp:
        fig.supylabel(f"Memory usage ({unit}s)", y=0.5, x=0.02, rotation=90)
    fig.supxlabel(f"Normalized Time", x=0.51)

    fig.legend(
        handles=[
            Line2D([], [], color=col, label=f"{cfg}", linestyle="-")
            for col, cfg in zip(colours, benchmarks["configuration"].unique().tolist())
        ],
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.51, 1.02),
    )
    plt.tight_layout(rect=[0.01, 0.01, 1, 1])
    plt.savefig(outfile, format="svg", bbox_inches="tight")
    print_success(
        f"Plotted graph: {outfile.parts[-4]}:{outfile.parts[-3]}:{outfile.stem}:profiles"
    )
    plt.close()


def parse_rt_metrics(dir, kind):
    files = glob.glob(f"{dir / "runtime"}/*.csv")
    data = []
    for f in files:
        flags = (
            pd.read_csv(f, usecols=[0, 1, 2])
            .tail(1)
            .replace({"true": True, "false": False})
        )
        df = pd.read_csv(f, usecols=list(range(3, 12))).tail(1).astype(float)
        base = os.path.splitext(os.path.basename(f))[0].split(".")

        exp = base[2].split("-")[0]
        cfg = base[2].split("-")[1]

        if exp == "gcvs":
            assert flags.all().all()
        elif exp == "premopt":
            assert flags["elision enabled"].all()
            if cfg == "opt":
                assert flags["pfp enabled"].all()
                assert flags["premopt enabled"].all()
            elif cfg == "naive":
                assert flags["pfp enabled"].all()
                assert (~flags)["premopt enabled"].all()
            else:
                assert (~flags)["pfp enabled"].all()
                assert (~flags)["premopt enabled"].all()
        elif exp == "elision":
            assert flags["pfp enabled"].all()
            assert flags["premopt enabled"].all()
            if cfg == "opt":
                assert flags["elision enabled"].all()
            else:
                assert (~flags)["elision enabled"].all()
        else:
            print_error(f"Unknown experiment {exp}")
            sys.exit(1)

        df["suite"] = base[0].rstrip("-harness")
        df["pexec"] = base[1]
        df["configuration"] = base[2]
        df["benchmark"] = base[3]
        data.append(df)

    if not data:
        return pd.DataFrame()

    df = pd.concat(data, ignore_index=True)
    if kind == "perf":
        return df

    return parse_ht_summary(dir).merge(
        df, on=["suite", "configuration", "benchmark", "pexec"]
    )


def parse_ht_summary(dir):
    files = glob.glob(f"{dir / "heaptrack"}/*summary.csv")
    data = []
    for f in files:
        df = pd.read_csv(f).tail(1).astype(float)
        base = os.path.splitext(os.path.basename(f))[0].split(".")
        df["suite"] = base[0].rstrip("-harness")
        df["pexec"] = base[1]
        df["configuration"] = base[2]
        df["benchmark"] = base[3]
        df = df.drop(columns=["temporary allocations"])
        data.append(df)

    if not data:
        return pd.DataFrame()
    return pd.concat(data, ignore_index=True)


def parse_heaptrack(dir):
    files = glob.glob(f"{dir}/*.massif")
    data = []
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0].split(".")
        with open(f, "r") as f:
            for line in f:
                if line.startswith("snapshot="):
                    snapshot = int(line.split("=")[1])
                elif line.startswith("time="):
                    time = float(line.split("=")[1])
                elif line.startswith("mem_heap_B="):
                    mem_heap = int(line.split("=")[1])
                    data.append(
                        {
                            "suite": base[0].rstrip("-harness"),
                            "pexec": base[1],
                            "configuration": base[2],
                            "benchmark": base[3],
                            "snapshot": snapshot,
                            "time": time,
                            "mem": mem_heap,
                        }
                    )
    return pd.DataFrame(data)


def parse_sampler(dir):
    csvs = glob.glob(f"{dir}/*.csv")
    data = []
    for f in csvs:
        df = pd.read_csv(f, header=0, names=["time", "mem"]).astype(float)
        df = df.assign(snapshot=range(0, len(df)))
        base = os.path.splitext(os.path.basename(f))[0].split(".")
        df["pexec"] = base[0]
        df["configuration"] = base[1]
        df["benchmark"] = base[2]
        data.append(df)
    return pd.concat(data, ignore_index=True)


def parse_results(expdir):
    results = {}
    for prog in os.scandir(expdir):
        if not prog.is_dir():
            print_warning(f"Skipping unknown file {prog}...")
            continue

        if prog.name not in BENCHMARKS:
            continue

        data = Path(prog.path) / "perf.csv"

        perf = pd.read_csv(data, sep="\t", comment="#", index_col="suite")
        pexecs = int(perf["invocation"].max())
        perf = perf[perf["criterion"] == "total"].rename(
            columns={"value": "wallclock", "executor": "configuration"}
        )
        perf = perf[["benchmark", "configuration", "wallclock"]]
        perf_metrics = parse_rt_metrics(
            Path(prog.path) / "perf" / "metrics", kind="perf"
        )

        mem = parse_heaptrack(Path(prog.path) / "heaptrack")
        mem_metrics = parse_rt_metrics(Path(prog.path) / "mem" / "metrics", kind="mem")

        results[prog.name] = (perf, mem, perf_metrics, mem_metrics)
    return results


def process_rt_metrics(metrics):
    def gmean_zeroes(series):
        positive_series = series[(series > 0)]
        if len(positive_series) == 0:
            return np.nan
        return np.exp(np.log(positive_series).mean())

    df = (
        metrics.groupby(["suite", "configuration", "benchmark"])
        .mean(numeric_only=True)
        .reset_index()
        .drop(columns=["benchmark"])
    )

    df = df.groupby(["suite", "configuration"]).apply(gmean_zeroes).round().fillna(0)

    return df


def process_stats(df, experiment, summary=False):

    def diff(a, b):
        diff = a - b
        ratio = a / b
        pct = (ratio - 1) * 100
        return {"diff_raw": diff, "diff_ratio": ratio, "diff_pct": pct}

    def baseline(suite, data, col, lower, upper, kind):
        bl = BASELINE[suite] if experiment == "gcvs" else BASELINE[experiment]
        if col != bl:
            # We compare the cfg with a common baseline
            val = data[("value", bl)]
            si = not (upper < data[("lower"), bl] or lower > data[("upper"), bl])
        else:
            others = data["value"].drop(col)
            blidx = others.idxmin() if kind == "best" else others.idxmax()
            val = data[("value", blidx)]
            si = not (upper < data[("lower"), blidx] or lower > data[("upper"), blidx])
        return (val, si)

    def calculate_diffs(df):
        worst = []
        best = []
        worst_all = []
        best_all = []
        for idx, row in df.iterrows():
            suite = idx[0] if not summary else idx
            for config in df["value"].columns:
                value = row[("value", config)]
                lower = row[("lower", config)]
                upper = row[("upper", config)]
                d = {
                    "suite": suite,
                    "configuration": config,
                    "value": value,
                    "upper": upper,
                    "lower": lower,
                }

                if not summary:
                    d["benchmark"] = idx[1]

                (best_other, best_si) = baseline(
                    suite, row, config, lower, upper, "best"
                )
                (worst_other, worst_si) = baseline(
                    suite, row, config, lower, upper, "worst"
                )

                worst_data = d | diff(value, best_other)
                best_data = d | diff(worst_other, value)

                if not worst_si:
                    worst.append(worst_data)
                if not best_si:
                    best.append(best_data)

                worst_all.append(worst_data)
                best_all.append(best_data)

        return {
            "worst": pd.DataFrame(worst),
            "best": pd.DataFrame(best),
            "worst_all": pd.DataFrame(worst_all),
            "best_all": pd.DataFrame(best_all),
        }

    df = df.copy()
    # df["lower"] = df["value"] - df["ci"]
    # df["upper"] = df["value"] + df["ci"]
    if summary:
        index = "suite"
    else:
        index = ["suite", "benchmark"]
    pivot = df.pivot_table(
        index=index,
        columns="configuration",
        values=["value", "lower", "upper"],
    )

    diffs = calculate_diffs(pivot)
    if summary:
        statidx = "configuration"
    else:
        statidx = ["suite", "configuration"]

    stats = pd.concat(
        {
            k: (
                v.loc[v.groupby(statidx)["diff_raw"].idxmax()].set_index(statidx)
                if not v.empty
                else pd.DataFrame()
            )
            for k, v in diffs.items()
        },
        axis=1,
    )

    stats.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1]) for col in stats.columns]
    )

    return stats

    # for k, v in stats.items():
    #     print_info(k)
    #     print(v)

    # worst_df = (
    #     worst_df.loc[
    #         worst_df.groupby(["suite", "configuration"])["diff_raw"].idxmax()
    #     ].set_index(["suite", "configuration"])
    #     if not worst_df.empty
    #     else pd.DataFrame()
    # )
    #
    # best_df = (
    #     best_df.loc[
    #         best_df.groupby(["suite", "configuration"])["diff_raw"].idxmax()
    #     ].set_index(["suite", "configuration"])
    #     if not best_df.empty
    #     else pd.DataFrame()
    # )
    #
    # worst_all_df = worst_all_df.loc[
    #     worst_all_df.groupby(["suite", "configuration"])["diff_raw"].idxmax()
    # ].set_index(["suite", "configuration"])
    #
    # best_all_df = best_all_df.loc[
    #     best_all_df.groupby(["suite", "configuration"])["diff_raw"].idxmax()
    # ].set_index(["suite", "configuration"])

    # print_info("worst all df")
    # print(worst_all_df)
    # print_info("best all df")
    # print(best_all_df)
    # print_info("worst df")
    # print(worst_df)
    # print_info("best df")
    # print(best_df)

    # Concatenate DataFrames
    # stats = pd.concat(
    #     {
    #         "worst": worst_df,
    #         "best": best_df,
    #         "worst_all": worst_all_df,
    #         "best_all": best_all_df,
    #     },
    #     axis=1,
    # )


def process_summary(df):
    print(df)
    gmean = (
        df.groupby(["suite", "configuration"])["value"]
        .apply(geomean_with_ci)
        .unstack()
        .reset_index()
    )
    return gmean


def process_perf(df, prog, experiment):
    pexecs = df.groupby(["suite", "configuration", "benchmark"]).size().max()
    # perf = df.groupby(["suite", "configuration", "benchmark"])["wallclock"].apply(
    #     arith_mean_ci, pexecs=pexecs
    # )
    # perf = perf.unstack().reset_index()

    # # Apply function and flatten result
    # perf = (
    #     df.groupby(["suite", "configuration", "benchmark"])["wallclock"]
    #     .apply(lambda x: ci_inl(x, pexecs))  # Apply function correctly
    #     .reset_index()  #
    # )
    # Apply function and **FIX multi-index issue**
    perf = (
        df.groupby(["suite", "configuration", "benchmark"])["wallclock"]
        .apply(arith_mean_ci)  # Get single value
        .unstack()
        .reset_index()  # Flatten multi-index
    )

    if perf["benchmark"].nunique() > 1:
        stats = process_stats(perf, experiment)
        write_stats(stats, experiment, fmt="perf")

    plt = PLOT_DIR / experiment / prog / "perf.svg"
    plt.parent.mkdir(parents=True, exist_ok=True)

    plot_perf(
        plt,
        perf,
        rows=perf["suite"].nunique(),
        cols=1,
    )
    return perf


def process_mem(df, prog, experiment):
    pexecs = df.groupby(["suite", "configuration", "benchmark"]).size().max()
    mem = (
        df.copy()
        .groupby(["suite", "configuration", "benchmark"])["mem"]
        .apply(arith_mean_ci)
        .unstack()
        .reset_index()
    )

    if mem["benchmark"].nunique() > 1:
        stats = process_stats(mem, experiment)
        write_stats(stats, experiment, fmt="mem")

    for suite, results in df.groupby("suite"):
        profile = PLOT_DIR / experiment / prog / "profiles" / f"{suite}.svg"
        profile.parent.mkdir(parents=True, exist_ok=True)
        m = interpolate(normalize_time(results), oversampling=0.1)

        bms = m["benchmark"].nunique()
        if bms == 1:
            rows = 1
            cols = 1
        else:
            rows = 7
            cols = 4
        cmp = True if experiment in ["premopt", "elision"] else False
        plot_mem_time_series(profile, m, rows, cols, cmp=cmp)
    return mem


# for row in gmean.itertuples():
#     s = row.suite.replace("-", "")
#     e = row.configuration.split("-")[0]
#     c = row.configuration.split("-")[1]
#     write_stat(f"\n% {row.suite} geomeans stats")
#     write_stat(f"\\newcommand\\{e}{c}{s}gmean{{{row.gmean:0.2f}\\xspace}}")
#     write_stat(
#         f"\\newcommand\\{e}{c}{s}gmeanci{{\\footnotesize{{±{max(row.gerr_lower,row.gerr_upper):0.3f}}}\\xspace}}"
#     )

# distinguishable = df[df["overlaps baseline"] != True]
# for s in df["suite"].unique():
#     write_stat(f"\n% {s} summary stats")
#     for c in df["configuration"].unique():
#         latex_name = experiment + s.replace("-", "") + c.split("-")[1]
#         write_stat(
#             f"\\newcommand\\{latex_name}best{{\\jake{{No statistically distinguishable best benchmark}}\\xspace}}"
#         )
#         write_stat(
#             f"\\newcommand\\{latex_name}worst{{\\jake{{No statistically distinguishable worst benchmark}}\\xspace}}"
#         )
#
#         cfg = df.loc[(df["suite"] == s) & (df["configuration"] == c)]
#
#         worst = df.iloc[df["value"].idxmax()]
#
#         print(worst)
#

# pivot_df = df.pivot(index="benchmark", columns="configuration", values="value")
#
# # Calculate slowdown ratio (gcvs-gc / gcvs-arc)
# slowdown_series = pivot_df["gcvs-gc"] / pivot_df["gcvs-arc"]
#
# # Add the slowdown column only to gcvs-gc rows
# df["slowdown"] = np.nan
# df.loc[df["configuration"] == "gcvs-gc", "slowdown"] = df["benchmark"].map(
#     slowdown_series
# )

# data[data['configuration'] == BASELINE[suite]].iloc[0])
# for suite, data in df.groupby("suite"):
#     print(data)
#     for row in data.sort_values(by='value'):
#         print(row)
#     for row in data.loc[data.groupby('configuration')['value'].idxmin()].itertuples():
#         print(row)


def process_conversion_stats(metrics):
    totals = (
        metrics.groupby(["suite", "configuration", "benchmark"])
        .mean(numeric_only=True)
        .reset_index()
        .drop(columns=["benchmark"])
        .groupby(["suite", "configuration"])
        .sum()
        .reset_index()
    )
    conv = pd.DataFrame()

    totals["pct_rc"] = pdiff(
        totals["Arc allocated"] + totals["Rc allocated"], totals["allocations"]
    )
    totals["pct_gc"] = pdiff(totals["Gc allocated"], totals["allocations"])
    totals["pct_gc"] = pdiff(totals["Gc allocated"], totals["allocations"])
    totals["pct_managed"] = pdiff(totals["Gc reclaimed"], totals["allocations"])

    # totals = totals[cols]
    # totals = totals.set_index(['suite','configuration'])

    for suite, data in totals.groupby("suite"):
        rc_leaks = round(
            totals[totals["configuration"] == BASELINE[suite]][
                "leaked allocations"
            ].iloc[0]
        )
        gc = data[data["configuration"] == "gcvs-gc"]
        gc_leaks = gc["leaked allocations"].iloc[0]
        all = gc["allocations"].iloc[0]
        rcs = gc["Rc allocated"].iloc[0] + gc["Arc allocated"].iloc[0]
        gcs = gc["Gc allocated"].iloc[0]
        box_ctors = gc["Box allocated"].iloc[0]
        num_swept = gc["Gc reclaimed"].iloc[0]
        all = gc["allocations"].iloc[0]
        finalizers_run = gc["finalizers completed"].iloc[0]
        freg = gc["finalizers registered"].iloc[0]
        elided = gc["finalizers elidable"].iloc[0]

        # We use the difference between RC and GC leaks as a proxy for the
        # number of 'GC' objects still on the heap (because Rc will
        # deterministically drop on exit, whereas Gc will not). This is not
        # hugely accurate, but will always serve as a 'lower bound'. There
        # is however one case where we need to be a bit careful: benchmarks
        # with cycles can cause large amounts of cyclic Rc garbage which
        # can exceed the number of GC leaks. In such cases, we don't want
        # to include this as it would cause us to erroneously think we
        # converted fewer objects to GC than we had.
        gc_leaks = max(gc["leaked allocations"].iloc[0] - rc_leaks, 0)

        pct_gc = (gcs / all) * 100
        gcs_trans = num_swept + gc_leaks + finalizers_run
        pct_gt = (gcs_trans / all) * 100
        print_info(f"{suite}")
        print_info(f"F reg: {format_number(freg)}")
        print_info(f"F run: {format_number(finalizers_run)}")
        print_info(f"F eli: {format_number(elided)}")
        print("finalizers registered {format_number(freg)}")
        print_info(f"Total objects {format_number(all)}")
        print_info(f"Box constructors {format_number(box_ctors)}")
        print_info(f"Managed objects {format_number(gcs_trans)}")
        print_info(f"RC leaks {format_number(rc_leaks)}")
        print_info(f"GC leaks {format_number(gc_leaks)}")
        print_info(f"GC constructors {format_number(gcs)}")
        print_info(f"(a)RC constructors {format_number(rcs)}")
        print_info(f"PCT managed: {pct_gt:.2f}")
        print_info(f"PCT explicit GC:{pct_gc:.2f}")

        s = suite.replace("-", "")

        write_stat(f"% Conversion statistics for {suite}")
        write_stat(f"\\newcommand\\{s}heapgcpct{{{pct_gc:.2f}\\%\\xspace}}")
        write_stat(f"\\newcommand\\{s}heapgctpct{{{pct_gt:.2f}\\%\\xspace}}")
        write_stat(f"\\newcommand\\{s}heapall{{{format_number(all)}\\xspace}}")
        write_stat(f"\\newcommand\\{s}heapgcs{{{format_number(gcs)}\\xspace}}")
        write_stat(f"\\newcommand\\{s}heapgcts{{{format_number(gcs_trans)}\\xspace}}")


def process_experiment(experiment):
    print_info(f"Processing {experiment} results...")
    results = parse_results(RESULTS_DIR / experiment)
    perfs = []
    mems = []

    def sanity_check(prog, df):
        if df.empty:
            print_error(f"{experiment}:{prog} has missing data")
            return False
        runs = df.groupby(["suite", "configuration", "benchmark"]).size()
        if (runs != runs.iloc[0]).all():
            print_error(
                f"{experiment}:{prog} has an inconsistent number of pexecs: {pruns}"
            )
            return False
        return True

    for prog, (perfraw, memraw, perfmetrics, memmetrics) in results.items():
        print_info(f"Processing {prog}...")

        perfs_ok = "perf" in PLOTS and sanity_check(prog, perfraw)
        perfms_ok = "perf" in PLOTS and sanity_check(prog, perfmetrics)
        mems_ok = "mem" in PLOTS and sanity_check(prog, memraw)
        memms_ok = "mem" in PLOTS and sanity_check(prog, memmetrics)

        if experiment == "gcvs":
            if memms_ok:
                print_warning(f"Processing conversion stats")
                process_conversion_stats(memmetrics)

        if perfs_ok:
            perf = process_perf(perfraw, prog, experiment)
            if perf["benchmark"].nunique() == 1:
                perfs.append(perf.drop(columns=["benchmark"]))
            else:
                perfs.append(process_summary(perf))

        if mems_ok:
            mem = process_mem(memraw, prog, experiment)
            if mem["benchmark"].nunique() == 1:
                mems.append(mem.drop(columns=["benchmark"]))
            else:
                mems.append(process_summary(mem))

    if perfs_ok:
        perfs = pd.concat(perfs, ignore_index=True)
        stats = process_stats(perfs, experiment, summary=True)
        write_stats(stats, experiment, fmt="perf", summary=True)
        perfs["suite"] = perfs["suite"].replace(SUITES)
        perfs["configuration"] = perfs["configuration"].replace(CFGS)
        write_table(PLOT_DIR / experiment / "perf_summary.tex", perfs)

    if mems_ok:
        mems = pd.concat(mems, ignore_index=True)
        stats = process_stats(mems, experiment, summary=True)
        write_stats(stats, experiment, fmt="mem", summary=True)
        print(mems.isna().any())
        mems["suite"] = mems["suite"].replace(SUITES)
        mems["configuration"] = mems["configuration"].replace(CFGS)
        mems["latex_value"] = mems.apply(format_mem_ci, axis=1)
        memltx = mems.pivot(
            index="suite", columns="configuration", values="latex_value"
        )
        memltx = memltx.fillna("-")
        write_table(PLOT_DIR / experiment / "mem_summary.tex", memltx)


def main():
    global RESULTS_DIR
    global PLOT_DIR
    global STATS_FILE

    RESULTS_DIR = Path(sys.argv[2])
    PLOT_DIR = Path(sys.argv[1])
    STATS_FILE = PLOT_DIR / "experiment_stats.tex"

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if os.path.exists(STATS_FILE):
        print_warning(f"File {STATS_FILE} already exists. Removing...")
        os.remove(STATS_FILE)

    print_info(f"Will process the the following experiments: {EXPERIMENTS}")
    print_info(f"Will process the the following benchmarks: {BENCHMARKS}")
    print_info(f"Will generate the the following plots: {PLOTS}")

    if DRYRUN:
        print_warning(
            f"DRYRUN enabled: no plots will be generated and CIs will be incorrect."
        )

    if not BOOTSTRAP:
        print_warning(f"BOOTSTRAP disabled: CI formula will be used for arith. mean")

    for e in EXPERIMENTS:
        process_experiment(e)
    # if "gcvs" in EXPERIMENTS:
    #     process_experiment("gcvs")
    # if "premopt" in PROCESS_EXPERIMENT:
    #     process_experiment("premopt")
    # if "elision" in PROCESS_EXPERIMENT:
    #     process_experiment("elision")


if __name__ == "__main__":
    main()
