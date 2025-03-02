#! /usr/bin/env python

import csv
import gc
import glob
import math
import os
import pprint
import random
import sys
from os import listdir, stat
from pathlib import Path
from statistics import geometric_mean, stdev

import matplotlib
import matplotlib.lines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from scipy import stats

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
        # "legend.title_fontsize": 0,
        "errorbar.capsize": 2,
    }
)

results = {}
pp = pprint.PrettyPrinter(indent=4)

PEXECS = int(os.environ["PEXECS"])

EXPERIMENTS = {
    "gcrc": r"GcRc",
    "elision": r"Elision",
    "premopt": r"PremOpt",
}

SUITES = {
    "som-rs-ast": r"\somrsast",
    "som-rs-bc": r"\somrsbc",
    "yksom": r"\yksom",
}

CFGS = {
    "gcvs-gc": "Alloy",
    "gcvs-rc": "RC",
    "premopt-opt": "Barriers Opt",
    "premopt-naive": "Barriers Naive",
    "premopt-none": "Barriers None",
    "premopt-opt": "Barriers Opt",
    "elision-naive": "Elision Naive",
    "elision-opt": "Elision Opt",
}

METRICS = {
    "finalizers registered": "Finalizable Objects",
    "finalizers completed": "Total Finalized",
    "barriers visited": "Barrier Chokepoints",
    "Gc allocated": "Allocations (Gc)",
    "Box allocated": "Allocations (Box)",
    "Rc alocated": "Allocations (Rc)",
    "Arc allocated": "Allocations (Arc)",
    "STW pauses": r"Gc Cycles",
}


def bootstrap(
    values, kind, method, num_bootstraps=10000, confidence=0.99, symmetric=True
):
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
            "ci_lower": res.confidence_interval.low,
            "ci_upper": res.confidence_interval.high,
        }

    return pd.Series(data)


def bootstrap_geomean_ci(means, num_bootstraps=10000, confidence=0.99, symmetric=False):
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


def bootstrap_mean_ci(raw_data, num_bootstraps=10000, confidence=0.99):
    return bootstrap(
        raw_data, np.mean, "percentile", num_bootstraps, confidence, symmetric=True
    )


def bootstrap_max_ci(raw_data, num_bootstraps=10000, confidence=0.99):
    return bootstrap(
        raw_data, np.max, "percentile", num_bootstraps, confidence, symmetric=True
    )


def pretty_name(col):
    # print("here")
    print(col)
    l = col.split("-")
    if len(l) == 1:
        return CFGS[l[0]]
    else:
        return CFGS[l[1]]


def plot_bar(title, filename, data, width, unit):
    values = data[0]
    errs = data[1]
    fig, ax = plt.subplots(figsize=(width, 4))

    values = values.rename(columns=pretty_name)
    errs = errs.rename(columns=pretty_name)
    values.plot(kind="bar", ax=ax, width=0.8, yerr=errs)

    ax.legend().set_title(None)
    ax.set_xticklabels(values.index, rotation=45, ha="right")

    if unit == "ms":
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)
    elif unit == "b":
        ax.yaxis.set_major_formatter(FuncFormatter(human_readable_bytes))
    elif unit == "kb":
        ax.yaxis.set_major_formatter(FuncFormatter(human_readable_bytes))
    else:
        raise ValueError("Unknown unit")
    ax.set_ylabel(title)
    ax.xaxis.label.set_visible(False)
    plt.tight_layout()
    plt.savefig(filename, format="svg", bbox_inches="tight")
    print(f"==> Created plot: {filename}")


def normalize_time(df):
    group["normalized_time"] = (group["timestamp"] - group["timestamp"].min()) / (
        group["time"].max() - group["time"].min()
    )
    return group


def mini_plot(benchmark, data, outdir):
    fig, ax = plt.subplots(figsize=(5, 3))
    # ax.set_title(f"{benchmark}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Memory Usage")
    ax.yaxis.set_major_formatter(FuncFormatter(human_readable_bytes))

    palette = sns.color_palette("colorblind")
    configs = data["configuration"].unique()
    colours = [palette[i] for i in range(0, len(configs))]
    mems = []
    means = []
    peaks = []

    for i, (cfg, snapshot) in enumerate(data.groupby("configuration")):
        snapshot = snapshot.sort_values("normalized_time")
        (real,) = ax.plot(
            snapshot["normalized_time"],
            snapshot["mem"],
            label=f"{cfg}",
            color=colours[i],
        )

        ax.fill_between(
            snapshot["normalized_time"],
            snapshot["mem"] - snapshot["mem_ci"],
            snapshot["mem"] + snapshot["mem_ci"],
            alpha=0.2,
            color=colours[i],
        )

        # Plot mean heap usage as line
        mean = snapshot["mean_heap_usage"].iloc[0]
        mean_ci = snapshot["mean_heap_usage_ci"].iloc[0]
        mean = plt.axhline(
            y=mean,
            color=colours[i],
            linestyle=":",
            alpha=0.7,
            label=f"Mean ({human_readable_bytes(mean)} ${{\scriptstyle \\pm {human_readable_bytes(mean_ci)}}}$)",
        )

        # Plot peak heap usage as line
        peak = snapshot["peak_heap_usage"].iloc[0]
        peak_ci = snapshot["peak_heap_usage_ci"].iloc[0]
        peak = plt.axhline(
            y=peak,
            color=colours[i],
            linestyle="--",
            alpha=0.5,
            label=f"Peak ({human_readable_bytes(peak)} ${{\scriptstyle \\pm {human_readable_bytes(peak_ci)}}}$)",
        )

        mems.append(real)
        means.append(mean)
        peaks.append(peak)

    handles = mems
    plt.legend(
        title=f"{benchmark}",
        title_fontsize=12,
        labelspacing=1,
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        # columnspacing=0.5,
        bbox_to_anchor=(0.5, 1.35),
        frameon=False,
    )
    plt.tight_layout()
    # plt.tight_layout(rect=[0, 0.1, 1, 0.7])
    plt.savefig(outdir / f"{benchmark.lower()}.svg", format="svg", bbox_inches="tight")
    print(
        f"==> Saved time-series memory results to {outdir / f"{benchmark.lower()}.svg"}"
    )


def plot_mem_time_series(data, outdir):
    data["configuration"].replace(CFGS, inplace=True)
    for benchmark, cfgs in data.groupby("benchmark"):
        mini_plot(benchmark, cfgs, outdir)


def mk_table(filename, vals, cis, columns):
    df = pd.DataFrame()
    for v, c in zip(vals.columns, cis.columns):
        df[v] = vals[v].round(2).astype(str) + " \pm " + cis[c].round(3).astype(str)

    with open(filename, "w") as f:
        f.write(df.unstack().rename(columns=columns).to_latex(index=False))


def ci(row, pexecs):
    Z = 2.576  # 99% interval
    return Z * (row / math.sqrt(pexecs))


def parse_metrics(mdir):
    csvs = glob.glob(f"{mdir}/*.log")
    m = []
    for f in csvs:
        df = pd.read_csv(f)
        base = os.path.splitext(os.path.basename(f))[0].split("-")
        df["configuration"] = base[1]
        df["benchmark"] = base[2]
        df = df.drop(
            [
                "elision enabled",
                "premature finalizer prevention enabled",
                "premopt enabled",
            ],
            axis=1,
        )
        m.append(df)
    return pd.concat(m, ignore_index=True)


def human_readable_bytes(x, pos=None):
    if x < 1024:
        return f"{x} B"
    elif x < 1024**2:
        return f"{x/1024:.1f} KiB"
    elif x < 1024**3:
        return f"{x/1024**2:.1f} MiB"
    else:
        return f"{x/1024**3:.1f} GiB"


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
                            "configuration": base[1],
                            "benchmark": base[2],
                            "pexec": base[0],
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


def parse_perfdata(csv):
    print(f"==> parsing perf data from {csv}")
    df = pd.read_csv(csv, sep="\t", skiprows=4, index_col="benchmark")
    pexecs = int(df["invocation"].max())
    assert pexecs == PEXECS
    perf = df[df["criterion"] == "total"].rename(columns={"value": "wallclock"})
    perf = perf[["executor", "wallclock"]]
    rss = df[df["criterion"] == "MaxRSS"].rename(columns={"value": "maxrss"})
    rss = rss[["executor", "maxrss"]]
    df = pd.merge(perf, rss, on=["benchmark", "executor"]).groupby(
        ["benchmark", "executor"]
    )
    return df


def aggregate(grouped, col, method, unstack=True):
    df = grouped[col].apply(method).unstack()
    if unstack:
        df = df.unstack()
    else:
        df = df.reset_index()
    return (df["value"], df["ci"])


def process_perf(resultsdir, outdir):
    pdata = parse_perfdata(resultsdir / "perf.csv")

    perf = aggregate(pdata, "wallclock", bootstrap_mean_ci)
    # maxrss = aggregate(pdata, "maxrss", bootstrap_mean_ci)
    # print(perf)

    plot_bar(
        "Wall-clock time (ms)\n(lower is better)",
        outdir / "perf.svg",
        perf,
        width=8,
        unit="ms",
    )


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
        bootstrap_mean_ci,
        unstack=True,
    )
    mean_memory = mean_memory.unstack().reset_index()
    mean_memory.rename(columns={0: "mean_heap_usage"}, inplace=True)
    mean_ci = mean_ci.unstack().reset_index()
    mean_ci.rename(columns={0: "mean_heap_usage_ci"}, inplace=True)
    df = df.merge(mean_memory, on=["configuration", "benchmark"])
    df = df.merge(mean_ci, on=["configuration", "benchmark"])
    print(df)
    return df


def process_heaptrack(resultsdir, outdir):
    # df = parse_heaptrack(resultsdir / "heaptrack")
    # df = interpolate(normalize_time(df))

    # rss = parse_sampler(resultsdir / "samples")
    # rss = interpolate(normalize_time(rss))
    # rss["configuration"] = rss["configuration"] + " rss"

    mem = parse_heaptrack(resultsdir / "heaptrack")
    mem = interpolate(normalize_time(mem), oversampling=0.1)

    # mem = pd.concat([rss, allocs])

    # RSS data tends to be an unreliable metric since it includes memory of
    # shared libraries, heap, stack, and code segments. See [1]
    #
    # [1]: https://community.ibm.com/community/user/aiops/blogs/riley-zimmerman/2021/07/05/memory-measurements-part3
    plot_mem_time_series(mem, outdir / "mem")

    sys.exit(1)
    # mdata = mdata.groupby(["benchmark", "configuration"])
    # avgmem = aggregate(mdata, "mem_heap_B", bootstrap_mean_ci)
    # maxmem = aggregate(mdata, "mem_heap_B", bootstrap_max_ci)

    # # print(avgmem)
    # plot_bar(
    #     "Average heap usage (KiB)\n(lower is better)",
    #     outdir / "avg_heap.svg",
    #     avgmem,
    #     width=8,
    #     unit="b",
    # )
    # plot_bar(
    #     "Max heap usage (KiB)\n(lower is better)",
    #     outdir / "max_heap.svg",
    #     avgmem,
    #     width=8,
    #     unit="b",
    # )


def main():

    print(f"==> processing results for {sys.argv[1:]}")
    resultsdir = Path(sys.argv[2])
    outdir = Path(sys.argv[1])

    perf = resultsdir / "perf.csv"
    mem = resultsdir / "mem.csv"

    if not os.path.exists(perf) and not os.path.exists(mem):
        print(f"No results for {resultsdir}. Exiting...")
        sys.exit()

    # if os.path.exists(perf):
    #     process_perf(resultsdir, outdir)

    if os.path.exists(mem):
        process_heaptrack(resultsdir, outdir)


if __name__ == "__main__":
    main()

# raw = parse_metrics(resultsdir / "metrics").groupby(["benchmark", "configuration"])
# metrics = raw.mean()
# cis = raw.std().apply(ci, pexecs=10)
# mk_table(outdir / "metrics.tex", metrics, cis, columns=METRICS)
