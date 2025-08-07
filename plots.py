import glob
import json
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zstandard as zstd
from scipy.interpolate import interp1d

from build import GCVS, HS_MAP, Aggregation, Metric

plt.rcParams.update(
    {
        "axes.edgecolor": "#b0b0b0",
        "grid.color": "#cccccc",
        "grid.linewidth": 0.4,
        "xtick.color": "#b0b0b0",
        "ytick.color": "#b0b0b0",
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "axes.linewidth": 0.4,
        "axes.axisbelow": True,
        "axes.grid": True,
        "axes.labelcolor": "black",
        "text.color": "black",
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
        "font.size": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 13,
        "legend.frameon": False,
        "legend.columnspacing": 1.0,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "text.usetex": True,
        "text.latex.preamble": r"""
                \usepackage{contour}
                \contourlength{0.5pt}
                \renewcommand{\familydefault}{\sfdefault}
            """,
    }
)


@dataclass
class ExperimentConfig:
    colors: List[str]
    outline: List[str]
    geomean: List[str]
    alphas: List[float]


@dataclass
class SimplePlotConfig:
    figsize: Tuple[int, int] = (10, 8)
    group_width: float = 0.8
    show_legend: bool = True


class SimplePlotter:
    EXPERIMENTS = {
        "gcvs": ExperimentConfig(
            colors=["#3A87D9", "#FF8F2E"],
            outline=["#1A5C85", "#D66000"],
            geomean=["#1A5C85", "#994400"],
            alphas=[0.1, 0.15],
        ),
        "heapsizes": ExperimentConfig(
            colors=["#3A87D9", "#FF8F2E", "#228B22", "#800080", "#DC143C"],
            outline=["#1A5C85", "#D66000", "#145214", "#4B004B", "#8B0A1E"],
            geomean=["#1A5C85", "#994400", "#0F3A10", "#330033", "#5C0B16"],
            alphas=[0.1, 0.15, 0.1, 0.1],
        ),
        "premopt": ExperimentConfig(
            colors=["#34495E", "#3A87D9"],
            outline=["#2C3E50", "#1A5C85"],
            geomean=["#1B2631", "#1A5C85"],
            alphas=[0.1, 0.1],
        ),
        "elision": ExperimentConfig(
            colors=["#3A87D9", "#E74C3C"],
            outline=["#1A5C85", "#C0392B"],
            geomean=["#1A5C85", "#922B21"],
            alphas=[0.1, 0.1],
        ),
    }

    def __init__(self, config: SimplePlotConfig = None):
        self.config = config

    def plot(
        self,
        df: pd.DataFrame,
        experiment: str,
        value_col: str,
        output_file: str = "plot.png",
        xlim: Optional[Tuple] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
    ):
        exp_config = self.EXPERIMENTS[experiment]

        suite_data = df[df["agg_type"] == Aggregation.SUITE_GEO]
        suites = sorted(suite_data["suite"].unique())
        configs = sorted(suite_data["configuration"].unique())

        fig, ax = plt.subplots(figsize=self.config.figsize)
        self._draw_ci_shadows(ax, df, value_col, configs, exp_config)
        ax.axvline(x=1, color="black", linewidth=0.5, zorder=30)
        bar_width = self.config.group_width / len(configs)

        for i_suite, suite in enumerate(suites):
            group_start = i_suite - self.config.group_width / 2

            for i_config, config in enumerate(configs):
                row_data = suite_data[
                    (suite_data["suite"] == suite)
                    & (suite_data["configuration"] == config)
                ]

                if row_data.empty:
                    continue

                row = row_data.iloc[0]
                y_pos = group_start + i_config * bar_width
                color_idx = i_config % len(exp_config.colors)

                # Bar
                ax.barh(
                    y_pos,
                    row[value_col],
                    height=bar_width,
                    color=exp_config.colors[color_idx],
                    alpha=0.9,
                    edgecolor=exp_config.outline[color_idx],
                    linewidth=0.5,
                    zorder=10,
                    align="edge",
                )

                # Error bars
                ci_lower_col = f"{value_col}_lower"
                ci_upper_col = f"{value_col}_upper"

                if (
                    ci_lower_col in row
                    and ci_upper_col in row
                    and not pd.isna(row[ci_lower_col])
                    and not pd.isna(row[ci_upper_col])
                ):
                    err_low = max(0, row[value_col] - row[ci_lower_col])
                    err_high = max(0, row[ci_upper_col] - row[value_col])

                    ax.errorbar(
                        row[value_col],
                        y_pos + bar_width / 2,
                        xerr=[[err_low], [err_high]],
                        fmt="none",
                        color="#2C2C2C",
                        capsize=0.8,
                        linewidth=0.5,
                        zorder=15,
                    )

        # Format
        ax.set_ylim(len(suites) - 0.5, -0.5)
        ax.set_yticks(range(len(suites)))
        ax.set_yticklabels([s for s in suites])
        ax.grid(True, axis="x", zorder=5)
        ax.grid(False, axis="y")

        if xlim:
            ax.set_xlim(xlim)
        if xlabel:
            ax.set_xlabel(xlabel)
        if title:
            ax.set_title(title)

        if self.config.show_legend:
            self._add_legend(plt, configs, exp_config)

        plt.tight_layout()
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close()

        return output_file

    def _draw_ci_shadows(
        self,
        ax,
        df: pd.DataFrame,
        value_col: str,
        configs: List[str],
        exp_config: ExperimentConfig,
    ):
        global_data = df[df["agg_type"] == Aggregation.GLOBAL_GEO]
        if global_data.empty:
            return

        ranges = []

        for i, config in enumerate(configs):
            config_data = global_data[global_data["configuration"] == config]
            if config_data.empty:
                continue

            row = config_data.iloc[0]
            ci_lower_col = f"{value_col}_lower"
            ci_upper_col = f"{value_col}_upper"

            if (
                ci_lower_col in row
                and ci_upper_col in row
                and not pd.isna(row[ci_lower_col])
                and not pd.isna(row[ci_upper_col])
            ):
                r_min, r_max = row[ci_lower_col], row[ci_upper_col]
                ranges.append((r_min, r_max))
                color_idx = i % len(exp_config.colors)

                # CI shadow
                ax.axvspan(
                    r_min,
                    r_max,
                    facecolor=exp_config.colors[color_idx],
                    alpha=exp_config.alphas[color_idx],
                    edgecolor="none",
                    zorder=2,
                )

                # CI edges
                for edge in [r_min, r_max]:
                    ax.axvline(
                        edge,
                        color=exp_config.outline[color_idx],
                        linewidth=0.5,
                        linestyle="--",
                        alpha=0.5,
                        zorder=3,
                    )

                # Geomean line
                if value_col in row and not pd.isna(row[value_col]):
                    ax.axvline(
                        row[value_col],
                        color=exp_config.geomean[color_idx],
                        linewidth=0.5,
                        zorder=5,
                    )

        # Overlaps
        if len(ranges) >= 2:
            boundaries = sorted(set(b for r in ranges for b in r))
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i + 1]
                overlap_count = sum(1 for r in ranges if r[0] < end and r[1] > start)

                if overlap_count > 1:
                    alpha = min(0.05 * (overlap_count - 1), 0.1)
                    ax.axvspan(
                        start,
                        end,
                        facecolor="lightgray",
                        alpha=alpha,
                        edgecolor="none",
                        zorder=4,
                    )

    def _add_legend(self, ax, configs: List[str], exp_config: ExperimentConfig):
        handles = []
        for i, config in enumerate(configs):
            color_idx = i % len(exp_config.colors)
            handles.append(
                plt.Line2D(
                    [],
                    [],
                    color=exp_config.colors[color_idx],
                    lw=4,
                    label=str(config.latex),
                )
            )

        ax.legend(
            handles=handles,
            loc="upper center",
            frameon=False,
            ncols=2,
            handlelength=1.0,
            fontsize=8,
            bbox_to_anchor=(0.5, -0.15),
        )


def ltx_ms_to_s(val_ms, err_ms):
    if pd.isna(val_ms) or pd.isna(err_ms):
        return "--", "--"
    return (
        f"{val_ms / 1_000:.3f}",
        f"\\scriptsize\\textcolor{{gray!60}}{{$\\pm${err_ms / 1_000:.3f}}}",
    )


def ltx_default_fmt(val, err):
    if pd.isna(val) or pd.isna(err):
        return "--", "--"
    return f"{val:.2f}", f"\\scriptsize\\textcolor{{gray!60}}{{\\pm{err:.2f}}}"


def ltx_ratio_fmt(ratio, lo, hi):
    if pd.isna(ratio):
        return "--", "--"
    return (
        f"{ratio:.2f}$\\times$",
        f"\\scriptsize\\textcolor{{gray!60}}{{[{lo:.2f},{hi:.2f}]}}",
    )


def ltx_ms_to_s_geo(val_ms, lo, hi):
    if pd.isna(val_ms):
        return "--", "--"
    return (
        f"{val_ms:.2f}",
        f"\\scriptsize\\textcolor{{gray!60}}{{[{lo / 1_000:.2f},{hi / 1_000:.2f}]}}",
    )


def mk_heap_table(
    df,
    formatter=ltx_ratio_fmt,
    metric_title="Ratio",
    fail_title="Benchmarks failed",
    output_file=None,
):
    """
    Required DataFrame columns (one row per Suite/Heap):
        suite, heap_mib, ratio, ci_lo, ci_hi, fails   (fails may be NaN/empty)
    """
    colspec = "l r r r "
    lines = [
        f"\\begin{{tabular}}{{{colspec}}}",
        "    Suite         & Heap Size (MiB) & " f"{metric_title}  &  \\\\",
        "\\hline",
    ]

    for suite in df["suite"].unique():
        block = df[df["suite"] == suite]
        for _, row in block.iterrows():
            ratio_cell, ci_cell = formatter(
                row["ratio"], row["ratio_lower"], row["ratio_upper"]
            )
            lines.append(
                f"{suite.latex} & {HS_MAP[suite.name][row['configuration'].full]} & "
                f"{ratio_cell} & {ci_cell}  \\\\"
            )
        lines.append("\\midrule")

    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    table = "\n".join(lines)
    with open(output_file, "w") as f:
        f.write(table)


def mk_alloc_table(
    df, value_fmt=ltx_ms_to_s_geo, ratio_fmt=ltx_ratio_fmt, output_file=None
):
    colspec = "l l r@{\\hspace{3pt}}l r@{\\hspace{3pt}}l"

    allocators = {
        GCVS.BASELINE: "jemalloc",
        GCVS.ARC: "gcmalloc",
        GCVS.RC: "gcmalloc",
    }

    lines = [
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        "Suite & Allocator & \\multicolumn{2}{c}{Wall-clock time (s)} & "
        "\\multicolumn{2}{c}{Slowdown} \\\\",
        "\\midrule",
    ]

    suites = df["suite"].unique()
    for suite in suites:
        block = df[df["suite"] == suite]
        for idx, row in block.iterrows():
            t_val, t_ci = value_fmt(row["value"], row["ci_lower"], row["ci_upper"])
            sd_val, sd_ci = (
                "--",
                "--",
            )
            if pd.notna(row["ratio"]) or row["ratio"] == 1:
                sd_val, sd_ci = ratio_fmt(
                    row["ratio"], row["ratio_lower"], row["ratio_upper"]
                )

            suite_cell = suite.latex if idx == block.index[0] else " " * len(suites)
            lines.append(
                f"{suite_cell} & \\texttt{{{allocators[row['configuration']]}}} & "
                f"{t_val} & {t_ci} & {sd_val} & {sd_ci} \\\\"
            )

        lines[-1] = lines[-1] + "[2pt]"

    lines += ["\\bottomrule", "\\end{tabular}"]
    table = "\n".join(lines)
    with open(output_file, "w") as f:
        f.write(table)


def tabulate_suites(df, formatter=ltx_default_fmt, header=None, output_file=None):
    df = df.copy()
    df["err"] = df["ci_upper"] - df["value"]
    df[["val_fmt", "err_fmt"]] = df.apply(
        lambda r: pd.Series(formatter(r["value"], r["err"])), axis=1
    )

    colspec = "l r@{\\hspace{3pt}}l"

    lines = [
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        f"Suite & \\multicolumn{{2}}{{c}}{{{header}}} \\\\",
        "\\midrule",
    ]

    for suite, row in df.groupby("suite").first().iterrows():
        lines.append(f"{suite} & {row['val_fmt']} & {row['err_fmt']} \\\\")

    lines += ["\\bottomrule", "\\end{tabular}"]
    table = "\n".join(lines)
    with open(output_file, "w") as f:
        f.write(table)


def tabulate_benchmarks(
    df,
    formatter=None,
    header=None,
    output_file=None,
):
    if df.empty:
        return
    df = df.copy()
    df["err"] = df["ci_upper"] - df["value"]
    df[["val_fmt", "err_fmt"]] = df.apply(
        lambda r: pd.Series(formatter(r["value"], r["err"])), axis=1
    )

    if header == None:
        header = df["metric"].iloc[0].latex

    long_cells = df.melt(
        id_vars=["suite", "benchmark", "configuration"],
        value_vars=["val_fmt", "err_fmt"],
        var_name="part",
        value_name="text",
    ).sort_values("part")

    suites = long_cells["suite"].unique()
    configurations = long_cells["configuration"].unique()

    colspec = "ll" + "@{\\hspace{6pt}}r@{\\hspace{3pt}}l" * len(configurations)

    lines = [
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        f"Suite & Benchmark & \\multicolumn{{{2 * len(configurations)}}}{{c}}{{{header}}} \\\\",
        " &  & "
        + " & ".join(
            f"\\multicolumn{{2}}{{c}}{{{cfg.latex}}}" for cfg in configurations
        )
        + " \\\\",
        "\\midrule",
    ]

    for suite in suites:
        subset_suite = long_cells[long_cells["suite"] == suite]
        benches = subset_suite["benchmark"].unique()
        span = len(benches)

        for i, bench in enumerate(benches):
            row = []
            row.append(
                f"\\multirow{{{span}}}{{*}}{{\\rotatebox{{90}}{{{suite}}}}}"
                if i == 0
                else ""
            )
            row.append(bench.latex)

            for cfg in configurations:
                pair = subset_suite[
                    (subset_suite["benchmark"] == bench)
                    & (subset_suite["configuration"] == cfg)
                ]["text"].tolist()
                row.extend(pair if pair else ["--", "--"])

            lines.append(" & ".join(row) + " \\\\")

        lines.append("\\midrule")

    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")

    table = "\n".join(lines)

    with open(output_file, "w") as f:
        f.write(table)


def plot_ripgrep_subset(df, outfile=None):
    colour_map = {
        GCVS.GC: ["#3A87D9", "#1A5C85", 0.8, "-"],
        GCVS.ARC: ["#FF8F2E", "#D66000", 0.8, "-"],
        GCVS.RC: ["#FF8F2E", "#D66000", 0.8, "-"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))

    i = 0
    handles, labels = [], []
    benchmarks = ["linux_word", "linux_unicode_greek", "linux_alternates"]
    df = df[df["benchmark"].isin(benchmarks)]
    if df.empty:
        return
    df["benchmark"] = df["benchmark"].astype("category")
    df["benchmark"] = df["benchmark"].cat.reorder_categories(benchmarks)
    for benchmark, results in df.groupby("benchmark"):
        ax = axes[i]
        for config, snapshot in results.groupby("configuration"):
            heap_size_mb = snapshot["value"] / (1024 * 1024)
            upper_mb = snapshot["ci_upper"] / (1024 * 1024)
            lower_mb = snapshot["ci_lower"] / (1024 * 1024)
            ax.plot(
                snapshot["normalized_time"],
                heap_size_mb,
                label=config.latex,
                linewidth=1,
                linestyle=colour_map[config][3],
                alpha=colour_map[config][2],
                color=colour_map[config][0],
            )
            ax.fill_between(
                snapshot["normalized_time"],
                lower_mb,
                upper_mb,
                alpha=0.1,
                color=colour_map[config][1],
            )
            ax.margins(x=0, y=0)
        ax.set_ylim(0, 10)
        ax.set_title(benchmark, fontsize=16, y=1.02)

        if i == 0:
            handles, labels = [], []
            for cfg in [GCVS.ARC, GCVS.GC]:
                color = colour_map[cfg][0]
                handles.append(
                    plt.Line2D([], [], color=color, lw=2, linestyle=colour_map[cfg][3])
                )
                labels.append(str(cfg.latex))
            handles.reverse()
            labels.reverse()
            legend = ax.legend(
                handles,
                labels,
                handlelength=1.7,
                frameon=True,
                fontsize=10,
            )
            frame = legend.get_frame()
            frame.set_linewidth(0.5)

        i += 1

    fig.supylabel("Heap Size (MiB)", fontsize=12, y=0.55)
    fig.supxlabel("Normalized Time (0→1)", fontsize=12, y=0.05)
    plt.tight_layout()
    plt.savefig(outfile, format="svg", bbox_inches="tight")
    return outfile


def plot_binary_trees_mem(df):
    colour_map = {
        GCVS.GC: ["#3A87D9", "#1A5C85", 0.8, "-"],
        GCVS.ARC: ["#FF8F2E", "#D66000", 0.8, "-"],
        GCVS.RC: ["#FF8F2E", "#D66000", 0.8, "-"],
        GCVS.TYPED_ARENA: [
            "grey",
            "grey",
            0.6,
            "-",
        ],
        GCVS.RUST_GC: [
            "#008080",
            "#005757",
            0.6,
            "-",
        ],
    }

    df = df[df["suite"] == "binary_trees"]
    if df.empty:
        return
    fig, axes = plt.subplots(1, 1, figsize=(4.5, 4.5))

    axes = [axes]

    for idx, (benchmark, results) in enumerate(df.groupby("benchmark")):
        ax = axes[idx]

        for config, snapshot in results.groupby("configuration"):
            heap_size_mb = snapshot["value"] / (1024 * 1024)
            upper_mb = snapshot["ci_upper"] / (1024 * 1024)
            lower_mb = snapshot["ci_lower"] / (1024 * 1024)

            ax.plot(
                snapshot["normalized_time"],
                heap_size_mb,
                label=config.latex,
                linewidth=1,
                linestyle=colour_map[config][3],
                alpha=colour_map[config][2],
                color=colour_map[config][0],
            )
            ax.fill_between(
                snapshot["normalized_time"],
                lower_mb,
                upper_mb,
                alpha=0.1,
                color=colour_map[config][1],
            )
            ax.margins(x=0, y=0)

        ax.set_ylim(0, 5)
        ax.set_ylabel("Heap size (MiB)", fontsize=14, labelpad=10)
        ax.set_xlabel("Normalized Time (0→1)", fontsize=14, labelpad=10)

        handles, labels = [], []
        for cfg in [GCVS.ARC, GCVS.TYPED_ARENA, GCVS.GC, GCVS.RUST_GC]:
            color = colour_map[cfg][0]
            handles.append(
                plt.Line2D([], [], color=color, lw=2, linestyle=colour_map[cfg][3])
            )
            labels.append(str(cfg.latex))
        handles
        labels
        legend = ax.legend(
            handles,
            labels,
            ncols=2,
            handlelength=1,
            frameon=True,
            fontsize=10,
        )
        frame = legend.get_frame()
        frame.set_linewidth(0.5)

    plt.tight_layout()
    plt.savefig("plots/binary_trees_mem.svg", format="svg", bbox_inches="tight")


def plot_ripgrep_full(df, outfile=None):
    colour_map = {
        GCVS.GC: ["#3A87D9", "#1A5C85", 0.8, "-"],
        GCVS.ARC: ["#FF8F2E", "#D66000", 0.8, "-"],
        GCVS.RC: ["#FF8F2E", "#D66000", 0.8, "-"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))

    i = 0
    handles, labels = [], []
    benchmarks = ["linux_word", "linux_unicode_greek", "linux_alternates"]
    df = df[df["benchmark"].isin(benchmarks)]
    df["benchmark"] = df["benchmark"].astype("category")
    df["benchmark"] = df["benchmark"].cat.reorder_categories(benchmarks)
    for benchmark, results in df.groupby("benchmark"):
        ax = axes[i]
        for config, snapshot in results.groupby("configuration"):
            heap_size_mb = snapshot["value"] / (1024 * 1024)
            upper_mb = snapshot["ci_upper"] / (1024 * 1024)
            lower_mb = snapshot["ci_lower"] / (1024 * 1024)
            ax.plot(
                snapshot["normalized_time"],
                heap_size_mb,
                label=config.latex,
                linewidth=1,
                linestyle=colour_map[config][3],
                alpha=colour_map[config][2],
                color=colour_map[config][0],
            )
            ax.fill_between(
                snapshot["normalized_time"],
                lower_mb,
                upper_mb,
                alpha=0.1,
                color=colour_map[config][1],
            )
            ax.margins(x=0, y=0)
        ax.set_ylim(0, 10)
        ax.set_title(benchmark, fontsize=16, y=1.02)

        if i == 0:
            handles, labels = [], []
            for cfg in [GCVS.ARC, GCVS.GC]:
                color = colour_map[cfg][0]
                handles.append(
                    plt.Line2D([], [], color=color, lw=2, linestyle=colour_map[cfg][3])
                )
                labels.append(str(cfg.latex))
            handles.reverse()
            labels.reverse()
            legend = ax.legend(
                handles,
                labels,
                handlelength=1.7,
                frameon=True,
                fontsize=10,
            )
            frame = legend.get_frame()
            frame.set_linewidth(0.5)

        i += 1

    fig.supylabel("Heap Size (MiB)", fontsize=12, y=0.55)
    fig.supxlabel("Normalized Time (0→1)", fontsize=12, y=0.05)
    plt.tight_layout()
    plt.savefig(outfile, format="svg", bbox_inches="tight")
    return outfile
