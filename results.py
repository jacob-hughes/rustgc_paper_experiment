import glob
import json
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
import zstandard as zstd
from scipy.interpolate import interp1d
import warnings
from scipy.sparse import SparseEfficiencyWarning

warnings.filterwarnings("ignore", module=r"^(scipy|numpy)\..*")

from build import (
    GCVS,
    Aggregation,
    BenchmarkSuite,
    Elision,
    HeapSize,
    Measurement,
    Metric,
    PremOpt,
)
from helpers import (
    BASELINES,
    EXCLUDE,
    arithmetic_mean,
    cache,
    geometric_mean,
    metadata,
    parallelise,
)
from plots import (
    SimplePlotConfig,
    SimplePlotter,
    mk_alloc_table,
    mk_heap_table,
    plot_binary_trees_mem,
    plot_ripgrep_subset,
    tabulate_benchmarks,
    tabulate_suites,
)


class Results:
    def __init__(self, df):
        self.df = df.copy()
        self._results = None

    @staticmethod
    def _to_benchmark(row):
        for b in row["suite"].benchmarks:
            if b.name.lower() == row["benchmark"].lower():
                return b
        raise ValueError(f"Benchmark for {row['benchmark']} not found.")

    @classmethod
    def concat(cls, l):
        if not l:
            return cls(pd.DataFrame())
        df = pd.concat([r.df for r in l], ignore_index=True)
        return cls(df)

    def _prepare_data(self, experiment, baseline=None):
        df = self.df.copy()

        profiles = [e.full for e in list(experiment)] + ["default"]
        df = df[df["configuration"].isin(profiles)]

        df["configuration"] = df["configuration"].map(experiment)
        if baseline is not None:
            df["is_baseline"] = df[df["configuration"] == baseline]
        else:
            df["is_baseline"] = df["configuration"].isin(BASELINES)

        group_cols = ["suite", "benchmark", "invocation", "metric"]
        baselines = df[df["is_baseline"]][group_cols + ["value"]].rename(
            columns={"value": "baseline_value"}
        )
        df = df.merge(baselines, on=group_cols, how="left")
        df["ratio"] = df["value"] / df["baseline_value"]

        df["suite"] = df["suite"].map({b.name: b for b in BenchmarkSuite.all()})
        df["benchmark"] = df.apply(self._to_benchmark, axis=1)

        return df

    def aggregate(self, experiment, baseline=None):
        if self.df.empty:
            return Means(self.df, experiment.__name__.lower())
        df = self._prepare_data(experiment, baseline=None)

        results = []

        # Individual benchmark level
        individual_groups = ["benchmark", "configuration", "suite", "metric"]

        individual_ratios = (
            df.groupby(individual_groups)["ratio"]
            .apply(geometric_mean)
            .unstack()
            .reset_index()
            .rename(
                columns={
                    "value": "ratio",
                    "ci_lower": "ratio_lower",
                    "ci_upper": "ratio_upper",
                }
            )
        )

        individual_values = (
            df.groupby(individual_groups)["value"]
            .apply(arithmetic_mean)
            .unstack()
            .reset_index()
        )

        individual = pd.merge(
            individual_ratios, individual_values, on=individual_groups
        )
        individual["agg_type"] = Aggregation.INDIVIDUAL
        results.append(individual)

        # Suite level
        suite_groups = ["suite", "configuration", "metric"]

        suite_values = (
            df.groupby(suite_groups)["value"]
            .apply(geometric_mean)
            .unstack()
            .reset_index()
        )

        suite_ratios = (
            df.groupby(suite_groups)["ratio"]
            .apply(geometric_mean)
            .unstack()
            .reset_index()
            .rename(
                columns={
                    "value": "ratio",
                    "ci_lower": "ratio_lower",
                    "ci_upper": "ratio_upper",
                }
            )
        )

        suite = pd.merge(suite_values, suite_ratios, on=suite_groups)
        suite["agg_type"] = Aggregation.SUITE_GEO
        results.append(suite)

        # Global level
        global_groups = ["configuration", "metric"]

        global_values = (
            df.groupby(global_groups)["value"]
            .apply(geometric_mean)
            .unstack()
            .reset_index()
        )

        global_ratios = (
            df.groupby(global_groups)["ratio"]
            .apply(geometric_mean)
            .unstack()
            .reset_index()
            .rename(
                columns={
                    "value": "ratio",
                    "ci_lower": "ratio_lower",
                    "ci_upper": "ratio_upper",
                }
            )
        )

        global_result = pd.merge(global_values, global_ratios, on=global_groups)
        global_result["agg_type"] = Aggregation.GLOBAL_GEO
        results.append(global_result)

        # Combine all results
        res = pd.concat(results, ignore_index=True)
        return Means(res, experiment.__name__.lower())


class Means(Results):
    def __init__(self, df, experiment):
        self.df = df
        self.experiment = experiment

    def plot(self, metric, label=None, xlim=None, show_legend=False):
        if self.df.empty:
            return
        Path("plots").mkdir(parents=True, exist_ok=True)
        df = self.df
        df = df[df["metric"] == metric]
        df = df[~df["configuration"].isin(EXCLUDE)]
        output_file = f"plots/{self.experiment}_{metric.pathname}.svg"

        cfg = SimplePlotConfig(
            figsize=(3.2, 2.5), group_width=0.8, show_legend=show_legend
        )
        plot = SimplePlotter(cfg)
        plot.plot(
            df,
            self.experiment,
            "ratio",
            output_file=output_file,
            xlim=xlim,
        )

    def tabulate(self, metric, agg_type, format_func=None, split=None):
        df = self.df
        if df.empty:
            return
        df = df[df["metric"] == metric]
        df = df[df["agg_type"] == agg_type]
        if agg_type == Aggregation.INDIVIDUAL:
            out = f"tables/appendix_{self.experiment}_{metric.pathname}"
            if not split:
                tabulate_benchmarks(
                    df,
                    header=metric.latex,
                    formatter=format_func,
                    output_file=f"{out}.tex",
                )
                return

            left, right = split
            ldf = df[df["suite"].apply(lambda s: s.name in left)]
            rdf = df[df["suite"].apply(lambda s: s.name in right)]
            tabulate_benchmarks(
                ldf,
                header=metric.latex,
                formatter=format_func,
                output_file=f"{out}_1.tex",
            )
            tabulate_benchmarks(
                rdf,
                header=metric.latex,
                formatter=format_func,
                output_file=f"{out}_2.tex",
            )
        elif agg_type == Aggregation.SUITE_GEO:
            out = f"tables/{self.experiment}_{metric.pathname}"
            tabulate_suites(
                df, formatter=format_func, header=metric.latex, output_file=f"{out}.tex"
            )

    def tabulate_binary_trees(self):
        df = self.df
        df = df[df["metric"] == Metric.WALLCLOCK]
        df = df[df["agg_type"] == Aggregation.SUITE_GEO]
        df = df[df["suite"].apply(lambda s: s.name == "binary_trees")]

        table = (
            df.loc[:, ["configuration", "ratio", "ratio_lower", "ratio_upper"]]
            .rename(
                columns={
                    "configuration": "Configuration",
                    "ratio": "Wall-clock ratio",
                    "ratio_lower": "CI lower",
                    "ratio_upper": "CI upper",
                }
            )
            .round(2)
            .style.format(
                {
                    "Wall-clock ratio": "{:.2f}",
                    "CI lower": "{:.2f}",
                    "CI upper": "{:.2f}",
                }
            )
            .hide(axis="index")
            .to_latex(
                hrules=True,
                column_format="lrrr",
                position="htbp",
            )
            .replace("CI lower", r"\scriptsize\textcolor{gray!60}{CI lower}")
            .replace("CI upper", r"\scriptsize\textcolor{gray!60}{CI upper}")
            .replace(" & ", " & ", 1)  # keep first space only (tidy spacing)
        )

        with open("tables/binary_trees.tex", "w") as f:
            f.write(table)

    def tabulate_allocators(self):
        df = self.df
        df = df[df["metric"] == Metric.WALLCLOCK]
        df = df[df["agg_type"] == Aggregation.SUITE_GEO]
        df = df[df["configuration"].isin([GCVS.ARC, GCVS.RC, GCVS.BASELINE])]
        mk_alloc_table(df, output_file="tables/gcvs_perf_summary.tex")

    def tabulate_fixed_size(self):
        df = self.df
        df = df[df["metric"] == Metric.WALLCLOCK]
        df = df[df["configuration"] != HeapSize.DEFAULT]
        df = df[df["agg_type"] == Aggregation.SUITE_GEO]
        mk_heap_table(df, output_file="tables/fixed_heaps.tex")


@dataclass
class AllocInfo:
    size: int
    trace_id: int


class Heaptrack(Results):
    def __init__(self, df, time_series):
        self.df = df
        self.time_series = time_series

    def aggregate_time_series(self, experiment):
        df = self.time_series.copy()
        df = df[
            df["configuration"].isin(
                ["default", "gcvs-arc", "gcvs-typed-arena", "gcvs-rust-gc"]
            )
        ]
        df["configuration"] = df["configuration"].map(experiment)
        return (
            df.groupby(["suite", "configuration", "benchmark", "normalized_time"])[
                "heap_size"
            ]
            .apply(arithmetic_mean)
            .rename({"mean": "heap_size"})
            .unstack()
            .reset_index()
        )

    def plot_time_series(self):
        df = self.aggregate_time_series(GCVS)
        plot_ripgrep_subset(df, outfile="plots/ripgrep_subset.svg")
        plot_binary_trees_mem(df)

    @staticmethod
    @cache()
    def parse(file, snapshot_interval):
        timestamps, heap_sizes, num_allocs_list = [], [], []
        alloc_infos = []
        alloc_counts, alloc_total_size = {}, {}
        current_allocs = (
            current_heap
        ) = total_allocs = total_frees = timestamp = event_count = 0

        SKIP_OPS = {"v", "X", "I", "s", "t", "i", "R"}

        def snapshot():
            timestamps.append(timestamp)
            heap_sizes.append(current_heap)
            num_allocs_list.append(current_allocs)

        with open(file, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                buf = b""
                while True:
                    chunk = reader.read(8192)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line or line.startswith(b"#"):
                            continue
                        parts = line.decode("utf-8").split()
                        op, *args = parts
                        try:
                            if op in SKIP_OPS:
                                continue
                            if op == "a" and len(args) >= 2:
                                alloc_infos.append(
                                    AllocInfo(int(args[0], 16), int(args[1], 16))
                                )
                            elif op == "+" and len(args) >= 1:
                                idx = int(args[0], 16)
                                if idx < len(alloc_infos):
                                    info = alloc_infos[idx]
                                    alloc_counts[idx] = alloc_counts.get(idx, 0) + 1
                                    alloc_total_size[idx] = (
                                        alloc_total_size.get(idx, 0) + info.size
                                    )
                                    current_allocs += 1
                                    current_heap += info.size
                                    total_allocs += 1
                                    event_count += 1
                            elif op == "-" and len(args) >= 1:
                                idx = int(args[0], 16)
                                if alloc_counts.get(idx, 0) > 0:
                                    info = alloc_infos[idx]
                                    alloc_counts[idx] -= 1
                                    alloc_total_size[idx] -= info.size
                                    if alloc_counts[idx] == 0:
                                        alloc_counts.pop(idx)
                                        alloc_total_size.pop(idx)
                                    current_allocs -= 1
                                    current_heap -= info.size
                                    total_frees += 1
                                    event_count += 1
                            elif op == "c" and len(args) >= 1:
                                timestamp = int(args[0], 16)
                            elif op == "A":
                                current_allocs = (
                                    current_heap
                                ) = total_allocs = total_frees = event_count = 0
                                alloc_counts.clear()
                                alloc_total_size.clear()
                                continue
                            if event_count > 0 and event_count % snapshot_interval == 0:
                                snapshot()
                        except Exception:
                            continue

        if event_count > 0 and (
            not timestamps
            or event_count % snapshot_interval != 0
            or (heap_sizes and heap_sizes[-1] != current_heap)
            or (num_allocs_list and num_allocs_list[-1] != current_allocs)
        ):
            snapshot()

        df = (
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "heap_size": heap_sizes,
                    "num_allocs": num_allocs_list,
                }
            )
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        return df

    @staticmethod
    @cache()
    def process_single_pexec(df):
        return pd.DataFrame(
            {"value": [df["heap_size"].mean()], "metric": [Metric.MEM_HSIZE_AVG]}
        )

    @classmethod
    def _oversample(cls, df, min_points=200, max_points=200000):
        duration_ms = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
        duration_seconds = duration_ms / 1_000
        calculated_points = int(200 + (duration_seconds * 1000))
        return max(min_points, min(calculated_points, max_points))

    @classmethod
    def _interpolate(
        cls,
        df,
        target_time,
        time_col="normalized_time",
    ):
        f_heap = interp1d(
            df[time_col], df["heap_size"], kind="linear", fill_value="extrapolate"
        )
        heap_values = f_heap(target_time).astype(int)
        heap_values[0] = 0
        if len(df) > 0:
            heap_values[-1] = df["heap_size"].iloc[-1]

        return pd.DataFrame(
            {
                "normalized_time": target_time,
                "heap_size": heap_values,
            }
        )

    @classmethod
    def _align_to_grid(cls, df, points=1000):
        if len(df) == points:
            return df

        time = np.linspace(0.0, 1.0, points)
        return cls._interpolate(df, time)

    @classmethod
    def _normalize_and_resample(cls, df, num_points=200):
        if df.empty:
            return df

        df = df.sort_values("timestamp").reset_index(drop=True)

        t0, t1 = df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
        if t1 == t0:
            df["normalized_time"] = 0.0
        else:
            df["normalized_time"] = (df["timestamp"] - t0) / (t1 - t0)

        zero_point = pd.DataFrame(
            {
                "timestamp": [t0],
                "heap_size": [0],
                "normalized_time": [0.0],
            }
        )
        df = pd.concat([zero_point, df]).reset_index(drop=True)
        df = df.drop_duplicates(subset=["normalized_time"]).reset_index(drop=True)

        uniform_t = np.linspace(0.0, 1.0, num_points)
        uniform_t[0] = 0.0
        uniform_t[-1] = 1.0

        return cls._interpolate(df, uniform_t)

    @classmethod
    @cache()
    def process_single_profile(cls, df):
        num_points = cls._oversample(df)
        normalized_df = cls._normalize_and_resample(df, num_points)
        return cls._align_to_grid(normalized_df, 1000)

    @classmethod
    def from_file(cls, f, snapshot_interval=1000):
        df = cls.parse(f, snapshot_interval)
        m = metadata(f)
        time_series = cls.process_single_profile(df.copy()).assign(**m)
        pexec = cls.process_single_pexec(df).assign(**m)
        return cls(pexec, time_series)

    @classmethod
    def concat(cls, l):
        if not l:
            return cls(pd.DataFrame())
        df = pd.concat([h.df for h in l], ignore_index=True)
        time_series = pd.concat([h.time_series for h in l], ignore_index=True)
        return cls(df, time_series)

    @classmethod
    @cache()
    def process(cls, exps):
        return cls.concat(
            parallelise(
                cls.from_file,
                exps.results(Measurement.HEAPTRACK),
                desc="Processing heaptrack profiles",
            )
        )


class Perf(Results):
    def __init__(self, df):
        self.df = df

    @classmethod
    def from_file(cls, f):
        return cls(cls.parse(f))

    @staticmethod
    # @cache()
    def parse(file):
        path = Path(file)
        suite = path.parts[-3]

        def to_cfg(value):
            return value.removeprefix(f"{suite}-")

        df = pd.read_csv(
            file,
            sep="\t",
            comment="#",
            index_col="suite",
            converters={"criterion": Metric, "executor": to_cfg},
        )

        df = df.rename(
            columns={"executor": "configuration", "criterion": "metric"}
        ).reset_index()[["benchmark", "configuration", "value", "metric", "invocation"]]

        df["suite"] = suite
        return df

    @classmethod
    # @cache()
    def process(cls, exps):
        return cls.concat(
            parallelise(
                cls.from_file,
                exps.results(Measurement.PERF),
                desc="Processing performance results",
            )
        )


class Metrics(Results):
    @staticmethod
    @cache()
    def parse(f):
        records = []
        with open(f, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                new_row = {}
                for key, value in data.items():
                    try:
                        metric = Metric(key)
                        new_row[metric] = value
                    except ValueError:
                        pass
                records.append(new_row)

        df = pd.DataFrame(records)
        gcs = df[df[Metric.COLLECTION_NUMBER] != -1].copy()
        on_exit = df[df[Metric.COLLECTION_NUMBER] == -1].copy()[
            [
                Metric.OBJ_ALLOCD_ARC,
                Metric.OBJ_ALLOCD_BOX,
                Metric.OBJ_ALLOCD_GC,
                Metric.OBJ_ALLOCD_RC,
                Metric.FLZ_REGISTERED,
                Metric.FLZ_RUN,
                Metric.FLZ_ELIDED,
            ]
        ]

        gcs[Metric.MEM_HSIZE_AVG] = gcs[Metric.MEM_HSIZE_EXIT]
        gcs[Metric.TOTAL_COLLECTIONS] = gcs[Metric.COLLECTION_NUMBER]
        gcs = gcs[
            [Metric.MEM_HSIZE_AVG, Metric.TIME_TOTAL, Metric.TOTAL_COLLECTIONS]
        ].agg(
            {
                Metric.MEM_HSIZE_AVG: "mean",
                Metric.TIME_TOTAL: "sum",
                Metric.TOTAL_COLLECTIONS: "count",
            }
        )

        on_exit_data = on_exit.iloc[0].to_dict() if not on_exit.empty else {}

        merged_data = {**on_exit_data, **gcs.to_dict()}
        merged = pd.DataFrame([merged_data])
        merged = merged.melt(var_name="metric", value_name="value")

        return merged

    @classmethod
    def from_file(cls, f):
        m = metadata(f)
        return cls(cls.parse(f).assign(**m))

    @classmethod
    @cache()
    def process(cls, exps):
        return cls.concat(
            parallelise(
                cls.from_file,
                exps.results(Measurement.METRICS),
                desc="Processing GC metrics",
            )
        )
