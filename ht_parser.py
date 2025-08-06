import hashlib
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zstandard as zstd
from scipy.interpolate import interp1d

from build import GCVS, Aggregation
from helpers import BMS, arithmetic_mean, to_cfg

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
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "legend.frameon": False,
        "legend.columnspacing": 1.0,
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


class AllocationInfo:
    def __init__(self, size: int, trace_id: int):
        self.size = size
        self.trace_id = trace_id


def clear_cache():
    cache_dir = os.path.expanduser("~/.heaptrack_cache")
    if os.path.exists(cache_dir):
        import shutil

        shutil.rmtree(cache_dir)
        print("Cache cleared")
    else:
        print("No cache directory found")


def get_cache_key(filepath: str, snapshot_interval: int) -> str:
    mtime = os.path.getmtime(filepath)
    cache_input = f"{filepath}:{mtime}:{snapshot_interval}"
    return hashlib.md5(cache_input.encode()).hexdigest()


def get_cache_path(cache_key: str) -> str:
    cache_dir = os.path.expanduser("~/.heaptrack_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{cache_key}.pkl")


def save_to_cache(cache_key: str, df: pd.DataFrame, stats: Dict):
    cache_path = get_cache_path(cache_key)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"df": df, "stats": stats}, f)
        print(f"Cached results to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")


def load_from_cache(cache_key: str) -> Optional[Tuple[pd.DataFrame, Dict]]:
    cache_path = get_cache_path(cache_key)
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["df"], data["stats"]
    except Exception as e:
        print(f"Warning: Could not load cache: {e}")
        try:
            os.remove(cache_path)
        except:
            pass
        return None


def parse_heaptrack_profile(
    filepath: str, snapshot_interval: int = 1000
) -> Tuple[pd.DataFrame, Dict]:
    # Check cache first
    cache_key = get_cache_key(filepath, snapshot_interval)
    cached_result = load_from_cache(cache_key)
    if cached_result is not None:
        return cached_result

    with open(filepath, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            chunks = []
            while True:
                chunk = reader.read(8192)
                if not chunk:
                    break
                chunks.append(chunk)
        decompressed_data = b"".join(chunks)

    content = decompressed_data.decode("utf-8")
    lines = content.strip().split("\n")

    timestamps = []
    heap_sizes = []
    num_allocations_list = []
    allocation_infos: List[AllocationInfo] = []

    allocation_counts = {}
    allocation_total_size = {}

    current_allocations = 0
    current_total_size = 0
    total_allocations_made = 0
    total_deallocations_made = 0
    timestamp = 0
    event_count = 0

    for line_num, line in enumerate(lines):
        if not line.strip() or line.startswith("#"):
            continue

        parts = line.split()
        if not parts:
            continue

        op = parts[0]
        args = parts[1:] if len(parts) > 1 else []

        try:
            if op == "v":
                # Version info - skip
                continue
            elif op == "X":
                # Executable info - skip
                continue
            elif op == "I":
                # System info - skip
                continue
            elif op == "s":
                # String definition - skip
                continue
            elif op == "t":
                # Trace definition - skip
                continue
            elif op == "i":
                # Instruction pointer - skip
                continue
            elif op == "a":
                # Allocation info: a <size> <trace_id>
                if len(args) >= 2:
                    size = int(args[0], 16)
                    trace_id = int(args[1], 16)
                    allocation_infos.append(AllocationInfo(size, trace_id))

            elif op == "+":
                if len(args) >= 1:
                    alloc_info_index = int(args[0], 16)
                    if alloc_info_index < len(allocation_infos):
                        info = allocation_infos[alloc_info_index]

                        allocation_counts[alloc_info_index] = (
                            allocation_counts.get(alloc_info_index, 0) + 1
                        )
                        allocation_total_size[alloc_info_index] = (
                            allocation_total_size.get(alloc_info_index, 0) + info.size
                        )

                        current_allocations += 1
                        current_total_size += info.size
                        total_allocations_made += 1
                        event_count += 1

            elif op == "-":
                if len(args) >= 1:
                    alloc_info_index = int(args[0], 16)
                    if (
                        alloc_info_index in allocation_counts
                        and allocation_counts[alloc_info_index] > 0
                    ):

                        info = allocation_infos[alloc_info_index]
                        allocation_counts[alloc_info_index] -= 1
                        allocation_total_size[alloc_info_index] -= info.size

                        if allocation_counts[alloc_info_index] == 0:
                            del allocation_counts[alloc_info_index]
                            del allocation_total_size[alloc_info_index]

                        current_allocations -= 1
                        current_total_size -= info.size
                        total_deallocations_made += 1
                        event_count += 1

            elif op == "c":
                # Timestamp: c <timestamp>
                if len(args) >= 1:
                    timestamp = int(args[0], 16)

            elif op == "R":
                # RSS info - skip
                continue
            elif op == "A":
                # Attached mode - reset counters
                current_allocations = 0
                current_total_size = 0
                total_allocations_made = 0
                total_deallocations_made = 0
                allocation_counts.clear()
                allocation_total_size.clear()
                event_count = 0
                continue
            else:
                continue

            if event_count > 0 and event_count % snapshot_interval == 0:
                timestamps.append(timestamp)
                heap_sizes.append(current_total_size)
                num_allocations_list.append(current_allocations)

        except (ValueError, IndexError) as e:
            continue

    needs_final_snapshot = event_count > 0 and (
        not timestamps
        or event_count % snapshot_interval != 0
        or heap_sizes[-1] != current_total_size
        or num_allocations_list[-1] != current_allocations
    )

    if needs_final_snapshot:
        timestamps.append(timestamp)
        heap_sizes.append(current_total_size)
        num_allocations_list.append(current_allocations)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "heap_size": heap_sizes,
            "num_allocations": num_allocations_list,
        }
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    leaked_bytes = sum(allocation_total_size.values())
    num_leaks = sum(allocation_counts.values())

    summary_stats = {
        "num_leaks": num_leaks,
        "leaked_bytes": leaked_bytes,
        "total_allocations": total_allocations_made,
        "total_deallocations": total_deallocations_made,
        "peak_heap_size": df["heap_size"].max() if not df.empty else 0,
        "peak_num_allocations": df["num_allocations"].max() if not df.empty else 0,
        "final_heap_size": current_total_size,
        "final_num_allocations": current_allocations,
    }

    save_to_cache(cache_key, df, summary_stats)
    return df, summary_stats


def interpolate(
    df: pd.DataFrame, target_time: np.ndarray, time_col: str = "normalized_time"
) -> pd.DataFrame:
    """
    Common interpolation logic for resampling data to a target time grid.
    """
    if df.empty or len(df) == 1:
        # Handle edge cases consistently
        return pd.DataFrame(
            {
                "normalized_time": target_time,
                "heap_size": [0] * len(target_time),
                "num_allocations": [0] * len(target_time),
            }
        )

    # Create interpolation functions
    f_heap = interp1d(
        df[time_col], df["heap_size"], kind="linear", fill_value="extrapolate"
    )
    f_alloc = interp1d(
        df[time_col], df["num_allocations"], kind="linear", fill_value="extrapolate"
    )

    # Interpolate values
    heap_values = f_heap(target_time).astype(int)
    alloc_values = f_alloc(target_time).astype(int)

    # Force exact endpoints
    heap_values[0] = 0
    alloc_values[0] = 0
    if len(df) > 0:
        heap_values[-1] = df["heap_size"].iloc[-1]
        alloc_values[-1] = df["num_allocations"].iloc[-1]

    return pd.DataFrame(
        {
            "normalized_time": target_time,
            "heap_size": heap_values,
            "num_allocations": alloc_values,
        }
    )


def normalize_and_resample(df: pd.DataFrame, num_points: int = 200) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Normalize time
    t0, t1 = df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
    if t1 == t0:
        df["normalized_time"] = 0.0
    else:
        df["normalized_time"] = (df["timestamp"] - t0) / (t1 - t0)

    # Add zero starting point
    zero_point = pd.DataFrame(
        {
            "timestamp": [t0],
            "heap_size": [0],
            "num_allocations": [0],
            "normalized_time": [0.0],
        }
    )
    df = pd.concat([zero_point, df]).reset_index(drop=True)
    # df = df.drop_duplicates(subset=["normalized_time"]).reset_index(drop=True)

    # Create target grid and interpolate
    uniform_t = np.linspace(0.0, 1.0, num_points)
    uniform_t[0] = 0.0
    uniform_t[-1] = 1.0

    return interpolate(df, uniform_t)


def align_to_grid(df, points=1000):
    if len(df) == points:
        return df

    time = np.linspace(0.0, 1.0, points)
    return interpolate(df, time)


def calculate_points_by_duration(
    df: pd.DataFrame, min_points: int = 200, max_points: int = 200000
) -> int:

    duration_ms = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
    duration_seconds = duration_ms / 1_000

    # Simple linear scaling: 1000 points per second of runtime
    calculated_points = int(200 + (duration_seconds * 1000))
    print(f"Calculated points {calculated_points}")

    return max(min_points, min(calculated_points, max_points))


def plot_time_series(df: pd.DataFrame):
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
        GCVS.BASELINE: ["#000000", "grey", 0.6, "--"],
    }

    # Count unique benchmarks and determine grid layout
    print(df["suite"].unique())
    df = df[df["suite"] == "binary_trees"]
    benchmarks = df["benchmark"].unique()
    n_benchmarks = len(benchmarks)
    print(n_benchmarks)

    # Calculate grid dimensions (adjust as needed)
    cols = 1  # You can adjust this based on your preference
    rows = (n_benchmarks + cols - 1) // cols  # Ceiling division

    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))

    # Handle case where there's only one subplot
    if n_benchmarks == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # Flatten axes array for easier indexing
    axes_flat = axes.flatten() if n_benchmarks > 1 else axes

    for idx, (benchmark, results) in enumerate(df.groupby("benchmark")):
        ax = axes_flat[idx]

        for config, snapshot in results.groupby("configuration"):
            heap_size_mb = snapshot["heap_size"] / (1024 * 1024)
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
        # ax.set_title("Heap profile", fontsize=16, y=1.05)
        ax.set_ylabel("Heap size (MiB)", fontsize=14, labelpad=10)
        ax.set_xlabel("Normalized Time (0→1)", fontsize=14, labelpad=10)

        # Add legend to each subplot (or only to the first one if you prefer)
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

    # Hide unused subplots
    for idx in range(n_benchmarks, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Add common labels
    # fig.supylabel("Heap Size (MiB)", fontsize=12, x=-0.002)
    # fig.supxlabel("Normalized Time (0→1)", fontsize=12)

    plt.tight_layout()
    plt.savefig("memplots/binary_trees.svg", format="svg", bbox_inches="tight")


def plot_ripgrep_subset(df, benchmark=None, suite=None, title=None, outfile=None):

    colour_map = {
        GCVS.GC: ["#3A87D9", "#1A5C85", 0.8, "-"],
        GCVS.ARC: ["#FF8F2E", "#D66000", 0.8, "-"],
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
            heap_size_mb = snapshot["heap_size"] / (1024 * 1024)
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
    plt.savefig(f"plots/ripgrep_subset.svg", format="svg", bbox_inches="tight")
    return outfile


def process_single_run(fp):
    parts = os.path.splitext(fp)[0].split("results/", 1)[-1].split("/")
    suite = parts[0]
    exp = parts[2]
    benchmark, invocation = parts[-1].rsplit("-", 1)
    time_interval_ns = 1000
    raw_df, stats = parse_heaptrack_profile(fp, time_interval_ns)

    num_points = calculate_points_by_duration(raw_df)
    normalized_df = normalize_and_resample(raw_df, num_points)
    df = align_to_grid(normalized_df, 1000)

    df["benchmark"] = benchmark
    df["suite"] = suite
    df["configuration"] = to_cfg(f"{suite}-{exp}")

    df["invocation"] = invocation

    return df


def aggregate_data(data):
    suite, benchmark, df = data
    df = (
        df.groupby(["configuration", "normalized_time"])["heap_size"]
        .apply(arithmetic_mean)
        .rename({"mean": "heap_size"})
        .unstack()
        .reset_index()
    )
    df["suite"] = suite
    df["benchmark"] = benchmark
    return df


def normalize_data(df):
    data = [
        (suite, benchmarks, results)
        for (suite, benchmarks), results in df.groupby(["suite", "benchmark"])
    ]

    per_benchmark = []

    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        future_to_plot = {executor.submit(aggregate_data, d): d for d in data}
        for future in as_completed(future_to_plot):
            (suite, benchmark, df) = future_to_plot[future]
            try:
                result = future.result()
                per_benchmark.append(result)
            except Exception as e:
                print(f"✗ Error processing {suite}-{benchmark}: {e}")
    return pd.concat(per_benchmark, ignore_index=True)


def parse_heaptrack(
    filepaths: List[str],
    time_interval_ns: int = 10000,
):

    results = []

    parse_start = time.time()
    with ProcessPoolExecutor(max_workers=24) as executor:
        future_to_file = {
            executor.submit(process_single_run, fp): fp for fp in filepaths
        }

        for future in as_completed(future_to_file):
            filepath = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    print(f"Collected {len(results)} results in {time.time() - parse_start:.2f}s")
    df = pd.concat(results, ignore_index=True)

    snapshots = normalize_data(df.copy())
    # plot_ripgrep_subset(snapshots)
    plot_time_series(snapshots)

    return (
        df.groupby(["suite", "benchmark", "configuration", "invocation"])["heap_size"]
        .mean()
        .reset_index()
        .rename(columns={"heap_size": "value"})
    )
