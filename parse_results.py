import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path

import pandas as pd
import zstandard as zstd
from tqdm import tqdm

from build import Measurement, Metric
from helpers import cache


def parse(files, experiment, suite, measurement):
    if measurement == Measurement.PERF:
        return parse_perf(files.pop(), experiment, suite)
    if measurement == Measurement.METRICS:
        df = parse_parallel(parse_metrics, files, experiment, suite)
        return df
    if measurement == Measurement.HEAPTRACK:
        df = parse_parallel(parse_heaptrack, files, experiment, suite)
        return df


@cache()
def process_heaptrack_profile(df):
    hsz = {
        "value": df["heap_size"].mean(),
        "experiment": df.iloc[0]["experiment"],
        "invocation": df.iloc[0]["invocation"],
        "benchmark": df.iloc[0]["benchmark"],
        "configuration": df.iloc[0]["configuration"],
        "suite": df.iloc[0]["suite"],
        "metric": Metric.MEM_HSIZE_AVG,
    }
    return pd.DataFrame([hsz])

    # invocations = []
    # grouped = raw.groupby(
    #     ["experiment", "invocation", "benchmark", "configuration", "suite"]
    # )
    # for group_keys, group_df in grouped:
    #     print(f"Processing: {group_keys}...")
    #     df = group_df["heap_size"].mean().rename(columns={"heap_size": "value"})
    #
    #     for col, val in zip(
    #         ["experiment", "invocation", "benchmark", "configuration", "suite"],
    #         group_keys,
    #     ):
    #         df[col] = val
    #     invocations.append(df)
    #
    # return pd.concat(invocations, ignore_index=True)


def parse_metrics(f, experiment, suite):
    return parse_metric_file(f, experiment, suite)


def process_ht(f, experiment, suite):
    parsed = parse_heaptrack_profile(f, experiment, suite)
    return process_heaptrack_profile(parsed)


@cache()
def parse_metric_file(f, experiment, suite):
    path = Path(f)
    benchmark, invocation = path.stem.rsplit("-", 1)
    configuration = experiment(path.parent.name)
    records = []
    expname = experiment.__name__.lower()
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
    gcs = gcs[[Metric.MEM_HSIZE_AVG, Metric.TIME_TOTAL, Metric.TOTAL_COLLECTIONS]].agg(
        {
            Metric.MEM_HSIZE_AVG: "mean",
            Metric.TIME_TOTAL: "sum",
            Metric.TOTAL_COLLECTIONS: "count",
        }
    )

    merged = pd.DataFrame([{**on_exit.iloc[0].to_dict(), **gcs.to_dict()}])
    merged = merged.melt(var_name="metric", value_name="value")

    merged["benchmark"] = benchmark
    merged["configuration"] = configuration
    merged["experiment"] = expname
    merged["invocation"] = invocation

    return merged


@cache()
def parse_perf(file, experiment, suite):

    def to_cfg(name):
        s = name.split("-")[-2:]
        try:
            return experiment(s)
        except:
            return None

    df = pd.read_csv(
        file,
        sep="\t",
        comment="#",
        index_col="suite",
        converters={
            "criterion": Metric,
        },
    )

    df = df.rename(
        columns={"executor": "configuration", "criterion": "metric"}
    ).reset_index()[["benchmark", "configuration", "value", "metric", "invocation"]]

    df["experiment"] = experiment
    return df


@dataclass
class AllocInfo:
    size: int
    trace_id: int


@cache()
def parse_heaptrack_profile(file, experiment, suite, snapshot_interval=1000):
    path = Path(file)
    benchmark, invocation = path.stem.rsplit("-", 1)
    configuration = experiment(path.parent.name)
    expname = experiment.__name__.lower()

    # Snapshot accumulators
    timestamps, heap_sizes, num_allocs_list = [], [], []
    alloc_infos = []
    alloc_counts, alloc_total_size = {}, {}

    # State
    current_allocs = current_heap = total_allocs = total_frees = timestamp = (
        event_count
    ) = 0

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
                            current_allocs = current_heap = total_allocs = (
                                total_frees
                            ) = event_count = 0
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
                "benchmark": benchmark,
                "suite": suite,
                "invocation": invocation,
                "configuration": configuration,
                "experiment": expname,
            }
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return df


# @cache()
# def parse_heaptrack_profile(file, experiment, suite, snapshot_interval=1000):
#     path = Path(file)
#     benchmark, invocation = path.stem.rsplit("-", 1)
#     configuration = experiment(path.parent.name)
#
#     expname = experiment.__name__.lower()
#     with open(file, "rb") as f:
#         dctx = zstd.ZstdDecompressor()
#         with dctx.stream_reader(f) as reader:
#             chunks = []
#             while True:
#                 chunk = reader.read(8192)
#                 if not chunk:
#                     break
#                 chunks.append(chunk)
#         decompressed_data = b"".join(chunks)
#
#     content = decompressed_data.decode("utf-8")
#     lines = content.strip().split("\n")
#
#     timestamps = []
#     heap_sizes = []
#     num_allocations_list = []
#     allocation_infos = []
#
#     allocation_counts = {}
#     allocation_total_size = {}
#
#     current_allocations = 0
#     current_total_size = 0
#     total_allocations_made = 0
#     total_deallocations_made = 0
#     timestamp = 0
#     event_count = 0
#
#     for line_num, line in enumerate(lines):
#         if not line.strip() or line.startswith("#"):
#             continue
#
#         parts = line.split()
#         if not parts:
#             continue
#
#         op = parts[0]
#         args = parts[1:] if len(parts) > 1 else []
#
#         try:
#             if op == "v":
#                 # Version info - skip
#                 continue
#             elif op == "X":
#                 # Executable info - skip
#                 continue
#             elif op == "I":
#                 # System info - skip
#                 continue
#             elif op == "s":
#                 # String definition - skip
#                 continue
#             elif op == "t":
#                 # Trace definition - skip
#                 continue
#             elif op == "i":
#                 # Instruction pointer - skip
#                 continue
#             elif op == "a":
#                 # Allocation info: a <size> <trace_id>
#                 if len(args) >= 2:
#                     size = int(args[0], 16)
#                     trace_id = int(args[1], 16)
#                     allocation_infos.append(AllocationInfo(size, trace_id))
#
#             elif op == "+":
#                 if len(args) >= 1:
#                     alloc_info_index = int(args[0], 16)
#                     if alloc_info_index < len(allocation_infos):
#                         info = allocation_infos[alloc_info_index]
#
#                         allocation_counts[alloc_info_index] = (
#                             allocation_counts.get(alloc_info_index, 0) + 1
#                         )
#                         allocation_total_size[alloc_info_index] = (
#                             allocation_total_size.get(alloc_info_index, 0) + info.size
#                         )
#
#                         current_allocations += 1
#                         current_total_size += info.size
#                         total_allocations_made += 1
#                         event_count += 1
#
#             elif op == "-":
#                 if len(args) >= 1:
#                     alloc_info_index = int(args[0], 16)
#                     if (
#                         alloc_info_index in allocation_counts
#                         and allocation_counts[alloc_info_index] > 0
#                     ):
#
#                         info = allocation_infos[alloc_info_index]
#                         allocation_counts[alloc_info_index] -= 1
#                         allocation_total_size[alloc_info_index] -= info.size
#
#                         if allocation_counts[alloc_info_index] == 0:
#                             del allocation_counts[alloc_info_index]
#                             del allocation_total_size[alloc_info_index]
#
#                         current_allocations -= 1
#                         current_total_size -= info.size
#                         total_deallocations_made += 1
#                         event_count += 1
#
#             elif op == "c":
#                 # Timestamp: c <timestamp>
#                 if len(args) >= 1:
#                     timestamp = int(args[0], 16)
#
#             elif op == "R":
#                 # RSS info - skip
#                 continue
#             elif op == "A":
#                 # Attached mode - reset counters
#                 current_allocations = 0
#                 current_total_size = 0
#                 total_allocations_made = 0
#                 total_deallocations_made = 0
#                 allocation_counts.clear()
#                 allocation_total_size.clear()
#                 event_count = 0
#                 continue
#             else:
#                 continue
#
#             if event_count > 0 and event_count % snapshot_interval == 0:
#                 timestamps.append(timestamp)
#                 heap_sizes.append(current_total_size)
#                 num_allocations_list.append(current_allocations)
#
#         except (ValueError, IndexError) as e:
#             continue
#
#     needs_final_snapshot = event_count > 0 and (
#         not timestamps
#         or event_count % snapshot_interval != 0
#         or heap_sizes[-1] != current_total_size
#         or num_allocations_list[-1] != current_allocations
#     )
#
#     if needs_final_snapshot:
#         timestamps.append(timestamp)
#         heap_sizes.append(current_total_size)
#         num_allocations_list.append(current_allocations)
#
#     df = pd.DataFrame(
#         {
#             "timestamp": timestamps,
#             "heap_size": heap_sizes,
#             "num_allocations": num_allocations_list,
#             "benchmark": benchmark,
#             "suite": suite,
#             "invocation": invocation,
#             "configuration": configuration,
#             "experiment": expname,
#         }
#     )
#
#     df = df.sort_values("timestamp").reset_index(drop=True)
#
#     leaked_bytes = sum(allocation_total_size.values())
#     num_leaks = sum(allocation_counts.values())
#
#     summary_stats = {
#         "num_leaks": num_leaks,
#         "leaked_bytes": leaked_bytes,
#         "total_allocations": total_allocations_made,
#         "total_deallocations": total_deallocations_made,
#         "peak_heap_size": df["heap_size"].max() if not df.empty else 0,
#         "peak_num_allocations": df["num_allocations"].max() if not df.empty else 0,
#         "final_heap_size": current_total_size,
#         "final_num_allocations": current_allocations,
#     }
#
#     return df
