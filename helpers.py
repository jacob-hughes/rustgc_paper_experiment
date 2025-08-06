import hashlib
import inspect
import pickle
from collections.abc import Collection
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from build import GCVS, BenchmarkSuite, Elision, HeapSize, PremOpt

CONFIDENCE_LEVEL = 0.99
BOOTSTRAP_SAMPLES = 100


ALL_CFGS = [GCVS, Elision, PremOpt, HeapSize]
BASELINES = [
    PremOpt.NONE,
    Elision.NAIVE,
    GCVS.RC,
    GCVS.ARC,
    HeapSize.DEFAULT,
]

EXCLUDE = BASELINES + [GCVS.BASELINE, GCVS.TYPED_ARENA]

BMS = {
    # alacritty
    "cursor_motion": "Cur. Motion",
    "dense_cells": "Dense Cells",
    "light_cells": "Light Cells",
    "scrolling": "Scroll",
    "scrolling_bottom_region": "Scroll Btm",
    "scrolling_bottom_small_region": "Scroll Btm (small)",
    "scrolling_fullscreen": "Scroll (fullscreen)",
    "scrolling_top_region": "Scroll Top",
    "scrolling_top_small_region": "Scroll Top (small)",
    "unicode": "Unicode",
    # ripgrep
    "linux_alternates": "Alternates",
    "linux_alternates_casei": "Alternates (-i)",
    "linux_literal": "Literal",
    "linux_literal_casei": "Literal (-i)",
    "linux_literal_default": "Literal (default)",
    "linux_literal_mmap": "Literal (mmap)",
    "linux_literal_casei_mmap": "Literal (mmap, -i)",
    "linux_word": "Word",
    "linux_unicode_greek": "UTF Greek",
    "linux_unicode_greek_casei": "UTF Greek (-i)",
    "linux_unicode_word_1": "UTF Word",
    "linux_unicode_word_2": "UTF Word (alt.)",
    "linux_re_literal_suffix": "Literal (regex)",
    # fd
    "command-execution": "Cmd Exec.",
    "command-execution-large-output": "Cmd Exec. (large)",
    "file-extension": "File Extension",
    "file-type": "File Type",
    "no-pattern": "No Pattern",
    "simple-pattern": "Simple",
    "simple-pattern-HI": "Simple (-HI)",
    # grmtools
    "eclipse": "Eclipse",
    "hadoop": "Hadoop",
    "jenkins": "Jenkins",
    "spring": "Spring",
    # som
    "Bounce": "Bounce",
    "BubbleSort": "BubbleSort",
    "DeltaBlue": "DeltaBlue",
    "Dispatch": "Dispatch",
    "Fannkuch": "Fannkuch",
    "Fibonacci": "Fibonacci",
    "FieldLoop": "FieldLoop",
    "GraphSearch": "GraphSearch",
    "IntegerLoop": "IntegerLoop",
    "JsonSmall": "JsonSmall",
    "List": "List",
    "Loop": "Loop",
    "Mandelbrot": "Mandelbrot",
    "NBody": "NBody",
    "PageRank": "PageRank",
    "Permute": "Permute",
    "Queens": "Queens",
    "QuickSort": "QuickSort",
    "Recurse": "Recurse",
    "Richards": "Richards",
    "Sieve": "Sieve",
    "Storage": "Storage",
    "Sum": "Sum",
    "Towers": "Towers",
    "TreeSort": "TreeSort",
    "WhileLoop": "WhileLoop",
}

SUITE_MAP = {b.name: b for b in BenchmarkSuite.all()}

BENCHMARK_SUITES = [
    "binary_trees",
    "regex_redux",
    "som-rs-ast",
    "som-rs-bc",
    "ripgrep",
    "fd",
    "alacritty",
    "grmtools",
]

SUITES = {
    "binary_trees": "Binary Trees",
    "regex_redux": r"\texttt{regex-redux}",
    "som-rs-ast": r"\texttt{som-rs-ast}",
    "som-rs-bc": r"\texttt{som-rs-bc}",
    "ripgrep": r"\texttt{ripgrep}",
    "fd": r"\texttt{fd}",
    "alacritty": "Alacritty",
    "grmtools": r"\texttt{grmtools}",
}

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def hash_object(obj):
    if isinstance(obj, Collection) and not isinstance(
        obj, (str, bytes, pd.Series, pd.DataFrame, np.ndarray, dict)
    ):
        if len(obj) > 0 and all(isinstance(x, (str, Path)) for x in obj):
            per_path_hashes = []
            for x in obj:
                p = Path(x)
                if p.exists():
                    per_path_hashes.append(f"{p.resolve()}:{p.stat().st_mtime}")
                else:
                    per_path_hashes.append(str(p))
            per_path_hashes.sort()
            combined = "|".join(per_path_hashes)
            return hashlib.md5(combined.encode()).hexdigest()

    if (isinstance(obj, (Path, str))) and Path(obj).exists():
        p = Path(obj)
        return f"{p.resolve()}:{p.stat().st_mtime}"
    if isinstance(obj, pd.DataFrame):
        return hashlib.md5(
            pd.util.hash_pandas_object(obj, index=True).values
        ).hexdigest()
    if isinstance(obj, pd.Series):
        return hashlib.md5(
            pd.util.hash_pandas_object(obj, index=True).values
        ).hexdigest()
    if isinstance(obj, np.ndarray):
        return hashlib.md5(obj.tobytes()).hexdigest()
    try:
        return hashlib.md5(pickle.dumps(obj)).hexdigest()
    except Exception:
        return hashlib.md5(str(obj).encode()).hexdigest()


class Cache:
    @staticmethod
    def make_key(static_deps, args, kwargs, func=None):
        key_parts = [hash_object(x) for x in static_deps]
        key_parts += [hash_object(x) for x in args]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={hash_object(v)}")
        if func is not None:
            try:
                source = inspect.getsource(func)
                key_parts.append(hashlib.md5(source.encode()).hexdigest())
            except Exception:
                pass
        key_str = "|".join(map(str, key_parts))
        return hashlib.md5(key_str.encode()).hexdigest()

    def __call__(self, *static_deps):
        def decorator(func):
            def wrapper(*args, **kwargs):
                key = self.make_key(static_deps, args, kwargs, func=func)
                file = CACHE_DIR / (key + ".pkl")
                if file.exists():
                    # print(f"[Cache hit] {func.__name__} → {file.name}")
                    with open(file, "rb") as f:
                        return pickle.load(f)
                # print(f"[Cache miss] {func.__name__} → {file.name}")
                result = func(*args, **kwargs)
                with open(file, "wb") as f:
                    pickle.dump(result, f)
                return result

            return wrapper

        return decorator


cache = Cache()


def to_cfg(name):
    s = name.partition("-")[2]
    if s == "baseline":
        return GCVS.BASELINE
    for c in ALL_CFGS:
        try:
            return c(s)
        except ValueError:
            continue
    raise ValueError(f"Configuration for {name} not found.")


def arithmetic_mean(series):
    arr = np.array(series)
    if len(arr) == 0:
        return pd.Series({"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan})
    if len(arr) == 1:
        val = arr[0]
        return pd.Series({"value": val, "ci_lower": val, "ci_upper": val})
    mean = arr.mean()
    sem = stats.sem(arr)
    alpha = 1 - CONFIDENCE_LEVEL
    h = sem * stats.t.ppf(1 - alpha / 2, len(arr) - 1)
    return pd.Series({"value": mean, "ci_lower": mean - h, "ci_upper": mean + h})


def geometric_mean(series):
    arr = np.array(series[series > 0])
    if len(arr) == 0:
        return pd.Series({"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan})
    if len(arr) == 1:
        val = arr[0]
        return pd.Series({"value": val, "ci_lower": val, "ci_upper": val})
    res = stats.bootstrap(
        (arr,),
        statistic=lambda y: stats.gmean(y, axis=0),
        n_resamples=BOOTSTRAP_SAMPLES,
        confidence_level=CONFIDENCE_LEVEL,
        method="percentile",
        axis=0,
    )
    return pd.Series(
        {
            "value": stats.gmean(arr),
            "ci_lower": res.confidence_interval.low,
            "ci_upper": res.confidence_interval.high,
        }
    )


def parallelise(fn, args, j=cpu_count(), desc="Processing"):
    results = []

    with ProcessPoolExecutor(max_workers=j) as executor:
        futures = {executor.submit(fn, a): a for a in args}

        with tqdm(
            desc=f"[{j}/{cpu_count()} threads]: {desc}",
            total=len(futures),
        ) as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing job: {e}")
                pbar.update(1)

    return results


EXPERIMENTS = {
    "gcvs": GCVS,
    "elision": Elision,
    "premopt": PremOpt,
}


def metadata(file):
    path = Path(file)
    benchmark, invocation = path.stem.rsplit("-", 1)
    suite = path.parts[-4]
    configuration = path.parent.name
    return {
        "invocation": int(invocation),
        "benchmark": benchmark,
        "suite": suite,
        "configuration": configuration,
    }


def fmt_ms_to_s(value):
    seconds = value / 1000.0
    return f"{seconds:.2f}"


def fmt_big_num(value):
    return f"{round(value):,}"


def fmt_bytes_to_mb(value):
    return f"{(value / 1024 / 1024):.2f}"
