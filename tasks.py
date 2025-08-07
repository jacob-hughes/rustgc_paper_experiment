import os
from pathlib import Path

from invoke import task

from build import (
    GCVS,
    HEAPTRACK,
    Aggregation,
    BenchmarkSuite,
    Elision,
    Experiments,
    HeapSize,
    Measurement,
    Metric,
    PremOpt,
)
from plots import ltx_ms_to_s
from results import Heaptrack, Metrics, Perf


def _parse_args(pexecs, exps=None, suites=None, measurements=None):
    def _to_list(val):
        return val.split() if isinstance(val, str) else val

    return Experiments.all(pexecs).filter(
        experiments=_to_list(exps),
        suites=_to_list(suites),
        measurements=_to_list(measurements),
    )


def _build_alloy(c, experiments, warn_if_empty=False):
    cfgs = experiments.alloy_variants(only_missing=True)
    if not cfgs:
        if warn_if_empty:
            print("Nothing to do")
        return

    print(
        f"{len(cfgs)} Alloy variant(s) require installing for {len(experiments.experiments)} experiment(s):"
    )
    [print(f"  {a.name}") for a in cfgs]
    for a in cfgs:
        a.build_alloy(c)


def list_exps(c):
    r = Experiments.all()
    es = r.experiments
    for e in es:
        for c in e.configurations():
            print(f"{c.name} --> {c.baseline}")


@task
def build_alloy(c, experiments=None, measurements=None):
    """Build all alloy configurations"""
    exps = _parse_args(pexecs=0, exps=experiments, measurements=measurements)
    _build_alloy(c, exps, warn_if_empty=True)


@task
def build_heaptrack(c):
    HEAPTRACK.build(c)


@task
def build_benchmarks(c, experiments=None, suites=None, measurements=None):
    """Build all benchmarks for all configurations"""
    exps = _parse_args(
        pexecs=0, exps=experiments, suites=suites, measurements=measurements
    )
    _build_alloy(c, exps)

    cfgs = exps.configurations()
    if not cfgs:
        print("Nothing to do")
        return
    print(f"Found {len(cfgs)} benchmark configuration(s):")
    [print(f"  {cfg.name}") for cfg in cfgs]
    for cfg in cfgs:
        cfg.build(c=c)


@task
def prerequisites(c):
    BenchmarkSuite.prerequisites(c)


@task
def run_benchmarks(c, pexecs, experiments=None, suites=None, measurements=None):
    exps = _parse_args(
        int(pexecs), exps=experiments, suites=suites, measurements=measurements
    )

    exps.run(c, pexecs)
    process_results(c, experiments, suites, measurements)


@task
def clean_alloy(c, experiments=None):
    exps = Experiments.new()

    if experiments:
        exps = exps.filter_experiments(experiments)

    to_remove = set(cfg.alloy for cfg in exps.configurations() if cfg.alloy.installed)

    if not to_remove:
        print("No Alloy configurations installed")
        return

    print(f"Found {len(to_remove)} Alloy configuration(s):")
    [print(f"  {os.path.relpath(a.bin)}") for a in to_remove]

    response = input(f"Are you sure you want to remove them? (y/n): ").strip().lower()
    if response == "y":
        for alloy in to_remove:
            alloy.remove(including_src=False)
        print(f"{len(alloy)} benchmark configurations removed.")
    else:
        print("cancelled.")


@task
def process_results(c, experiments=None, suites=None, measurements=None):
    exps = _parse_args(0, exps=experiments, suites=suites, measurements=measurements)

    Path("tables").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)

    htraw = Heaptrack.process(exps)
    htraw.plot_time_series()
    ht = htraw.aggregate(GCVS)
    ht.plot(Metric.MEM_HSIZE_AVG)

    gcvs = Perf.process(exps).aggregate(GCVS)
    gcvs.plot(Metric.WALLCLOCK)
    gcvs.tabulate(
        Metric.WALLCLOCK,
        Aggregation.INDIVIDUAL,
        format_func=ltx_ms_to_s,
        split=(["alacritty", "fd", "som-rs-ast"], ["grmtools", "ripgrep", "som-rs-bc"]),
    )
    gcvs.tabulate_binary_trees()
    gcvs.tabulate(
        Metric.USER,
        Aggregation.INDIVIDUAL,
        format_func=ltx_ms_to_s,
        split=(["alacritty", "fd", "som-rs-ast"], ["grmtools", "ripgrep", "som-rs-bc"]),
    )
    gcvs.tabulate(
        Metric.USER,
        Aggregation.SUITE_GEO,
        format_func=ltx_ms_to_s,
    )
    jemalloc = Perf.process(exps).aggregate(GCVS, baseline=GCVS.BASELINE)
    jemalloc.tabulate_allocators()

    pelide = Perf.process(exps).aggregate(Elision)
    pelide.plot(Metric.WALLCLOCK, xlim=(0, 1.5))
    pelide.plot(Metric.USER, xlim=(0, 1.5))
    pelide.tabulate(
        Metric.WALLCLOCK,
        Aggregation.INDIVIDUAL,
        format_func=ltx_ms_to_s,
        split=(["alacritty", "fd", "som-rs-ast"], ["grmtools", "ripgrep", "som-rs-bc"]),
    )
    pelide.tabulate(
        Metric.USER,
        Aggregation.INDIVIDUAL,
        format_func=ltx_ms_to_s,
        split=(["alacritty", "fd", "som-rs-ast"], ["grmtools", "ripgrep", "som-rs-bc"]),
    )

    pprem = Perf.process(exps).aggregate(PremOpt)
    pprem.plot(Metric.WALLCLOCK, xlim=(0.6, 1.4))

    melide = Metrics.process(exps).aggregate(Elision)
    melide.plot(Metric.MEM_HSIZE_AVG, xlim=(0, 1.5))
    melide.plot(Metric.TIME_TOTAL, xlim=(0, 1.5))
    melide.plot(Metric.TOTAL_COLLECTIONS, xlim=(0, 1.5))

    melide.tabulate(
        Metric.MEM_HSIZE_AVG,
        Aggregation.INDIVIDUAL,
        format_func=ltx_ms_to_s,
        split=(["alacritty", "fd", "som-rs-ast"], ["grmtools", "ripgrep", "som-rs-bc"]),
    )

    melide.tabulate(
        Metric.OBJ_ALLOCD_GC,
        Aggregation.INDIVIDUAL,
        format_func=ltx_ms_to_s,
        split=(["alacritty", "fd", "som-rs-ast"], ["grmtools", "ripgrep", "som-rs-bc"]),
    )

    fixed = Perf.process(exps).aggregate(HeapSize)
    fixed.tabulate_fixed_size()


@task
def clean_results(c, suites=None):
    exps = Experiments()
    if suites:
        exps = exps.filter_suites(suites)

    exps.clean()


@task
def clean_benchmarks(c, suites=None, including_src=False):
    """Clean all benchmarks for all configurations"""

    exps = Experiments.new()
    if suites:
        exps = exps.filter_suites(suites)

    cfgs = exps.configurations(only_installed=True)

    if not cfgs:
        print("No benchmark programs installed")
        return
    print(f"Found {len(cfgs)} configurations:")
    [print(f"  {os.path.relpath(cfg.bin)}") for cfg in cfgs]

    response = input(f"Are you sure you want to remove them? (y/n): ").strip().lower()
    if response == "y":
        for cfg in cfgs:
            cfg.remove(including_src)
        print(f"{len(cfgs)} benchmark configurations removed.")
    else:
        print("cancelled.")


@task
def clean(c):
    response = (
        input("Are you sure you want to remove everything? (y/n): ").strip().lower()
    )
    if response == "y":
        clean_alloy(c)
        clean_benchmarks(c)
    else:
        print("cancelled.")
