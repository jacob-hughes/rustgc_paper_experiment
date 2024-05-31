#! /usr/bin/env python

import gc, math, random, os, sys
from os import listdir, stat
from statistics import geometric_mean, stdev
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'cm'
matplotlib.rcParams.update({'errorbar.capsize': 2})
from matplotlib.ticker import ScalarFormatter
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.pyplot as plt

def mean(l):
    return math.fsum(l) / float(len(l))

def confidence_interval(l):
    Z = 2.576  # 99% interval
    return Z * (stdev(l) / math.sqrt(len(l)))

def process_graph(name, p, rc, gc, rc_boehm):
    results = {}
    with open(p) as f:
        for l in f.readlines():
            if l.startswith("#"):
                continue
            l = l.strip()
            if len(l) == 0:
                continue
            s = [x.strip() for x in l.split()]

            if s[4] != "total":
                continue

            bm = s[5]
            if bm not in results:
                results[bm] = {}

            cfg = s[6]

            if cfg not in results[bm]:
                results[bm][cfg] = []

            results[bm][cfg].append(float(s[2]))

    benchmarks = []
    rc_means = []
    gc_means = []
    rc_boehm_means = []

    rc_cis = []
    gc_cis = []
    rc_boehm_cis = []

    for bm, runs in dict(sorted(results.items())).items():
        # if rc not in runs:
        #     print("No results for ", bm)
        #     continue
        benchmarks.append(bm)
        rc_runs = []
        gc_runs = []
        rc_boehm_runs = []
        for r, g, b in zip(runs[rc], runs[gc], runs[rc_boehm]):
            rc_runs.append(r)
            gc_runs.append(g)
            rc_boehm_runs.append(b)

        print(rc_runs)

        rc_means.append(mean(rc_runs))
        rc_cis.append(confidence_interval(rc_runs))
        gc_means.append(mean(gc_runs))
        gc_cis.append(confidence_interval(gc_runs))
        rc_boehm_means.append(mean(rc_boehm_runs))
        rc_cis.append(confidence_interval(rc_boehm_runs))

    sns.set(style="whitegrid")
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    df = pd.DataFrame(zip(rc_means, rc_boehm_means, gc_means), index=benchmarks)
    errs = pd.DataFrame(zip(rc_cis, rc_boehm_cis, gc_cis), index=benchmarks)
    plot = df.plot(kind='bar', width=0.8, ax=ax, yerr=errs)
    plot.margins(x=0.01)
    ax.legend(['RC', 'RC(boehm)' 'GC'])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Benchmark')
    ax.set_ylabel('Wall-clock time (ms)\n(lower is better)')
    ax.grid(linewidth=0.25)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_tick_params(which='minor', size=0)
    ax.yaxis.set_tick_params(which='minor', width=0)
    plt.xticks(range(0, len(benchmarks)), benchmarks, rotation = "vertical")
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.savefig(name, format="svg", bbox_inches="tight")

process_graph("som_rs_perf.svg", "raw_data/som-rs-perf.data", 'som-rs-rc', 'som-rs-gc', 'som-rs-rc-boehm')
