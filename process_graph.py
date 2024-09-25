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

def process_graph(input, output):
    print("Processing results from %s" % input)
    results = {}
    cfgs = set()
    with open(input) as f:
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
            cfgs.add(cfg)

            if cfg not in results[bm]:
                results[bm][cfg] = []

            results[bm][cfg].append(float(s[2]))

    benchmarks = []
    means = []
    cis = []
    labels = list(sorted(cfgs))
    for bm, cfgs in dict(sorted(results.items())).items():
        if len(cfgs.keys()) != len(labels):
              print("Not all cfgs have results for %s, skipping..." % bm)
              continue
        benchmarks.append(bm)

        means.append(tuple([mean(v) for (k,v) in sorted(cfgs.items())]))
        cis.append(tuple([confidence_interval(v) for (k,v) in sorted(cfgs.items())]))

    sns.set(style="whitegrid")
    plt.rc('text', usetex=False)
    plt.rc('font', family='sans-serif')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    df = pd.DataFrame(means, index=benchmarks)
    errs = pd.DataFrame(cis, index=benchmarks)
    plot = df.plot(kind='bar', width=0.8, ax=ax, yerr=errs)
    plot.margins(x=0.01)
    ax.legend(labels)

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
    plt.savefig(output, format="svg", bbox_inches="tight")
    print("Graph saved to '%s'" % output)


infile = sys.argv[1]
outfile = sys.argv[2]
process_graph(infile, outfile)
