#! /usr/bin/env python

import gc, math, random, os, sys
from os import listdir, stat
from statistics import geometric_mean, stdev
import numpy as np
import pandas as pd
import pprint

import matplotlib
matplotlib.use('Agg')
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['font.family'] = 'sans-serif'
# matplotlib.rcParams['font.sans-serif'] = 'cm'
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': False,
    'pgf.rcfonts': False,
})

# matplotlib.rcParams.update({'errorbar.capsize': 2})
from matplotlib.ticker import ScalarFormatter
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.pyplot as plt

results = {}
pp = pprint.PrettyPrinter(indent=4)
class PExec:
    def __init__(self, num, cfg, benchmark, iters):
        self.num = num
        self.cfg = cfg
        self.benchmark = benchmark
        self.iters = iters

    def __repr__(self):
        return f"Pexec('{self.num}', '{self.cfg}', '{self.benchmark}', '{self.iters}')"

class Experiment:
    def __init__(self, vm, name, pexecs):
        self.vm = vm
        self.name = name
        self.pexecs = pexecs

    def geomean(self, cfg, benchmark='all'):
        return geometric_mean(self.iters(cfg, benchmark))

    def mean(self, cfg, benchmark='all'):
        l = self.iters(cfg, benchmark)
        return math.fsum(l) / float(len(l))

    def speedup(self, normalised_to, benchmark):
        pass

    def num_pexecs(self):
        return max([pexec.num for pexec in self.pexecs])

    def num_iters(self):
        return len(self.pexecs[0].iters)

    def cfgs(self):
        return sorted(list({p.cfg for p in self.pexecs}))

    def iters(self, cfg, benchmark):
        if benchmark == 'all':
            return flatten([p.iters for p in self.pexecs if p.cfg == cfg])
        else:
            return flatten([p.iters for p in self.pexecs if (p.cfg == cfg and p.benchmark == benchmark)])

    def benchmarks(self):
        return sorted(list({p.benchmark for p in self.pexecs}))

    def diff(self, cfg, baseline, benchmark = 'all'):
        return self.mean(baseline, benchmark) - self.mean(cfg, benchmark)

    def speedup(self, cfg, baseline, benchmark = 'all'):
        return self.mean(baseline, benchmark) / self.mean(cfg, benchmark)

    def dump_stats(self):
        mapper = {
            "perf_gc": "gc",
            "perf_rc": "rc",
            "all" : "all"
        }
        stats = dict((mapper[cfg], {}) for cfg in self.cfgs())
        baseline = 'perf_rc'
        for cfg in self.cfgs():
            for benchmark in self.benchmarks() + ['all']:
                stats[mapper[cfg]][benchmark.lower()] = {
                    'mean' : f"{self.mean(cfg, benchmark):.2f}",
                    'diff' : f"{self.diff(cfg, baseline, benchmark):.2f}",
                    'speedup' : f"{self.speedup(cfg, baseline, benchmark):.2f}",
                }

        return stats

def load_exp(vm, exp):
    datafile = os.path.join(os.environ['RESULTS_DIR'], vm, exp, os.environ['REBENCH_DATA'])
    pexecs = {}
    with open(datafile) as f:
        for l in f.readlines():
            if l.startswith("#"):
                continue
            l = l.strip()
            if len(l) == 0:
                continue
            s = [x.strip() for x in l.split()]

            if s[4] != "total":
                continue

            # A PExec can be uniquely identified by a tuple of (invocation, benchmark, cfg)
            invocation = int(s[0])
            benchmark = s[5]
            cfg = s[6]
            iter = s[1]
            time = float(s[2])

            if (invocation, benchmark, cfg) not in pexecs:
                pexecs[(invocation, benchmark, cfg)] = PExec(invocation, cfg, benchmark, [time])
            else:
                pexecs[(invocation, benchmark, cfg)].iters.append(time)

    return Experiment(vm, exp, list(pexecs.values()))

def mean(l):
    return math.fsum(l) / float(len(l))

def confidence_interval(l):
    Z = 2.576  # 99% interval
    return Z * (stdev(l) / math.sqrt(len(l)))

def flatten(l):
  return [y for x in l for y in x]

def plot_bar(exp):
    means = [[mean(exp.iters(cfg, benchmark)) for benchmark in exp.benchmarks()] for cfg in exp.cfgs()]
    cis = [[confidence_interval(exp.iters(cfg, benchmark)) for benchmark in exp.benchmarks()] for cfg in exp.cfgs()]

    sns.set(style="whitegrid")
    # plt.rc('text', usetex=False)
    # plt.rc('font', family='sans-serif')
    fig, ax = plt.subplots(figsize=(8, 4))
    df = pd.DataFrame(zip(*means), index=exp.benchmarks())
    print(exp.benchmarks())
    plot = df.plot(kind='bar', width=0.8, ax=ax)
    plot.margins(x=0.01)
    ax.legend(exp.cfgs())

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('Wall-clock time (ms)\n(lower is better)')
    ax.grid(linewidth=0.25)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_tick_params(which='minor', size=0)
    ax.yaxis.set_tick_params(which='minor', width=0)
    plt.xticks(range(0, len(exp.benchmarks())), exp.benchmarks(), rotation = 45, ha="right")
    # ax.set_yticks(range, len(exp.benchmarks()))
    # ax.set_ytickslabels(exp.benchmarks())
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.savefig(f"{exp.vm}_{exp.name}.svg", format="svg", bbox_inches="tight")
    print("Graph saved to '%s'" % f"{exp.vm}_{exp.name}.svg")

def write_stats(f, exp):
    def depth(d):
        depth = 0
        while(1):
            if not isinstance(d, dict):
                break
            d = d[next(iter(d))]
            depth += 1
        return depth

    def make_args(d, arg):
        if not isinstance(d, dict):
            f.write(f"{d}")
            return
        for k, v in d.items():
            f.write(f"\\ifthenelse{{\equal{{#{arg}}}{{{k}}}}}{{%\n")
            make_args(v, arg + 1)
            f.write(f"}}%\n")
            f.write(f"{{")
        f.write(f"\\error{{Invalid argument}}%\n")

        for k, v in d.items():
            f.write(f"}}")
    stats = exp.dump_stats()
    f.write(f"\\newcommand{{{exp.latex_name}}}[{depth(stats)}]{{%\n")
    make_args(stats, 1)
    f.write(f"}}")

    # somrsperfdata
    # cfg: bench: mean, geomean, diff, speedup, finalised, objects, peakmem, avgmem
    # cfg: all: mean, geomean, diff, speedup, finalised, objects, peakmem, avgmem

    # \pexecs{somrsperf}
    # \iters{somrsperf}
    # \somrsperf{gc}{all}{geomean}

results_dir = os.environ['RESULTS_DIR']
experiments = []

e = load_exp('som-rs', 'perf')
e.latex_name = r'\somrsperf'

e.baseline = 'perf_rc'
experiments.append(e)

for e in experiments:
    plot_bar(e)

with open("experiment_stats.tex", "w") as f:
    for e in experiments:
        write_stats(f, e)
