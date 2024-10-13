#! /usr/bin/env python

import gc, math, random, os, sys
from os import listdir, stat
from statistics import geometric_mean, stdev
import numpy as np
import pandas as pd
import pprint
import csv
from scipy.stats import t

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
import numpy as np

results = {}
pp = pprint.PrettyPrinter(indent=4)

PEXECS = int(os.environ['PEXECS'])
ITERS = int(os.environ['ITERS'])

def mean(l):
    return math.fsum(l) / float(len(l))

def confidence_interval(l):
    Z = 2.576  # 99% interval
    return Z * (stdev(l) / math.sqrt(len(l)))


def flatten(l):
  return [y for x in l for y in x]


class PExec:
    def __init__(self, num, cfg, benchmark, iters):
        self.num = num
        self.cfg = cfg
        self.benchmark = benchmark
        self.iters = iters

    def __repr__(self):
        return f"Pexec('{self.num}', '{self.cfg}', '{self.benchmark}', '{self.iters}')"

class Experiment:
    def __init__(self, name, pexecs):
        self.name = name
        self.pexecs = pexecs

    def latex_name(self):
        return f"\\{self.name}".replace("_","")

    def geomean(self, cfg, benchmark='all'):
        return geometric_mean(self.iters(cfg, benchmark))

    def ci_geomean(self, cfg):
        l = self.iters(cfg, 'all')
        log = np.log(l)
        n = len(log)
        mean_log = np.mean(log)
        std_log = np.std(log, ddof=1)
        confidence = 0.99
        degrees_of_freedom = n - 1
        t_value = np.abs(t.ppf((1 - confidence) / 2, degrees_of_freedom))

        margin_of_error = t_value * (std_log / np.sqrt(n))
        ci_lower = np.exp(mean_log - margin_of_error)
        ci_upper = np.exp(mean_log + margin_of_error)
        return (ci_lower, ci_upper)

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

    def diff(self, cfg, baseline, benchmark = 'all', geomean = False):
        if geomean:
            return self.geomean(baseline, benchmark) - self.geomean(cfg, benchmark)
        return self.mean(baseline, benchmark) - self.mean(cfg, benchmark)

    def speedup(self, cfg, baseline, benchmark = 'all', geomean = False):
        if geomean:
            return self.geomean(baseline, benchmark) / self.geomean(cfg, benchmark)
        return self.mean(baseline, benchmark) / self.mean(cfg, benchmark)

    def percent(self, cfg, baseline, benchmark = 'all', geomean = False):
        a = self.geomean(cfg, benchmark)
        b = self.geomean(baseline, benchmark)
        return (abs(a - b) / b) * 100

    def dump_stats(self):
        mapper = {
            "perf_gc": "gc",
            "perf_rc": "rc",
            "finalise_naive": "fnaive",
            "finalise_elide": "felide",
            "barriers_none": "bnone",
            "barriers_naive": "bnaive",
            "barriers_opt": "bopt",
            "all" : "all"
        }
        baseline = {
            "perf": "perf_rc",
            "elision": "finalise_naive",
            "barriers": "barriers_none",
        }
        stats = dict((mapper[cfg], {}) for cfg in self.cfgs())
        for cfg in self.cfgs():
            for benchmark in self.benchmarks():
                stats[mapper[cfg]][benchmark.lower()] = {
                    'mean' : f"{self.mean(cfg, benchmark):.2f}",
                    'diff' : f"{self.diff(cfg, baseline[self.name], benchmark):.2f}",
                    'speedup' : f"{self.speedup(cfg, baseline[self.name], benchmark):.2f}",
                }
            # Use geomean for all benchmarks
            stats[mapper[cfg]]['all'] = {
                'geomean' : f"{self.geomean(cfg, benchmark):.2f}",
                'diff' : f"{self.diff(cfg, baseline[self.name], benchmark, geomean = True):.2f}",
                'speedup' : f"{self.speedup(cfg, baseline[self.name], benchmark, geomean = True):.2f}",
                'percent' : f"{self.percent(cfg, baseline[self.name], benchmark, geomean = True):.2f}\\%",
            }
            print(f"{cfg}: {stats[mapper[cfg]]['all']['percent']}")

        return stats

def load_exp(exp_name):
    exp_dir = os.path.join(os.getcwd(), 'results', exp_name)
    rbdata = os.path.join(exp_dir, "results.data")
    pexecs = {}
    stats =  {}
    with open(rbdata) as f:
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

            if cfg not in stats:
                stats[cfg] = {}
            if benchmark not in stats[cfg]:
                stats[cfg][benchmark] = {
                    'finalisers_registered': [],
                    'finalisers_run': [],
                    'gc_allocations': [],
                    'rust_allocations': [],
                    'gc_cycles': [],
                }

            if (invocation, benchmark, cfg) not in pexecs:
                pexecs[(invocation, benchmark, cfg)] = PExec(invocation, cfg, benchmark, [time])
            else:
                pexecs[(invocation, benchmark, cfg)].iters.append(time)

    # print(pexecs)
    experiment = Experiment(exp_name, list(pexecs.values()))

    for cfg in stats.keys():
        p = os.path.join(exp_dir, f"{cfg}.log")
        if os.path.exists(p):
            with open(p, "r") as f:
                for l in f.readlines():
                    s = [x.strip() for x in l.split(',')]
                    bm = s[0]
                    stats[cfg][bm]['finalisers_registered'].append(int(s[1]))
                    stats[cfg][bm]['finalisers_run'].append(int(s[2]))
                    stats[cfg][bm]['gc_allocations'].append(int(s[3]))
                    stats[cfg][bm]['rust_allocations'].append(int(s[4]))
                    stats[cfg][bm]['gc_cycles'].append(int(s[5]))
    experiment.gc_stats = stats
    return experiment


def plot_bar(exp):
    means = [[mean(exp.iters(cfg, benchmark)) for benchmark in exp.benchmarks()] for cfg in exp.cfgs()]
    cis = [[confidence_interval(exp.iters(cfg, benchmark)) for benchmark in exp.benchmarks()] for cfg in exp.cfgs()]

    sns.set(style="whitegrid")
    # plt.rc('text', usetex=False)
    # plt.rc('font', family='sans-serif')
    fig, ax = plt.subplots(figsize=(4, 4))
    df = pd.DataFrame(zip(*means), index=exp.benchmarks())
    errs = pd.DataFrame(zip(*cis), index=exp.benchmarks())
    plot = df.plot(kind='bar', width=0.8, ax=ax, yerr=errs)
    plot.margins(x=0.01)
    ax.legend(exp.cfgs()).remove()

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
    plt.savefig(f"plots/{exp.name}.svg", format="svg", bbox_inches="tight")
    print("Graph saved to '%s'" % f"{exp.name}.svg")

def make_table(exp):
    bms = exp.benchmarks()
    with open(f"plots/{exp.name}.tex", "w") as f:
            # f.write(f" & {bm}")
        for (cfg, values) in exp.gc_stats.items():
            f.write(f"{cfg} &")
            for bm in exp.benchmarks():
                f.write(f" {mean(values[bm]['finalisers_run'])} &\n")


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
    f.write(f"\\newcommand{{{exp.latex_name()}}}[{depth(stats)}]{{%\n")
    make_args(stats, 1)
    f.write(f"}}%\n")

summary = False
args = sys.argv[1:]
if sys.argv[1] == "summary":
    summary = True
    args = sys.argv[2:]

for arg in args:
    print(f"Called on {arg}")
    e = load_exp(arg)
    # make_table(e)
    # plot_bar(e)
    if summary:
        overview = {
            'perf_rc',
            'perf_gc',
        }

        with open("summary.csv", "a") as f:
            for cfg in e.cfgs():
                if cfg in overview:
                    (ci_l, ci_u) = e.ci_geomean(cfg)
                    f.write(f"{e.name},{cfg}, {e.geomean(cfg)}, {ci_l}, {ci_u}\n")
                    continue
                ci = confidence_interval(e.iters(cfg, 'all'))
                f.write(f"{e.name},{cfg}, {e.mean(cfg)}, {ci}, {ci}\n")
    #
    # with open("plots/experiment_stats.tex", "a") as f:
    #         write_stats(f, e)
