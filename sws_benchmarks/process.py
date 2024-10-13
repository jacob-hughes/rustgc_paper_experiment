#! /usr/bin/env python

import gc, math, random, os, sys
from os import listdir, stat
from statistics import geometric_mean, stdev
import pandas as pd

def mean(l):
    return math.fsum(l) / float(len(l))

def confidence_interval(l):
    Z = 2.576  # 99% interval
    return Z * (stdev(l) / math.sqrt(len(l)))

gc = pd.read_csv('data/benchmark_gc.csv', header=None).values.tolist()
arc = pd.read_csv('data/benchmark_arc.csv', header=None).values.tolist()

with open('summary.csv', 'w') as f:
    gc_reqs = [rs[2] for rs in gc]
    arc_reqs = [rs[2] for rs in arc]
    gc_ci = confidence_interval(gc_reqs)
    arc_ci = confidence_interval(arc_reqs)
    f.write(f"\\sws,\\ourgc,{mean(gc_reqs):2f},{gc_ci:3f},{gc_ci:3f}\n")
    f.write(f"\\sws,\\arc,{mean(arc_reqs):2f},{arc_ci:3f},{arc_ci:3f}\n")
