#! /bin/sh

set -e

mkdir -p raw_data

usage() {
  echo "usage"
}

run_perf() {
  # Run som-rs perf
  cd som-rs
  git checkout -- .
  git apply ../configs/som-rs/tune_benchmarks.patch
  git apply ../configs/som-rs/bdwgc_allocator.patch
  git apply ../configs/som-rs/rebench_perf.patch
  git apply ../configs/som-rs/rebench_iters.patch
  rebench -R --experiment all rebench.conf all || true
  mv rebench.data ../raw_data/som-rs-perf.data
  rm boehm_rc.sh
  cd ..
}

run_finalisers() {
  # Run som-rs elision
  cd som-rs
  git checkout -- .
  git apply ../configs/som-rs/tune_benchmarks.patch
  git apply ../configs/som-rs/rebench_finaliser_elision.patch
  mkdir -p naive_counts
  mkdir -p elision_counts
  rebench -R --experiment all rebench.conf all || true
  mv rebench.data ../raw_data/som-rs-finaliser_elision.data
  mv naive_counts ../raw_data/
  mv elision_counts ../raw_data/
  cd ..
}

run_barriers() {
  cd som-rs
  git checkout -- .
  git apply ../configs/som-rs/tune_benchmarks.patch
  git apply ../configs/som-rs/rebench_barriers.patch
  rebench -R --experiment all rebench.conf all || true
  mv rebench.data ../raw_data/som-rs-barriers.data
  cd ..
}

run_memory() {
  cd som-rs
  git checkout -- .
  git apply ../configs/som-rs/tune_benchmarks.patch
  git apply ../configs/som-rs/rebench_iters_mem.patch

}

if [ $# -ne 1 ]; then
  run_all
elif [ "$@" = "perf" ]; then
  run_perf
elif [ "$@" = "elision" ]; then
  run_finalisers
elif [ "$@" = "barriers" ]; then
  run_barriers
elif [ "$@" = "memory" ]; then
  run_memory
else
  usage
fi


