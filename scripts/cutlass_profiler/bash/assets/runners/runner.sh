#!/bin/bash

if [[ -z "${APP_DIR}" ]]; then
  echo "WARNING: APP_DIR environment variable is not set. Using current directory (\".\") as APP_DIR."
  APP_DIR="."
fi

if [[ -z "${EXTRA_DIST_ARGS}" ]]; then
  echo "WARNING: EXTRA_DIST_ARGS environment variable is not set."
fi

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --scope|-s) scope="$2"; shift ;;
    --kernel|-k) kernel="$2"; shift ;;
    --mask_ratio|-mr) mask_ratio="$2"; shift ;;
    --operation|-op) operation="$2"; shift ;;
    --output|-o) output_path="$2"; shift ;;
    --tags|-t) tags="$2"; shift ;;
    --warmup-iterations|-wi) warmup_iterations="$2"; shift ;;
    --profiling-iterations|-pi) profiling_iterations="$2"; shift ;;
    --dump_data) dump_data="$2"; shift ;;
    --dry) dry_run="true"; shift ;;  # Add dry run support
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

scope=${scope:-5}
kernel=${kernel:-*}
output_path=${output_path:-output}
mask_ratio=${mask_ratio:-0}
warmup_iterations=${warmup_iterations:-500}
profiling_iterations=${profiling_iterations:-2000}
dump_data=${dump_data:-false}
dry_run=${dry_run:-false}

optional_args=""
if [[ -n "$operation" ]]; then
  optional_args="${optional_args} --operation=$operation"
fi
if [[ -n "$tags" ]]; then
  optional_args="${optional_args} --tags=$tags"
fi
if [[ "$dump_data" == "true" ]]; then
  optional_args="${optional_args} --save-workspace=always"
fi

extra_dist_args=${EXTRA_DIST_ARGS:-""}
if [[ "$extra_dist_args" != *"mask_ratio"* ]]; then
  extra_dist_args="${extra_dist_args},mask_ratio:${mask_ratio}"
fi

problem_shape_args=${PROBLEM_SHAPE_ARGS:-""}
if [[ -z "${problem_shape_args}" ]]; then
  problem_shape_args="--m=16384 --n=16384 --k=16384"
fi

profiler_app="${APP_DIR}/tools/profiler/cutlass_profiler"

if [[ "$dry_run" != "true" ]]; then
  if [[ ! -f "$profiler_app" ]]; then
    echo "ERROR: Profiler not found at $profiler_app"
    exit 2
  fi
fi

cmd="$profiler_app --kernels=${kernel} ${problem_shape_args} --providers=cutlass \
  --sleep-duration=3000 --warmup-iterations=${warmup_iterations} --profiling-iterations=${profiling_iterations} \
  --print-kernel-before-running=true --verification-enabled=false --initialization-provider=device \
  --dist=uniform,min:-${scope},max:${scope},scale:-1${extra_dist_args} --output=\"${output_path}.csv\" ${optional_args} 2>&1 | tee -a \"${output_path}.log.txt\""

if [[ "$dry_run" == "true" ]]; then
  echo "[DRY RUN] $cmd"
else
  eval $cmd
fi
