#!/bin/bash

root_dir="/dataset/shuangpeng/project/cutlass/cutlass/baseline_build/"
profiler_app="tools/profiler/cutlass_profiler"

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --mode|-m) mode="$2"; shift ;;
    --scope|-s) scope="$2"; shift ;;
    --kernel|-k) kernel="$2"; shift ;;
    --mask_ratio|-mr) mask_ratio="$2"; shift ;;
    --operation|-op) operation="$2"; shift ;;
    --output|-o) output_path="$2"; shift ;;
    --tags|-t) tags="$2"; shift ;;
    --warmup-iterations|-wi) warmup_iterations="$2"; shift ;;
    --profiling-iterations|-pi) profiling_iterations="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

mode=${mode:-0}
scope=${scope:-5}
kernel=${kernel:-*}
output_path=${output_path:-output}
mask_ratio=${mask_ratio:-0}
warmup_iterations=${warmup_iterations:-500}
profiling_iterations=${profiling_iterations:-2000}

optional_args=""
if [[ -n "$operation" ]]; then
  optional_args="${optional_args} --operation=$operation"
fi
if [[ -n "$tags" ]]; then
  optional_args="${optional_args} --tags=$tags"
fi

${root_dir}/$profiler_app \
  --kernels=${kernel} --m=16384 --n=16384 --k=16384 --providers=cutlass \
  --sleep-duration=3000 --warmup-iterations=${warmup_iterations} --profiling-iterations=${profiling_iterations} \
  --print-kernel-before-running=true --verification-enabled=false --initialization-provider=device --save-workspace=always \
  --dist=uniform,min:-${scope},max:${scope},scale:-1 --output="${output_path}.csv" ${optional_args} 2>&1 | tee -a "${output_path}.log.txt"