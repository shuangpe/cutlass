#!/bin/bash

root_dir="/dataset/shuangpeng/project/cutlass/cutlass/build/"

# 解析参数
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --mode|-m) mode="$2"; shift ;;
    --scope|-s) scope="$2"; shift ;;
    --kernel|-k) kernel="$2"; shift ;;
    --operation|-op) operation="$2"; shift ;;
    --output|-o) output_path="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

mode=${mode:-0}
scope=${scope:-5}
kernel=${kernel:-*}
output_path=${output_path:-output}

if [[ "$mode" -eq 0 ]]; then
  profiler_app="before_hacking/tools/profiler/cutlass_profiler"
else
  profiler_app="after_hacking/tools/profiler/cutlass_profiler"
fi

${root_dir}/$profiler_app \
  --operation=${operation} --kernels=${kernel} --m=16384 --n=16384 --k=16384 --providers=cutlass \
  --sleep-duration=3000 --warmup-iterations=500 --profiling-iterations=2000 \
  --print-kernel-before-running=true --verification-enabled=false \
  --dist=uniform,min:-${scope},max:${scope} \
  --output="${output_path}.csv" 2>&1 | tee -a "${output_path}.log.txt"