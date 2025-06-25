#!/usr/bin/env bash

scope_array=(0.5 2 5 6)
int_scale_array=(-1 0 1)
exclude_zero_array=(-1 0 1)

for scope in "${scope_array[@]}"; do
    for int_scale in "${int_scale_array[@]}"; do
        for exclude_zero in "${exclude_zero_array[@]}"; do
            echo "scope: $scope, int_scale: $int_scale, exclude_zero: $exclude_zero"
            tags="ScopeMin:-${scope},ScopeMax:${scope},IntScale:${int_scale},ExcludeZero:${exclude_zero}"
            ./tools/profiler/cutlass_profiler \
                --operation=block_scaled_gemm --providers=cutlass \
                --kernels=ue4m3xe2m1_ue4m3xe2m1_f32_void_ue4m3xe2m1 \
                --m=16384 --n=16384 --k=16384 \
                --sleep-duration=3000 \
                --warmup-iterations=500 \
                --profiling-iterations=2000 \
                --tags=$tags \
                --dist=uniform,min:-$scope,max:$scope,scale:$int_scale,exclude_zero:$exclude_zero \
                --append=true --output=cutlass_profile_nvfp4_dist_verify_oobMhz.csv
            sleep 5
        done
    done
done
