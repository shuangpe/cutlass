#!/usr/bin/env bash

scope_array=(0.5 2 5 6)
int_scale_array=(-1 0 1)
exclude_zero_array=(-1 0 1)
init_providers=(device host)

# Calculate total iterations
total_iterations=$((${#scope_array[@]} * ${#int_scale_array[@]} * ${#exclude_zero_array[@]} * ${#init_providers[@]}))
current_iteration=0

# Record start time
start_time=$(date +%s)

# Progress bar function
function show_progress {
    local current=$1
    local total=$2
    local percent=$((current * 100 / total))
    local elapsed_time=$(($(date +%s) - start_time))

    # Calculate estimated remaining time
    local estimated_remaining=0
    if [ $current -gt 0 ]; then
        estimated_remaining=$(( elapsed_time * (total - current) / current ))
    fi

    # Convert time format
    local elapsed_formatted=$(printf "%02d:%02d:%02d" $((elapsed_time/3600)) $((elapsed_time%3600/60)) $((elapsed_time%60)))
    local remaining_formatted=$(printf "%02d:%02d:%02d" $((estimated_remaining/3600)) $((estimated_remaining%3600/60)) $((estimated_remaining%60)))

    # Create progress bar
    local bar_length=30
    local filled_length=$((bar_length * current / total))
    local bar=""
    for ((i=0; i<filled_length; i++)); do
        bar="${bar}#"
    done
    for ((i=filled_length; i<bar_length; i++)); do
        bar="${bar}-"
    done

    # Output progress information
    printf "\r[%s] %3d%% | %d/%d | Elapsed time: %s | Estimated remaining: %s" "$bar" "$percent" "$current" "$total" "$elapsed_formatted" "$remaining_formatted"
}

echo "Starting tests, total iterations: $total_iterations"

for scope in "${scope_array[@]}"; do
    for int_scale in "${int_scale_array[@]}"; do
        for exclude_zero in "${exclude_zero_array[@]}"; do
            for init_provider in "${init_providers[@]}"; do
                ((current_iteration++))

                # Display progress bar
                show_progress $current_iteration $total_iterations

                echo -e "\nscope: $scope, int_scale: $int_scale, exclude_zero: $exclude_zero, init_provider: $init_provider"
                tags="ScopeMin:-${scope},ScopeMax:${scope},IntScale:${int_scale},ExcludeZero:${exclude_zero},InitProvider:${init_provider}"
                ./tools/profiler/cutlass_profiler \
                    --operation=block_scaled_gemm --providers=cutlass \
                    --kernels=ue4m3xe2m1_ue4m3xe2m1_f32_void_ue4m3xe2m1 \
                    --m=16384 --n=16384 --k=16384 \
                    --sleep-duration=3000 \
                    --warmup-iterations=500 \
                    --profiling-iterations=2000 \
                    --tags=$tags \
                    --initialization-provider=$init_provider \
                    --dist=uniform,min:-$scope,max:$scope,scale:$int_scale,exclude_zero:$exclude_zero \
                    --append=true --output=cutlass_profile_nvfp4_dist_verify_oobMhz.csv
                sleep 5
            done
        done
    done
done

# Display final progress and add a new line
show_progress $total_iterations $total_iterations
total_time=$(($(date +%s) - start_time))
echo -e "\n\nCompleted! Total time: $(printf "%02d:%02d:%02d" $((total_time/3600)) $((total_time%3600/60)) $((total_time%60)))"
