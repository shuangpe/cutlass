#!/bin/bash
# Automation workload execution script

#===== Global Variables =====
SCRIPT_NAME=$(basename "$0")
LOG_PREFIX="[$(date '+%Y-%m-%d %H:%M:%S')]"
CONFIG_FILE="batch_profile.cfg"
OUTPUT_DIR=""
BENCHNAME="Automation Workload"

total_runs=0
current_run=0
total_scopes_maskratios=0

#===== Helper Functions =====
log_info() {
  local timestamp="[$(date '+%Y-%m-%d %H:%M:%S')]"
  echo "${timestamp} INFO: $1"
}

log_error() {
  local timestamp="[$(date '+%Y-%m-%d %H:%M:%S')]"
  echo "${timestamp} ERROR: $1" >&2
}

log_warning() {
  local timestamp="[$(date '+%Y-%m-%d %H:%M:%S')]"
  echo "${timestamp} WARNING: $1" >&2
}

# Check if running as root
check_root() {
  if [[ $EUID -ne 0 ]]; then
    log_error "This script must be run as root!"
    exit 1
  fi
}

# Load configuration file
load_config() {
  local config_file="$1"

  if [ -f "$config_file" ]; then
    log_info "Reading configuration file: $config_file"
    source "$config_file"
    return 0
  else
    log_error "Configuration file not found: $config_file"
    return 1
  fi
}

# Check all configuration parameters
check_all_configs() {
  if [ -z "$script_runner" ]; then
    log_error "Profiler script runner is not set. Please set 'script_runner' in the configuration file."
    exit 1
  fi

  freq=($frequency)
  mode=($profile_mode)

  log_info "Scan Configuration:"
  log_info "  Profiler Script: $script_runner"
  log_info "  ${#mode[@]} Models: ${mode[*]}"
  log_info "  ${#freq[@]} Frequencies: ${freq[*]}"
  log_info "  ${#mask_ratios[@]} MaskRatios: ${mask_ratios[*]}"
  log_info "  ${#profile_iterations[@]} ProfileIterations:"
  for ((i=0; i<${#profile_iterations[@]}; i++)); do
    log_info "    [$i]: ${profile_iterations[$i]}"
  done
  log_info "  ${#kernel_array[@]} Kernels:"

  local total_scopes=0
  for ((i=0; i<${#kernel_array[@]}; i++)); do
    IFS=',' read -r kernel_name operation <<< "${kernel_array[$i]}"
    local kernel_scopes=$(get_init_scope "$kernel_name")
    local scope_count=$(echo "$kernel_scopes" | wc -w)
    total_scopes=$((total_scopes + scope_count))
    log_info "    [$i]: ${kernel_array[$i]} (scopes: $kernel_scopes, count: $scope_count)"
  done
  log_info "  Total scopes across all kernels: $total_scopes"

  total_scopes_maskratios=$((total_scopes * ${#mask_ratios[@]}))

  # Calculate total number of runs using the total number of scopes
  total_runs=0  # Reset the global variable
  for ((i=0; i<${#kernel_array[@]}; i++)); do
    IFS=',' read -r kernel_name operation <<< "${kernel_array[$i]}"
    local kernel_scopes=$(get_init_scope "$kernel_name")
    local scope_count=$(echo "$kernel_scopes" | wc -w)
    local kernel_runs=$((${#mode[@]} * ${#freq[@]} * scope_count * ${#mask_ratios[@]} * ${#profile_iterations[@]}))
    total_runs=$((total_runs + kernel_runs))
    log_info "    Kernel [$i]: ${kernel_array[$i]} will run $kernel_runs tests"
  done
  log_info "Total runs: $total_runs"
}

# Create output directory
create_output_directory() {
  local date_prefix=$(date +%m%d)
  local dir_num=0
  local dir_suffix=""

  # Add tag to directory name if provided
  if [ -n "$USER_TAG" ]; then
    dir_suffix="_${USER_TAG}"
  fi

  while true; do
    OUTPUT_DIR="${date_prefix}_cutlass_profile_${dir_num}"
    if [ ! -d "$OUTPUT_DIR" ]; then
      OUTPUT_DIR="${OUTPUT_DIR}${dir_suffix}"
      mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/data"
      log_info "Created output directory: $OUTPUT_DIR"
      return
    fi
    dir_num=$((dir_num + 1))
  done
}

# Process file path
process_file_path() {
  local file_path="$1"
  local file_name=$(basename "$file_path")
  echo "${OUTPUT_DIR}/${file_name}"
}

# NVIDIA SMI logging
nvsmi_log() {
  if [ "$DRY_RUN" = "true" ]; then
    if [ "$1" = "start" ]; then
      log_info "DRY RUN: Would start GPU monitoring: $gpu_monitor_cmd"
    elif [ "$1" = "stop" ]; then
      log_info "DRY RUN: Would stop GPU monitoring"
    fi
  else
    if [ "$1" = "start" ]; then
      log_info "Starting GPU monitoring: $gpu_monitor_cmd"
      eval "$gpu_monitor_cmd" &
    elif [ "$1" = "stop" ]; then
      log_info "Stopping GPU monitoring"
      pkill -f nvidia-smi
      sleep 5
    fi
  fi
}

# Rename log file
rename_log() {
  mv "$1" "$2"
}

# Format seconds into hours, minutes, seconds string
format_time() {
  local seconds=$1
  local hours=$((seconds / 3600))
  local minutes=$(((seconds % 3600) / 60))
  local seconds=$((seconds % 60))
  echo "${hours}h ${minutes}m ${seconds}s"
}

# Profile a single kernel
profile_kernel() {
  local kernel_name="$1"
  local operation="$2"
  local mask_ratio="$3"
  local scope="$4"
  local freq="$5"
  local current_run="$6"
  local total_runs="$7"
  local profile_type="$8"
  local warmup_iterations="$9"
  local profiling_iterations="${10}"

  # Record the start time of the task
  local task_start_time=$(date +%s)

  # Display progress
  local progress=$((current_run * 100 / total_runs))
  local bar_length=20
  local filled_length=$((progress * bar_length / 100))
  local bar=""
  for ((i=0; i<filled_length; i++)); do
    bar+="#"
  done
  for ((i=filled_length; i<bar_length; i++)); do
    bar+="-"
  done

  # Prepare countdown
  if [ ${start_delay:-0} -gt 0 ]; then
    local secs=$start_delay
    echo -ne "Start kernel analysis in $secs seconds ..."
    while [ $secs -gt 0 ]; do
      local current_timestamp="[$(date '+%Y-%m-%d %H:%M:%S')]"
      echo -ne "\r${current_timestamp} INFO: Start kernel analysis in $secs seconds ...   "
      sleep 1
      ((secs--))
    done
    echo -ne "\r"
  fi

  local dist_output=${OUTPUT_DIR}/data/${kernel_name}_mask${mask_ratio}_scope${scope}
  if [ "$total_scopes_maskratios" -gt 15 ]; then
    dist_output=${OUTPUT_DIR}/data/mask${mask_ratio}_scope${scope}
  fi

  local output=${OUTPUT_DIR}/${kernel_name}_${freq}Mhz_mask${mask_ratio}_scope${scope}_mode${profile_type}_run${current_run}
  local tags="Freq:${freq},Kernel:${kernel_name},Hacking:${profile_type},ScopeMin:-${scope},ScopeMax:${scope},MaskRatio:${mask_ratio},WarmupIter:${warmup_iterations},ProfileIter:${profiling_iterations}"
  log_info "${script_runner} --mode ${profile_type} --scope ${scope} --mask_ratio ${mask_ratio} --kernel ${kernel_name} --operation ${operation} --tags ${tags} --output ${output} --warmup-iterations ${warmup_iterations} --profiling-iterations ${profiling_iterations}"

  dump_data=false
  if [ ! -f "${dist_output}.data_dist.csv" ]; then
    dump_data=true
    rm -rf *.mat
  fi

  if [ "$DRY_RUN" = "false" ]; then
    nvsmi_log start
    ${script_runner} --mode ${profile_type} --scope ${scope} --mask_ratio ${mask_ratio} --kernel ${kernel_name} --operation ${operation} \
      --dump_data ${dump_data} --tags ${tags} --output ${output} --warmup-iterations ${warmup_iterations} --profiling-iterations ${profiling_iterations}
    nvsmi_log stop
    if [ $dump_data = true ]; then
      log_info "Analyzing data distribution and moving *.mat files..."
      ./analyze_distribution.py --tags ${tags} --csv ${dist_output}.data_dist
      rm -rf *.mat
      # for file in ./*_A.mat; do
      #   [ -e "$file" ] || continue
      #   mv "$file" "${dist_output}_A.mat"
      # done
      # for file in ./*_B.mat; do
      #   [ -e "$file" ] || continue
      #   mv "$file" "${dist_output}_B.mat"
      # done
    fi
    rename_log nvsmi.csv "${output}_nvsmi.txt"
  fi

  # Calculate the duration of this task and update statistics
  local task_end_time=$(date +%s)
  local task_duration=$((task_end_time - task_start_time))

  # Update total and average execution time
  total_execution_time=$((total_execution_time + task_duration))
  local avg_time=$((total_execution_time / current_run))
  local remaining_tasks=$((total_runs - current_run))
  local estimated_remaining_time=$((avg_time * remaining_tasks))

  # Format time values using the helper function
  local task_time=$(format_time $task_duration)
  local avg_time_fmt=$(format_time $avg_time)
  local est_remaining=$(format_time $estimated_remaining_time)

  local now_time=$(date +%s)
  local total_elapsed_sec=$((now_time - start_time))
  local total_elapsed_fmt=$(format_time $total_elapsed_sec)

  log_info "Progress: [${current_run}/${total_runs} ${bar}] Elapsed: ${total_elapsed_fmt} | Est. remaining: ${est_remaining} | Avg Task: ${avg_time_fmt}"
}

# Get init_scope value for a specific kernel
get_init_scope() {
  local kernel_name="$1"
  if [[ "$kernel_name" == *"ue4m3xe2m1_ue4m3xe2m1_f32"* ]]; then
    echo "6"
    return
  fi
  if [[ "$kernel_name" == *"ue8m0xe4m3_ue8m0xe4m3_f32"* ]] || [[ "$kernel_name" == *"f16_f16_f32"* ]]; then
    echo "0.5 5"
    return
  fi
  echo "5"
}

# Apply frequency and run kernel analysis
apply_frequency_and_run() {
  local freq_value="$1"
  local kernel_name="$2"
  local operation="$3"

  local freq_cmd_value=$frequency_cmd

  if [ "$DRY_RUN" = "true" ]; then
    if [ "$freq_value" = "-1" ] || [ "$freq_value" = "oob" ]; then
      log_info "DRY RUN: nvidia-smi -i ${gpu_id} --reset-gpu-clocks"
    else
      freq_cmd_value=${freq_cmd_value//frequency/$freq_value}
      log_info "DRY RUN: $freq_cmd_value"
    fi
  else
    if [ "$freq_value" = "-1" ] || [ "$freq_value" = "oob" ]; then
      log_info "Resetting GPU clocks..."
      nvidia-smi -i ${gpu_id} --reset-gpu-clocks
    else
      freq_cmd_value=${freq_cmd_value//frequency/$freq_value}
      log_info "$freq_cmd_value"
      eval $freq_cmd_value
    fi
  fi

  local kernel_specific_scope=$(get_init_scope "$kernel_name")
  log_info "Using scope value for kernel $kernel_name: $kernel_specific_scope"

  # Outer loop - iterate over modes
  for profile_type in "${mode[@]}"; do
    for scope in $kernel_specific_scope; do
      for mask_ratio in "${mask_ratios[@]}"; do
        for iteration_tuple in "${profile_iterations[@]}"; do
          IFS=',' read -r warmup_iterations profiling_iterations <<< "$iteration_tuple"
          current_run=$((current_run + 1))
          profile_kernel "$kernel_name" "$operation" "$mask_ratio" "$scope" "$freq_value" "$current_run" "$total_runs" "$profile_type" "$warmup_iterations" "$profiling_iterations"
        done
      done
    done
  done
}

#===== Main Function =====
main() {
  # Process command line arguments
  DRY_RUN="false"
  USER_TAG=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dry)
        DRY_RUN="true"
        log_info "Running in DRY RUN mode - commands will be shown but not executed"
        shift
        ;;
      --tag)
        if [[ -n "$2" && "$2" != --* ]]; then
          USER_TAG="$2"
          log_info "Using tag for output directory: $USER_TAG"
          shift 2
        else
          log_error "Option --tag requires an argument"
          exit 1
        fi
        ;;
      *)
        # If it's a configuration file (first non-option argument)
        if [[ -f "$1" && $# -eq 1 ]]; then
          CONFIG_FILE="$1"
        else
          log_error "Unknown option: $1"
          log_info "Usage: $0 [--dry] [--tag tagname] [config_file]"
          exit 1
        fi
        shift
        ;;
    esac
  done

  # Initial checks
  check_root

  # Load configuration
  if [ -n "$CONFIG_FILE" ]; then
    if ! load_config "$CONFIG_FILE"; then
      exit 2
    fi
  elif [ -f "$CONFIG_FILE" ]; then
    load_config "$CONFIG_FILE"
  else
    log_error "Automation configuration file not found, this file is required"
    exit 2
  fi

  check_all_configs
  create_output_directory

  # Set CUDA visible devices
  if [ "$DRY_RUN" = "true" ]; then
    log_info "DRY RUN: Would set CUDA_VISIBLE_DEVICES=$gpu_id"
  else
    export CUDA_VISIBLE_DEVICES=$gpu_id
    log_info "Setting CUDA visible devices: $gpu_id"
  fi

  log_info "Starting $BENCHNAME (total tasks: $total_runs)"

  current_run=0
  local start_time=$(date +%s)
  total_execution_time=0

  for kernel_tuple in "${kernel_array[@]}"; do
    IFS=',' read -r kernel_name operation <<< "$kernel_tuple"
    for freq_value in "${freq[@]}"; do
      apply_frequency_and_run "$freq_value" "$kernel_name" "$operation"
      log_info "Completed kernel $kernel_name frequency $freq_value runs: current progress $current_run/$total_runs"
    done
  done

  local end_time=$(date +%s)
  local duration=$((end_time - start_time))
  local hours=$((duration / 3600))
  local minutes=$(((duration % 3600) / 60))
  local seconds=$((duration % 60))

  if [ "$DRY_RUN" = "true" ]; then
    log_info "DRY RUN: nvidia-smi -i ${gpu_id} --reset-gpu-clocks"
  else
    nvidia-smi -i ${gpu_id} --reset-gpu-clocks
  fi
  log_info "$BENCHNAME completed! Total duration: ${hours}h ${minutes}m ${seconds}s"

  refer_app="/dataset/shuangpeng/project/cutlass/HPC-Kernels-CUDA/xgemm/xgemm_cublasLt/xgemm_scope5"
  refer_app_args="-type=8 -iter=2000 -warmup=500 -m=16384 -n=16384 -k=16384 -tc=1 -bs=1 -transa=1 -transb=0 -verify=0 -fastAccum=0"
  if [ -f "$refer_app" ]; then
    log_info "Execute reference measurement application to ensure environment is expected"
    log_info "RUN xgemm with type=fp8 scope=[-5,5] freq=oob"
    nvidia-smi -i ${gpu_id} --reset-gpu-clocks > /dev/null
    $refer_app $refer_app_args | tee "${OUTPUT_DIR}/xgemm_scope5.log"
  else
    log_warning "Referencing application not found: $refer_app"
  fi

  mv *log*.txt ${OUTPUT_DIR}
  zip -r ${OUTPUT_DIR}.zip ${OUTPUT_DIR} -x "*.mat" > /dev/null
  FULL_PATH=$(realpath "${OUTPUT_DIR}.zip")
  echo -e "\n  scp b200:${FULL_PATH} ."
}

main "$@"
