#!/bin/bash
# Automation workload execution script

#===== Global Variables =====
SCRIPT_NAME=$(basename "$0")
LOG_PREFIX="[$(date '+%Y-%m-%d %H:%M:%S')]"
CONFIG_FILE="batch_profile.cfg"
OUTPUT_DIR=""
BENCHNAME="Automation Workload"
current_run=0  # Now a global variable

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
  freq=($frequency)
  scope=($init_scope)
  mode=($profile_mode)

  # Output all configuration information
  log_info "Scan Configuration:"
  log_info "  ${#mode[@]} Models: ${mode[*]}"
  log_info "  ${#freq[@]} Frequencies: ${freq[*]}"
  log_info "  ${#scope[@]} InitScopes: ${scope[*]}"
  log_info "  ${#mask_ratios[@]} MaskRatios: ${mask_ratios[*]}"
  log_info "  ${#profile_iterations[@]} ProfileIterations:"
  # Print each profile iteration line by line
  for ((i=0; i<${#profile_iterations[@]}; i++)); do
    log_info "    [$i]: ${profile_iterations[$i]}"
  done
  log_info "  ${#kernel_array[@]} Kernels:"
  # Print each kernel line by line
  for ((i=0; i<${#kernel_array[@]}; i++)); do
    log_info "    [$i]: ${kernel_array[$i]}"
  done

  # Calculate total number of runs
  total_runs=$((${#mode[@]} * ${#freq[@]} * ${#kernel_array[@]} * ${#scope[@]} * ${#mask_ratios[@]} * ${#profile_iterations[@]}))
  log_info "Total runs: $total_runs"
}

# Create output directory
create_output_directory() {
  local date_prefix=$(date +%m%d)
  local dir_num=0

  while true; do
    OUTPUT_DIR="${date_prefix}_cutlass_profile_${dir_num}"
    if [ ! -d "$OUTPUT_DIR" ]; then
      mkdir -p "$OUTPUT_DIR"
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

  local output=${OUTPUT_DIR}/${kernel_name}_${freq}Mhz_mask${mask_ratio}_scope${scope}_mode${profile_type}_wi${warmup_iterations}_pi${profiling_iterations}
  local tags="Freq:${freq},Kernel:${kernel_name},Hacking:${profile_type},ScopeMin:-${scope},ScopeMax:${scope},MaskRatio:${mask_ratio},WarmupIter:${warmup_iterations},ProfileIter:${profiling_iterations}"
  log_info "./cutlass_profiler_16k.sh --mode ${profile_type} --scope ${scope} --mask_ratio ${mask_ratio} --kernel ${kernel_name} --operation ${operation} --tags ${tags} --output ${output} --warmup-iterations ${warmup_iterations} --profiling-iterations ${profiling_iterations}"

  if [ "$DRY_RUN" = "false" ]; then
    nvsmi_log start
    ./cutlass_profiler_16k.sh --mode ${profile_type} --scope ${scope} --mask_ratio ${mask_ratio} --kernel ${kernel_name} --operation ${operation} --tags ${tags} --output ${output} --warmup-iterations ${warmup_iterations} --profiling-iterations ${profiling_iterations}
    nvsmi_log stop
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

# Apply frequency and run kernel analysis
apply_frequency_and_run() {
  local freq_value="$1"

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

  # Outer loop - iterate over modes
  for profile_type in "${mode[@]}"; do
    # For each kernel, execute with different mask_ratio and scope combinations
    for kernel_tuple in "${kernel_array[@]}"; do
      IFS=',' read -r kernel_name operation <<< "$kernel_tuple"
      for scope in $init_scope; do
        for mask_ratio in "${mask_ratios[@]}"; do
          for iteration_tuple in "${profile_iterations[@]}"; do
            IFS=',' read -r warmup_iterations profiling_iterations <<< "$iteration_tuple"
            current_run=$((current_run + 1))
            profile_kernel "$kernel_name" "$operation" "$mask_ratio" "$scope" "$freq_value" "$current_run" "$total_runs" "$profile_type" "$warmup_iterations" "$profiling_iterations"
          done
        done
      done
    done
  done
}

#===== Main Function =====
main() {
  # Process command line arguments
  DRY_RUN="false"
  for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
      DRY_RUN="true"
      log_info "Running in DRY RUN mode - commands will be shown but not executed"
      shift
    fi
  done

  # Initial checks
  check_root

  # Load configuration
  if [ $# -eq 1 ]; then
    if ! load_config "$1"; then
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

  current_run=0  # 现在是全局变量
  local start_time=$(date +%s)
  total_execution_time=0

  for freq_value in "${freq[@]}"; do
    apply_frequency_and_run "$freq_value"
    log_info "Completed frequency $freq_value runs: current progress $current_run/$total_runs"
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
}

main "$@"
