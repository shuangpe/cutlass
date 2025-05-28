#!/bin/bash

# cmake .. -DCUTLASS_NVCC_ARCHS=100a -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_UNITY_BUILD_ENABLED=ON -DCUTLASS_LIBRARY_OPERATIONS=gemm -DCUTLASS_LIBRARY_KERNELS=gemm_*_void*_2sm -DCUTLASS_LIBRARY_IGNORE_KERNELS=stream
# make cutlass_profiler -j32

# Function to display help information
show_help() {
  echo "Usage: $0 -g <gpu_id> [-d] [-h]"
  echo "Options:"
  echo "  -g  Specify GPU ID (required)"
  echo "  -d  Dry run mode - print commands without executing"
  echo "  -h  Display this help message"
  echo "Example: $0 -g 0"
}

# Default parameters
gpu_id=""
dry_run=false

# Hardcoded frequency profiles as tuples (min_freq max_freq)
freq_profiles=(
  "1500 1500"
  "1300 1300"
)

# Parse command line arguments
while getopts "g:dh" opt; do
  case ${opt} in
    g )
      gpu_id=$OPTARG
      ;;
    d )
      dry_run=true
      ;;
    h )
      show_help
      exit 0
      ;;
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      show_help
      exit 1
      ;;
    : )
      echo "Option $OPTARG requires an argument" 1>&2
      show_help
      exit 1
      ;;
  esac
done

# Check if GPU ID is provided
if [ -z "$gpu_id" ]; then
  echo "Error: GPU ID must be specified"
  show_help
  exit 1
fi

# Function to validate GPU ID
validate_gpu_id() {
  local gpu_id=$1
  local gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits)

  if [ "$gpu_id" -ge "$gpu_count" ] || [ "$gpu_id" -lt 0 ]; then
    echo "Error: Invalid GPU ID. System has $gpu_count GPU(s), IDs range from 0 to $((gpu_count-1))"
    return 1
  fi
  return 0
}

# Main execution flow
if $dry_run; then
  echo "DryRun mode enabled. No scripts will be executed."
else
  # Validate GPU ID in non-dry-run mode
  if ! validate_gpu_id "$gpu_id"; then
    echo "Invalid GPU ID. Exiting."
    exit 1
  fi
fi

# Get current date in YYYYMMDD format
current_date=$(date +"%Y%m%d")

# Create report directory with date - use absolute path
report_dir="$(pwd)/report_${current_date}"
mkdir -p "$report_dir"

# Create a log file to record all main script output
log_file="${report_dir}/run_profiler_${current_date}.log"

# Start capturing script output to both terminal and log file
exec > >(tee -a "$log_file") 2>&1

# Function to write content to a file (overwrites existing content)
write_to_file() {
  local file="$1"
  local content="$2"
  echo "$content" > "$file"
}

# Function to append content to a file
append_to_file() {
  local file="$1"
  local content="$2"
  echo "$content" >> "$file"
}

# Function to append multiple lines to a file from an array
append_lines_to_file() {
  local file="$1"
  shift
  local lines=("$@")

  for line in "${lines[@]}"; do
    append_to_file "$file" "$line"
  done
}

# Function to write multiple lines to a file (overwrites existing content)
write_lines_to_file() {
  local file="$1"
  shift
  local lines=("$@")

  # Check if there are any lines to write
  if [ ${#lines[@]} -eq 0 ]; then
    return
  fi

  # Write first line (overwrite file)
  write_to_file "$file" "${lines[0]}"

  # Append remaining lines if any
  if [ ${#lines[@]} -gt 1 ]; then
    for ((i=1; i<${#lines[@]}; i++)); do
      append_to_file "$file" "${lines[$i]}"
    done
  fi
}

# Create main runnable script
main_run_script="${report_dir}/run.sh"
main_script_header=(
  "#!/bin/bash"
  "# Script to reproduce all profiling runs from ${current_date} for GPU ID ${gpu_id}"
  ""
  "echo \"Starting profiling runs for GPU ID: ${gpu_id}\""
  ""
)

write_to_file "$main_run_script" "${main_script_header[0]}"
unset "main_script_header[0]"
append_lines_to_file "$main_run_script" "${main_script_header[@]}"
chmod +x "$main_run_script"

# Create script template sections - these don't change with frequency profiles
script_header_template=(
  "#!/bin/bash"
  "# Command to reproduce profiling at %MAX_FREQ%MHz for GPU ID %GPU_ID%"
  ""
  "# Exit immediately if a command exits with a non-zero status"
  "set -e"
  ""
  "# Define variables"
  "gpu_id=%GPU_ID%"
  "min_freq=%MIN_FREQ%"
  "max_freq=%MAX_FREQ%"
  ""
  "# Save original CUDA_VISIBLE_DEVICES value"
  "ORIGINAL_CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-}"
  "# Force CUDA_VISIBLE_DEVICES to use specified GPU ID"
  "export CUDA_VISIBLE_DEVICES=\$gpu_id"
  ""
  "# Create logs directory and log file for profiler output"
  "mkdir -p \"${report_dir}/logs\""
  "log_filename=\"${report_dir}/logs/profile-gemm-\${max_freq}mhz-gpu\${gpu_id}.log\""
  ""
)

set_frequency_section=(
  "# Set GPU frequency"
  "echo \"Setting GPU \$gpu_id frequency range to \$min_freq MHz ~ \$max_freq MHz...\""
  "if nvidia-smi --id=\"\$gpu_id\" --lock-gpu-clocks=\"\$min_freq\",\"\$max_freq\"; then"
  "  echo \"GPU frequency setting successful\""
  "  echo \"Current GPU settings:\""
  "  nvidia-smi --id=\"\$gpu_id\" --query-gpu=name,clocks.gr,clocks.max.gr --format=csv"
  "else"
  "  echo \"GPU frequency setting failed\""
  "  exit 1"
  "fi"
  ""
)

run_profiler_section=(
  "# Run profiler"
  "echo \"Running profiler with GPU \$gpu_id at frequency range \$min_freq MHz ~ \$max_freq MHz...\""
  "/workspace/cutlass/build/tools/profiler/cutlass_profiler \\"
  "  --operation=Gemm --op_class=tensorop \\"
  "  --profiling-iterations=100 --warmup-iterations=10 \\"
  "  --m=8192 --n=8192 --k=256,512,1024,2048,4096,8192 \\"
  "  --providers=cutlass \\"
  "  --output=\"${report_dir}/profile-gemm-\${max_freq}mhz-gpu\${gpu_id}.csv\" \\"
  "  > \"\$log_filename\" 2>&1"
  "echo \"Profiler output saved to \$log_filename\""
  ""
)

reset_frequency_section=(
  "# Reset GPU frequency"
  "echo \"Resetting GPU \$gpu_id frequency to default...\""
  "if nvidia-smi --id=\"\$gpu_id\" --reset-gpu-clocks; then"
  "  echo \"GPU frequency reset successful\""
  "  echo \"Current GPU settings:\""
  "  nvidia-smi --id=\"\$gpu_id\" --query-gpu=name,clocks.gr,clocks.max.gr --format=csv"
  "else"
  "  echo \"GPU frequency reset failed\""
  "fi"
  ""
  "# Restore original CUDA_VISIBLE_DEVICES setting"
  "if [ -z \"\$ORIGINAL_CUDA_VISIBLE_DEVICES\" ]; then"
  "  unset CUDA_VISIBLE_DEVICES"
  "else"
  "  export CUDA_VISIBLE_DEVICES=\$ORIGINAL_CUDA_VISIBLE_DEVICES"
  "fi"
  ""
)

# Function to generate a frequency profiler script
generate_freq_script() {
  local script_path="$1"
  local gpu_id="$2"
  local min_freq="$3"
  local max_freq="$4"

  # Create script header with substituted values
  local script_header=()
  for line in "${script_header_template[@]}"; do
    line="${line//%GPU_ID%/$gpu_id}"
    line="${line//%MIN_FREQ%/$min_freq}"
    line="${line//%MAX_FREQ%/$max_freq}"
    script_header+=("$line")
  done

  # Write all sections to the file
  write_lines_to_file "$script_path" "${script_header[@]}"
  append_lines_to_file "$script_path" "${set_frequency_section[@]}"
  append_lines_to_file "$script_path" "${run_profiler_section[@]}"
  append_lines_to_file "$script_path" "${reset_frequency_section[@]}"

  chmod +x "$script_path"
}

# Generate individual frequency profile scripts and add them to main script
for profile in "${freq_profiles[@]}"; do
  read -r min_freq max_freq <<< "$profile"

  # Create individual run script for this frequency with GPU ID in filename
  freq_run_script="${report_dir}/run_${max_freq}mhz_gpu${gpu_id}.sh"

  # Generate the frequency script
  generate_freq_script "$freq_run_script" "$gpu_id" "$min_freq" "$max_freq"

  # Add this frequency run to the main script with absolute path
  main_script_addition=(
    "echo \"Running profile for ${max_freq}MHz...\""
    "\"${freq_run_script}\""
    ""
  )
  append_lines_to_file "$main_run_script" "${main_script_addition[@]}"
done

append_to_file "$main_run_script" "echo \"All profiles completed.\""

# Execute the main run script only if not in dry run mode
if ! $dry_run; then
  echo "Executing generated run script..."
  bash "$main_run_script" "$gpu_id"
    echo "Profiling process complete."
    echo "Reports saved to: $report_dir"
else
  echo "To run manually: bash $main_run_script"
fi

