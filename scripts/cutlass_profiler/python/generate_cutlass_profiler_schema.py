#!/usr/bin/env python3

"""
Python script to generate cutlass_profiler profiling shell scripts, refactored from run_profiler.sh
"""
import os
import sys
import argparse
import datetime

# Default operations and frequency profiles
default_operations = ["Gemm", "BlockScaledGemm"]
default_freq_profiles = [(1500, 1500), (1305, 1305), (1005, 1005), (-1, -1)]  # -1 means no frequency limit

def parse_args():
    parser = argparse.ArgumentParser(description="Generate cutlass_profiler profiling scripts")
    parser.add_argument("-g", "--gpu_id", type=int, required=True, help="GPU ID (required)")
    parser.add_argument("-d", "--dry_run", action="store_true", help="Dry run mode: only print commands")
    parser.add_argument("-o", "--out_dir", type=str, default=None, help="Output directory for generated scripts")
    parser.add_argument("--operations", nargs="*", default=None, help="Operations to test (default: Gemm BlockScaledGemm)")
    parser.add_argument("--freqs", nargs="*", type=int, default=None, help="Frequency profiles, e.g. 1500 1305 (applies to both min/max)")
    return parser.parse_args()


def validate_gpu_id(gpu_id):
    import subprocess
    try:
        # Use nvidia-smi -L, each line represents one GPU
        result = subprocess.run([
            "nvidia-smi", "-L"
        ], capture_output=True, text=True, check=True)
        gpu_count = len([line for line in result.stdout.strip().split('\n') if line.strip()])
        if gpu_id < 0 or gpu_id >= gpu_count:
            print(f"Error: Invalid GPU ID. System has {gpu_count} GPU(s), IDs range from 0 to {gpu_count-1}")
            return False
        return True
    except Exception as e:
        print(f"Warning: Could not validate GPU ID: {e}")
        return True  # Allow to proceed if nvidia-smi not available


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_lines_to_file(filepath, lines):
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def append_lines_to_file(filepath, lines):
    with open(filepath, 'a') as f:
        f.write('\n'.join(lines) + '\n')


def generate_freq_script(script_path, gpu_id, min_freq, max_freq, operation, kernel_filter, report_dir, csv_dir):
    """Generate a shell script for a specific freq/operation (no save_log_to_file, always print to console)"""
    # Create directory for frequency monitoring data inside csv_dir
    freq_logs_dir = os.path.join(csv_dir, "freq_logs")
    ensure_dir(freq_logs_dir)

    # Calculate GPU frequency monitoring script path based on current script path
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_script_dir)
    gpu_freq_collector_path = os.path.join(parent_dir, "gpu_freq_monitor", "gpu_freq_collector.py")

    # Frequency string for filenames/logs
    freq_str = "oob" if max_freq == -1 else str(max_freq)
    min_freq_str = "oob" if min_freq == -1 else str(min_freq)

    script_base = os.path.splitext(os.path.basename(script_path))[0]
    perf_csv_filename = os.path.join(csv_dir, f"{script_base}.csv")
    freq_log_filename = os.path.join(freq_logs_dir, f"freq_{script_base}.csv")

    lines = [
        "#!/bin/bash",
        f"# Command to reproduce profiling at {freq_str}MHz for GPU ID {gpu_id} with operation {operation}",
        "",
        "set -e",
        f"gpu_id={gpu_id}",
        f"min_freq={min_freq}",
        f"max_freq={max_freq}",
        f"operation=\"{operation}\"",
        f"kernel_filter=\"{kernel_filter}\"",
        "ORIGINAL_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}",
        "export CUDA_VISIBLE_DEVICES=$gpu_id",
        f"mkdir -p \"{report_dir}/logs\"",
        f"mkdir -p \"{freq_logs_dir}\"",
        f"perf_csv_filename=\"{perf_csv_filename}\"",
        f"freq_log_filename=\"{freq_log_filename}\"",
        "start_time=$(date +%s)",
        "echo \"Started at $(date '+%Y-%m-%d %H:%M:%S')\"",
        "",
        "# Check dependencies",
        "if ! pip3 show pynvml pandas > /dev/null 2>&1; then",
        "  echo \"Installing required Python libraries...\"",
        "  pip3 install pynvml pandas",
        "fi",
        ""
    ]

    # Set GPU frequency only if not -1
    if max_freq != -1 and min_freq != -1:
        lines += [
            "# Set GPU frequency",
            "echo \"Setting GPU $gpu_id frequency range to $min_freq MHz ~ $max_freq MHz...\"",
            "if nvidia-smi --id=\"$gpu_id\" --lock-gpu-clocks=\"$min_freq\",\"$max_freq\"; then",
            "  echo \"GPU frequency setting successful\"",
            "  echo \"Current GPU settings:\"",
            "  nvidia-smi --id=\"$gpu_id\" --query-gpu=name,clocks.gr,clocks.max.gr --format=csv",
            "else",
            "  echo \"GPU frequency setting failed\"",
            "  exit 1",
            "fi",
            ""
        ]
    else:
        lines += [
            "# No frequency lock applied (min_freq or max_freq is -1, treated as out-of-band)",
            "echo \"No GPU frequency lock applied for GPU $gpu_id (min_freq or max_freq is -1)\"",
            ""
        ]

    lines += [
        "# Start GPU frequency monitoring",
        "echo \"Starting GPU frequency monitoring...\"",
        f"python3 {gpu_freq_collector_path} \\",
        "  --gpu-id=$gpu_id \\",
        "  --output=$freq_log_filename \\",
        "  --interval=0.001 &",
        "",
        "MONITOR_PID=$!",
        "echo \"GPU frequency monitoring started with PID: $MONITOR_PID\"",
        "",
        "# Wait a few seconds to ensure the monitoring script is running",
        "sleep 2",
        "",
        "# Run profiler",
        f"echo \"Running profiler for operation $operation with GPU $gpu_id at frequency range {min_freq_str} MHz ~ {freq_str} MHz...\"",
        "profiler_start_time=$(date +%s)",
        "profiler_start_formatted=$(date '+%Y-%m-%d %H:%M:%S')",
        f"/workspace/cutlass/build/tools/profiler/cutlass_profiler \\",
        "  --operation=$operation --kernels=$kernel_filter \\",
        "  --profiling-iterations=100 --warmup-iterations=10 \\",
        "  --m=16384 --n=16384 --k=256,512,1024,2048,4096,8192,16384 \\",
        "  --providers=cutlass --dist=uniform,min:-5,max:5 \\",
        f"  --output=$perf_csv_filename",
        "profiler_end_time=$(date +%s)",
        "profiler_end_formatted=$(date '+%Y-%m-%d %H:%M:%S')",
        "profiler_duration=$((profiler_end_time - profiler_start_time))",
        "echo \"-------------------------------------------------------------\"",
        "echo \"Profiler execution summary:\"",
        "echo \"  Started at:  $profiler_start_formatted\"",
        "echo \"  Finished at: $profiler_end_formatted\"",
        "echo \"  Runtime:     $((profiler_duration / 60)) minutes and $((profiler_duration % 60)) seconds\"",
        "echo \"-------------------------------------------------------------\"",
        "",
        "# Stop GPU frequency monitoring",
        "echo \"Stopping GPU frequency monitoring...\"",
        "kill $MONITOR_PID",
        "wait $MONITOR_PID 2>/dev/null || true",
        "echo \"GPU frequency monitoring stopped. Frequency data saved to: $freq_log_filename\"",
        ""
    ]

    # Reset GPU frequency only if not -1
    if max_freq != -1 and min_freq != -1:
        lines += [
            "# Reset GPU frequency",
            "echo \"Resetting GPU $gpu_id frequency to default...\"",
            "if nvidia-smi --id=\"$gpu_id\" --reset-gpu-clocks; then",
            "  echo \"GPU frequency reset successful\"",
            "  echo \"Current GPU settings:\"",
            "  nvidia-smi --id=\"$gpu_id\" --query-gpu=name,clocks.gr,clocks.max.gr --format=csv",
            "else",
            "  echo \"GPU frequency reset failed\"",
            "fi",
        ]
    else:
        lines += [
            "# No GPU frequency reset needed (min_freq or max_freq is -1)",
            "echo \"No GPU frequency reset needed for GPU $gpu_id (min_freq or max_freq is -1)\"",
        ]

    lines += [
        "end_time=$(date +%s)",
        "duration=$((end_time - start_time))",
        "echo \"Finished at $(date '+%Y-%m-%d %H:%M:%S')\"",
        "echo \"Total runtime: $((duration / 60)) minutes and $((duration % 60)) seconds\"",
        "echo \"GPU frequency monitoring data saved to: $freq_log_filename\"",
        "if [ -z \"$ORIGINAL_CUDA_VISIBLE_DEVICES\" ]; then",
        "  unset CUDA_VISIBLE_DEVICES",
        "else",
        "  export CUDA_VISIBLE_DEVICES=$ORIGINAL_CUDA_VISIBLE_DEVICES",
        "fi",
        ""
    ]
    write_lines_to_file(script_path, lines)
    os.chmod(script_path, 0o755)


def main():
    args = parse_args()
    gpu_id = args.gpu_id
    dry_run = args.dry_run
    operations = args.operations if args.operations else default_operations
    freq_profiles = (
        [(f, f) for f in args.freqs] if args.freqs else default_freq_profiles
    )

    # Validate GPU ID (only if not dry-run)
    if not dry_run and not validate_gpu_id(gpu_id):
        sys.exit(1)

    # Output/report directory
    current_date = datetime.datetime.now().strftime("%m%d")
    report_dir = args.out_dir or os.path.join(os.getcwd(), f"{current_date}_cutlass_profiler_schema")
    ensure_dir(report_dir)
    scripts_dir = os.path.join(report_dir, "scripts")
    csv_base = f"{current_date}_cutlass_profiler_csv"
    csv_dir = os.path.join(report_dir, csv_base)
    ensure_dir(scripts_dir)
    ensure_dir(csv_dir)
    # Create directory for frequency monitoring data inside csv_dir
    freq_logs_dir = os.path.join(csv_dir, "freq_logs")
    ensure_dir(freq_logs_dir)

    # Main run script
    main_run_script = os.path.join(report_dir, "run.sh")
    main_script_lines = [
        "#!/bin/bash",
        f"# Script to reproduce all profiling runs from {current_date} for GPU ID {gpu_id}",
        f"mkdir -p \"{report_dir}/logs\"",
        "overall_start_time=$(date +%s)",
        f"echo \"Starting profiling runs for GPU ID: {gpu_id} at $(date)\"",
        ""
    ]

    # Generate freq/operation scripts and add to main script
    for operation in operations:
        operation_lower = operation.lower()
        kernel_filters = ["f16_f16_f32_void_f16", "ue8m0xe4m3_ue8m0xe4m3_f32_void_ue8m0xe4m3"] if operation_lower == "gemm" else ["ue4m3xe2m1_ue4m3xe2m1_f32_void_ue4m3xe2m1"]
        for kernel_filter in kernel_filters:
            precision_str = "mxfp8"
            if kernel_filter == "ue4m3xe2m1_ue4m3xe2m1_f32_void_ue4m3xe2m1":
                precision_str = "nvfp4"
            elif kernel_filter == "f16_f16_f32_void_f16":
                precision_str = "fp16"

            for min_freq, max_freq in freq_profiles:
                freq_str = "oob" if max_freq == -1 else str(max_freq)
                freq_run_script = os.path.join(scripts_dir, f"profile_{precision_str}_{operation_lower}_{freq_str}mhz_gpu{gpu_id}.sh")
                generate_freq_script(freq_run_script, gpu_id, min_freq, max_freq, operation, kernel_filter, report_dir, csv_dir)
                # Main script calls subscripts and redirects output to log file
                script_base = os.path.splitext(os.path.basename(freq_run_script))[0]
                log_filename = os.path.join(report_dir, "logs", f"{script_base}.log")
                main_script_lines.extend([
                    f"echo \"Running profile for {precision_str} {operation} at {max_freq}MHz...\"",
                    f'"{freq_run_script}" > "{log_filename}" 2>&1',
                    ""
                ])

    main_script_lines.append("echo \"All profiles completed.\"")
    main_script_lines.extend([
        "# Calculate and display total runtime",
        "overall_end_time=$(date +%s)",
        "overall_duration=$((overall_end_time - overall_start_time))",
        "echo \"Started at $(date -d @$overall_start_time)\"",
        "echo \"Finished at $(date -d @$overall_end_time)\"",
        "echo \"Total profiling time: $((overall_duration / 3600)) hours, $(((overall_duration % 3600) / 60)) minutes and $((overall_duration % 60)) seconds\"",
    ])
    write_lines_to_file(main_run_script, main_script_lines)
    os.chmod(main_run_script, 0o755)

    print(f"All scripts generated. To start profiling, run: bash {main_run_script}")
    print(f"Scripts are located in: {report_dir}")
    print("No profiling was executed automatically.")

if __name__ == "__main__":
    main()
