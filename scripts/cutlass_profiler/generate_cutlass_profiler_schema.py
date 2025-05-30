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
default_freq_profiles = [(1500, 1500), (1300, 1300)]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate cutlass_profiler profiling scripts")
    parser.add_argument("-g", "--gpu_id", type=int, required=True, help="GPU ID (required)")
    parser.add_argument("-d", "--dry_run", action="store_true", help="Dry run mode: only print commands")
    parser.add_argument("-o", "--out_dir", type=str, default=None, help="Output directory for generated scripts")
    parser.add_argument("--operations", nargs="*", default=None, help="Operations to test (default: Gemm BlockScaledGemm)")
    parser.add_argument("--freqs", nargs="*", type=int, default=None, help="Frequency profiles, e.g. 1500 1300 (applies to both min/max)")
    return parser.parse_args()


def validate_gpu_id(gpu_id):
    import subprocess
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"
        ], capture_output=True, text=True, check=True)
        gpu_count = int(result.stdout.strip())
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


def generate_freq_script(script_path, gpu_id, min_freq, max_freq, operation, report_dir, csv_dir):
    """Generate a shell script for a specific freq/operation (no save_log_to_file, always print to console)"""
    lines = [
        "#!/bin/bash",
        f"# Command to reproduce profiling at {max_freq}MHz for GPU ID {gpu_id} with operation {operation}",
        "",
        "set -e",
        f"gpu_id={gpu_id}",
        f"min_freq={min_freq}",
        f"max_freq={max_freq}",
        f"operation=\"{operation}\"",
        "ORIGINAL_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}",
        "export CUDA_VISIBLE_DEVICES=$gpu_id",
        f"mkdir -p \"{report_dir}/logs\"",
        f"log_filename=\"{report_dir}/logs/profile-${{operation}}-${{max_freq}}mhz-gpu${{gpu_id}}.log\"",
        "start_time=$(date +%s)",
        "echo \"Started at $(date)\"",
        "",
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
        "",
        "# Run profiler",
        "echo \"Running profiler for operation $operation with GPU $gpu_id at frequency range $min_freq MHz ~ $max_freq MHz...\"",
        f"/workspace/cutlass/build/tools/profiler/cutlass_profiler \\",
        "  --operation=$operation \\",
        "  --profiling-iterations=100 --warmup-iterations=10 \\",
        "  --m=16384 --n=16384 --k=256,512,1024,2048,4096,8192,16384 \\",
        "  --providers=cutlass --dist=uniform,min:-5,max:5 \\",
        f"  --output=\"{csv_dir}/profile-${{operation}}-${{max_freq}}mhz-gpu${{gpu_id}}.csv\"",
        "",
        "# Reset GPU frequency",
        "echo \"Resetting GPU $gpu_id frequency to default...\"",
        "if nvidia-smi --id=\"$gpu_id\" --reset-gpu-clocks; then",
        "  echo \"GPU frequency reset successful\"",
        "  echo \"Current GPU settings:\"",
        "  nvidia-smi --id=\"$gpu_id\" --query-gpu=name,clocks.gr,clocks.max.gr --format=csv",
        "else",
        "  echo \"GPU frequency reset failed\"",
        "fi",
        "end_time=$(date +%s)",
        "duration=$((end_time - start_time))",
        "echo \"Finished at $(date)\"",
        "echo \"Total runtime: $((duration / 60)) minutes and $((duration % 60)) seconds\"",
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

    # Main run script
    main_run_script = os.path.join(report_dir, "run.sh")
    main_script_lines = [
        "#!/bin/bash",
        f"# Script to reproduce all profiling runs from {current_date} for GPU ID {gpu_id}",
        "overall_start_time=$(date +%s)",
        f"echo \"Starting profiling runs for GPU ID: {gpu_id} at $(date)\"",
        ""
    ]

    # Generate freq/operation scripts and add to main script
    for operation in operations:
        operation_lower = operation.lower()
        for min_freq, max_freq in freq_profiles:
            freq_run_script = os.path.join(scripts_dir, f"profile_{operation_lower}_{max_freq}mhz_gpu{gpu_id}.sh")
            generate_freq_script(freq_run_script, gpu_id, min_freq, max_freq, operation, report_dir, csv_dir)
            # 主脚本调用子脚本并重定向输出到log文件
            main_script_lines.extend([
                f"echo \"Running profile for {operation} at {max_freq}MHz...\"",
                f'"{freq_run_script}" > "{report_dir}/logs/profile-{operation_lower}-{max_freq}mhz-gpu{gpu_id}.log" 2>&1',
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
