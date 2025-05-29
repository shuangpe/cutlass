#!/usr/bin/env python3

import os
import argparse
import datetime
import itertools
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Generate CUTLASS test shell script')
    parser.add_argument('--q', type=int, default=1024, help='Sequence length Q parameter')
    parser.add_argument('--k', type=int, default=4096, help='Sequence length K parameter')
    parser.add_argument('--h', type=int, default=16, help='Number of heads parameter')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use for testing')
    parser.add_argument('--commented', action='store_true', default=True, help='Include commented test options (default: True)')
    parser.add_argument('--out_dir', type=str, default='', help='Output directory for generated scripts')
    parser.add_argument('--exec_dir', type=str, default='/workspace/cutlass/build/examples',
                        help='Directory containing executable files')
    return parser.parse_args()

# Fixed frequency profiles (MHz)
FREQ_PROFILES = [1500, 1300]

# Hardcoded test parameters
TEST_PARAMS = {
    "TEST_BASIC": "--b=1 --h=4 --q=512 --k=512 --d=128 --verify --mask=no",
    "TEST_CAUSAL": "--b=1 --h=4 --q=512 --k=512 --d=128 --verify --mask=causal",
    "TEST_VARLEN": "--b=1 --h=4 --q=512 --k=512 --d=128 --verify --mask=residual --varlen",
    "TEST_HDIM64": "--b=2 --h=4 --q=512 --k=512 --d=64 --verify",
    "TEST_GQA": "--b=2 --h=4 --h_k=2 --q=512 --k=512 --d=64 --verify",

    "TEST_GEN_BASIC": "--b=1 --h=4 --k=512 --d=128 --verify",
    "TEST_GEN_VARLEN": "--b=1 --h=4 --k=512 --d=128 --verify --varlen",
    "TEST_GEN_HDIM64": "--b=2 --h=4 --k=512 --d=64 --verify",
    "TEST_GEN_GQA": "--b=2 --h=4 --h_k=2 --k=512 --d=64 --verify",
    "TEST_GEN_REMAP": "--b=2 --h=4 --h_k=2 --k=512 --d=128 --verify --remap",
    "TEST_GEN_CACHEONLY": "--b=2 --h=4 --h_k=2 --k=512 --d=128 --verify --cache-only",

    "TEST_MLA_BASIC": "--b=1 --k=512 --verify"
}

# Hardcoded targets configuration
# Format: target_name, source_file, [active_options], [commented_options]
TARGETS = [
    ("77_blackwell_fmha", "77_blackwell_fmha.cu",
     ["TEST_BASIC"],
     ["TEST_CAUSAL", "TEST_VARLEN", "TEST_HDIM64", "TEST_GQA"]),

    ("77_blackwell_fmha_gen", "77_blackwell_fmha_gen.cu",
     ["TEST_GEN_BASIC"],
     ["TEST_GEN_VARLEN", "TEST_GEN_HDIM64", "TEST_GEN_GQA", "TEST_GEN_REMAP", "TEST_GEN_CACHEONLY"]),

    ("77_blackwell_mla_2sm", "77_blackwell_mla.cu",
     ["TEST_MLA_BASIC"],
     []),

    ("77_blackwell_mla_2sm_cpasync", "77_blackwell_mla.cu",
     ["TEST_MLA_BASIC"],
     []),

    ("77_blackwell_mla_b2b_2sm", "77_blackwell_mla.cu",
     ["TEST_MLA_BASIC"],
     []),

    ("77_blackwell_fmha_bwd", "77_blackwell_fmha_bwd.cu",
     ["TEST_BASIC"],
     ["TEST_GEN_VARLEN", "TEST_GEN_HDIM64", "TEST_GEN_GQA", "TEST_GEN_REMAP", "TEST_GEN_CACHEONLY"]),

    ("77_blackwell_fmha_bwd_sat", "77_blackwell_fmha_bwd.cu",
     ["TEST_BASIC", "TEST_GEN_HDIM64"],
     ["TEST_GEN_VARLEN", "TEST_GEN_GQA", "TEST_GEN_REMAP", "TEST_GEN_CACHEONLY"])
]

def replace_params(param_str, user_params):
    """Replace parameters according to user specifications, maintaining ratios"""
    params = param_str.strip().split()
    result = []

    # Extract original h and h_k values
    original_h = None
    original_h_k = None
    for param in params:
        if param.startswith('--h='):
            original_h = int(param.split('=')[1])
        elif param.startswith('--h_k='):
            original_h_k = int(param.split('=')[1])

    # Calculate h_k to h ratio if both exist
    h_k_ratio = None
    if original_h is not None and original_h_k is not None:
        h_k_ratio = original_h_k / original_h

    # Replace parameters
    for param in params:
        parts = param.split('=', 1)
        name = parts[0]

        # Handle flags without values
        if len(parts) == 1:
            result.append(param)
            continue

        value = parts[1]

        if name == '--q' and '--q' in user_params:
            param = f"{name}={user_params['--q']}"
        elif name == '--k' and '--k' in user_params:
            param = f"{name}={user_params['--k']}"
        elif name == '--h' and '--h' in user_params:
            param = f"{name}={user_params['--h']}"
            # Add h_k parameter if not present
            if original_h_k is None and '--h_k' not in [p.split('=')[0] for p in params]:
                result.append(f"--h_k={user_params['--h']}")
        elif name == '--h_k' and '--h' in user_params and h_k_ratio is not None:
            # Maintain original ratio
            new_h_k = int(user_params['--h'] * h_k_ratio)
            param = f"{name}={new_h_k}"
        elif name == '--h_k' and '--h' in user_params and h_k_ratio is None:
            # If no ratio exists, set h_k equal to h
            param = f"{name}={user_params['--h']}"

        result.append(param)

    # Add common parameters that should always be included
    result.append("--verbose")
    result.append("--iterations=100")

    return ' '.join(result)

def group_tests_by_executable(targets, include_commented, user_params):
    """Group test configurations by executable"""
    tests_by_executable = defaultdict(list)

    for target_base, _, active_options, commented_options in targets:
        options_to_test = active_options.copy()
        if include_commented:
            options_to_test.extend(commented_options)

        for option_name in options_to_test:
            if option_name in TEST_PARAMS:
                original_params = TEST_PARAMS[option_name]
                custom_params = replace_params(original_params, user_params)

                for prec in ['fp8', 'fp16']:
                    executable = f"{target_base}_{prec}"
                    tests_by_executable[executable].append((option_name, custom_params))

    return tests_by_executable

def generate_executable_script(out_dir, freq, executable, tests, gpu_id, exec_dir):
    """Generate a shell script for a specific executable at a specific frequency"""
    script_filename = os.path.join(out_dir, "scripts", f"{executable}_{freq}mhz.sh")

    # Get current date for logs directory
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    logs_dir_name = f"{current_date}_sm100_fmha_logs"

    script_lines = [
        "#!/bin/bash",
        "",
        f"# CUTLASS SM100 FMHA Test Script for {executable} at {freq}MHz",
        f"# Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "# Configuration variables",
        f"GPU_ID={gpu_id}",
        f"LOCK_FREQ={freq}",
        f"EXECUTABLE=\"{executable}\"",
        f"EXEC_DIR=\"{exec_dir}\"",
        "SCRIPT_DIR=$(dirname \"$(readlink -f \"$0\")\")",
        "BASE_DIR=$(dirname \"$SCRIPT_DIR\")",
        f"LOGS_DIR=\"$BASE_DIR/{logs_dir_name}\"",
        "",
        "# Create logs directory",
        "mkdir -p \"$LOGS_DIR\"",
        "",
        "# Start time",
        "START_TIME=$(date +%s)",
        "echo \"Starting tests for $EXECUTABLE at $(date)\"",
        "echo \"GPU ID: $GPU_ID, Frequency: $LOCK_FREQ MHz\"",
        "echo \"Logs directory: $LOGS_DIR\"",
        "echo \"Executable directory: $EXEC_DIR\"",
        "echo",
        "",
        "# Set GPU ID",
        "export CUDA_VISIBLE_DEVICES=$GPU_ID",
        "",
        "# Lock GPU frequency",
        "echo \"Setting GPU $GPU_ID frequency to $LOCK_FREQ MHz...\"",
        "if nvidia-smi --id=$GPU_ID --lock-gpu-clocks=$LOCK_FREQ,$LOCK_FREQ; then",
        "    echo \"GPU frequency locking successful\"",
        "    echo \"Current GPU settings:\"",
        "    nvidia-smi --id=$GPU_ID --query-gpu=name,clocks.gr,clocks.max.gr --format=csv",
        "else",
        "    echo \"GPU frequency locking failed, continuing with default frequency\"",
        "fi",
        "echo",
        "",
    ]

    # Generate test functions for each test configuration
    for option_name, params in tests:
        # For each test configuration, generate variations with different cache options
        cache_options = [
            "",                              # No cache options
            "--clear-cache",                 # Only clear-cache
            "--cache-only",                  # Only cache-only
            "--clear-cache --cache-only"     # Both options
        ]

        for cache_params in cache_options:
            # Skip cache-only option for tests that already have it
            if "cache-only" in params and "--cache-only" in cache_params:
                continue

            full_params = f"{params} {cache_params}".strip()

            script_lines.extend([
                f"# Test $EXECUTABLE with {option_name}",
                f"echo \"Running: $EXECUTABLE with {option_name}\"",
                f"echo \"Parameters: {full_params}\"",
                f"echo \"[{option_name}/${{LOCK_FREQ}}MHz] $EXECUTABLE {full_params}\"",
                f"\"$EXEC_DIR/$EXECUTABLE\" {full_params}",
                f"echo",
                "",
            ])

    # Reset frequency and add footer
    script_lines.extend([
        "# Reset GPU frequency",
        "echo \"Resetting GPU $GPU_ID frequency to default...\"",
        "if nvidia-smi --id=$GPU_ID --reset-gpu-clocks; then",
        "    echo \"GPU frequency reset successful\"",
        "    echo \"Current GPU settings:\"",
        "    nvidia-smi --id=$GPU_ID --query-gpu=name,clocks.gr,clocks.max.gr --format=csv",
        "else",
        "    echo \"GPU frequency reset failed\"",
        "fi",
        "echo",
        "",
        "# Test summary",
        "echo \"Testing complete for $EXECUTABLE!\"",
        "END_TIME=$(date +%s)",
        "DURATION=$((END_TIME - START_TIME))",
        "echo \"Total runtime: $((DURATION / 60)) minutes and $((DURATION % 60)) seconds\"",
        "",
        "exit 0"
    ])

    # Create scripts directory if it doesn't exist
    os.makedirs(os.path.dirname(script_filename), exist_ok=True)

    # Write to output file
    with open(script_filename, 'w') as f:
        f.write('\n'.join(script_lines))

    # Set executable permission
    os.chmod(script_filename, 0o755)

    return script_filename

def generate_freq_main_script(out_dir, freq, executable_scripts, gpu_id, exec_dir):
    """Generate a main script for a specific frequency"""
    script_filename = os.path.join(out_dir, "scripts", f"run_{freq}mhz.sh")

    # Get current date for logs directory
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    logs_dir_name = f"{current_date}_sm100_fmha_logs"

    script_lines = [
        "#!/bin/bash",
        "",
        f"# CUTLASS SM100 FMHA Main Test Script - {freq}MHz",
        f"# Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "# Configuration variables",
        f"GPU_ID={gpu_id}",
        f"LOCK_FREQ={freq}",
        f"EXEC_DIR=\"{exec_dir}\"",
        "SCRIPT_DIR=$(dirname \"$(readlink -f \"$0\")\")",
        "BASE_DIR=$(dirname \"$SCRIPT_DIR\")",
        f"LOGS_DIR=\"$BASE_DIR/{logs_dir_name}\"",
        "",
        "# Create logs directory",
        "mkdir -p \"$LOGS_DIR\"",
        "",
        "# Start time",
        "START_TIME=$(date +%s)",
        f"echo \"Starting all tests at {freq}MHz at $(date)\"",
        f"echo \"GPU ID: {gpu_id}\"",
        f"echo \"Executable directory: {exec_dir}\"",
        "echo",
        "",
    ]

    # Add calls to each executable script with output redirection to log file
    for script_path in executable_scripts:
        script_name = os.path.basename(script_path)
        executable_name = script_name.replace(f"_{freq}mhz.sh", "")
        log_file = f"$LOGS_DIR/{executable_name}_{freq}mhz.log"
        
        script_lines.extend([
            f"echo \"=========================================================================\"",
            f"echo \"Running tests for {executable_name} at {freq}MHz...\"",
            f"echo \"=========================================================================\"",
            f"echo \"Log file: {log_file}\"",
            f"# Initialize log file",
            f"> \"{log_file}\"",
            f"# Run the script and redirect output to log file",
            f"\"$SCRIPT_DIR/{script_name}\" > \"{log_file}\" 2>&1",  # Add redirection to log file
            "echo",
            "",
        ])

    # Add footer
    script_lines.extend([
        f"# {freq}MHz tests summary",
        f"echo \"All {freq}MHz tests complete!\"",
        "END_TIME=$(date +%s)",
        "DURATION=$((END_TIME - START_TIME))",
        "echo \"Total runtime: $((DURATION / 60)) minutes and $((DURATION % 60)) seconds\"",
        "",
        "exit 0"
    ])

    # Write to output file
    with open(script_filename, 'w') as f:
        f.write('\n'.join(script_lines))

    # Set executable permission
    os.chmod(script_filename, 0o755)

    return script_filename

def generate_main_script(out_dir, freq_scripts, exec_dir):
    """Generate the main run.sh script that calls frequency-specific scripts"""
    script_filename = os.path.join(out_dir, "run.sh")

    script_lines = [
        "#!/bin/bash",
        "",
        "# CUTLASS SM100 FMHA Main Test Script",
        f"# Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "# Configuration variables",
        f"EXEC_DIR=\"{exec_dir}\"",
        "SCRIPT_DIR=$(dirname \"$(readlink -f \"$0\")\")",
        "TIMESTAMP=$(date +\"%Y%m%d_%H%M%S\")",
        "",
        "# Start time",
        "START_TIME=$(date +%s)",
        "echo \"Starting all frequency tests at $(date)\"",
        "echo \"Executable directory: $EXEC_DIR\"",
        "echo \"Timestamp: $TIMESTAMP\"",
        "echo",
        "",
    ]

    # Add calls to each frequency script
    for freq_script in freq_scripts:
        script_name = os.path.basename(freq_script)
        freq = script_name.replace("run_", "").replace("mhz.sh", "")
        script_lines.extend([
            f"echo \"=========================================================================\"",
            f"echo \"Running tests at {freq}MHz...\"",
            f"echo \"=========================================================================\"",
            f"\"$SCRIPT_DIR/scripts/{script_name}\"",
            "if [ $? -ne 0 ]; then",
            f"    echo \"Warning: Tests at {freq}MHz failed\"",
            "fi",
            "echo",
            "",
        ])

    # Add summary
    script_lines.extend([
        "# All tests summary",
        "echo \"All frequency tests complete!\"",
        "END_TIME=$(date +%s)",
        "DURATION=$((END_TIME - START_TIME))",
        "echo \"Total runtime: $((DURATION / 3600)) hours, $(((DURATION % 3600) / 60)) minutes and $((DURATION % 60)) seconds\"",
        "",
        "exit 0"
    ])

    # Write to output file
    with open(script_filename, 'w') as f:
        f.write('\n'.join(script_lines))

    # Set executable permission
    os.chmod(script_filename, 0o755)

    return script_filename

def main():
    args = parse_args()

    # Set default output directory if not specified
    if not args.out_dir:
        current_date = datetime.datetime.now().strftime("%m%d")
        args.out_dir = f"{current_date}_generated_sm100_fmha"

    # Make sure output directory is absolute
    if not os.path.isabs(args.out_dir):
        # Use current working directory, not script directory
        args.out_dir = os.path.join(os.getcwd(), args.out_dir)

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "scripts"), exist_ok=True)

    # User parameters
    user_params = {
        '--q': args.q,
        '--k': args.k,
        '--h': args.h
    }

    # Group tests by executable
    tests_by_executable = group_tests_by_executable(TARGETS, args.commented, user_params)

    # Generate scripts for each frequency
    freq_scripts = []
    for freq in FREQ_PROFILES:
        # Generate executable-specific scripts for this frequency
        executable_scripts = []
        for executable, tests in tests_by_executable.items():
            script_path = generate_executable_script(args.out_dir, freq, executable, tests, args.gpu_id, args.exec_dir)
            executable_scripts.append(script_path)
            print(f"Generated script for {executable} at {freq}MHz: {script_path}")

        # Generate main script for this frequency
        freq_script = generate_freq_main_script(args.out_dir, freq,
                                             [os.path.basename(s) for s in executable_scripts],
                                             args.gpu_id, args.exec_dir)
        freq_scripts.append(freq_script)
        print(f"Generated frequency script for {freq}MHz: {freq_script}")

    # Generate main run script
    main_script = generate_main_script(args.out_dir, [os.path.basename(s) for s in freq_scripts], args.exec_dir)
    print(f"Generated main run script: {main_script}")

    print(f"All scripts generated in directory: {args.out_dir}")
    print(f"Using executables from: {args.exec_dir}")

if __name__ == "__main__":
    main()