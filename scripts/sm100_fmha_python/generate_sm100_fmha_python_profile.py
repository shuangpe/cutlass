#!/usr/bin/env python3

import os
import argparse
import itertools
import datetime

batch_size = [1]
q_seq_len = [1024, 2048, 4096, 8192, 16384]
kv_seq_len = [4096, 8192, 16384]
head_num = [8, 16]
head_size = [128]
dtype = ["Float16", "Float8E4M3FN"]
is_persist = [True, False]
has_casual_mask = [True, False]

FREQ_PROFILES = [1500, 1300, 1005]

def parse_args():
    parser = argparse.ArgumentParser(description='Generate python fmha test shell script')
    parser.add_argument('-o', '--out_dir', type=str, default='', help='Output directory for generated scripts')
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID to use for testing')
    parser.add_argument('-e', '--exec_dir', type=str, default='/workspace/cutlass/examples/python/CuTeDSL/blackwell', help='Directory containing fmha.py')
    return parser.parse_args()

def generate_freq_script(scripts_dir, freq, gpu_id, exec_dir):
    script_filename = os.path.join(scripts_dir, f"run_python_fmha_{freq}mhz.sh")
    with open(script_filename, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write(f"# Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"GPU_ID={gpu_id}\n")
        f.write(f"LOCK_FREQ={freq}\n")
        f.write(f"EXEC_DIR=\"{exec_dir}\"\n")
        f.write("SCRIPT_DIR=$(dirname \"$(readlink -f \"$0\")\")\n")
        f.write("START_TIME=$(date +%s)\n")
        f.write("echo 'Starting python fmha tests at ' $(date)\n\n")
        f.write("export CUDA_VISIBLE_DEVICES=$GPU_ID\n")
        f.write("echo \"Setting GPU $GPU_ID frequency to $LOCK_FREQ MHz...\"\n")
        f.write("if nvidia-smi --id=$GPU_ID --lock-gpu-clocks=$LOCK_FREQ,$LOCK_FREQ; then\n")
        f.write("    echo \"GPU frequency locking successful\"\n")
        f.write("else\n")
        f.write("    echo \"GPU frequency locking failed, continuing with default frequency\"\n")
        f.write("fi\n\n")
        idx = 0
        for b, ql, kl, hn, hs, dt, persist, mask in itertools.product(
            batch_size, q_seq_len, kv_seq_len, head_num, head_size, dtype, is_persist, has_casual_mask
        ):
            cmd = [
                f"python3 $EXEC_DIR/fmha.py",
                f"--in_dtype {dt}",
                f"--out_dtype {dt}",
                "--qk_acc_dtype Float32",
                "--pv_acc_dtype Float32",
                "--mma_tiler_mn 128,128",
                f"--q_shape {b},{ql},{hn},{hs}",
                f"--k_shape {b},{kl},{hn},{hs}",
                "--warmup_iterations 10",
                "--iterations 30",
                "--skip_ref_check"
            ]
            if persist:
                cmd.append("--is_persistent")
            if mask:
                cmd.append("--has_casual_mask")
            cmd_str = " ".join(cmd)
            f.write("echo '================================================================================'\n")
            f.write(f"echo 'Test #{idx}'\n")
            f.write(f"echo 'Command to be executed:'\n")
            f.write(f"echo '{cmd_str}'\n")
            f.write(f"{cmd_str}\n")
            f.write("echo\n")
            idx += 1
        f.write("echo \"Resetting GPU $GPU_ID frequency to default...\"\n")
        f.write("if nvidia-smi --id=$GPU_ID --reset-gpu-clocks; then\n")
        f.write("    echo \"GPU frequency reset successful\"\n")
        f.write("else\n")
        f.write("    echo \"GPU frequency reset failed\"\n")
        f.write("fi\n\n")
        f.write("END_TIME=$(date +%s)\n")
        f.write("DURATION=$((END_TIME - START_TIME))\n")
        f.write("echo 'All python fmha tests complete!'\n")
        f.write("echo 'Total runtime: $((DURATION / 60)) minutes and $((DURATION % 60)) seconds'\n")
        f.write("exit 0\n")
    os.chmod(script_filename, 0o755)
    return script_filename

def main():
    args = parse_args()
    if not args.out_dir:
        current_date = datetime.datetime.now().strftime("%m%d")
        args.out_dir = f"{current_date}_generated_python_fmha"
    if not os.path.isabs(args.out_dir):
        args.out_dir = os.path.join(os.getcwd(), args.out_dir)
    scripts_dir = os.path.join(args.out_dir, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)

    # Generate scripts for each frequency
    freq_scripts = []
    for freq in FREQ_PROFILES:
        script_path = generate_freq_script(scripts_dir, freq, args.gpu_id, args.exec_dir)
        freq_scripts.append(script_path)

    # Generate main entry script
    main_script = os.path.join(scripts_dir, "run_python_fmha.sh")
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    logs_dir_name = f"{current_date}_sm100_fmha_python_logs"
    with open(main_script, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write(f"# Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("START_TIME=$(date +%s)\n")
        f.write("SCRIPT_DIR=$(dirname \"$(readlink -f \"$0\")\")\n")
        f.write("BASE_DIR=$(dirname \"$SCRIPT_DIR\")\n")
        f.write(f"LOGS_DIR=\"$BASE_DIR/{logs_dir_name}\"\n")
        f.write("mkdir -p \"$LOGS_DIR\"\n\n")
        f.write("echo 'Starting all python fmha frequency tests at ' $(date)\n\n")
        for freq in FREQ_PROFILES:
            freq_script = os.path.basename(f"run_python_fmha_{freq}mhz.sh")
            log_file = f"$LOGS_DIR/python_fmha_{freq}mhz.log"
            f.write(f"echo '================ {freq}MHz ================'\n")
            f.write(f"> \"{log_file}\"\n")
            f.write(f"bash ${{BASH_SOURCE%/*}}/{freq_script} > \"{log_file}\" 2>&1\n")
            f.write("echo\n")
        f.write("END_TIME=$(date +%s)\n")
        f.write("DURATION=$((END_TIME - START_TIME))\n")
        f.write("echo 'All python fmha frequency tests complete!'\n")
        f.write("echo 'Total runtime: $((DURATION / 60)) minutes and $((DURATION % 60)) seconds'\n")
        f.write("exit 0\n")
    os.chmod(main_script, 0o755)
    print(f"Python FMHA benchmark main entry shell script generated:\n  {main_script}")
    print(f"You can run all benchmarks with:\n  bash {main_script}")

if __name__ == "__main__":
    main()