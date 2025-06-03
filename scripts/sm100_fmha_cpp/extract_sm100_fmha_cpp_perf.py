#!/usr/bin/env python3

import os
import re
import csv
import sys

def extract_gpu_freq(filename):
    m = re.search(r'fmha_(\d+)mhz', filename)
    return int(m.group(1)) if m else None

def extract_type_from_exec(cmd):
    # e.g. $EXEC_DIR/77_blackwell_fmha_fp16 or ..._fp8
    m = re.search(r'fmha_(fp\d+)', cmd)
    return m.group(1) if m else ''

def parse_test_block(block, gpu_freq):
    # 提取命令行
    cmd_match = re.search(r'Command to be executed:\n(.+)', block)
    cmd = cmd_match.group(1) if cmd_match else ''
    TypeIn = TypeOut = extract_type_from_exec(cmd)
    # 优先从######行提取参数
    header_match = re.search(r'###### B (\d+) H (\d+) H_K (\d+) Q (\d+) K (\d+) D (\d+)[^#]*#SM (\d+)', block)
    if header_match:
        Batch = header_match.group(1)
        HeadNum = header_match.group(2)
        HeadNum_K = header_match.group(3)
        Q_SeqLen = header_match.group(4)
        KV_SeqLen = header_match.group(5)
        HeadSize = header_match.group(6)
        SMs = header_match.group(7)
    else:
        def extract_arg(pattern, text):
            m = re.search(pattern, text)
            return m.group(1) if m else ''
        Batch = extract_arg(r'--b=(\d+)', cmd)
        Q_SeqLen = extract_arg(r'--q=(\d+)', cmd)
        KV_SeqLen = extract_arg(r'--k=(\d+)', cmd)
        HeadNum = extract_arg(r'--h=(\d+)', cmd)
        HeadNum_K = extract_arg(r'--h_k=(\d+)', cmd)
        HeadSize = extract_arg(r'--d=(\d+)', cmd)
        SMs = ''
    Mask = ''
    m = re.search(r'--mask=(\w+)', cmd)
    if m:
        Mask = m.group(1)
    # 提取tma ws行，兼容任意空白符
    tma_lines = re.findall(r'\[--\] tma ws (\d+)x(\d+) acc (\w+) (persistent|individual) : ([\d.]+) TFLOPS/s\n\s*t=([\d.]+)ms', block)
    rows = []
    for CtaTile_m, CtaTile_n, acc_type, mode, tflops, avg_runtime in tma_lines:
        AccTypeQK = AccTypePV = acc_type
        IsPersistent = '1' if mode == 'persistent' else '0'
        row = [
            gpu_freq, tflops, TypeIn, TypeOut, AccTypeQK, AccTypePV, CtaTile_m, CtaTile_n,
            Batch, Q_SeqLen, KV_SeqLen, HeadNum, HeadNum_K, HeadSize, IsPersistent, Mask, avg_runtime, SMs
        ]
        rows.append(row)
    return rows

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract FMHA cpp perf logs to CSV')
    parser.add_argument('in_dir', help='Input log directory')
    parser.add_argument('out_csv_file', nargs='?', default=None, help='Output CSV file (optional)')
    args = parser.parse_args()

    log_dir = args.in_dir
    if args.out_csv_file:
        out_csv = args.out_csv_file
    else:
        out_csv = os.path.basename(os.path.normpath(log_dir)) + '.combined.csv'

    CSV_HEADER = [
        'GPUFreq', 'TFLOPS', 'TypeIn', 'TypeOut', 'AccTypeQK', 'AccTypePV', 'CtaTile_m', 'CtaTile_n',
        'Batch', 'Q_SeqLen', 'KV_SeqLen', 'HeadNum', 'HeadNum_K', 'HeadSize', 'IsPersistent', 'Mask', 'AvgRuntime (ms)', 'SMs'
    ]

    rows = []
    for fname in os.listdir(log_dir):
        if not fname.endswith('.log'):
            continue
        gpu_freq = extract_gpu_freq(fname)
        with open(os.path.join(log_dir, fname), 'r') as f:
            content = f.read()
        # 兼容无Test #的情况，按Test #分割但每个block都处理
        tests = re.split(r'(?=Test #\d+)', content)
        for block in tests:
            block_rows = parse_test_block(block, gpu_freq)
            rows.extend(block_rows)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)
    print(f"[INFO] Extraction finished. Output CSV: {os.path.abspath(out_csv)}. Total records: {len(rows)}")

if __name__ == '__main__':
    main()
