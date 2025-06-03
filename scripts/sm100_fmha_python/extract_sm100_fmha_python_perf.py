#!/usr/bin/env python3

import os
import re
import csv
import sys

def extract_gpu_freq(filename):
    m = re.search(r'fmha_(\d+)mhz', filename)
    return int(m.group(1)) if m else None

def parse_test_block(block):
    # Extract all fields using regex
    def extract(pattern, text, group=1, default=None, type_cast=str):
        m = re.search(pattern, text)
        if m:
            return type_cast(m.group(group))
        return default

    # Mapping for dtype and acc type
    dtype_map = {
        'Float16': 'fp16',
        'Float8E4M3FN': 'fp8',
        'Float32': 'fp32',
        'BFloat16': 'bf16',
    }
    acc_map = {
        'Float32': 'fp32',
        'Float16': 'fp16',
        'BFloat16': 'bf16',
    }
    TFLOPS = extract(r'TFLOPS: ([\d.]+)', block, 1, None, float)
    if TFLOPS is None:
        raise ValueError('TFLOPS not found in block')
    raw_typein = extract(r'in_dtype: (\w+)', block, 1, None)
    if raw_typein is None:
        raise ValueError('in_dtype not found in block')
    TypeIn = dtype_map.get(raw_typein)
    if TypeIn is None:
        raise ValueError(f'Unknown in_dtype: {raw_typein}')
    raw_typeout = extract(r'out_dtype: (\w+)', block, 1, None)
    if raw_typeout is None:
        raise ValueError('out_dtype not found in block')
    TypeOut = dtype_map.get(raw_typeout)
    if TypeOut is None:
        raise ValueError(f'Unknown out_dtype: {raw_typeout}')
    raw_accqk = extract(r'qk_acc_dtype: (\w+)', block, 1, None)
    if raw_accqk is None:
        raise ValueError('qk_acc_dtype not found in block')
    AccTypeQK = acc_map.get(raw_accqk)
    if AccTypeQK is None:
        raise ValueError(f'Unknown qk_acc_dtype: {raw_accqk}')
    raw_accpv = extract(r'pv_acc_dtype: (\w+)', block, 1, None)
    if raw_accpv is None:
        raise ValueError('pv_acc_dtype not found in block')
    AccTypePV = acc_map.get(raw_accpv)
    if AccTypePV is None:
        raise ValueError(f'Unknown pv_acc_dtype: {raw_accpv}')
    CtaTile_m = CtaTile_n = ''
    m = re.search(r'mma_tiler_mn: \((\d+), (\d+)\)', block)
    if m:
        CtaTile_m, CtaTile_n = m.group(1), m.group(2)
    Batch = Q_SeqLen = HeadNum = HeadSize = ''
    m = re.search(r'q_shape: \((\d+), (\d+), (\d+), (\d+)\)', block)
    if m:
        Batch, Q_SeqLen, HeadNum, HeadSize = m.group(1), m.group(2), m.group(3), m.group(4)
    KV_SeqLen = HeadNum_K = ''
    m = re.search(r'k_shape: \((\d+), (\d+), (\d+), (\d+)\)', block)
    if m:
        KV_SeqLen, HeadNum_K = m.group(2), m.group(3)
    IsPersistent = extract(r'is_persistent: (True|False)', block)
    Mask = extract(r'has_casual_mask: (True|False)', block)
    if Mask == 'True':
        Mask = 'causal'
    elif Mask == 'False':
        Mask = 'no'
    FLOPS = extract(r'flops\(final\)=([\d.]+)', block, 1, '', float)
    FLOPS = int(float(FLOPS)) if FLOPS != '' else ''
    AvgRuntime = extract(r'Average runtime: ([\d.]+) ms', block, 1, '', float)
    return [
        '', TFLOPS, TypeIn, TypeOut, AccTypeQK, AccTypePV, CtaTile_m, CtaTile_n,
        Batch, Q_SeqLen, KV_SeqLen, HeadNum, HeadNum_K, HeadSize, IsPersistent, Mask, FLOPS, AvgRuntime
    ]

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract FMHA python perf logs to CSV')
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
        'Batch', 'Q_SeqLen', 'KV_SeqLen', 'HeadNum', 'HeadNum_K', 'HeadSize', 'IsPersistent', 'Mask', 'FLOPS', 'AvgRuntime (ms)'
    ]

    rows = []
    for fname in os.listdir(log_dir):
        if not fname.endswith('.log'):
            continue
        gpu_freq = extract_gpu_freq(fname)
        with open(os.path.join(log_dir, fname), 'r') as f:
            content = f.read()
        tests = re.split(r'(?=Test #\d+)', content)
        for block in tests:
            if 'Test #' not in block:
                continue
            row = parse_test_block(block)
            row[0] = gpu_freq
            rows.append(row)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)
    print(f"[INFO] Extraction finished. Output CSV: {os.path.abspath(out_csv)}. Total records: {len(rows)}")

if __name__ == '__main__':
    main()
