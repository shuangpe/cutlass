#!/usr/bin/env python3

import csv
import re
import sys
import os
import glob

def process_csv(input_file, all_data=None):
    """Process CUTLASS performance CSV file and extract relevant data"""
    # Extract GPU frequency from filename
    freq_match = re.search(r'(\d+)mhz', input_file.lower())
    gpu_freq = freq_match.group(1) if freq_match else "unknown"

    columns_to_keep = ['Operation', 'm', 'n', 'k', 'Bytes', 'Flops', 'Flops/Byte', 'Runtime', 'GB/s']

    if all_data is None:
        all_data = []

    with open(input_file, 'r') as f:
        reader = csv.DictReader(f.readlines())

    for row in reader:
        filtered_row = {col: row[col] for col in columns_to_keep if col in row}

        if gpu_freq != "unknown":
            filtered_row['GPUFreq'] = gpu_freq

        if 'GFLOPs' not in row:
            raise ValueError(f"GFLOPs column missing in row: {row}")

        gflops = float(row['GFLOPs'])
        filtered_row['TFLOPs'] = str(gflops / 1000)

        operation = row['Operation']
        datatypes_match = re.search(r'tensorop_gemm_(.*?)_\d+x\d+x\d+', operation)
        filtered_row['DataTypes'] = datatypes_match.group(1) if datatypes_match else 'unknown'
        config_match = re.search(r'(\d+x\d+x\d+.*)', operation)
        filtered_row['Config'] = config_match.group(1) if config_match else 'unknown'

        all_data.append(filtered_row)

    output_columns = ['DataTypes', 'Config', 'TFLOPs'] + columns_to_keep
    if all_data and 'GPUFreq' in all_data[0]:
        output_columns = ['GPUFreq'] + output_columns

    return all_data, output_columns

def write_csv_file(output_file, column_names, data):
    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=column_names)
            writer.writeheader()
            writer.writerows(data)
        print(f"Results saved as: {output_file}")
        return True
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")
        return False

def get_default_output_path(input_path):
    """Generate default output path based on input path"""
    if os.path.isfile(input_path):
        base_name = os.path.basename(input_path)
        return os.path.join(os.path.dirname(input_path), f"revised_{base_name}")
    else:
        directory_path = input_path.rstrip('/')
        return f"{directory_path}_combined_results.csv"

def process_directory(input_path):
    """Process all CSV files in a directory and combine results"""
    all_data = []
    output_columns = []
    files_processed = 0

    csv_files = glob.glob(os.path.join(input_path, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_path}")
        exit(1)

    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        all_data, output_columns = process_csv(csv_file, all_data)
        files_processed += 1

    print(f"Processed {files_processed} file(s)")
    return all_data, output_columns

def main():
    if len(sys.argv) < 2:
        print("Usage: python revise_perf_csv.py <input_path> [output_csv_file]")
        print("       <input_path> can be a single CSV file or a directory containing CSV files")
        sys.exit(1)

    input_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if output_file is None:
        output_file = get_default_output_path(input_path)

    if os.path.isfile(input_path):
        data, columns = process_csv(input_path)
    else:
        data, columns = process_directory(input_path)

    if data:
        write_csv_file(output_file, columns, data)

if __name__ == "__main__":
    main()
