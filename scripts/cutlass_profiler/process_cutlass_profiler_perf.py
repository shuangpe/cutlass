#!/usr/bin/env python3

import csv
import re
import sys
import os
import glob
import logging
import argparse
import collections
from typing import List, Dict, Tuple, Optional, Any

# Define constants

def setup_logging(verbose: bool = False) -> None:
    """Set up logging level and format"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=level)

def extract_gpu_frequency(filename: str) -> str:
    """Extract GPU frequency from filename"""
    freq_match = re.search(r'(\d+)mhz', filename.lower())
    return freq_match.group(1) if freq_match else "oob"

def extract_gemmkind(filename: str) -> str:
    """Extract GemmKind information from filename (was MMAOP)"""
    base_name = os.path.basename(filename)
    gemmkind_match = re.search(r'\.([^.]+)\.csv$', base_name)
    return gemmkind_match.group(1) if gemmkind_match else "unknown"

def extract_datatypes(operation: str) -> str:
    """Extract data types information from operation name"""
    datatypes_match = re.search(r'tensorop_gemm_(.*?)_\d+x\d+x\d+', operation)
    datatypes = datatypes_match.group(1) if datatypes_match else 'unknown'
    return datatypes

def extract_stream_k(operation: str) -> str:
    """Extract stream_k information from operation name"""
    return "Y" if 'stream_k' in operation else "N"

def process_csv_row(row: Dict[str, str], gpu_freq: str, gemmkind: str) -> Dict[str, str]:
    """Process a single row of CSV data, adding shape columns to the left and keeping all columns"""
    # Add composite columns
    problem_shape = f"{row.get('m', '')}x{row.get('n', '')}x{row.get('k', '')}"
    cta_shape = f"{row.get('cta_m', '')}x{row.get('cta_n', '')}x{row.get('cta_k', '')}"
    cluster_shape = f"{row.get('cluster_m', '')}x{row.get('cluster_n', '')}x{row.get('cluster_k', '')}"
    warps_shape = f"{row.get('warps_m', '')}x{row.get('warps_n', '')}x{row.get('warps_k', '')}"
    instruct_shape = f"{row.get('inst_m', '')}x{row.get('inst_n', '')}x{row.get('inst_k', '')}"

    # Keep all original columns
    filtered_row = dict(row)

    if gemmkind != "unknown":
        filtered_row['GemmKind'] = gemmkind

    gflops = float(row['GFLOPs'])
    datatypes = extract_datatypes( row['Operation'])
    stream_k = extract_stream_k(row['Operation'])

    # New columns are placed at the left, followed by the original columns
    new_row = {
        'TFLOPs': str(gflops / 1000),  # Convert GFLOPs to TFLOPs
        'GPUFreq': gpu_freq,
        'DataTypes': datatypes,
        'ProblemShape': problem_shape,
        'CtaShape': cta_shape,
        'ClusterShape': cluster_shape,
        'WarpsShape': warps_shape,
        'InstructShape': instruct_shape,
        'StreamK': stream_k
    }
    new_row.update(filtered_row)
    return new_row

def process_csv(input_file: str, all_data: Optional[List[Dict[str, str]]] = None) -> Tuple[List[Dict[str, str]], List[str]]:
    """Process CUTLASS performance CSV file and extract relevant data"""
    if all_data is None:
        all_data = []

    # Extract file metadata
    gpu_freq = extract_gpu_frequency(input_file)
    gemmkind = extract_gemmkind(input_file)

    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()

        # Check if the file only has a header row
        if len(lines) <= 1:
            logging.warning(f"{input_file} only has a header row, skipping this file")
            return all_data, []

        reader = csv.DictReader(lines)
        row_count = 0

        for row in reader:
            row_count += 1
            try:
                filtered_row = process_csv_row(row, gpu_freq, gemmkind)
                all_data.append(filtered_row)
            except ValueError as e:
                logging.error(f"Error processing row: {e}")

        # Check if there are no actual data rows
        if row_count == 0:
            logging.warning(f"{input_file} has no valid data rows, skipping this file")
            return all_data, []

        # Prepare output columns
        output_columns = generate_output_columns(all_data)
        return all_data, output_columns

    except Exception as e:
        logging.error(f"Error processing file {input_file}: {e}")
        return all_data, []

def generate_output_columns(all_data: List[Dict[str, str]]) -> List[str]:
    """Generate column names for output CSV, with new columns at the left, all original columns preserved"""
    if not all_data:
        return []
    # New column order
    new_columns = [
        'TFLOPs', 'GPUFreq', 'DataTypes',
        'ProblemShape', 'CtaShape', 'ClusterShape', 'WarpsShape', 'InstructShape', 'StreamK', 'GemmKind'
    ]
    # Collect all encountered fields
    all_keys = []
    for row in all_data:
        for k in row.keys():
            if k not in all_keys:
                all_keys.append(k)
    # New columns first, then the rest of the original fields in order
    output_columns = [col for col in new_columns if col in all_keys]
    output_columns += [k for k in all_keys if k not in output_columns]
    return output_columns

def write_csv_file(output_file: str, column_names: List[str], data: List[Dict[str, str]]) -> bool:
    """Write processed data to a CSV file, ensure all rows have all columns filled"""
    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=column_names)
            writer.writeheader()
            for row in data:
                # Fill all columns
                full_row = {col: row.get(col, "") for col in column_names}
                writer.writerow(full_row)
        return True
    except Exception as e:
        logging.error(f"Error writing to {output_file}: {e}")
        return False

def get_default_output_path(input_path: str) -> str:
    """Generate default output path based on input path"""
    if os.path.isfile(input_path):
        base_name = os.path.basename(input_path)
        return os.path.join(os.path.dirname(input_path), f"revised_{base_name}")
    else:
        directory_path = input_path.rstrip('/')
        return f"{directory_path}_combined_results.csv"

def scan_and_group_csv_files(input_path: str) -> Tuple[Dict[int, List[str]], List[str], Dict[int, List[str]]]:
    """Scan all csv files, group by column names, return group dict, empty file list, and columns for each group"""
    csv_files = glob.glob(os.path.join(input_path, "*.csv"))
    group_dict = collections.defaultdict(list)  # key: group_id, value: list of files
    group_columns = {}  # key: group_id, value: columns
    empty_csv = []
    columns_to_group = {}
    for csv_file in csv_files:
        try:
            with open(csv_file, 'r') as f:
                lines = f.readlines()
            if len(lines) <= 1:
                empty_csv.append(csv_file)
                continue
            reader = csv.DictReader(lines)
            columns = tuple(reader.fieldnames)
            if columns not in columns_to_group:
                group_id = len(columns_to_group) + 1
                columns_to_group[columns] = group_id
                group_columns[group_id] = list(columns)
            group_id = columns_to_group[columns]
            group_dict[group_id].append(csv_file)
        except Exception:
            empty_csv.append(csv_file)
    return group_dict, empty_csv, group_columns

def process_grouped_csv_files(group_dict: Dict[int, List[str]], group_columns: Dict[int, List[str]]) -> Dict[int, Tuple[List[Dict[str, str]], List[str]]]:
    """Process csv files by group, return merged data and columns (including new columns) for each group"""
    group_data = {}
    for group_id, files in group_dict.items():
        all_data = []
        for csv_file in files:
            gpu_freq = extract_gpu_frequency(csv_file)
            gemmkind = extract_gemmkind(csv_file)
            try:
                with open(csv_file, 'r') as f:
                    lines = f.readlines()
                reader = csv.DictReader(lines)
                for row in reader:
                    try:
                        filtered_row = process_csv_row(row, gpu_freq, gemmkind)
                        all_data.append(filtered_row)
                    except Exception:
                        continue
            except Exception:
                continue
        # Use all data rows to generate the final column names, ensuring order and content match
        output_columns = generate_output_columns(all_data)
        group_data[group_id] = (all_data, output_columns)
    return group_data

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process CUTLASS performance CSV files')
    parser.add_argument('input_path', help='Input CSV file or directory containing CSV files')
    parser.add_argument('-o', '--output', help='Path for output CSV file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    return parser.parse_args()

def main() -> None:
    """Main function"""
    args = parse_arguments()
    setup_logging(args.verbose)

    input_path = args.input_path
    output_file = args.output or get_default_output_path(input_path)

    if os.path.isfile(input_path):
        data, columns = process_csv(input_path)
    else:
        # Output files are placed in the current directory, named as "input_dirname_combined_xx.csv"
        input_dir_name = os.path.basename(os.path.abspath(input_path.rstrip('/')))
        group_dict, empty_csv, group_columns = scan_and_group_csv_files(input_path)
        group_data = process_grouped_csv_files(group_dict, group_columns)
        combined_files = {}
        for group_id, (data, columns) in group_data.items():
            output_file = f"{input_dir_name}_combined_{group_id}.csv"
            write_csv_file(output_file, columns, data)
            combined_files[output_file] = group_dict[group_id]
        print("\n===== Processing Results List =====")
        print("\n[Empty CSV files]")
        for f in empty_csv:
            print(f)
        print("\n[Combined files and their source CSVs]")
        for out, srcs in combined_files.items():
            print(f"{out} <- ")
            for s in srcs:
                print(f"    {s}")
        return

    if data:
        write_csv_file(output_file, columns, data)
    else:
        logging.error("No data to write, no output file was generated")

if __name__ == "__main__":
    main()
