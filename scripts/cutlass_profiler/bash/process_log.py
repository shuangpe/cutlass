#!/usr/bin/env python3

import os
import sys
import re
import csv
import subprocess
import argparse
from pathlib import Path

# ====================== HELPER FUNCTIONS ======================

def _safe_convert_for_sort(value):
    """
    Helper function to safely convert values for sorting.
    
    Args:
        value: The value to convert for sorting
        
    Returns:
        Converted value suitable for sorting
    """
    # Special case for 'oob' frequency
    if value == 'oob':
        return float('inf')  # Make 'oob' sort last

    # Try numeric conversion
    try:
        return float(value)
    except (ValueError, TypeError):
        # If not numeric, return as string
        return str(value)

# ====================== FILE DISCOVERY FUNCTIONS ======================

def find_log_pairs(directory_path, filters=None):
    """
    Traverse the directory to find all _nvsmi.txt files and their corresponding *gemm*.csv files.

    Args:
        directory_path (str): Log files directory path
        filters (list, optional): List of filter keywords

    Returns:
        list: List of tuples (nvsmi_file_path, gemm_file_path)
    """
    # Make sure the directory path is absolute
    directory_path = os.path.abspath(directory_path)

    # Get all files in the directory at once
    all_files = os.listdir(directory_path)

    # Filter nvsmi files and perf files
    nvsmi_files = [f for f in all_files if f.endswith("_nvsmi.txt")]
    perf_files = [f for f in all_files if "gemm" in f and f.endswith(".csv")]

    # Apply filters if specified
    if filters:
        nvsmi_files = [f for f in nvsmi_files if any(keyword in f for keyword in filters)]

    result_pairs = []

    for nvsmi_file in nvsmi_files:
        # Extract file name prefix (remove _nvsmi.txt part)
        file_prefix = nvsmi_file.replace("_nvsmi.txt", "")

        # Use regular expression to find matching perf files
        pattern = re.compile(f"^{re.escape(file_prefix)}.*gemm.*\\.csv$")
        matching_perf_files = [f for f in perf_files if pattern.match(f)]

        # If no corresponding perf file is found, raise an exception
        if not matching_perf_files:
            raise FileNotFoundError(f"Cannot find matching gemm file for {nvsmi_file}")

        # Take the first matching perf file
        perf_file = matching_perf_files[0]

        # Convert to absolute path
        nvsmi_abs_path = os.path.abspath(os.path.join(directory_path, nvsmi_file))
        perf_abs_path = os.path.abspath(os.path.join(directory_path, perf_file))

        # Add to result list
        result_pairs.append((nvsmi_abs_path, perf_abs_path))

    return result_pairs

def group_log_pairs_by_data_type(log_pairs):
    """
    Group log file pairs by data type extracted from the file name.

    Args:
        log_pairs (list): List of tuples (nvsmi_file_path, gemm_file_path)

    Returns:
        dict: Key is data type string (TA_TB_TAcc_TC_TD), value is the list of log file pairs for that type
    """
    grouped_pairs = {}

    for nvsmi_file, gemm_file in log_pairs:
        # Extract base name from file name (remove path)
        nvsmi_basename = os.path.basename(nvsmi_file)

        # Remove _nvsmi.txt suffix
        name_without_suffix = nvsmi_basename.replace("_nvsmi.txt", "")

        # Split by underscore
        parts = name_without_suffix.split('_')

        # Find the boundary of the data type part by looking for NxNxN pattern
        data_type_end_idx = None
        for i, part in enumerate(parts):
            if re.match(r'\d+x\d+x\d+', part):
                data_type_end_idx = i
                break

        if data_type_end_idx is None:
            # If no size part is found, use the default method (assume first 5 parts)
            data_type = '_'.join(parts[:5])
        else:
            # Size part found, data type is all parts before it
            data_type = '_'.join(parts[:data_type_end_idx])

        # Add file pairs to the corresponding group
        if data_type not in grouped_pairs:
            grouped_pairs[data_type] = []

        grouped_pairs[data_type].append((nvsmi_file, gemm_file))

    return grouped_pairs

# ====================== DATA PROCESSING FUNCTIONS ======================

def get_combined_headers(nvsmi_file, perf_file, parse_script_path):
    """
    Get combined CSV headers.

    Args:
        nvsmi_file (str): NVSMI log file path
        perf_file (str): Performance CSV file path
        parse_script_path (str): parse_nvsim_log.py script path

    Returns:
        list: Combined header list, or None if failed
    """
    # Get statistics headers from nvsmi file
    cmd_headers = ["python3", parse_script_path, nvsmi_file, "--csv-headers", "--stats-type", "meanStable"]
    try:
        nvsmi_headers = subprocess.check_output(cmd_headers, universal_newlines=True).strip()
    except subprocess.CalledProcessError as e:
        print(f"Error: Cannot get headers from {nvsmi_file}: {e}")
        return None

    # Read performance file headers
    try:
        with open(perf_file, 'r', newline='') as f:
            reader = csv.reader(f)
            perf_headers = next(reader)  # Read header row
    except Exception as e:
        print(f"Error: Cannot read {perf_file}: {e}")
        return None

    # Merge headers
    return perf_headers + nvsmi_headers.split(',')


def get_combined_data(nvsmi_file, perf_file, parse_script_path):
    """
    Get combined CSV data.

    Args:
        nvsmi_file (str): NVSMI log file path
        perf_file (str): Performance CSV file path
        parse_script_path (str): parse_nvsim_log.py script path

    Returns:
        list: Combined data list, or None if failed
    """
    # Get statistics data from nvsmi file
    cmd_data = ["python3", parse_script_path, nvsmi_file, "--csv", "--stats-type", "meanStable"]
    try:
        nvsmi_data = subprocess.check_output(cmd_data, universal_newlines=True).strip()
    except subprocess.CalledProcessError as e:
        print(f"Error: Cannot get data from {nvsmi_file}: {e}")
        return None

    # Read performance file data
    try:
        with open(perf_file, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row
            perf_data = next(reader, None)  # Read first data row
            if not perf_data:
                print(f"Warning: No data rows in {perf_file}")
                return None
    except Exception as e:
        print(f"Error: Cannot read {perf_file}: {e}")
        return None

    # Merge data
    return perf_data + nvsmi_data.split(',')


def reorder_and_enhance_data(headers, data_rows):
    """
    Reorder columns, add TFLOPsPerWatt column, and sort data rows.

    Args:
        headers (list): CSV header list
        data_rows (list): CSV data rows list

    Returns:
        tuple: (reordered_headers, reordered_data_rows)
    """
    # Define new column order
    new_order = [
        "Freq", "Kernel", "Hacking", "MaskRatio", "TFLOPs", "TFLOPsPerWatt",
        "ScopeMin", "ScopeMax", "WarmupIter", "ProfileIter", "GPUAvgPowerMeanstable(W)", "MemAvgPowerMeanstable(W)",
        "GPUInstPowerMeanstable(W)", "GPUTempMeanstable(C)", "MemTempMeanstable(C)",
        "SMClocksMeanstable(MHz)", "MemClocksMeanstable(MHz)", "GPUUtilMeanstable(%)",
        "MemUtilMeanstable(%)"
    ]

    # Get original column indices
    orig_indices = {header: i for i, header in enumerate(headers)}

    # Create new headers list
    new_headers = []
    for col in new_order:
        if col == "TFLOPsPerWatt":  # New column
            new_headers.append(col)
        elif col in orig_indices:
            new_headers.append(col)

    # Create new data rows with reordered columns and TFLOPsPerWatt calculation
    new_data_rows = []
    for row in data_rows:
        new_row = []

        for col in new_order:
            if col == "TFLOPsPerWatt":
                # Calculate TFLOPsPerWatt = TFLOPs / (GPU Power - Memory Power)
                if "TFLOPs" in orig_indices and "GPUAvgPowerMeanstable(W)" in orig_indices and "MemAvgPowerMeanstable(W)" in orig_indices:
                    tflops = float(row[orig_indices["TFLOPs"]])
                    gpu_power = float(row[orig_indices["GPUAvgPowerMeanstable(W)"]])
                    mem_power = float(row[orig_indices["MemAvgPowerMeanstable(W)"]])

                    power_diff = gpu_power - mem_power
                    if power_diff > 0:
                        tflops_per_watt = tflops / power_diff
                    else:
                        tflops_per_watt = 0  # Avoid division by zero or negative

                    new_row.append(f"{tflops_per_watt:.6f}")
                else:
                    new_row.append("0")
            elif col in orig_indices:
                new_row.append(row[orig_indices[col]])

        new_data_rows.append(new_row)

    # Sort data by Freq, Hacking, ScopeMax, MaskRatio
    sort_keys = ["Freq", "Hacking", "ScopeMax", "MaskRatio"]
    sort_indices = []

    for key in sort_keys:
        if key in new_headers:
            sort_indices.append(new_headers.index(key))

    if sort_indices:
        # Use safer sorting with _safe_convert_for_sort
        new_data_rows_sorted = sorted(new_data_rows, key=lambda row: [
            _safe_convert_for_sort(row[idx]) for idx in sort_indices
        ])
        return new_headers, new_data_rows_sorted

    return new_headers, new_data_rows

def process_with_pandas(input_csv, data_type, directory_path, headers=None, combined_records=None):
    """
    Process the combined CSV file using pandas.

    Args:
        input_csv (str): Input CSV file path
        data_type (str): Data type string for output file naming
        directory_path (str): Directory path for output
        headers (list, optional): Original headers for fallback processing
        combined_records (list, optional): Original data for fallback processing
    """
    try:
        import pandas as pd

        # Skip pandas CSV reading which has issues, use the records directly
        if not headers or not combined_records:
            print("Error: Missing headers or data, falling back to original processing method.")
            return False

        # Create DataFrame directly from our data
        df = pd.DataFrame(combined_records, columns=headers)

        # Calculate TFLOPsPerWatt
        if "TFLOPs" in df.columns and "GPUAvgPowerMeanstable(W)" in df.columns and "MemAvgPowerMeanstable(W)" in df.columns:
            # Avoid division by zero or negative values
            df["TFLOPsPerWatt"] = df.apply(
                lambda row: float(row["TFLOPs"]) / (float(row["GPUAvgPowerMeanstable(W)"]) - float(row["MemAvgPowerMeanstable(W)"]))
                if pd.to_numeric(row["GPUAvgPowerMeanstable(W)"], errors='coerce') - pd.to_numeric(row["MemAvgPowerMeanstable(W)"], errors='coerce') > 0
                else 0,
                axis=1
            )
        else:
            df["TFLOPsPerWatt"] = 0

        # Define the new column order
        new_columns = [
            "Freq", "Kernel", "Hacking", "MaskRatio", "TFLOPs", "TFLOPsPerWatt",
            "ScopeMin", "ScopeMax", "GPUAvgPowerMeanstable(W)", "MemAvgPowerMeanstable(W)",
            "GPUInstPowerMeanstable(W)", "GPUTempMeanstable(C)", "MemTempMeanstable(C)",
            "SMClocksMeanstable(MHz)", "MemClocksMeanstable(MHz)", "GPUUtilMeanstable(%)",
            "MemUtilMeanstable(%)"
        ]

        # Only keep columns that exist in the DataFrame
        columns_to_use = [col for col in new_columns if col in df.columns]

        # Reorder columns
        df = df[columns_to_use]

        # Sort by Freq, Hacking, ScopeMax, MaskRatio
        sort_columns = []
        for col in ["Freq", "Hacking", "ScopeMax", "MaskRatio"]:
            if col in df.columns:
                sort_columns.append(col)

        if sort_columns:
            # Convert string columns to numeric if possible (for proper sorting)
            for col in sort_columns:
                df[col] = pd.to_numeric(df[col], errors='ignore')

            df = df.sort_values(by=sort_columns)

        # Save the processed file
        output_csv = os.path.join(os.path.abspath(directory_path), f"combined_results_{data_type}.csv")
        df.to_csv(output_csv, index=False)
        print(f"Processed results saved to: {output_csv}")
        return True

    except Exception as e:
        print(f"Error processing with pandas: {e}")
        if headers and combined_records:
            print("Falling back to original processing method.")
            fallback_process(headers, combined_records, data_type, directory_path)
        else:
            print("Error: Missing headers or data for fallback processing.")
        return False

def fallback_process(headers, combined_records, data_type, directory_path):
    """
    Fallback processing method when pandas is not available or fails.

    Args:
        headers (list): CSV header list
        combined_records (list): CSV data rows list
        data_type (str): Data type for file naming
        directory_path (str): Output directory path
    """
    reordered_headers, reordered_records = reorder_and_enhance_data(headers, combined_records)
    type_result_csv = os.path.join(os.path.abspath(directory_path), f"combined_results_{data_type}.csv")
    save_csv(type_result_csv, reordered_headers, reordered_records)

# ====================== FILE I/O FUNCTIONS ======================

def save_csv(output_file, headers, data_rows):
    """
    Save CSV file.

    Args:
        output_file (str): Output CSV file path
        headers (list): CSV header list
        data_rows (list): CSV data rows list

    Returns:
        bool: True if saved successfully, otherwise False
    """
    if not headers or not data_rows:
        print("No valid records to write")
        return False

    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(data_rows)
        print(f"Merge results saved to: {output_file}")
        return True
    except Exception as e:
        print(f"Error: Cannot write result file: {e}")
        return False

# ====================== MAIN FUNCTION ======================

def main():
    """Main function for processing log files."""
    parser = argparse.ArgumentParser(description="Process log files.")
    parser.add_argument("directory_path", help="Log files directory path")
    parser.add_argument("--filter", help="Filter keywords separated by semicolons (e.g., 'keyword1;keyword2')", default=None)
    args = parser.parse_args()

    directory_path = args.directory_path
    filters = args.filter.split(';') if args.filter else None

    try:
        log_pairs = find_log_pairs(directory_path, filters)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not log_pairs:
        print("No matching file pairs found")
        sys.exit(1)

    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    parse_script_path = os.path.abspath(current_dir / "../../parsers/parse_nvsim_log.py")

    if not os.path.exists(parse_script_path):
        print(f"Error: Script {parse_script_path} not found")
        sys.exit(1)

    # Group file pairs by data type
    grouped_pairs = group_log_pairs_by_data_type(log_pairs)

    for data_type, type_log_pairs in grouped_pairs.items():
        print(f"\nProcessing data type: {data_type} (found {len(type_log_pairs)} file pairs)")

        combined_records = []

        first_nvsmi_file, first_perf_file = type_log_pairs[0]
        headers = get_combined_headers(first_nvsmi_file, first_perf_file, parse_script_path)
        if headers is None:
            print(f"Error: Cannot get headers for {data_type}")
            continue

        # Process each file pair in the group and show progress bar
        total_pairs = len(type_log_pairs)
        for idx, (nvsmi_file, perf_file) in enumerate(type_log_pairs):
            progress = (idx + 1) / total_pairs
            bar_length = 20
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f"\r[{bar}] {progress:.1%} ({idx+1}/{total_pairs}) Processing: {os.path.basename(nvsmi_file)}", end="")

            combined_data = get_combined_data(nvsmi_file, perf_file, parse_script_path)
            if combined_data:
                combined_records.append(combined_data)

        print()

        if combined_records:
            # First save the raw combined results
            raw_result_csv = os.path.join(os.path.abspath(directory_path), f"raw_combined_results_{data_type}.csv")
            save_csv(raw_result_csv, headers, combined_records)

            # Process directly with fallback_process
            try:
                fallback_process(headers, combined_records, data_type, directory_path)
            except Exception as e:
                print(f"Error processing data: {e}")

if __name__ == "__main__":
    main()