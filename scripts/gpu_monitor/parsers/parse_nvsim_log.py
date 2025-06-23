#!/usr/bin/env python3

import os
import re
import csv
import glob
import argparse
import numpy as np
from datetime import datetime

# Define fields to extract and their corresponding regex patterns
FIELDS = {
    'SMClocks': {
        'pattern': r'SM\s+:\s+(\d+)\s+MHz',
        'csv_header': 'SMClocks(MHz)'
    },
    'MemoryClocks': {
        'pattern': r'Memory\s+:\s+(\d+)\s+MHz',
        'csv_header': 'MemClocks(MHz)'
    },
    'GPUAveragePower': {
        'pattern': r'GPU Power Readings[\s\S]*?Average Power Draw\s+:\s+([\d.]+)\s+W',
        'csv_header': 'GPUAvgPower(W)'
    },
    'GPUInstantPower': {
        'pattern': r'GPU Power Readings[\s\S]*?Instantaneous Power Draw\s+:\s+([\d.]+)\s+W',
        'csv_header': 'GPUInstPower(W)'
    },
    'MemoryAveragePower': {
        'pattern': r'GPU Memory Power Readings[\s\S]*?Average Power Draw\s+:\s+([\d.]+)\s+W',
        'csv_header': 'MemAvgPower(W)'
    },
    'CurrentPowerLimit': {
        'pattern': r'Current Power Limit\s+:\s+([\d.]+)\s+W',
        'csv_header': 'PowerLimit(W)'
    },
    'GPUUtilization': {
        'pattern': r'Utilization[\s\S]*?GPU\s+:\s+(\d+)\s+%',
        'csv_header': 'GPUUtil(%)'
    },
    'MemoryUtilization': {
        'pattern': r'Utilization[\s\S]*?Memory\s+:\s+(\d+)\s+%',
        'csv_header': 'MemUtil(%)'
    },
    'GPUTemperature': {
        'pattern': r'GPU Current Temp\s+:\s+(\d+)\s+C',
        'csv_header': 'GPUTemp(C)'
    },
    'MemoryTemperature': {
        'pattern': r'Memory Current Temp\s+:\s+(\d+)\s+C',
        'csv_header': 'MemTemp(C)'
    },
    'ProductName': {
        'pattern': r'Product Name\s+:\s+(.*)',
        'csv_header': 'ProductName'
    },
    'ProductArchitecture': {
        'pattern': r'Product Architecture\s+:\s+(.*)',
        'csv_header': 'Architecture'
    },
    'PerformanceState': {
        'pattern': r'Performance State\s+:\s+(.*)',
        'csv_header': 'PerfState'
    },
    'Timestamp': {
        'pattern': r'Timestamp\s+:\s+(.*)',
        'csv_header': 'Timestamp'
    }
}

# Define default metrics order used in multiple functions
DEFAULT_METRICS = [
    'GPUAveragePower', 'MemoryAveragePower', 'GPUInstantPower',
    'GPUTemperature', 'MemoryTemperature', 'SMClocks', 'MemoryClocks',
    'GPUUtilization', 'MemoryUtilization'
]

def parse_log_entry(entry):
    """Parse a single log entry, extract specified fields"""
    result = {}

    for field, config in FIELDS.items():
        match = re.search(config['pattern'], entry)
        if match:
            # Special handling for timestamp
            if field == 'Timestamp':
                timestamp_str = match.group(1)
                try:
                    dt = datetime.strptime(timestamp_str, '%a %b %d %H:%M:%S %Y')
                    result[field] = dt.strftime('%Y%m%d-%H:%M:%S')
                except ValueError:
                    result[field] = timestamp_str
            else:
                result[field] = match.group(1)
        else:
            result[field] = 'N/A'

    return result

def parse_log_file(log_file):
    """Parse the entire log file, return all entries"""
    with open(log_file, 'r') as f:
        content = f.read()

    # Split into individual records
    entries = content.split('==============NVSMI LOG==============')

    # Filter out empty records
    entries = [entry.strip() for entry in entries if entry.strip()]

    # Parse each record
    parsed_entries = []
    for entry in entries:
        parsed_entry = parse_log_entry(entry)
        if any(v != 'N/A' for v in parsed_entry.values()):  # Only keep meaningful records
            parsed_entries.append(parsed_entry)

    return parsed_entries

def write_to_csv(data, output_file):
    """Write parsed data to CSV file"""
    if not data:
        return

    # Get CSV headers
    csv_headers = [FIELDS[field]['csv_header'] for field in FIELDS]

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(FIELDS.keys()))
        # Write custom headers
        writer.writerow({field: FIELDS[field]['csv_header'] for field in FIELDS})

        # Write data
        for entry in data:
            writer.writerow(entry)

def calculate_stable_mean(values, stability_window=5, stability_threshold=0.05):
    """
    Calculate the mean value during the stable period of the metric.

    Args:
        values: List of numeric values
        stability_window: Window size to check for stability (number of consecutive points)
        stability_threshold: Maximum allowed relative change within window to be considered stable

    Raises:
        ValueError: If stable mean cannot be calculated due to insufficient data points
                   or inability to find a stable period

    Returns:
        Mean value during the stable period
    """
    if not values:
        raise ValueError(f"Cannot calculate stable mean: Empty input data. Expected at least {stability_window*2} data points, got 0.")

    if len(values) < stability_window * 2:
        raise ValueError(f"Cannot calculate stable mean: Insufficient data points. Expected at least {stability_window*2} data points, got {len(values)}.")

    # Find the maximum value to identify potential stable periods at peak performance
    max_value = max(values)
    max_index = values.index(max_value)

    # For metrics like power, find stable period around the maximum value
    # First, find a continuous period where values are close to the maximum
    stable_start = max_index
    stable_end = max_index

    # Search backward from max_index to find start of stable period
    for i in range(max_index, 0, -1):
        if abs(values[i] - values[i-1]) / max(values[i], 1) > stability_threshold:
            stable_start = i
            break

    # Search forward from max_index to find end of stable period
    for i in range(max_index, len(values)-1):
        if abs(values[i] - values[i+1]) / max(values[i], 1) > stability_threshold:
            stable_end = i
            break

    # If the stable period is too short, try to find the longest stable period
    if stable_end - stable_start < stability_window:
        best_stable_len = 0
        best_stable_start = 0
        best_stable_end = 0

        i = 0
        while i < len(values) - 1:
            j = i + 1
            while j < len(values) and abs(values[j] - values[j-1]) / max(values[j-1], 1) <= stability_threshold:
                j += 1

            if j - i > best_stable_len:
                best_stable_len = j - i
                best_stable_start = i
                best_stable_end = j - 1

            i = j

        if best_stable_len >= stability_window:
            stable_start = best_stable_start
            stable_end = best_stable_end

    # Calculate mean of the stable period
    if stable_end - stable_start + 1 >= stability_window:
        return np.mean(values[stable_start:stable_end+1])

    # No stable period found - raise an exception with detailed information
    raise ValueError(
        f"Cannot calculate stable mean: No stable period of at least {stability_window} points found. "
        f"Data may be too volatile (threshold={stability_threshold}). "
        f"Longest stable sequence found: {stable_end-stable_start+1} points."
    )

def calculate_statistics(data, field, stats_types=None):
    """Calculate statistics (median, mean, max, min, meanStable) for a specific field"""
    if stats_types is None:
        stats_types = ['meanStable', 'median', 'mean', 'max', 'min']

    values = []
    for entry in data:
        if entry[field] != 'N/A' and entry[field].strip():
            try:
                values.append(float(entry[field]))
            except (ValueError, TypeError):
                continue

    if not values:
        return {stat_type: 'N/A' for stat_type in stats_types}

    stats = {}
    if 'median' in stats_types:
        stats['median'] = np.median(values)
    if 'mean' in stats_types:
        stats['mean'] = np.mean(values)
    if 'max' in stats_types:
        stats['max'] = np.max(values)
    if 'min' in stats_types:
        stats['min'] = np.min(values)
    if 'meanStable' in stats_types:
        try:
            stats['meanStable'] = calculate_stable_mean(values)
        except ValueError as e:
            print(f"Warning when calculating stable mean for {field}: {str(e)}")
            stats['meanStable'] = 'N/A'

    return stats

def filter_idle_periods(data, utilization_field='GPUUtilization', retain_count=0):
    """
    Filter out consecutive GPU utilization records with 0% usage at the beginning and end,
    but retain a specified number of idle records for reference.

    Args:
        data: List of parsed data entries
        utilization_field: GPU utilization field name (internal field name, not CSV column name)
        retain_count: Number of idle records to keep at the beginning and end

    Returns:
        Filtered list of data entries
    """
    if not data:
        return data

    # Find the index of the maximum utilization
    max_idx = 0
    max_value = -1
    for i, entry in enumerate(data):
        try:
            value = int(entry[utilization_field])
            if value >= max_value:
                max_value = value
                max_idx = i
        except (ValueError, TypeError):
            continue

    # Scan from the maximum index to the left to find the second zero
    left_zero_count = 0
    start_idx = max_idx
    for i in range(max_idx, -1, -1):
        try:
            if data[i][utilization_field] == 'N/A' or int(data[i][utilization_field]) == 0:
                start_idx = i+1
                break
        except (ValueError, TypeError):
            continue

    # Scan from the maximum index to the right to find the second zero
    right_zero_count = 0
    end_idx = max_idx
    for i in range(max_idx, len(data)):
        try:
            if data[i][utilization_field] == 'N/A' or int(data[i][utilization_field]) == 0:
                end_idx = i-1
                break
        except (ValueError, TypeError):
            continue

    # Adjust indices to retain some idle records
    start_idx = max(0, start_idx - retain_count)
    end_idx = min(len(data) - 1, end_idx + retain_count)

    return data[start_idx:end_idx + 1]

def get_csv_headers(metrics=None, stats_types=None):
    """Get CSV headers for metrics with statistics types"""
    if metrics is None:
        metrics = DEFAULT_METRICS

    if stats_types is None:
        stats_types = ['meanStable', 'median', 'mean', 'max', 'min']

    headers = []
    for stat_type in stats_types:
        for metric in metrics:
            # Get the unit for this metric (if any)
            header = FIELDS[metric]['csv_header']
            base_name = header.split('(')[0] if '(' in header else header
            unit = header.split('(')[1].rstrip(')') if '(' in header else ""

            # Create new header format: SMClocksMedian(MHz)
            if unit:
                headers.append(f"{base_name}{stat_type.capitalize()}({unit})")
            else:
                headers.append(f"{base_name}{stat_type.capitalize()}")

    return ','.join(headers)

def format_stats_as_csv(data, metrics=None, stats_types=None):
    """Format statistics as CSV row"""
    if metrics is None:
        metrics = DEFAULT_METRICS

    if stats_types is None:
        stats_types = ['meanStable', 'median', 'mean', 'max', 'min']

    csv_values = []
    for stat_type in stats_types:
        for metric in metrics:
            try:
                stats = calculate_statistics(data, metric, [stat_type])
                value = stats.get(stat_type, 'N/A')
                if value != 'N/A':
                    # Format as integer if it's a whole number
                    if value == int(value):
                        csv_values.append(f"{int(value)}")
                    else:
                        csv_values.append(f"{value:.2f}")
                else:
                    csv_values.append('N/A')
            except Exception as e:
                print(f"Error calculating {stat_type} for {metric}: {str(e)}")
                csv_values.append('N/A')

    return ','.join(csv_values)

def print_statistics(data):
    """Print statistics information to console"""
    metrics = DEFAULT_METRICS

    print("\nStatistics for key metrics:")
    print("-" * 80)
    print(f"{'Metric':<20} {'Mean(Stable)':<12} {'Median':<12} {'Mean':<12} {'Max':<12} {'Min':<12}")
    print("-" * 80)

    for metric in metrics:
        stats = calculate_statistics(data, metric)
        header = FIELDS[metric]['csv_header']

        # Format statistics data - use integer format for whole numbers
        median = stats['median']
        mean = stats['mean']
        max_val = stats['max']
        min_val = stats['min']
        meanStable = stats.get('meanStable', 'N/A')

        # Format each value
        median = f"{int(median)}" if median != 'N/A' and median == int(median) else f"{median:.2f}" if median != 'N/A' else 'N/A'
        mean = f"{int(mean)}" if mean != 'N/A' and mean == int(mean) else f"{mean:.2f}" if mean != 'N/A' else 'N/A'
        max_val = f"{int(max_val)}" if max_val != 'N/A' and max_val == int(max_val) else f"{max_val:.2f}" if max_val != 'N/A' else 'N/A'
        min_val = f"{int(min_val)}" if min_val != 'N/A' and min_val == int(min_val) else f"{min_val:.2f}" if min_val != 'N/A' else 'N/A'
        meanStable = f"{int(meanStable)}" if meanStable != 'N/A' and meanStable == int(meanStable) else f"{meanStable:.2f}" if meanStable != 'N/A' else 'N/A'

        print(f"{header:<20} {meanStable:<12} {median:<12} {mean:<12} {max_val:<12} {min_val:<12}")

    print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="Parse NVIDIA-SMI log files, extract GPU monitoring data and save as CSV")
    parser.add_argument("path", nargs='?', help="Log file path or directory containing log files")
    parser.add_argument("--csv", action="store_true", help="Output statistics in CSV format to terminal")
    parser.add_argument("--csv-headers", action="store_true", help="Output only CSV headers")
    parser.add_argument("--stats-type", choices=["median", "mean", "max", "min", 'meanStable'], nargs="+",
                      help="Specify statistics types to include (default: all)")

    args = parser.parse_args()

    # If only CSV headers are needed
    if args.csv_headers:
        headers = get_csv_headers(stats_types=args.stats_type)
        print(headers)
        return

    # For other operations, path is required
    if not args.path:
        parser.error("the following arguments are required: path")

    path = args.path

    # Collect all files to process
    log_files = []
    if os.path.isfile(path):
        log_files = [path]
    elif os.path.isdir(path):
        log_files = glob.glob(os.path.join(path, "*.log.txt"))
        if not log_files:
            print(f"No log files found in directory {path}")
            return
    else:
        print(f"Error: Path {path} does not exist")
        return

    # Process all files
    for log_file in log_files:
        # Parse data
        parsed_data = parse_log_file(log_file)
        filtered_data = filter_idle_periods(parsed_data)

        csv_file = log_file.rsplit('.', 1)[0] + '.csv'
        write_to_csv(parsed_data, csv_file)
        filtered_csv_file = log_file.rsplit('.', 1)[0] + '.filtered.csv'
        write_to_csv(filtered_data, filtered_csv_file)

        # If CSV output mode, only output CSV format data
        if args.csv:
            csv_data = format_stats_as_csv(filtered_data, stats_types=args.stats_type)
            print(csv_data)
        else:
            # Print statistics information
            print(f"Parsing file: {log_file}")
            print(f"Original data points: {len(parsed_data)}, After filtering: {len(filtered_data)}")
            print(f"Data has been written to: {csv_file}")
            print(f"Filtered data has been written to: {filtered_csv_file}")
            print_statistics(filtered_data)

if __name__ == "__main__":
    main()