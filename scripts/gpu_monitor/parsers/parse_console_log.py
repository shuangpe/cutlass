#!/usr/bin/env python3

import re
import json
import sys
import argparse

# Define default keys order used in multiple functions
DEFAULT_KEYS = [
    "problem_size", "stages", "tile_shape", "disposition",
    "iterations", "avg_runtime", "scope_range",
    "hack_load_g2l", "mask_ratio", "tflops"
]

def parse_test_output(text):
    """
    Parse CUTLASS test output to extract key metrics.

    Args:
        text (str): The console output from a CUTLASS test run

    Returns:
        dict: Extracted metrics
    """
    result = {
        "problem_size": "",
        "stages": "",
        "tile_shape": "",
        "grid_dims": "",
        "hack_load_g2l": "",
        "tflops": "",
        "disposition": "",
        "iterations": "",
        "avg_runtime": "",
        "mask_ratio": "",
        "scope_range": ""
    }

    patterns = {
        "problem_size": r"Problem Size:\ ([0-9x]+)",
        "stages": r"Stages:\ ([0-9]+)",
        "tile_shape": r"TileShape:\ ([0-9x]+)",
        "grid_dims": r"GridDims:\ ([0-9x]+)",
        "hack_load_g2l": r"HackLoadG2L:\ ([0-9]+)",
        "tflops": r"TFLOPS:\ ([0-9.]+)",
        "disposition": r"Disposition:\ ([A-Za-z]+)",
        "iterations": r"Start profiling CUTLASS kernel for ([0-9]+) iterations",
        "avg_runtime": r"Avg runtime: ([0-9.]+) ms",
        "mask_ratio": r"MaskRatio:\ ([0-9.%]+)",
        "scope_range": r"ScopeRange:\ (\[[^]]+\])"  # Capture the entire range including brackets
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            result[key] = match.group(1)

    if result["mask_ratio"] == "":
        result["mask_ratio"] = "0%"

    return result

def format_as_csv_row(data, keys=None):
    """
    Format the extracted data as a CSV row

    Args:
        data (dict): The extracted data
        keys (list): Optional list of keys to include in specific order

    Returns:
        str: CSV formatted row
    """
    if keys is None:
        keys = DEFAULT_KEYS

    values = []
    for key in keys:
        value = data.get(key, "")
        if "," in value:
            value = f'"{value}"'
        values.append(value)

    return ",".join(values)

def get_csv_headers(keys=None):
    """
    Get CSV headers for the specified keys

    Args:
        keys (list): Optional list of keys to include in specific order

    Returns:
        str: CSV headers
    """
    if keys is None:
        keys = DEFAULT_KEYS

    field_display_names = {
        "problem_size": "ProblemSize",
        "stages": "Stages",
        "tile_shape": "TileShape",
        "grid_dims": "GridDims",
        "hack_load_g2l": "HackLoadG2L",
        "disposition": "Disposition",
        "tflops": "TFLOPS",
        "iterations": "Iterations",
        "avg_runtime": "AvgRuntime",
        "mask_ratio": "MaskRatio",
        "scope_range": "ScopeRange"
    }

    headers = [field_display_names.get(key, key) for key in keys]

    return ",".join(headers)

def main():
    parser = argparse.ArgumentParser(description="Parse CUTLASS test output")
    parser.add_argument("--input", "-i", type=str, help="Input file (default: stdin)")
    parser.add_argument("--output", "-o", type=str, help="Output file (default: stdout)")
    parser.add_argument("--format", "-f", choices=["json", "csv"], default="json",
                        help="Output format (default: json)")
    parser.add_argument("--csv-keys", type=str, help="Comma-separated list of keys for CSV output")
    parser.add_argument("--csv-headers", action="store_true",
                        help="Output only CSV headers (for table creation)")
    parser.add_argument("--csv", action="store_true", help="Shortcut for --format csv")

    args = parser.parse_args()

    if args.csv:
        args.format = "csv"

    if args.csv_keys:
        keys = args.csv_keys.split(',')
    else:
        keys = None

    if args.csv_headers:
        output = get_csv_headers(keys)
    else:
        if args.input:
            with open(args.input, 'r') as f:
                text = f.read()
        else:
            text = sys.stdin.read()

        result = parse_test_output(text)

        if args.format == "json":
            output = json.dumps(result, indent=2)
        else:  # csv
            output = format_as_csv_row(result, keys)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
    else:
        print(output)

if __name__ == "__main__":
    main()

