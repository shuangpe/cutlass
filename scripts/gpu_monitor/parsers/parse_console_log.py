#!/usr/bin/env python3

import re
import json
import sys
import argparse

def parse_test_output(text):
    """
    Parse CUTLASS test output to extract key metrics.

    Args:
        text (str): The console output from a CUTLASS test run

    Returns:
        dict: Extracted metrics
    """
    # Initialize result dictionary with empty values
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
        "mask_ratio": ""
    }

    # Define regex patterns for each field
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
        "mask_ratio": r"MaskRatio:\ ([0-9.%]+)"
    }

    # Extract values using regex
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            result[key] = match.group(1)

    # Set default value "0%" for MaskRatio if not found
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
        keys = ["problem_size", "stages", "tile_shape", "grid_dims",
                "hack_load_g2l", "disposition", "tflops", "iterations", 
                "avg_runtime"]

    # Extract values in the order specified by keys
    values = [data.get(key, "") for key in keys]

    # Return comma-separated values
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
        keys = ["problem_size", "stages", "tile_shape", "grid_dims",
                "hack_load_g2l", "disposition", "tflops", "iterations", 
                "avg_runtime"]

    # Map internal field names to display names
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
        "mask_ratio": "MaskRatio"  # 添加MaskRatio的显示名称
    }

    # Get display names for the specified keys
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

    args = parser.parse_args()

    # Get CSV keys if specified
    if args.csv_keys:
        keys = args.csv_keys.split(',')
    else:
        keys = None

    # Output CSV headers only if requested
    if args.csv_headers:
        output = get_csv_headers(keys)
    else:
        # Read input
        if args.input:
            with open(args.input, 'r') as f:
                text = f.read()
        else:
            text = sys.stdin.read()

        # Parse the text
        result = parse_test_output(text)

        # Format output
        if args.format == "json":
            output = json.dumps(result, indent=2)
        else:  # csv
            output = format_as_csv_row(result, keys)

    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
    else:
        print(output)

if __name__ == "__main__":
    main()

