#!/usr/bin/env python3

import os
from collections import Counter
from multiprocessing import Pool
import argparse
import csv
from dataclasses import dataclass, field


@dataclass
class NVFP4BinMapper:
    """Bin mapper for NVFP4 kernel."""
    bins: list = field(default_factory=lambda: [0, -0.5, 0.5, -1, 1, -1.5, 1.5, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6])

    def map_to_bin(self, value):
        # Return the bin value that is closest to the input value
        return min(self.bins, key=lambda x: abs(x - value))


@dataclass
class GeneralBinMapper:
    """Bin mapper for general kernels."""
    bins: list = field(default_factory=lambda: [
        0, -0.5, 0.5, -1, 1, -1.5, 1.5, -2, 2, -2.5, 2.5, -3, 3, -3.5, 3.5,
        -4, 4, -4.5, 4.5, -5, 5
    ])
    thresholds: list = field(default_factory=lambda: [
        (-0, 0),  # 0
        (-0.75, 0),  # -0.5
        (0, 0.75),  # 0.5
        (-1.25, -0.75),  # -1
        (0.75, 1.25),  # 1
        (-1.75, -1.25),  # -1.5
        (1.25, 1.75),  # 1.5
        (-2.25, -1.75),  # -2
        (1.75, 2.25),  # 2
        (-2.75, -2.25),  # -2.5
        (2.25, 2.75),  # 2.5
        (-3.25, -2.75),  # -3
        (2.75, 3.25),  # 3
        (-3.75, -3.25),  # -3.5
        (3.25, 3.75),  # 3.5
        (-4.25, -3.75),  # -4
        (3.75, 4.25),  # 4
        (-4.75, -4.25),  # -4.5
        (4.25, 4.75),  # 4.5
        (-5, -4.75),  # -5
        (4.75, 5)  # 5
    ])

    def map_to_bin(self, value):
        """Map a value to the closest bin for general kernels."""
        if value == 0:
            return 0  # Special case for 0
        for (lower, upper), bin_value in zip(self.thresholds, self.bins):
            if lower < value <= upper:
                return bin_value
        return 5 if value > 5 else -5


def find_mat_files(kernel_name):
    """Return a list of all .mat files containing the kernel_name and ending with _A.mat or _B.mat."""
    result = []
    for file in os.listdir('.'):
        if (file.endswith('_A.mat') or file.endswith('_B.mat')):
            if kernel_name=="" or kernel_name in file:
                result.append(file)
    return result


def load_and_count_values(args):
    file_path, bin_mapper = args
    value_counts = Counter()

    with open(file_path, 'r') as f:
        for line in f:
            values = [float(x.strip()) for x in line.split(',') if x.strip()]
            binned_values = [bin_mapper.map_to_bin(val) for val in values]
            value_counts.update(binned_values)

    return {'counts': value_counts, 'file': file_path}


def print_count(file_counts):
    file_output = ""
    if 'file' in file_counts.keys() and file_counts['file']:
        file_output += f"File: {file_counts['file']}\n"
    file_output += "-" * 80 + "\n"
    total_elements = sum(file_counts['counts'].values())
    file_output += f"{'Value':>10} | {'Count':>15} | {'Percentage':>10}\n"
    file_output += "-" * 80 + "\n"
    for value, count in sorted(file_counts['counts'].items()):
        percentage = (count / total_elements) * 100
        file_output += f"{value:>10.1f} | {count:>15} | {percentage:>9.2f}%\n"
    file_output += "\n"
    print(file_output)


def write_to_csv(file_counts, bins, csv_file, tags):
    header = list(tags.keys()) if tags else []
    row = list(tags.values()) if tags else []
    total_elements = sum(file_counts['counts'].values())

    for value in bins:
        header.append(f"Count{value}")
        count = file_counts['counts'].get(value, 0)
        percentage = count / total_elements if total_elements > 0 else 0
        row.append(f"{percentage:.4f}")

    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def merge_counts(results):
    """Merge a list of count dicts into one."""
    merged = Counter()
    for item in results:
        merged.update(item['counts'])
    return {'counts': merged}


def main():
    parser = argparse.ArgumentParser(description="Count values in .mat files.")
    parser.add_argument("--separate", action="store_true", help="Separate the output for A and B files (default: combined output).")
    parser.add_argument("--csv", type=str, help="Write the statistics to a CSV file. Provide the base filename (e.g., 'output').")
    parser.add_argument("--tags", type=str, help="Add tags to the CSV file. Format: <column:tag,...> (e.g., 'Experiment:Test1,Run:42').")
    args = parser.parse_args()

    tags = {}
    kernel_name = ""

    if args.tags:
        tags = dict(tag.split(':') for tag in args.tags.split(','))
        kernel_name = tags.get("Kernel", "")

    is_nvfp4 = "ue4m3xe2m1_ue4m3xe2m1" in kernel_name
    bin_mapper = NVFP4BinMapper() if is_nvfp4 else GeneralBinMapper()
    bins = bin_mapper.bins

    mat_files = find_mat_files(kernel_name)

    with Pool(processes=len(mat_files)) as pool:
        results = pool.map(load_and_count_values, [(file, bin_mapper) for file in mat_files])

    if args.csv:
        if args.separate:
            for i, counts in enumerate(results):
                if counts:
                    write_to_csv(counts, bins, f"{args.csv}_{i+1}.csv", tags)
        else:
            merged_counts = merge_counts(results)
            write_to_csv(merged_counts, bins, f"{args.csv}.csv", tags)
    else:
        if args.separate:
            for counts in results:
                if counts:
                    print_count(counts)
        else:
            merged_counts = merge_counts(results)
            print_count(merged_counts)


if __name__ == "__main__":
    main()
