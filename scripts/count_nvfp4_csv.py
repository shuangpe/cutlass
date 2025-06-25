import os
import numpy as np
from collections import Counter
from multiprocessing import Pool
import argparse
import csv

E2M1_VALUES = [0, -0.5, 0.5, -1, 1, -1.5, 1.5, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6]


def find_mat_files():
    a_file = None
    b_file = None
    for file in os.listdir('.'):
        if file.endswith('_A.mat'):
            a_file = file
        elif file.endswith('_B.mat'):
            b_file = file
    return a_file, b_file


def load_and_count_values(file_path):
    value_counts = Counter()
    with open(file_path, 'r') as f:
        for line in f:
            values = [float(x.strip()) for x in line.split(',') if x.strip()]
            value_counts.update(values)
    return {'counts': value_counts, 'file': file_path}


def format_as_readable(a_counts, b_counts):
    output = ""

    def format_file_counts(file_counts):
        file_output = f"File: {file_counts['file']}\n"
        file_output += "-" * 80 + "\n"
        total_elements = sum(file_counts['counts'].values())
        file_output += f"{'Value':>10} | {'Count':>15} | {'Percentage':>10}\n"
        file_output += "-" * 80 + "\n"
        for value, count in sorted(file_counts['counts'].items()):
            percentage = (count / total_elements) * 100
            file_output += f"{value:>10.1f} | {count:>15} | {percentage:>9.2f}%\n"
        file_output += "\n"
        return file_output

    if a_counts:
        output += format_file_counts(a_counts)
    if b_counts:
        output += format_file_counts(b_counts)
    return output


def write_to_csv(file_counts, csv_file, tags):
    header = [col for col, _ in tags] if tags else []
    row = [val for _, val in tags] if tags else []
    total_elements = sum(file_counts['counts'].values())
    for value in E2M1_VALUES:
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


def merge_counts(a_counts, b_counts):
    merged_counts = Counter(a_counts['counts'])
    merged_counts.update(b_counts['counts'])
    return {'counts': merged_counts, 'file': f"{a_counts['file']} + {b_counts['file']}"}


def main():
    parser = argparse.ArgumentParser(description="Count values in .mat files.")
    parser.add_argument("--separate", action="store_true", help="Separate the output for A and B files (default: combined output).")
    parser.add_argument("--csv", type=str, help="Write the statistics to a CSV file. Provide the base filename (e.g., 'output').")
    parser.add_argument("--tags", type=str, help="Add tags to the CSV file. Format: <column:tag,...> (e.g., 'Experiment:Test1,Run:42').")
    args = parser.parse_args()

    tags = []
    if args.tags:
        tags = [tuple(tag.split(':')) for tag in args.tags.split(',')]

    a_file, b_file = find_mat_files()

    with Pool(processes=2) as pool:
        results = pool.map(load_and_count_values, [a_file, b_file])

    a_counts, b_counts = results

    if args.csv:
        if args.separate:
            if a_counts:
                write_to_csv(a_counts, f"{args.csv}_A.csv", tags)
            if b_counts:
                write_to_csv(b_counts, f"{args.csv}_B.csv", tags)
        else:
            merged_counts = merge_counts(a_counts, b_counts)
            write_to_csv(merged_counts, f"{args.csv}.csv", tags)
    else:
        if args.separate:
            print(format_as_readable(a_counts, None), end='')
            print(format_as_readable(None, b_counts), end='')
        else:
            print(format_as_readable(a_counts, b_counts), end='')


if __name__ == "__main__":
    main()
