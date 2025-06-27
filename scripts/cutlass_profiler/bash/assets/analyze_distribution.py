#!/usr/bin/env python3

import os
from collections import Counter
from multiprocessing import Pool
import argparse
import csv
from dataclasses import dataclass, field
import time
import logging
import sys
import subprocess

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# FileHandler will be added in main() after output_dir is known

@dataclass
class NVFP4BinMapper:
    """Bin mapper for NVFP4 kernel."""
    bins: list = field(default_factory=lambda: [0, -0.5, 0.5, -1, 1, -1.5, 1.5, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6])

    def map_to_bin(self, value):
        """Return the bin value that is closest to the input value."""
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


def find_mat_files(scan_path, exclude):
    """Recursively return a list of all .mat files ending with _A.mat or _B.mat in scan_path."""
    result = []
    for root, dirs, files in os.walk(scan_path):
        for file in files:
            if (file.endswith('_A.mat') or file.endswith('_B.mat')):
                full_path = os.path.join(root, file)
                if full_path not in exclude:
                    result.append(full_path)
    return result


def load_and_count_values(file_path):
    """Load values from a file, map them to bins, and count occurrences."""

    parent_dirname = os.path.basename(os.path.dirname(file_path))
    is_nvfp4 = "ue4m3xe2m1_ue4m3xe2m1" in parent_dirname
    bin_mapper = NVFP4BinMapper() if is_nvfp4 else GeneralBinMapper()

    value_counts = Counter()

    total_lines = None
    progress_interval = 1000
    try:
        # Use wc -l for fast line counting
        result = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            total_lines = int(result.stdout.strip().split()[0])
            if total_lines and total_lines > 0:
                progress_interval = int(total_lines * 0.25)
        else:
            total_lines = None
    except Exception:
        total_lines = None

    start_time = time.time()

    try:
        with open(file_path, 'r') as f:
            line_count = 0
            for line in f:
                values = [float(x.strip()) for x in line.split(',') if x.strip()]
                binned_values = [bin_mapper.map_to_bin(val) for val in values]
                value_counts.update(binned_values)
                line_count += 1
                if line_count % progress_interval == 0:
                    if total_lines:
                        percent = 100.0 * line_count / total_lines
                        logger.info(f"Processing {parent_dirname}: {line_count}/{total_lines} lines ({percent:.2f}%)")
                    else:
                        logger.info(f"Processing {parent_dirname}: {line_count} lines processed")
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")

    elapsed = time.time() - start_time
    logger.info(f"Finished processing {parent_dirname} in {elapsed:.2f} seconds")
    return {'counts': value_counts, 'file': file_path}


def write_to_csv(counts, output_dir, remove_processed=False):
    """Write the bin counts to a CSV file."""
    file_path = counts['file']
    dirname = os.path.dirname(file_path)
    parent_dirname = os.path.basename(dirname)

    if remove_processed:
        try:
            os.remove(file_path)
            logger.info(f"Removed file: {file_path}")
            if os.path.isdir(dirname) and not os.listdir(dirname):
                os.rmdir(dirname)
                logger.info(f"Removed empty directory: {dirname}")
        except Exception as e:
            logger.warning(f"Failed to remove file or directory: {e}")

    # Parse parent_dirname: expected format is {kernel_name}.{mask_ratio}.{scope}.{run_id}
    kernel_name = mask_ratio = scope = run_id = ""
    parts = parent_dirname.rsplit('.', 3)
    if len(parts) == 4:
        kernel_name, mask_ratio, scope, run_id = parts

    is_nvfp4 = "ue4m3xe2m1_ue4m3xe2m1" in kernel_name
    bin_mapper = NVFP4BinMapper() if is_nvfp4 else GeneralBinMapper()
    bins = bin_mapper.bins

    csv_path = os.path.join(output_dir, f"{kernel_name}.csv")
    header = ["Kernel", "Scope", "MaskRatio", "RunID"]
    row = [kernel_name, scope, mask_ratio, run_id]
    total_elements = sum(counts['counts'].values())

    for value in bins:
        header.append(f"Count{value}")
        count = counts['counts'].get(value, 0)
        percentage = count / total_elements if total_elements > 0 else 0
        row.append(f"{percentage:.4f}")

    file_exists = os.path.isfile(csv_path)
    try:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)
        logger.info(f"Wrote statistics to {csv_path} for {file_path}")
    except Exception as e:
        logger.error(f"Error writing to CSV {csv_path}: {e}")


def process_files(scan_dir, output_dir, files_in_analysis, processes=8, remove_processed=False):
    """Process files in the scan directory."""
    results = []
    pool = Pool(processes=processes)
    try:
        while True:
            mat_files = find_mat_files(scan_dir, files_in_analysis)
            if not mat_files and results:
                break

            for file in mat_files:
                files_in_analysis.add(file)
                results.append((file, pool.apply_async(load_and_count_values, args=(file,))))

            # Check and process finished results
            i = 0
            while i < len(results):
                file, async_result = results[i]
                if async_result.ready():
                    counts = async_result.get()
                    write_to_csv(counts, output_dir, remove_processed=remove_processed)
                    logger.info(f"Processed file: {file}")
                    results.pop(i)
                else:
                    i += 1

            if not mat_files and results:
                time.sleep(10)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, terminating pool.")
        pool.terminate()
        pool.join()
        raise
    finally:
        pool.close()
        pool.join()


def main():
    parser = argparse.ArgumentParser(
        description="Count values in .mat files.",
        allow_abbrev=False  # Prevent ambiguous short options
    )
    parser.add_argument("-i", "--scan_dir", type=str, default=".", dest="scan_dir",
                        help="Directory to scan for .mat files (default: current directory).")
    parser.add_argument("-o", "--output_dir", type=str, default=".", dest="output_dir",
                        help="Directory to write the statistics CSV files (default: current directory).")
    parser.add_argument("-p", "--processes", type=int, default=8, dest="processes",
                        help="Number of processes to use (default: 8).")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        help="Disable all print statements and logging.")
    parser.add_argument("--remove", action="store_true", dest="remove_processed",
                        help="Remove processed .mat files after analysis.")
    args = parser.parse_args()

    # Ensure output_dir exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Add file handler for logging to output_dir
    log_file = os.path.join(args.output_dir, "analyze_distribution.log")
    fh = logging.FileHandler(log_file, mode='a')
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    if args.quiet:
        logger.setLevel(logging.CRITICAL)
        fh.setLevel(logging.INFO)  # Still log to file even if quiet

    files_in_analysis = set()

    try:
        logger.info(f"Starting analysis. Scanning: {args.scan_dir}, Output: {args.output_dir}, Processes: {args.processes}")
        process_files(args.scan_dir, args.output_dir, files_in_analysis, processes=args.processes, remove_processed=args.remove_processed)
        logger.info("Analysis completed.")
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
