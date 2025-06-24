#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import os
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def read_matrix_from_file(file_path):
    """Read a matrix of floating point numbers from a file"""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    matrix = []
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        if line.endswith(','):
            line = line[:-1]
        elements = [x.strip() for x in line.split(',')]
        row = [float(x) for x in elements if x]
        matrix.append(row)

    if not matrix:
        logger.warning(f"No valid data found in {file_path}")
    return matrix

def flatten_matrix(matrix):
    """Flatten a 2D matrix into a 1D list"""
    return [item for row in matrix for item in row]

def plot_histogram(data, bins=50, title="Matrix Data Histogram", xlabel="Value", ylabel="Percentage (%)", output=None):
    """Plot and show or save a histogram of the data with y-axis as percentage"""
    if not data:
        logger.error("No data to plot histogram")
        return

    plt.figure(figsize=(8, 6))
    counts, bin_edges, patches = plt.hist(data, bins=bins, edgecolor='black', density=False)
    total = len(data)
    # Convert y-axis scale to percentage
    plt.gca().set_ylim(0, max(counts) / total * 100 * 1.1)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}%'.format(y)))
    # Convert each bin's count to percentage
    for rect, count in zip(patches, counts):
        rect.set_height(count / total * 100)
    # Count the number and percentage of values equal to float(0.0)
    zero_count = sum(1 for v in data if v == 0.0)
    zero_percent = zero_count / total * 100 if total > 0 else 0.0
    # Add text to the figure
    plt.text(
        0.98, 0.95,
        f'Count of 0.0: {zero_count} ({zero_percent:.2f}%)',
        ha='right', va='top',
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.6)
    if output:
        try:
            plt.savefig(output)
            logger.info(f"Histogram saved to {output}")
        except Exception as e:
            logger.error(f"Failed to save histogram to {output}: {e}")
    else:
        plt.show()

def process_file(file_path, bins, output=None):
    """Process a single file to generate a histogram"""
    try:
        matrix = read_matrix_from_file(file_path)
        if not matrix:
            return

        data = flatten_matrix(matrix)
        # Determine output filename if not specified
        if output is None:
            base, _ = os.path.splitext(file_path)
            output = base + '.png'
        plot_histogram(data, bins=bins, output=output, title=f"Histogram of {os.path.basename(file_path)}")
    except IOError as e:
        logger.error(f"Error reading file {file_path}: {e}")
    except ValueError as e:
        logger.warning(f"Error parsing data in {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")

def collect_files_to_process(path):
    """
    Collect files to process based on the input path.
    
    Args:
        path: File path or directory path to process
        
    Returns:
        list: files to process
    """
    files_to_process = []

    # Check if path exists before processing
    if not os.path.exists(path):
        logger.error(f"Error: {path} does not exist")
        return files_to_process

    # Check file accessibility outside the try block
    if os.path.isfile(path):
        if os.access(path, os.R_OK):
            return [path]
        else:
            logger.error(f"File {path} exists but is not readable")
            return files_to_process

    if not os.path.isdir(path):
        logger.error(f"Error: {path} is neither a file nor a folder")
        return files_to_process

    try:
        # Folder processing: collect all .mat files
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.mat'):
                    files_to_process.append(os.path.join(root, file))

        if not files_to_process:
            logger.warning(f"No .mat files found in {path}")

    except Exception as e:
        logger.error(f"Error accessing path {path}: {e}")

    return files_to_process

def main():
    parser = argparse.ArgumentParser(description='Plot histogram of matrix data from file or folder')
    parser.add_argument('path', help='Matrix data file or folder containing .mat files')
    parser.add_argument('--bins', type=int, default=10, help='Number of histogram bins (default: 10)')
    args = parser.parse_args()

    files_to_process = collect_files_to_process(args.path)

    if not files_to_process:
        return

    # Process each file
    for i, file_path in enumerate(files_to_process):
        logger.info(f"Processing file {i+1}/{len(files_to_process)}: {file_path}")
        process_file(file_path, bins=args.bins)

    logger.info(f"Processed {len(files_to_process)} files")

if __name__ == "__main__":
    main()
