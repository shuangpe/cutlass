#!/usr/bin/env python3

import os
import argparse
import logging
import matplotlib.pyplot as plt
from collections import Counter
import sys
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# All possible values of E2M1 format
E2M1_VALUES = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6,
               -0.5, -1, -1.5, -2, -3, -4, -5, -6]

def count_e2m1_values_from_file(file_path, approximate=True):
    """
    Read data from file and count occurrences of E2M1 values directly
    
    Args:
        file_path: Path to the input file
        approximate: Whether to approximate the value to the nearest E2M1 value
    
    Returns:
        counter: Counter object containing the count of each E2M1 value
        total_elements: Total number of elements
    """
    counter = Counter()
    total_elements = 0

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            if line.endswith(','):
                line = line[:-1]

            elements = [x.strip() for x in line.split(',')]
            values = []

            for element in elements:
                if not element:
                    continue
                try:
                    values.append(float(element))
                except ValueError:
                    logger.warning(f"Line {line_num}: Could not convert '{element}' to float. Skipping.")

            for value in values:
                if approximate:
                    # Find the closest E2M1 value
                    closest_value = min(E2M1_VALUES, key=lambda x: abs(x - value))
                    counter[closest_value] += 1
                else:
                    # Use the value directly (assuming the input is already in E2M1 format)
                    counter[value] += 1

                total_elements += 1

                # Print progress regularly
                if total_elements % 1000000 == 0:
                    logger.info(f"Processed {total_elements:,} elements...")

    return counter, total_elements

def plot_distribution(counter, total_elements, output_path):
    """Plot a pie chart of the E2M1 value distribution with natural colors"""
    if total_elements == 0:
        logger.warning("No elements found, cannot plot distribution.")
        return

    # Ensure all E2M1 values are in the counter, even if count is 0
    for val in E2M1_VALUES:
        if val not in counter:
            counter[val] = 0

    values = sorted(counter.keys())
    counts = [counter[val] for val in values]
    percentages = [count / total_elements * 100 for count in counts]

    # Define a set of natural colors
    natural_colors = [
        "#FF9999", "#66B3FF", "#99FF99", "#FFCC99", "#CCCCFF",
        "#FFB3E6", "#B3B3CC", "#FF6666", "#66FF66", "#B3E6FF"
    ]

    # Ensure adjacent colors are not the same
    colors = [natural_colors[i % len(natural_colors)] for i in range(len(percentages))]
    for i in range(1, len(colors)):
        if colors[i] == colors[i - 1]:
            colors[i] = natural_colors[(i + 1) % len(natural_colors)]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Separate values with 0% for horizontal display below the pie chart
    zero_percentage_labels = [
        f"{val} (0%)" for val, percentage in zip(values, percentages) if percentage == 0
    ]
    non_zero_values = [
        val for val, percentage in zip(values, percentages) if percentage > 0
    ]
    non_zero_percentages = [
        percentage for percentage in percentages if percentage > 0
    ]
    non_zero_colors = [
        color for percentage, color in zip(percentages, colors) if percentage > 0
    ]

    # Create pie chart with labels directly
    wedges, texts = ax.pie(
        non_zero_percentages,
        labels=[f"{val}({percentage:.1f}%)" for val, percentage in zip(non_zero_values, non_zero_percentages)],
        startangle=90,
        colors=non_zero_colors
    )

    # Style the text
    for text in texts:
        text.set_fontsize(10)
        text.set_rotation(0)
        text.set_ha('center')
        text.set_va('center')

    # Display zero percentage labels below the pie chart
    if zero_percentage_labels:
        plt.figtext(
            0.5, 0.12,  # Move closer to the pie chart (was 0.02)
            ", ".join(zero_percentage_labels),
            ha='center', fontsize=10, color='gray'
        )

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Distribution plot saved to {output_path}")

def print_distribution(counter, total_elements):
    """Print the distribution of E2M1 values in table format"""
    if total_elements == 0:
        logger.warning("No elements found, cannot print distribution.")
        return

    # Ensure all E2M1 values are in the counter, even if count is 0
    for val in E2M1_VALUES:
        if val not in counter:
            counter[val] = 0

    values = sorted(counter.keys())

    print("\nE2M1 Value Distribution:")
    print("-" * 50)
    print(f"{'Value':<10} {'Count':<15} {'Percentage':<10}")
    print("-" * 50)

    for val in values:
        count = counter[val]
        percentage = (count / total_elements) * 100
        print(f"{val:<10} {count:<15} {percentage:.2f}%")

    print("-" * 50)
    print(f"Total: {total_elements} elements")

def main():
    parser = argparse.ArgumentParser(description='Analyze the distribution of E2M1 values in matrix files')
    parser.add_argument('input_file', help='Path to the input matrix file')
    parser.add_argument('--no-approximate', action='store_true',
                        help='Do not approximate values to the nearest E2M1 representation')

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        return 1

    # Automatically generate output file path
    output_file = os.path.splitext(args.input_file)[0] + '.png'

    logger.info(f"Reading and counting E2M1 values from {args.input_file}...")

    # Process the file and count E2M1 values
    try:
        counter, total_elements = count_e2m1_values_from_file(
            args.input_file,
            approximate=not args.no_approximate
        )
    except Exception as e:
        logger.error(f"Error processing file {args.input_file}: {e}")
        return 1

    if total_elements == 0:
        logger.warning("No valid elements found in the input file.")
        return 1

    # Print distribution
    print_distribution(counter, total_elements)

    # Generate plot
    try:
        plot_distribution(counter, total_elements, output_file)
        logger.info(f"Plot saved to {output_file}")
    except Exception as e:
        logger.error(f"Error generating plot: {e}")
        # Don't return 1 here, as we still printed the distribution successfully

    return 0

if __name__ == "__main__":
    sys.exit(main())
