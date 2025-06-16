#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

def parse_timestamp(timestamp):
    """Parse timestamp from format YYYYMMDD-HH:MM:SS.mmmmmm"""
    try:
        return datetime.strptime(timestamp, '%Y%m%d-%H:%M:%S.%f')
    except ValueError:
        # Try format without microseconds
        return datetime.strptime(timestamp, '%Y%m%d-%H:%M:%S')

def read_gpu_metrics(csv_file):
    """Read CSV file and convert data types"""
    # Read CSV file, skip comment lines starting with #
    with open(csv_file, 'r') as f:
        comments = []
        for line in f:
            if line.startswith('#'):
                comments.append(line.strip('# \n'))
            else:
                break

    # Read data section
    df = pd.read_csv(csv_file, comment='#')

    # Convert timestamp to datetime objects
    df['timestamp'] = df['timestamp'].apply(parse_timestamp)

    # Convert non-numeric values (like "N/A") to NaN
    for col in df.columns:
        if col != 'timestamp':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df, comments

def extract_title_info(comments):
    """Extract GPU model, CUDA version and driver version information from metadata"""
    gpu_model = ""
    cuda_version = ""
    driver_version = ""

    for comment in comments:
        if comment.startswith("GPU:"):
            gpu_model = comment.replace("GPU:", "").strip()
        elif "cuda_version" in comment:
            parts = comment.split(":", 1)
            if len(parts) > 1:
                cuda_parts = parts[1].strip().split(",")
                if len(cuda_parts) > 2:
                    cuda_version = cuda_parts[2].strip()
        elif "driver_version" in comment:
            # Extract driver version number precisely, only numbers and dots
            import re
            match = re.search(r'driver_version:\s*([\d\.]+)', comment)
            if match:
                driver_version = match.group(1)

    title = "GPU Monitoring Metrics"
    if gpu_model:
        title = f"{gpu_model} {title}"
    if cuda_version and driver_version:
        title = f"{title} (CUDA {cuda_version}, Driver {driver_version})"

    return title

def get_filtered_stats(df, column, filter_zeros=True):
    """
    Calculate statistics for the specified column, optionally only for periods when GPU utilization is non-zero

    Args:
        df: DataFrame containing GPU monitoring data
        column: Column name for which to calculate statistics
        filter_zeros: Whether to only consider periods when GPU utilization is non-zero

    Returns:
        (mean, median, min_val, max_val) tuple
    """
    if filter_zeros and 'gpu_utilization (%)' in df.columns:
        # Only select rows where GPU utilization is greater than 0
        filtered_df = df[df['gpu_utilization (%)'] > 0]
        if len(filtered_df) == 0:  # If no data after filtering, use original data
            filtered_df = df
    else:
        filtered_df = df

    if column not in filtered_df.columns or filtered_df[column].isna().all():
        return None

    return (
        filtered_df[column].mean(),
        filtered_df[column].median(),
        filtered_df[column].min(),
        filtered_df[column].max()
    )

def get_stats_text(series, label, df=None, filter_zeros=True, unit=""):
    """Generate statistics text for a data series"""
    if series.isna().all():
        return f"{label}: No data"

    # If full DataFrame is provided, use filtered statistics
    if df is not None and filter_zeros:
        stats = get_filtered_stats(df, series.name, filter_zeros)
        if stats:
            mean, median, min_val, max_val = stats
        else:
            # If filtering fails, fall back to regular statistics
            mean = series.mean()
            median = series.median()
            min_val = series.min()
            max_val = series.max()
    else:
        # Use regular statistics
        mean = series.mean()
        median = series.median()
        min_val = series.min()
        max_val = series.max()

    # Format text, put unit in parentheses
    unit_str = f" ({unit})" if unit else ""
    return (f"{label}{unit_str}: Med={median:.1f}, Min={min_val:.1f}, "
            f"Max={max_val:.1f}, Avg={mean:.1f}")

def plot_gpu_metrics(df, output_file=None, show=False, title_prefix="", comments=None):
    """Plot GPU metrics chart"""
    # Extract relative time (seconds from the first data point)
    start_time = df['timestamp'].iloc[0]
    df['relative_time'] = [(t - start_time).total_seconds() for t in df['timestamp']]

    # Generate rich title
    rich_title = "GPU Monitoring Metrics"
    if comments:
        extracted_title = extract_title_info(comments)
        rich_title = extracted_title

    if title_prefix:
        rich_title = f"{title_prefix} {rich_title}"

    # Create chart - modify chart width, increase by 30%
    fig, axes = plt.subplots(4, 1, figsize=(15.6, 16), sharex=True)
    fig.suptitle(rich_title, fontsize=16, y=0.98)  # Increase spacing between title and chart

    # Reorder charts, from top to bottom: Utilization, Power, Frequency, Temperature

    # Plot utilization chart (now in the first position)
    axes[0].set_title('GPU Utilization (%)')
    # Change color from blue to red for GPU Utilization
    axes[0].plot(df['relative_time'], df['gpu_utilization (%)'], label='GPU Utilization', color='red')
    # Change color from magenta to blue for Memory Utilization
    axes[0].plot(df['relative_time'], df['memory_utilization (%)'], label='Memory Utilization', color='blue')

    # Add median lines
    gpu_util_filtered = df[df['gpu_utilization (%)'] > 0]['gpu_utilization (%)']
    mem_util_filtered = df[df['gpu_utilization (%)'] > 0]['memory_utilization (%)']

    if not gpu_util_filtered.empty:
        gpu_median = gpu_util_filtered.median()
        # Use green for all median lines
        line = axes[0].axhline(y=gpu_median, linestyle='--', color='green', alpha=0.8)
        # Increase font size for median labels from 9 to 11
        axes[0].text(0.01, gpu_median, f"{gpu_median:.1f}%",
                    verticalalignment='center', color='red', fontsize=11,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

    if not mem_util_filtered.empty:
        mem_median = mem_util_filtered.median()
        # Use green for Memory median line
        line = axes[0].axhline(y=mem_median, linestyle='--', color='green', alpha=0.8)
        # Update text color to match line color
        axes[0].text(0.01, mem_median, f"{mem_median:.1f}%",
                    verticalalignment='center', color='blue', fontsize=11,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

    axes[0].set_ylabel('Utilization (%)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True)

    # Add statistics to the GPU utilization chart
    gpu_util_stats = get_stats_text(df['gpu_utilization (%)'], "GPU", df, unit="%")
    mem_util_stats = get_stats_text(df['memory_utilization (%)'], "MEM", df, unit="%")
    stats_text = f"{gpu_util_stats}\n{mem_util_stats}"
    # Increase font size for statistics text from 9 to 11
    axes[0].text(0.5, 0.08, stats_text, transform=axes[0].transAxes,
                fontsize=11, verticalalignment='bottom', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Plot power chart (now in second position)
    axes[1].set_title('GPU Power (W)')
    axes[1].plot(df['relative_time'], df['power_draw (W)'], 'g-', label='Power Draw')

    # Add HBM power plot if available
    has_hbm_power = 'hbm_power (W)' in df.columns and not df['hbm_power (W)'].isna().all()
    if has_hbm_power:
        axes[1].plot(df['relative_time'], df['hbm_power (W)'], 'b-', label='HBM Power')
        
        # Add HBM power median line
        hbm_power_filtered = df[df['gpu_utilization (%)'] > 0]['hbm_power (W)']
        if not hbm_power_filtered.empty:
            hbm_power_median = hbm_power_filtered.median()
            line = axes[1].axhline(y=hbm_power_median, linestyle='--', color='green', alpha=0.8)
            # Add median label on the left
            axes[1].text(0.01, hbm_power_median, f"{hbm_power_median:.1f}W",
                        verticalalignment='center', color='blue', fontsize=11,
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

    # Add power median line
    power_filtered = df[df['gpu_utilization (%)'] > 0]['power_draw (W)']
    if not power_filtered.empty:
        power_median = power_filtered.median()
        line = axes[1].axhline(y=power_median, linestyle='--', color='green', alpha=0.8)
        # Add median label on the left
        axes[1].text(0.01, power_median, f"{power_median:.1f}W",
                    verticalalignment='center', color='green', fontsize=11,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

    if not df['instantaneous_power (W)'].isna().all():
        axes[1].plot(df['relative_time'], df['instantaneous_power (W)'], 'g--', label='Instantaneous Power')

    axes[1].set_ylabel('Power (W)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)

    # Add statistics to power chart
    power_stats = get_stats_text(df['power_draw (W)'], "Power", df, unit="W")
    stats_text = power_stats
    
    # Add HBM power statistics if available
    if has_hbm_power:
        hbm_power_stats = get_stats_text(df['hbm_power (W)'], "HBM", df, unit="W")
        stats_text = f"{power_stats}\n{hbm_power_stats}"
    
    # Increase font size for statistics text
    axes[1].text(0.5, 0.08, stats_text, transform=axes[1].transAxes,
                fontsize=11, verticalalignment='bottom', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Plot frequency chart (now in third position)
    axes[2].set_title('GPU Frequency (MHz)')
    axes[2].plot(df['relative_time'], df['sm_clock (MHz)'], label='SM Clock', color='red')
    axes[2].plot(df['relative_time'], df['graphics_clock (MHz)'], label='Graphics Clock', color='green')

    # Add frequency median lines
    sm_freq_filtered = df[df['gpu_utilization (%)'] > 0]['sm_clock (MHz)']
    graphics_freq_filtered = df[df['gpu_utilization (%)'] > 0]['graphics_clock (MHz)']

    if not sm_freq_filtered.empty:
        sm_median = sm_freq_filtered.median()
        line = axes[2].axhline(y=sm_median, linestyle='--', color='green', alpha=0.8)
        # Add median label on the left
        axes[2].text(0.01, sm_median, f"{sm_median:.1f}MHz",
                    verticalalignment='center', color='red', fontsize=11,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

    if not graphics_freq_filtered.empty:
        gfx_median = graphics_freq_filtered.median()
        line = axes[2].axhline(y=gfx_median, linestyle='--', color='green', alpha=0.8)
        # Add median label on the left
        axes[2].text(0.01, gfx_median, f"{gfx_median:.1f}MHz",
                    verticalalignment='center', color='green', fontsize=11,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

    axes[2].set_ylabel('Frequency (MHz)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True)

    # Add statistics to frequency chart
    sm_freq_stats = get_stats_text(df['sm_clock (MHz)'], "SM", df, unit="MHz")
    graphics_freq_stats = get_stats_text(df['graphics_clock (MHz)'], "GFX", df, unit="MHz")
    freq_stats_text = f"{sm_freq_stats}\n{graphics_freq_stats}"
    # Increase font size for statistics text
    axes[2].text(0.5, 0.08, freq_stats_text, transform=axes[2].transAxes,
                fontsize=11, verticalalignment='bottom', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Plot temperature chart (now in fourth position)
    axes[3].set_title('GPU Temperature (°C)')
    axes[3].plot(df['relative_time'], df['temperature (°C)'], 'r-', label='Temperature')

    # Add temperature median line
    temp_filtered = df[df['gpu_utilization (%)'] > 0]['temperature (°C)']
    if not temp_filtered.empty:
        temp_median = temp_filtered.median()
        line = axes[3].axhline(y=temp_median, linestyle='--', color='green', alpha=0.8)
        # Add median label on the left
        axes[3].text(0.01, temp_median, f"{temp_median:.1f}°C",
                    verticalalignment='center', color='red', fontsize=11,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

    axes[3].set_ylabel('Temperature (°C)')
    axes[3].set_xlabel('Time (seconds)')
    axes[3].legend(loc='upper right')
    axes[3].grid(True)

    # Add statistics to temperature chart
    temp_stats = get_stats_text(df['temperature (°C)'], "Temp", df, unit="°C")
    axes[3].text(0.5, 0.08, temp_stats, transform=axes[3].transAxes,
                fontsize=11, verticalalignment='bottom', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Beautify X axis - 修改这部分
    from matplotlib.ticker import MultipleLocator

    # 设置适当的主要和次要刻度，但不显示标签
    for ax in axes:
        # 设置适当的主要刻度位置，但不显示标签
        time_range = df['relative_time'].max() - df['relative_time'].min()
        
        if time_range < 1:  # Less than 1 second
            ax.xaxis.set_major_locator(MultipleLocator(0.1))
        elif time_range < 10:  # 1-10 seconds
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
        elif time_range < 60:  # Less than 1 minute
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
        else:  # More than 1 minute
            ax.xaxis.set_major_locator(MultipleLocator(2))

        # 隐藏所有子图的x轴刻度标签
        ax.set_xticklabels([])

        ax.set_xlim(df['relative_time'].min(), df['relative_time'].max())

    # 在底部子图保留"Time"标签，但不显示具体值
    axes[3].set_xlabel('Time')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    # Save or display the chart
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {output_file}")

    if show:
        plt.show()

    plt.close()

def crop_zero_utilization(df, margin=5):
    """
    Crop continuous periods of zero GPU utilization at the beginning and end, keeping specified number of boundary points

    Args:
        df: DataFrame containing GPU monitoring data
        margin: Number of zero-utilization points to keep as margin when cropping

    Returns:
        Cropped DataFrame
    """
    if 'gpu_utilization (%)' not in df.columns or len(df) <= margin*2:
        return df  # If no utilization column or too few data points, return original data

    # Find the first index with non-zero utilization
    start_idx = 0
    for i, util in enumerate(df['gpu_utilization (%)']):
        if not pd.isna(util) and util > 0:
            start_idx = max(0, i - margin)  # Keep some margin
            break

    # Find the last index with non-zero utilization
    end_idx = len(df) - 1
    for i in range(len(df)-1, -1, -1):
        if not pd.isna(df['gpu_utilization (%)'].iloc[i]) and df['gpu_utilization (%)'].iloc[i] > 0:
            end_idx = min(len(df)-1, i + margin)  # Keep some margin
            break

    # If the entire sequence is zero, keep all data
    if start_idx >= end_idx:
        return df

    # Crop the data
    return df.iloc[start_idx:end_idx+1].reset_index(drop=True)

def collect_statistics(df, csv_filename):
    """Collect statistics for each metric into a dictionary"""
    stats_dict = {'filename': os.path.basename(csv_filename)}

    # Use warnings to filter empty data calculation warnings
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Collect statistics for each metric
        for col in df.columns:
            if col not in ['timestamp', 'relative_time']:
                try:
                    # Get more accurate statistics, only considering data points where GPU utilization is greater than 0
                    filtered_stats = get_filtered_stats(df, col)
                    if filtered_stats:
                        mean_val, median_val, min_val, max_val = filtered_stats
                        stats_dict[f"{col}_mean"] = mean_val
                        stats_dict[f"{col}_median"] = median_val
                        stats_dict[f"{col}_min"] = min_val
                        stats_dict[f"{col}_max"] = max_val
                except Exception as e:
                    print(f"  Error collecting statistics for {col}: {e}")

    return stats_dict

def process_csv_file(csv_path, output=None, show=False, title='', crop=True, margin=10):
    """Process single CSV file and generate chart"""
    # If no output file is specified, create png subdirectory in CSV directory by default
    if not output and not show:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_dir = os.path.dirname(csv_path)

        # Create png subdirectory
        png_dir = os.path.join(output_dir, "png")
        if not os.path.exists(png_dir):
            os.makedirs(png_dir)
            print(f"Created directory: {png_dir}")

        output = os.path.join(png_dir, f"{base_name}.png")

    # Read data
    df, comments = read_gpu_metrics(csv_path)

    # Crop the beginning and ending parts where GPU utilization is zero
    original_length = len(df)
    if crop:
        df = crop_zero_utilization(df, margin=margin)
        if len(df) < original_length:
            print(f"Data cropped from {original_length} to {len(df)} points, removed zero utilization periods")

    # Output some statistics
    print(f"Read {len(df)} records from {os.path.basename(csv_path)}")
    print("Statistics:")
    
    # Format and print column headers
    print("  {:<25} {:>10} {:>10} {:>10} {:>10}".format(
        "Metric", "Mean", "Median", "Min", "Max"))
    print("  {:<25} {:>10} {:>10} {:>10} {:>10}".format(
        "-" * 25, "-" * 10, "-" * 10, "-" * 10, "-" * 10))

    # Collect statistics
    stats_dict = collect_statistics(df, csv_path)

    # Suppress warnings for mean calculation on empty slices
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Calculate and print useful statistics with consistent formatting
        for col in df.columns:
            if col != 'timestamp' and col != 'relative_time':
                try:
                    mean_val = df[col].mean()
                    median_val = df[col].median()
                    max_val = df[col].max()
                    min_val = df[col].min()

                    # Print formatted statistics
                    print("  {:<25} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}".format(
                        col, mean_val, median_val, min_val, max_val))
                except Exception as e:
                    print(f"  {col:<25} Error: {e}")

    # Plot chart
    plot_gpu_metrics(df, output, show, title, comments)

    # Display useful comments/metadata
    print("\nMetadata information:")
    for comment in comments:
        print(f"  {comment}")

    return output, stats_dict

def save_stats_to_csv(stats_list, folder_path):
    """Save collected statistics to a CSV file"""
    if not stats_list:
        print("No statistics to save")
        return None

    # Create DataFrame
    stats_df = pd.DataFrame(stats_list)

    # Sort column names to maintain consistent order
    cols = ['filename']
    metric_cols = [col for col in stats_df.columns if col != 'filename']
    metric_cols.sort()
    cols.extend(metric_cols)

    # Reorder columns
    stats_df = stats_df[cols]

    # Save to CSV file
    stats_file = os.path.join(folder_path, "summary_statistics.csv")
    stats_df.to_csv(stats_file, index=False)
    print(f"\nStatistics summary saved to: {stats_file}")
    return stats_file

def merge_with_tflops_csv(folder_path, all_stats):
    """
    Merge GPU monitoring statistics with tflops.csv to generate tflops_full.csv file

    Args:
        folder_path: Folder path containing tflops.csv
        all_stats: List of statistics collected from GPU monitoring files
    """
    tflops_file = os.path.join(folder_path, "tflops.csv")
    if not os.path.exists(tflops_file):
        print(f"tflops.csv file does not exist: {tflops_file}")
        return

    # Read tflops.csv
    tflops_df = pd.read_csv(tflops_file)
    print(f"Read tflops.csv, containing {len(tflops_df)} rows")

    # Prepare statistics data dictionary for merging
    stats_dict = {}
    for stat in all_stats:
        filename = stat['filename']

        # Extract Executable and Frequency information from filename
        # First, remove the suffix Mhz.csv
        base_name = filename.replace("Mhz.csv", "")

        # Find the position of the last underscore
        last_underscore = base_name.rfind('_')
        if last_underscore == -1:
            continue

        # The part before is executable, the part after is frequency
        executable = base_name[:last_underscore]
        frequency = base_name[last_underscore+1:]

        key = (executable, frequency)
        stats_dict[key] = stat

    # Create result DataFrame
    result_df = tflops_df.copy()

    # Determine the order of metrics to add
    all_metrics = set()
    for stat in all_stats:
        for key in stat.keys():
            if key != 'filename':
                metric_base = key.rsplit('_', 1)[0]  # Remove _mean, _median etc. suffixes
                all_metrics.add(metric_base)

    # Define the priority order of metrics
    priority_metrics = [
        'gpu_utilization (%)',
        'power_draw (W)',
        'sm_clock (MHz)',
        'temperature (°C)'
    ]

    # Order metrics by priority
    ordered_metrics = []
    # First, add the prioritized metrics (if they exist in the data)
    for metric in priority_metrics:
        if metric in all_metrics:
            ordered_metrics.append(metric)
            all_metrics.remove(metric)

    # Then, add the remaining metrics (in alphabetical order)
    ordered_metrics.extend(sorted(all_metrics))

    # Function to convert metric names to camelCase
    def convert_to_camel_case(metric_name, stat_type):
        # Extract unit (if any)
        unit = ""
        if "(" in metric_name and ")" in metric_name:
            unit_start = metric_name.find("(")
            unit_end = metric_name.find(")")
            unit = metric_name[unit_start:unit_end+1]
            metric_name = metric_name[:unit_start].strip()

        # Split words and convert to camel case
        parts = metric_name.replace("_", " ").split()
        camel_case = "".join(word.capitalize() for word in parts)

        # Add statistical measure
        camel_case += stat_type.capitalize()

        # Add unit - at the end
        return camel_case + unit if unit else camel_case

    # Store mapping of old column names to new ones
    column_mapping = {}

    # Add new columns for each metric - in order: median, mean, max, min
    for suffix in ['median', 'mean', 'max', 'min']:
        for metric in ordered_metrics:
            old_col_name = f"{metric}_{suffix}"
            new_col_name = convert_to_camel_case(metric, suffix)
            column_mapping[old_col_name] = new_col_name
            result_df[old_col_name] = None

    # Fill statistics data
    for i, row in result_df.iterrows():
        executable = row['Executable']
        frequency = row['Frequency']
        key = (executable, frequency)

        if key in stats_dict:
            stat = stats_dict[key]
            # Fill in order: median, mean, max, min
            for suffix in ['median', 'mean', 'max', 'min']:
                for metric in ordered_metrics:
                    col_name = f"{metric}_{suffix}"
                    metric_key = f"{metric}_{suffix}"
                    if metric_key in stat:
                        # Format values to 2 decimal places
                        result_df.at[i, col_name] = float(f"{stat[metric_key]:.2f}")

    # Rename columns
    result_df = result_df.rename(columns=column_mapping)

    # Save to tflops_full.csv
    output_file = os.path.join(folder_path, "tflops_full.csv")
    result_df.to_csv(output_file, index=False)
    print(f"Merged data saved to: {output_file}")
    print(f"Column names converted to camelCase format")

def main():
    parser = argparse.ArgumentParser(description='Visualize GPU monitoring data')
    parser.add_argument('input_path', help='CSV data file or folder containing CSV files')
    parser.add_argument('-o', '--output', help='Output image file path (only valid when processing a single file)')
    parser.add_argument('-s', '--show', action='store_true', help='Display chart')
    parser.add_argument('-t', '--title', default='', help='Chart title prefix')
    parser.add_argument('-c', '--crop', action='store_true', help='Crop periods of zero GPU utilization at beginning and end', default=True)
    parser.add_argument('-m', '--margin', type=int, default=10, help='Number of zero-utilization data points to keep as margin when cropping')

    args = parser.parse_args()

    # Check if input path is a file or folder
    if os.path.isfile(args.input_path):
        # Process single file
        output_file, stats = process_csv_file(args.input_path, args.output, args.show, args.title, args.crop, args.margin)

        # 如果是单个文件，也保存统计信息
        if stats:
            save_stats_to_csv([stats], os.path.dirname(args.input_path))

    elif os.path.isdir(args.input_path):
        # Process all CSV files in the folder
        print(f"Scanning folder: {args.input_path}")
        csv_files = [f for f in os.listdir(args.input_path) if f.lower().endswith('.csv') and f != "tflops.csv" and f != "tflops_full.csv" and f != "summary_statistics.csv"]

        if not csv_files:
            print("No CSV files found!")
            return

        print(f"Found {len(csv_files)} CSV files")
        processed_files = []
        all_stats = []  # 收集所有统计信息

        for csv_file in csv_files:
            csv_path = os.path.join(args.input_path, csv_file)
            print(f"\nProcessing file: {csv_file}")
            try:
                output_file, stats = process_csv_file(csv_path, None, args.show, args.title, args.crop, args.margin)
                processed_files.append((csv_file, output_file))
                if stats:
                    all_stats.append(stats)
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")

        # 保存汇总统计信息
        if all_stats:
            save_stats_to_csv(all_stats, args.input_path)

            # 如果目录中存在 tflops.csv 文件，则合并统计信息
            if os.path.exists(os.path.join(args.input_path, "tflops.csv")):
                merge_with_tflops_csv(args.input_path, all_stats)

    else:
        print(f"Error: Input path '{args.input_path}' does not exist!")

if __name__ == "__main__":
    main()
