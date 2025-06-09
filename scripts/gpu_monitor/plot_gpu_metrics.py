#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
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

    # 如果提供了完整DataFrame，使用过滤后的统计
    if df is not None and filter_zeros:
        stats = get_filtered_stats(df, series.name, filter_zeros)
        if stats:
            mean, median, min_val, max_val = stats
        else:
            # 如果过滤统计失败，回退到普通统计
            mean = series.mean()
            median = series.median()
            min_val = series.min()
            max_val = series.max()
    else:
        # 使用普通统计
        mean = series.mean()
        median = series.median()
        min_val = series.min()
        max_val = series.max()

    # 格式化文本，将单位放在括号里
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
    axes[0].plot(df['relative_time'], df['gpu_utilization (%)'], label='GPU Utilization', color='blue')
    axes[0].plot(df['relative_time'], df['memory_utilization (%)'], label='Memory Utilization', color='magenta')  # Change to more visible magenta

    # Add median lines
    gpu_util_filtered = df[df['gpu_utilization (%)'] > 0]['gpu_utilization (%)']
    mem_util_filtered = df[df['gpu_utilization (%)'] > 0]['memory_utilization (%)']

    if not gpu_util_filtered.empty:
        gpu_median = gpu_util_filtered.median()
        line = axes[0].axhline(y=gpu_median, linestyle='--', color='lightblue', alpha=0.8)
        # Increase font size for median labels from 9 to 11
        axes[0].text(0.01, gpu_median, f"{gpu_median:.1f}%",
                    verticalalignment='center', color='blue', fontsize=11,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

    if not mem_util_filtered.empty:
        mem_median = mem_util_filtered.median()
        line = axes[0].axhline(y=mem_median, linestyle='--', color='lightpink', alpha=0.8)
        # Increase font size for median labels from 9 to 11
        axes[0].text(0.01, mem_median, f"{mem_median:.1f}%",
                    verticalalignment='center', color='magenta', fontsize=11,
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

    # 添加功率中值虚线
    power_filtered = df[df['gpu_utilization (%)'] > 0]['power_draw (W)']
    if not power_filtered.empty:
        power_median = power_filtered.median()
        line = axes[1].axhline(y=power_median, linestyle='--', color='lightgreen', alpha=0.8)
        # 在左侧添加中值标签
        axes[1].text(0.01, power_median, f"{power_median:.1f}W",
                    verticalalignment='center', color='green', fontsize=11,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

    if not df['instantaneous_power (W)'].isna().all():
        axes[1].plot(df['relative_time'], df['instantaneous_power (W)'], 'g--', label='Instantaneous Power')

    axes[1].set_ylabel('Power (W)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)

    # 添加统计信息到功率图
    power_stats = get_stats_text(df['power_draw (W)'], "Power", df, unit="W")
    # Increase font size for statistics text
    axes[1].text(0.5, 0.08, power_stats, transform=axes[1].transAxes,
                fontsize=11, verticalalignment='bottom', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Plot frequency chart (now in third position)
    axes[2].set_title('GPU Frequency (MHz)')
    axes[2].plot(df['relative_time'], df['sm_clock (MHz)'], label='SM Clock', color='red')
    axes[2].plot(df['relative_time'], df['graphics_clock (MHz)'], label='Graphics Clock', color='green')

    # 添加频率中值虚线
    sm_freq_filtered = df[df['gpu_utilization (%)'] > 0]['sm_clock (MHz)']
    graphics_freq_filtered = df[df['gpu_utilization (%)'] > 0]['graphics_clock (MHz)']

    if not sm_freq_filtered.empty:
        sm_median = sm_freq_filtered.median()
        line = axes[2].axhline(y=sm_median, linestyle='--', color='lightcoral', alpha=0.8)
        # 在左侧添加中值标签
        axes[2].text(0.01, sm_median, f"{sm_median:.1f}MHz",
                    verticalalignment='center', color='red', fontsize=11,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

    if not graphics_freq_filtered.empty:
        gfx_median = graphics_freq_filtered.median()
        line = axes[2].axhline(y=gfx_median, linestyle='--', color='lightgreen', alpha=0.8)
        # 在左侧添加中值标签
        axes[2].text(0.01, gfx_median, f"{gfx_median:.1f}MHz",
                    verticalalignment='center', color='green', fontsize=11,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

    axes[2].set_ylabel('Frequency (MHz)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True)

    # 添加统计信息到频率图
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

    # 添加温度中值虚线
    temp_filtered = df[df['gpu_utilization (%)'] > 0]['temperature (°C)']
    if not temp_filtered.empty:
        temp_median = temp_filtered.median()
        line = axes[3].axhline(y=temp_median, linestyle='--', color='lightcoral', alpha=0.8)
        # 在左侧添加中值标签
        axes[3].text(0.01, temp_median, f"{temp_median:.1f}°C",
                    verticalalignment='center', color='red', fontsize=11,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

    axes[3].set_ylabel('Temperature (°C)')
    axes[3].set_xlabel('Time (seconds)')
    axes[3].legend(loc='upper right')
    axes[3].grid(True)

    # 添加统计信息到温度图
    temp_stats = get_stats_text(df['temperature (°C)'], "Temp", df, unit="°C")
    axes[3].text(0.5, 0.08, temp_stats, transform=axes[3].transAxes,
                fontsize=11, verticalalignment='bottom', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Beautify X axis - modify this part
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    # Custom formatter function to convert seconds to "seconds.milliseconds" format
    def format_time(x, pos):
        total_seconds = int(x)  # Integer part is total seconds
        milliseconds = int((x % 1) * 1000)  # Decimal part converted to milliseconds
        # Display in 10ms units
        return f"{total_seconds}.{milliseconds//10:02d}"

    for ax in axes:
        ax.xaxis.set_major_formatter(FuncFormatter(format_time))
        # 设置合适的主刻度和次刻度
        time_range = df['relative_time'].max() - df['relative_time'].min()
        if time_range < 1:  # 小于1秒
            ax.xaxis.set_major_locator(MultipleLocator(0.1))  # 每0.1秒一个主刻度
        elif time_range < 10:  # 1-10秒
            ax.xaxis.set_major_locator(MultipleLocator(0.5))  # 每0.5秒一个主刻度
        elif time_range < 60:  # 小于1分钟
            ax.xaxis.set_major_locator(MultipleLocator(0.5))  # 每0.5秒一个主刻度
        else:  # 大于1分钟
            ax.xaxis.set_major_locator(MultipleLocator(2))  # 每2秒一个主刻度，大范围时保持更低密度

        # 设置横轴标签倾斜45度
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')  # 水平对齐方式为右对齐

        ax.set_xlim(df['relative_time'].min(), df['relative_time'].max())

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

def process_csv_file(csv_path, output=None, show=False, title='', crop=True, margin=10):
    """Process single CSV file and generate chart"""
    # If no output file is specified, use the same directory and base filename by default
    if not output and not show:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_dir = os.path.dirname(csv_path)
        output = os.path.join(output_dir, f"{base_name}.png")

    # Read data
    df, comments = read_gpu_metrics(csv_path)

    # Crop the beginning and ending parts where GPU utilization is zero
    original_length = len(df)
    if crop:
        df = crop_zero_utilization(df, margin=margin)
        if len(df) < original_length:
            print(f"Data cropped from {original_length} to {len(df)} points, removed zero utilization periods")

    # Output some statistics
    print(f"Read {len(df)} records")
    print("Statistics:")

    # Suppress warnings for mean calculation on empty slices
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Calculate and print useful statistics (including median) - one line per metric
        for col in df.columns:
            if col != 'timestamp' and col != 'relative_time':
                try:
                    mean_val = df[col].mean()
                    median_val = df[col].median()
                    max_val = df[col].max()
                    min_val = df[col].min()

                    # All statistics in one line
                    print(f"  {col}: Mean={mean_val:.2f}, Median={median_val:.2f}, Max={max_val:.2f}, Min={min_val:.2f}")
                except Exception as e:
                    print(f"  {col}: Processing error - {e}")

    # Plot chart
    plot_gpu_metrics(df, output, show, title, comments)

    # Display useful comments/metadata
    print("\nMetadata information:")
    for comment in comments:
        print(f"  {comment}")

    return output

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
        process_csv_file(args.input_path, args.output, args.show, args.title, args.crop, args.margin)
    elif os.path.isdir(args.input_path):
        # Process all CSV files in the folder
        print(f"Scanning folder: {args.input_path}")
        csv_files = [f for f in os.listdir(args.input_path) if f.lower().endswith('.csv')]

        if not csv_files:
            print("No CSV files found!")
            return

        print(f"Found {len(csv_files)} CSV files")
        processed_files = []

        for csv_file in csv_files:
            csv_path = os.path.join(args.input_path, csv_file)
            print(f"\nProcessing file: {csv_file}")
            try:
                output_file = process_csv_file(csv_path, None, args.show, args.title, args.crop, args.margin)
                processed_files.append((csv_file, output_file))
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")

        # Print processing summary
        print("\nProcessing summary:")
        for csv_file, output_file in processed_files:
            print(f"  {csv_file} -> {os.path.basename(output_file)}")
    else:
        print(f"Error: Input path '{args.input_path}' does not exist!")

if __name__ == "__main__":
    main()
