#!/usr/bin/env python3
import os
import re
import glob
import csv
import argparse
import datetime
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Extract performance data from CUTLASS SM100 MHA tests logs')
    parser.add_argument('--logs_dir', type=str, default='', help='Directory containing logs (default: look for logs_*mhz directories)')
    parser.add_argument('--output', type=str, default='performance_summary.csv', help='Output CSV file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    return parser.parse_args()

def extract_tflops(log_file):
    """Extract TFLOPS values from log file"""
    tflops_data = []
    with open(log_file, 'r') as f:
        content = f.read()
        # Look for patterns like ' [--] tma ws 256x128 acc fp32 persistent : 485.563 TFLOPS/s'
        matches = re.findall(r'\[--\]\s+([^:]+)\s*:\s*(\d+\.\d+)\s*TFLOPS', content)
        for desc, tflops in matches:
            tflops_data.append({
                'kernel': desc.strip(),
                'tflops': float(tflops)
            })
    return tflops_data

def extract_header_info(log_file):
    """Extract header information like ###### B 1 H 16 H_K 16 Q 1024 K 4096 D 128 Forward None #SM 148"""
    with open(log_file, 'r') as f:
        content = f.read()
        header_match = re.search(r'######\s+(.+)', content)
        if header_match:
            header = header_match.group(1)
            # Parse header into key-value pairs
            parts = header.split()
            header_info = {}
            for i in range(0, len(parts), 2):
                if i+1 < len(parts):
                    header_info[parts[i]] = parts[i+1]
            return header_info
    return {}

def extract_gpu_info(log_file):
    """Extract GPU information from log file"""
    with open(log_file, 'r') as f:
        content = f.read()
        # Look for GPU ID and frequency information
        gpu_id_match = re.search(r'GPU ID: (\d+)', content)
        freq_match = re.search(r'Frequency: (\d+) MHz', content)
        
        gpu_info = {}
        if gpu_id_match:
            gpu_info['GPU_ID'] = gpu_id_match.group(1)
        if freq_match:
            gpu_info['Frequency'] = freq_match.group(1)
            
        return gpu_info

def main():
    args = parse_args()
    
    # Determine base directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find all log directories
    if args.logs_dir:
        if os.path.isdir(args.logs_dir):
            log_dirs = [args.logs_dir]
        else:
            print(f"Error: Specified logs directory {args.logs_dir} does not exist")
            return 1
    else:
        log_dirs = glob.glob(os.path.join(script_dir, '**/logs_*mhz'), recursive=True)
    
    if not log_dirs:
        print("No log directories found")
        return 1
    
    if args.verbose:
        print(f"Found log directories: {log_dirs}")
    
    # Prepare CSV output
    output_file = args.output
    if not os.path.isabs(output_file):
        output_file = os.path.join(script_dir, output_file)
    
    # Define output field names
    fieldnames = [
        'Timestamp', 'Date', 'GPU_ID', 'Frequency', 'Executable', 'Test', 
        'B', 'H', 'H_K', 'Q', 'K', 'D', 'Mode', 'Kernel', 'TFLOPS'
    ]
    
    # Track statistics
    total_files = 0
    files_with_data = 0
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each log directory
        for log_dir in log_dirs:
            # Extract frequency from directory name
            freq_match = re.search(r'logs_(\d+)mhz', log_dir)
            default_frequency = freq_match.group(1) if freq_match else "unknown"
            
            log_files = glob.glob(os.path.join(log_dir, '*.log'))
            total_files += len(log_files)
            
            if args.verbose:
                print(f"Processing {len(log_files)} log files in {log_dir}")
            
            for log_file in log_files:
                # Extract timestamp from filename if available
                timestamp_match = re.search(r'(\d{8}_\d{6})', log_file)
                timestamp = timestamp_match.group(1) if timestamp_match else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                date = timestamp.split('_')[0] if '_' in timestamp else timestamp
                
                # Extract executable and test name from filename
                file_base = os.path.basename(log_file)
                name_parts = file_base.split('_')
                if len(name_parts) < 2:
                    if args.verbose:
                        print(f"Skipping file with invalid naming format: {file_base}")
                    continue
                
                # Extract test information parts
                if file_base.endswith('.log'):
                    executable = '_'.join(name_parts[:-1])
                    test = name_parts[-1].replace('.log', '')
                else:
                    # Fallback if naming doesn't follow expected pattern
                    executable = file_base
                    test = "unknown"
                
                # Extract TFLOPS data
                tflops_data = extract_tflops(log_file)
                if not tflops_data:
                    if args.verbose:
                        print(f"No TFLOPS data found in {file_base}")
                    continue
                
                files_with_data += 1
                
                # Extract header info and GPU info
                header_info = extract_header_info(log_file)
                gpu_info = extract_gpu_info(log_file)
                
                # Set frequency
                frequency = gpu_info.get('Frequency', default_frequency)
                
                # Write to CSV
                for entry in tflops_data:
                    row = {
                        'Timestamp': timestamp,
                        'Date': date,
                        'Frequency': frequency,
                        'GPU_ID': gpu_info.get('GPU_ID', 'unknown'),
                        'Executable': executable,
                        'Test': test,
                        'Kernel': entry['kernel'],
                        'TFLOPS': entry['tflops']
                    }
                    
                    # Add header info
                    for key, value in header_info.items():
                        if key in fieldnames:
                            row[key] = value
                    
                    writer.writerow(row)
    
    print(f"Processed {total_files} log files, found TFLOPS data in {files_with_data} files")
    print(f"Performance data extracted to {output_file}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
