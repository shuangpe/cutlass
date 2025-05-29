#!/usr/bin/env python3
import os
import re
import glob
import csv
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Extract performance data from CUTLASS SM100 MHA tests logs')
    parser.add_argument('--logs_dir', type=str, required=True, help='Directory containing logs')
    # 默认输出文件名为输入目录名+"combined.csv"
    default_output = None
    args, unknown = parser.parse_known_args()
    if args.logs_dir:
        dir_name = os.path.basename(os.path.normpath(args.logs_dir))
        default_output = dir_name + "_combined.csv"
    parser.add_argument('--output', type=str, default=default_output, help='Output CSV file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    return parser.parse_args()

def parse_filename(filename):
    # 例：77_blackwell_mla_2sm_cpasync_fp16_1300mhz.log
    base = os.path.basename(filename)
    # 去掉后缀
    base = re.sub(r'_\d+mhz\.log$', '', base)
    # 精度是最后一个下划线后的内容
    parts = base.split('_')
    if len(parts) >= 2:
        precision = parts[-1]
        test_name = '_'.join(parts[:-1])
    else:
        precision = ''
        test_name = base
    return test_name, precision

def parse_parameters(param_line, header_line=None):
    # 先从Parameters行提取
    params = {
        'b': '', 'h_k': '', 'h': '', 'q': '', 'k': '', 'd': '', 'mask': 'no', 'clear-cache': 0, 'cache-only': 0
    }
    for part in param_line.split():
        if part.startswith('--b='):
            params['b'] = part.split('=')[1]
        elif part.startswith('--h_k='):
            params['h_k'] = part.split('=')[1]
        elif part.startswith('--h='):
            params['h'] = part.split('=')[1]
        elif part.startswith('--q='):
            params['q'] = part.split('=')[1]
        elif part.startswith('--k='):
            params['k'] = part.split('=')[1]
        elif part.startswith('--d='):
            params['d'] = part.split('=')[1]
        elif part.startswith('--mask='):
            params['mask'] = part.split('=')[1]
        elif part == '--clear-cache':
            params['clear-cache'] = 1
        elif part == '--cache-only':
            params['cache-only'] = 1
    # 如果有key没取到，再从######行补全
    if header_line:
        # 例：###### B 1 H 16 Q 1024 K 4096 D 128 Backward Full #SM 148
        header = header_line.replace('#', '').strip()
        items = header.split()
        for i, item in enumerate(items):
            if item == 'B' and not params['b'] and i+1 < len(items):
                params['b'] = items[i+1]
            elif item == 'H' and not params['h'] and i+1 < len(items):
                params['h'] = items[i+1]
            elif item == 'Q' and not params['q'] and i+1 < len(items):
                params['q'] = items[i+1]
            elif item == 'K' and not params['k'] and i+1 < len(items):
                params['k'] = items[i+1]
            elif item == 'D' and not params['d'] and i+1 < len(items):
                params['d'] = items[i+1]
        # mask有可能在header里没有，保持原样
    return params

def extract_max_tflops(block):
    # 匹配所有TFLOPS
    matches = re.findall(r':\s*([\d\.]+)\s*TFLOPS', block)
    if matches:
        return max(float(x) for x in matches)
    return ''

def main():
    args = parse_args()
    logs_dir = args.logs_dir
    output_file = args.output
    if not os.path.isabs(output_file):
        output_file = os.path.join(os.getcwd(), output_file)
    fieldnames = ['GPUFreq', 'TestName', 'Precision', 'TestConfig', 'b', 'h_k', 'h', 'q', 'k', 'd', 'mask', 'clear-cache', 'cache-only', 'TFlOPS']
    log_files = glob.glob(os.path.join(logs_dir, '*.log'))
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for log_file in log_files:
            if '77_blackwell_mla' in os.path.basename(log_file):
                continue
            if '77_blackwell_fmha_gen' in os.path.basename(log_file):
                continue
            # GPUFreq从文件名
            m = re.search(r'_(\d+)mhz\.log$', log_file)
            gpufreq = m.group(1) if m else ''
            test_name, precision = parse_filename(log_file)
            with open(log_file, 'r') as lf:
                content = lf.read()
            # 按空行分割为多个测试块
            blocks = [b for b in re.split(r'\n\s*\n', content) if b.strip()]
            for block in blocks:
                # 找到TestConfig
                test_config = ''
                for line in block.splitlines():
                    m = re.match(r'Running: .* with (\S+)', line.strip())
                    if m:
                        test_config = m.group(1)
                        break
                # 找到Parameters行和######行
                param_line = None
                header_line = None
                for line in block.splitlines():
                    if line.strip().startswith('Parameters:'):
                        param_line = line.strip().replace('Parameters:', '').strip()
                    if line.strip().startswith('######'):
                        header_line = line.strip()
                if not param_line:
                    continue
                params = parse_parameters(param_line, header_line)
                tflops = extract_max_tflops(block)
                row = {
                    'GPUFreq': gpufreq,
                    'TestName': test_name,
                    'Precision': precision,
                    'TestConfig': test_config,
                    'b': params['b'],
                    'h_k': params['h_k'],
                    'h': params['h'],
                    'q': params['q'],
                    'k': params['k'],
                    'd': params['d'],
                    'mask': params['mask'],
                    'clear-cache': params['clear-cache'],
                    'cache-only': params['cache-only'],
                    'TFlOPS': tflops
                }
                writer.writerow(row)
    if args.verbose:
        print(f"Processed {len(log_files)} log files. Output: {output_file}")

if __name__ == "__main__":
    main()
