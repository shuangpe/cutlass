#!/bin/bash

# Script to process all log files in a directory and generate a CSV report

if [ $# -lt 1 ]; then
    echo "Usage: $0 <output_directory>"
    echo "  <output_directory>: Directory containing log files to process"
    exit 1
fi

OUTPUT_DIR="$1"
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Directory '$OUTPUT_DIR' does not exist"
    exit 1
fi

# Extract script directory path
SCRIPT_DIR=$(dirname "$0")
PARSER_ROOT="$SCRIPT_DIR/parsers"

# Initialize the CSV file with headers
RESULTS_FILE="$OUTPUT_DIR/test_results.csv"
echo -n "Executable,Frequency," > "$RESULTS_FILE"

# Get headers from parse_console_log.py
CONSOLE_HEADERS=$(python3 $PARSER_ROOT/parse_console_log.py --csv-headers)
echo -n "$CONSOLE_HEADERS," >> "$RESULTS_FILE"

# Get headers from parse_nvsim_log.py (only median stats type)
NVSMI_HEADERS=$(python3 $PARSER_ROOT/parse_nvsim_log.py --csv-headers)
echo "$NVSMI_HEADERS" >> "$RESULTS_FILE"

echo "Created results file: $RESULTS_FILE"

# Find all nvsmi log files in the directory
find "$OUTPUT_DIR" -name "*_nvsmi.txt" | sort | while read nvsmi_file; do
    # Extract base name without _nvsmi.txt
    base_name=$(basename "$nvsmi_file" _nvsmi.txt)

    # Find corresponding perf log file
    perf_file="$OUTPUT_DIR/${base_name}_perf.log"

    if [ ! -f "$perf_file" ]; then
        echo "Warning: No matching performance log found for $nvsmi_file"
        continue
    fi

    echo "Processing logs for: $base_name"

    # Extract app name and frequency from base_name using a simpler approach
    # Example: "70_blackwell_fp16_gemm_1005" -> app_name="70_blackwell_fp16_gemm", freq="1005"
    # Example: "70_blackwell_fp16_gemm_load_once_1830" -> app_name="70_blackwell_fp16_gemm_load_once", freq="1830"

    # 从最后一个下划线分割文件名
    freq="${base_name##*_}"
    app_name="${base_name%_$freq}"

    echo "  App: $app_name, Frequency: $freq"

    # Get console log data in CSV format
    CONSOLE_DATA=$(python3 $PARSER_ROOT/parse_console_log.py --input "$perf_file" --format csv)

    # Get nvidia-smi monitoring data in CSV format (only median stats type)
    NVSMI_DATA=$(python3 $PARSER_ROOT/parse_nvsim_log.py "$nvsmi_file" --csv)

    # Combine the results and append to CSV file
    echo "$app_name,$freq,$CONSOLE_DATA,$NVSMI_DATA" >> "$RESULTS_FILE"

    echo "  ✅ Processed"
done

echo -e "\n$(printf '%0.s=' {1..80})"
echo "✨ All logs processed"
echo "📊 Results saved in: $RESULTS_FILE"
exit 0
