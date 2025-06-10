#!/bin/bash

# Hardcoded GPU frequency settings
FREQUENCIES=("oob" "1500" "1305" "1005")

# Default parameters
GPU_ID=0
INTERVAL=0.05
WARMUP=1
COOLDOWN=1
ARGS=""

# Set GPU frequency
set_gpu_frequency() {
    local gpu_id=$1
    local frequency=$2

    echo "Setting GPU $gpu_id frequency to $frequency MHz"

    nvidia-smi -i $gpu_id -ac $(nvidia-smi -i $gpu_id --query-gpu=clocks.mem --format=csv,noheader | cut -d' ' -f1),$frequency

    if [ $? -eq 0 ]; then
        echo "Successfully set GPU $gpu_id frequency to $frequency MHz"
        return 0
    else
        echo "Failed to set GPU frequency"
        return 1
    fi
}

# Reset GPU frequency
reset_gpu_frequency() {
    local gpu_id=$1

    echo "Resetting GPU $gpu_id frequency to default"
    nvidia-smi -i $gpu_id -rac

    if [ $? -eq 0 ]; then
        echo "Successfully reset GPU $gpu_id frequency"
        return 0
    else
        echo "Failed to reset GPU frequency"
        return 1
    fi
}

# Help information
function show_help {
    echo "Run tests and monitor GPU metrics"
    echo "Usage: $0 -e EXECUTABLE -g GPU_ID [-a ARGS] [-w WARMUP] [-c COOLDOWN] [-i INTERVAL]"
    echo ""
    echo "Options:"
    echo "  -e, --executable   Test executable or directory"
    echo "  -g, --gpu          GPU ID"
    echo "  -a, --args         Arguments to pass to the executable"
    echo "  -w, --warmup       Warmup time (seconds) (default: 1)"
    echo "  -c, --cooldown     Cooldown time (seconds) (default: 1)"
    echo "  -i, --interval     Monitoring sampling interval (seconds) (default: 0.05)"
    echo "  -h, --help         Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -e|--executable)
            EXECUTABLE="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -a|--args)
            ARGS="$2"
            shift 2
            ;;
        -w|--warmup)
            WARMUP="$2"
            shift 2
            ;;
        -c|--cooldown)
            COOLDOWN="$2"
            shift 2
            ;;
        -i|--interval)
            INTERVAL="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Check required parameters
if [ -z "$EXECUTABLE" ]; then
    echo "Error: Must specify executable or directory (-e)"
    show_help
fi

if [ -z "$GPU_ID" ]; then
    echo "Error: Must specify GPU ID (-g)"
    show_help
fi

# Determine executables list and output directory
EXECUTABLES=()
if [ -d "$EXECUTABLE" ]; then
    echo "Directory specified: $EXECUTABLE"
    # Find all executable files in the directory
    for file in "$EXECUTABLE"/*; do
        if [ -f "$file" ] && [ -x "$file" ]; then
            EXECUTABLES+=("$file")
        fi
    done

    if [ ${#EXECUTABLES[@]} -eq 0 ]; then
        echo "No executable files found in directory: $EXECUTABLE"
        exit 1
    fi

    echo "Found ${#EXECUTABLES[@]} executable files"
    for exe in "${EXECUTABLES[@]}"; do
        echo "  - $(basename "$exe")"
    done

    # Create output directory based on directory name
    DIR_BASE=$(basename "$EXECUTABLE")
    OUTPUT_DIR=$(date +%Y%m%d)_gpu_metrics_${DIR_BASE}_0
    i=1
    while [ -d "$OUTPUT_DIR" ]; do
        OUTPUT_DIR=$(date +%Y%m%d)_gpu_metrics_${DIR_BASE}_$i
        ((i++))
    done
else
    # Single executable file
    if [ ! -f "$EXECUTABLE" ] || [ ! -x "$EXECUTABLE" ]; then
        echo "Error: $EXECUTABLE is not an executable file"
        exit 1
    fi
    EXECUTABLES=("$EXECUTABLE")

    # Create output directory based on executable name
    EXE_BASE=$(basename "$EXECUTABLE")
    EXE_NAME="${EXE_BASE%.*}"
    OUTPUT_DIR=$(date +%Y%m%d)_gpu_metrics_${EXE_NAME}_0
    i=1
    while [ -d "$OUTPUT_DIR" ]; do
        OUTPUT_DIR=$(date +%Y%m%d)_gpu_metrics_${EXE_NAME}_$i
        ((i++))
    done
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "All output files will be saved to: $OUTPUT_DIR"

# Run tests for each executable
for exe in "${EXECUTABLES[@]}"; do
    exe_name=$(basename "$exe")
    echo -e "\n$(printf '%0.s#' {1..100})"
    echo "Testing executable: $exe_name"
    echo "$(printf '%0.s#' {1..100})"

    # Test with each frequency
    for freq in "${FREQUENCIES[@]}"; do
        echo -e "\n$(printf '%0.s=' {1..80})"

        # Frequency setting
        if [ "$freq" = "oob" ]; then
            echo "Running test with default GPU frequency (resetting to default)"
            freq_suffix="oobMhz"
            # Reset GPU frequency to default
            reset_gpu_frequency $GPU_ID
        else
            echo "Running test with GPU frequency set to $freq MHz"
            freq_suffix="${freq}Mhz"
            # Set GPU frequency
            set_gpu_frequency $GPU_ID $freq
        fi

        # Create output file
        output_file="$OUTPUT_DIR/${exe_name%.*}_${freq_suffix}.csv"
        echo "Output will be saved to: $output_file"
        echo "$(printf '%0.s=' {1..80})"

        # Start GPU monitoring using path relative to this script
        SCRIPT_DIR="$(dirname "$0")"
        python3 "${SCRIPT_DIR}/gpu_monitor.py" -g $GPU_ID -i $INTERVAL -o $output_file &
        MONITOR_PID=$!

        # Wait for monitoring script to start
        sleep 1

        # Wait for warmup time
        echo "Collecting baseline metrics, warming up for $WARMUP seconds..."
        sleep $WARMUP

        # Run test program
        echo "Executing command: $exe $ARGS"
        "$exe" $ARGS
        echo "Test completed"

        # Wait for cooldown time
        echo "Test ended, continuing to monitor metrics cooldown for $COOLDOWN seconds..."
        sleep $COOLDOWN

        # Terminate monitoring process
        echo "Terminating GPU monitoring process..."
        kill $MONITOR_PID 2>/dev/null

        # Wait for monitoring process to complete
        wait_time=0
        while kill -0 $MONITOR_PID 2>/dev/null; do
            sleep 1
            ((wait_time++))
            if [ $wait_time -gt 10 ]; then
                echo "Monitoring process didn't terminate within expected time, forcing termination"
                kill -9 $MONITOR_PID 2>/dev/null
                break
            fi
        done

        # Reset GPU frequency (if needed)
        if [ "$freq" != "oob" ]; then
            reset_gpu_frequency $GPU_ID
        fi

        echo "GPU monitoring data saved to: $output_file"
    done
done

echo -e "\nAll tests completed. Results saved in $OUTPUT_DIR/"
exit 0
