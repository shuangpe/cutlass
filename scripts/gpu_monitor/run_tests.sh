#!/bin/bash

# Extract script directory path
SCRIPT_DIR=$(dirname "$0")
PARSER_ROOT="$SCRIPT_DIR/parsers"

# Fixed internal parameters (not exposed to users)
WARMUP=3
COOLDOWN=5

for arg in "$@"; do
    if [ "$arg" = "-h" ] || [ "$arg" = "--help" ]; then
        python3 $PARSER_ROOT/parse_options.py "$@"
        exit 0
    fi
done

CONFIG_JSON=$(python3 $PARSER_ROOT/parse_options.py "$@")
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo $CONFIG_JSON
    exit 1
fi

# Extract configuration values from JSON
EXECUTABLE=$(echo "$CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['executable'])")
GPU_ID=$(echo "$CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['gpu_id'])")
INTERVAL=$(echo "$CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['interval'])")
ARGS=$(echo "$CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['args'])")
FREQUENCIES=$(echo "$CONFIG_JSON" | python3 -c "import sys, json; print(' '.join(map(str, json.load(sys.stdin)['frequencies'])))")
FREQUENCIES=($FREQUENCIES)  # Convert to array

# Set GPU frequency
set_gpu_frequency() {
    local gpu_id=$1
    local frequency=$2

    nvidia-smi --id="$gpu_id" --lock-gpu-clocks="$frequency","$frequency"

    if [ $? -eq 0 ]; then
        echo "✓ GPU $gpu_id frequency set to $frequency MHz"
        return 0
    else
        echo "✗ Failed to set GPU frequency"
        return 1
    fi
}

# Reset GPU frequency
reset_gpu_frequency() {
    local gpu_id=$1

    nvidia-smi --reset-gpu-clocks --id="$gpu_id"

    if [ $? -eq 0 ]; then
        echo "✓ GPU $gpu_id frequency reset to default"
        return 0
    else
        echo "✗ Failed to reset GPU frequency"
        return 1
    fi
}

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

# Initialize the CSV file with headers from both parsers
RESULTS_FILE="$OUTPUT_DIR/test_results.csv"
echo -n "Executable,Frequency," > "$RESULTS_FILE"

# Get headers from parse_console_log.py
CONSOLE_HEADERS=$(python3 $PARSER_ROOT/parse_console_log.py --csv-headers)
echo -n "$CONSOLE_HEADERS," >> "$RESULTS_FILE"

# Get headers from parse_nvsim_log.py (only median stats type)
NVSMI_HEADERS=$(python3 $PARSER_ROOT/parse_nvsim_log.py --csv-headers --stats-type median)
echo "$NVSMI_HEADERS" >> "$RESULTS_FILE"

echo "Created results file: $RESULTS_FILE"

# Run tests for each executable
for exe in "${EXECUTABLES[@]}"; do
    EXE_NAME=$(basename "$exe")
    echo -e "\n$(printf '%0.s#' {1..80})"
    echo "# Testing: $EXE_NAME"
    echo "$(printf '%0.s#' {1..80})"

    # Test with each frequency
    for freq in "${FREQUENCIES[@]}"; do
        echo -e "\n$(printf '%0.s-' {1..60})"

        # Frequency setting
        if [ "$freq" = "oob" ]; then
            echo "> Running with default GPU frequency"
            FREQ_SUFFIX="oob"
            # Reset GPU frequency to default
            reset_gpu_frequency $GPU_ID
        else
            echo "> Running with GPU frequency: $freq MHz"
            FREQ_SUFFIX="${freq}"
            # Set GPU frequency
            set_gpu_frequency $GPU_ID $freq
        fi

        # Create output file for GPU monitoring
        NVSMI_FILE="$OUTPUT_DIR/${EXE_NAME%.*}_${FREQ_SUFFIX}.log.txt"

        # Start GPU monitoring with better timing control
        echo "  📊 Starting GPU monitoring..."
        nvidia-smi -i $GPU_ID -q -a --loop-ms=$INTERVAL | grep -v -e{Fan,N/A,JPEG,OFA} > "$NVSMI_FILE" &
        MONITOR_PID=$!

        echo "  🔄 Waiting for monitoring to stabilize and warming up GPU for $WARMUP seconds..."
        sleep $WARMUP

        # Run test program
        echo "  🚀 Executing: $EXE_NAME"
        TEST_OUTPUT=$("$exe" $ARGS 2>&1)

        # Extract data from test output using Python parser
        echo "  📋 Results:"

        # Save test output to temporary file
        TMP_OUTPUT_FILE=$(mktemp)
        echo "$TEST_OUTPUT" > "$TMP_OUTPUT_FILE"

        # Display important parts of test output
        echo "$TEST_OUTPUT" | grep -E "Problem Size:|Avg runtime:|TFLOPS:|Stages:|TileShape:|GridDims:|HackLoadG2L:|Disposition:" | while read line; do
            echo "     $line"
        done

        # Wait for cooldown time and capture post-execution metrics
        echo "  📉 Cooling down and capturing post-execution metrics for $COOLDOWN seconds..."
        sleep $COOLDOWN

        # Terminate monitoring process
        echo "  ⏹️ Terminating GPU monitoring..."
        kill $MONITOR_PID 2>/dev/null

        # Wait for monitoring process to complete
        wait_count=0
        while kill -0 $MONITOR_PID 2>/dev/null; do
            sleep 1
            ((wait_count++))
            if [ $wait_count -gt 10 ]; then
                kill -9 $MONITOR_PID 2>/dev/null
                break
            fi
        done

        # Reset GPU frequency (if needed)
        if [ "$freq" != "oob" ]; then
            reset_gpu_frequency $GPU_ID
        fi

        # Now that the test is complete, process the results
        echo "  📊 Processing results..."

        # Get console log data in CSV format
        CONSOLE_DATA=$(python3 $PARSER_ROOT/parse_console_log.py --input "$TMP_OUTPUT_FILE" --format csv)

        # Get nvidia-smi monitoring data in CSV format (only median stats type)
        NVSMI_DATA=$(python3 $PARSER_ROOT/parse_nvsim_log.py "$NVSMI_FILE" --csv --stats-type median)

        # Combine the results and append to CSV file
        echo "$EXE_NAME,$freq,$CONSOLE_DATA,$NVSMI_DATA" >> "$RESULTS_FILE"

        # Cleanup temporary file
        rm -f "$TMP_OUTPUT_FILE"

        echo "  ✅ Test completed and results recorded"
    done
done

echo -e "\n$(printf '%0.s=' {1..80})"
echo "✨ All tests completed"
echo "📊 Results saved in: $OUTPUT_DIR/"
echo "📈 Test results summary available in: $RESULTS_FILE"
exit 0
