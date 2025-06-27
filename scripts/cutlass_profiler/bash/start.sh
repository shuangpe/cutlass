#!/bin/bash
set -euo pipefail

# Global variables for argument parsing results
BUILD_APP=false
VERIFY_APP=false
APP_NAMES=()

# Directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(realpath "$SCRIPT_DIR/../../../build")"

# Global app name mapping
declare -A APP_MAP=( ["0"]="baseline" ["1"]="baseline.zeromask" ["2"]="hacking" )

# Function to build the application
build_baseline() {
  baseline_dir="$BUILD_DIR/baseline"
  echo "Building the baseline..."
  mkdir -p "$baseline_dir"
  cd "$baseline_dir"
  cmake ../.. \
    -DCUTLASS_NVCC_ARCHS=100a \
    -DCUTLASS_ENABLE_TESTS=OFF \
    -DCUTLASS_LIBRARY_OPERATIONS=gemm \
    -DCUTLASS_LIBRARY_KERNELS=gemm_*_2sm \
    -DCUTLASS_LIBRARY_IGNORE_KERNELS=stream
  make cutlass_profiler -j32
}

build_hacking() {
  hacking_dir="$BUILD_DIR/hacking"
  echo "Building the hacking..."
  mkdir -p "$hacking_dir"
  cd "$hacking_dir"
  cmake ../.. \
    -DCUTLASS_NVCC_ARCHS=100a \
    -DCUTLASS_ENABLE_TESTS=OFF \
    -DHACK_GEMM_WRITE_SLM_ONCE=1 \
    -DCUTLASS_LIBRARY_OPERATIONS=gemm \
    -DCUTLASS_LIBRARY_KERNELS=gemm_*_2sm \
    -DCUTLASS_LIBRARY_IGNORE_KERNELS=stream
  make cutlass_profiler -j32
}

build_app() {
  echo "Building the application..."
  build_baseline
  build_hacking
  echo "Build completed."
}

run_profile() {
  local assets_dir="$SCRIPT_DIR/assets"
  local cfg_file="$1"
  local out_dir="$2"
  local tag="$3"
  bash "$assets_dir/batch_profile.sh" --output "$out_dir" --tag "$tag" "$assets_dir/config/$cfg_file" | tee "full_log.$tag.txt"
}

run_app() {
  local app_names=("$@")
  local out_dir="$SCRIPT_DIR/b200_bench"
  for app_name in "${app_names[@]}"; do
    if [[ "$app_name" == "baseline" ]]; then
      echo "Running baseline application..."
      run_profile "baseline.cfg" "$out_dir" "baseline"
    elif [[ "$app_name" == "baseline.zeromask" ]]; then
      echo "Running baseline.zeromask application..."
      run_profile "baseline.zeromask.cfg" "$out_dir" "baseline.zeromask"
    elif [[ "$app_name" == "hacking" ]]; then
      echo "Running hacking application..."
      run_profile "hacking.cfg" "$out_dir" "hacking"
    else
      echo "Unknown application: $app_name"
      exit 1
    fi
  done
}

# Function to verify the application
verify_app() {
  echo "Verifying the application..."
}

# Function to parse command line arguments and set global variables
parse_args() {
  while [[ "$#" -gt 0 ]]; do
    case $1 in
      --build)
        BUILD_APP=true
        ;;
      --app)
        IFS=',' read -r -a app_array <<< "$2"
        for app in "${app_array[@]}"; do
          if [[ ${APP_MAP[$app]+_} ]]; then
            APP_NAMES+=("${APP_MAP[$app]}")
          else
            APP_NAMES+=("$app")
          fi
        done
        shift
        ;;
      --verify)
        VERIFY_APP=true
        ;;
      *)
        echo "Unknown parameter passed: $1"
        exit 1
        ;;
    esac
    shift
  done
}

# Main function
main() {
  parse_args "$@"

  if [[ "$BUILD_APP" == true ]]; then
    build_app
  fi

  if [[ "${#APP_NAMES[@]}" -gt 0 ]]; then
    run_app "${APP_NAMES[@]}"
  fi
}

main "$@"
