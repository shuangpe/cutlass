#!/usr/bin/env python3
import pynvml
import time
import argparse
import csv
import platform
import os
import subprocess
import signal
import sys
from datetime import datetime

# Global variable to track NVML initialization status
nvml_initialized = False

def initialize_nvml():
    """Initialize NVML library if not already initialized"""
    global nvml_initialized

    if nvml_initialized:
        return True

    try:
        pynvml.nvmlInit()
        nvml_initialized = True
        print(f"NVML initialization successful, driver version: {pynvml.nvmlSystemGetDriverVersion()}")
        return True
    except pynvml.NVMLError as err:
        print(f"NVML initialization failed: {err}")
        return False

def shutdown_nvml():
    """Shutdown NVML library if initialized"""
    global nvml_initialized

    if nvml_initialized:
        try:
            pynvml.nvmlShutdown()
            nvml_initialized = False
        except:
            pass

def get_system_info():
    """Get system and CUDA version information"""
    info = {}

    # System information
    info['system'] = platform.system()
    info['architecture'] = platform.machine()
    info['python_version'] = platform.python_version()

    # NVIDIA driver version using pynvml
    if initialize_nvml():
        try:
            info['driver_version'] = pynvml.nvmlSystemGetDriverVersion().decode('utf-8') if isinstance(pynvml.nvmlSystemGetDriverVersion(), bytes) else pynvml.nvmlSystemGetDriverVersion()
        except Exception as e:
            info['driver_version'] = f"Error: {str(e)}"
    else:
        info['driver_version'] = "Error: Failed to initialize NVML"

    # CUDA version using nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode('utf-8')
        for line in output.split('\n'):
            if 'release' in line.lower() and 'V' in line:
                info['cuda_version'] = line.strip()
                break
        else:
            info['cuda_version'] = "Not found in nvcc output"
    except Exception as e:
        info['cuda_version'] = f"Error: {str(e)}"

    return info

def get_gpu_metrics(handle):
    """Get all monitored GPU metrics"""
    metrics = {}

    # Get average power draw
    try:
        metrics['power_draw'] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
    except pynvml.NVMLError as err:
        metrics['power_draw'] = "N/A"
        print(f"Failed to get average power: {err}")

    # Try to get instantaneous power
    try:
        metrics['instantaneous_power'] = pynvml.nvmlDeviceGetInstantaneousPowerUsage(handle) / 1000.0
    except (pynvml.NVMLError, AttributeError) as err:
        metrics['instantaneous_power'] = "N/A"
        
    # Get HBM power usage (supported on Ampere and newer GPUs)
    try:
        metrics['hbm_power'] = pynvml.nvmlDeviceGetMemorySubsystemPowerUsage(handle) / 1000.0  # Convert to watts
    except (pynvml.NVMLError, AttributeError) as err:
        # Try alternative method if available
        try:
            # Define the field ID for memory subsystem power if available in the NVML version
            NVML_FI_DEV_MEMORY_SUBSYSTEM_POWER = 310  # This might vary based on NVML version
            field_ids = [NVML_FI_DEV_MEMORY_SUBSYSTEM_POWER]
            field_values = pynvml.nvmlDeviceGetFieldValues(handle, field_ids)
            
            if field_values[0].valueType == pynvml.NVML_VALUE_TYPE_UNSIGNED_LONG:
                metrics['hbm_power'] = field_values[0].value.ulVal / 1000.0
            else:
                metrics['hbm_power'] = "N/A"
        except (pynvml.NVMLError, AttributeError) as inner_err:
            metrics['hbm_power'] = "N/A"

    # Get GPU temperature
    try:
        metrics['temperature'] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    except pynvml.NVMLError as err:
        metrics['temperature'] = "N/A"
        print(f"Failed to get GPU temperature: {err}")

    # Get GPU clock frequencies
    # 1. Get current SM clock (also represents Tensor Core frequency)
    try:
        sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        metrics['sm_clock'] = sm_clock
    except pynvml.NVMLError as err:
        metrics['sm_clock'] = "N/A"
        print(f"Failed to get SM clock frequency: {err}")

    # 2. Get current memory clock
    try:
        metrics['mem_clock'] = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
    except pynvml.NVMLError as err:
        metrics['mem_clock'] = "N/A"
        print(f"Failed to get memory clock frequency: {err}")

    # 3. Get graphics clock
    try:
        graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        metrics['graphics_clock'] = graphics_clock
    except pynvml.NVMLError as err:
        metrics['graphics_clock'] = "N/A"
        print(f"Failed to get graphics clock frequency: {err}")

    # Get GPU utilization
    try:
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        metrics['gpu_utilization'] = utilization.gpu
        metrics['memory_utilization'] = utilization.memory
    except pynvml.NVMLError as err:
        metrics['gpu_utilization'] = "N/A"
        metrics['memory_utilization'] = "N/A"
        print(f"Failed to get GPU utilization: {err}")

    return metrics

def monitor_gpu(gpu_id, interval, output_file, system_info):
    """Continuously monitor GPU metrics and record to CSV file until terminated"""
    if not initialize_nvml():
        return

    try:
        # Get GPU handle
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        device_name = pynvml.nvmlDeviceGetName(handle)
        print(f"Monitoring GPU: {device_name}")

        # Get GPU capabilities
        capabilities = {
            "Instantaneous Power": False,
            "SM Clock": False,
            "Memory Clock": False,
            "Graphics Clock": False,
            "HBM Power": False
        }

        # Test which metrics are available
        try:
            pynvml.nvmlDeviceGetInstantaneousPowerUsage(handle)
            capabilities["Instantaneous Power"] = True
        except (pynvml.NVMLError, AttributeError):
            pass

        # Test for HBM power monitoring capability
        try:
            pynvml.nvmlDeviceGetMemorySubsystemPowerUsage(handle)
            capabilities["HBM Power"] = True
        except (pynvml.NVMLError, AttributeError):
            try:
                # Alternative method
                NVML_FI_DEV_MEMORY_SUBSYSTEM_POWER = 310
                field_ids = [NVML_FI_DEV_MEMORY_SUBSYSTEM_POWER]
                field_values = pynvml.nvmlDeviceGetFieldValues(handle, field_ids)
                if field_values[0].nvmlReturn == pynvml.NVML_SUCCESS:
                    capabilities["HBM Power"] = True
            except (pynvml.NVMLError, AttributeError):
                pass

        # Print capability information
        print("GPU Monitoring Capabilities:")
        for capability, available in capabilities.items():
            print(f"  {capability}: {'Available' if available else 'Not Available'}")

        # Ensure directory exists for the output file
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create CSV file and write header
        with open(output_file, 'w', newline='') as csvfile:
            # Write system information as comments at the top of the file
            csvfile.write("# GPU Monitoring Data\n")
            csvfile.write(f"# Date: {datetime.now().strftime('%Y%m%d-%H:%M:%S')}\n")
            csvfile.write(f"# GPU: {device_name}\n")

            # Write CUDA and system information
            for key, value in system_info.items():
                csvfile.write(f"# {key}: {value}\n")

            # Write capabilities
            for capability, available in capabilities.items():
                csvfile.write(f"# {capability}: {'Available' if available else 'Not Available'}\n")

            # Write note about HBM power monitoring
            csvfile.write("# Note: HBM power monitoring is available on Ampere and newer GPU architectures\n")
            
            # Write note about clock frequencies
            csvfile.write("# Note: Tensor Core does not have separate frequency monitoring; it runs at SM clock frequency\n")
            csvfile.write("# Note: On many NVIDIA GPU architectures, SM clock and Graphics clock report identical values\n")

            # CSV data fields with units
            fieldnames = [
                'timestamp',
                'power_draw (W)',
                'instantaneous_power (W)',
                'hbm_power (W)',
                'temperature (°C)',
                'sm_clock (MHz)',
                'mem_clock (MHz)',
                'graphics_clock (MHz)',
                'gpu_utilization (%)',
                'memory_utilization (%)'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Monitor continuously until terminated
            print("GPU monitoring started. Press Ctrl+C to stop...")
            try:
                while True:
                    # Modified timestamp format: YYYYMMDD-HH:MM:SS.mmmmmm
                    timestamp = datetime.now().strftime('%Y%m%d-%H:%M:%S.%f')
                    metrics = get_gpu_metrics(handle)

                    # Add timestamp to metrics data and rename keys to match fieldnames with units
                    metrics_with_units = {
                        'timestamp': timestamp,
                        'power_draw (W)': metrics['power_draw'],
                        'instantaneous_power (W)': metrics['instantaneous_power'],
                        'hbm_power (W)': metrics['hbm_power'],
                        'temperature (°C)': metrics['temperature'],
                        'sm_clock (MHz)': metrics['sm_clock'],
                        'mem_clock (MHz)': metrics['mem_clock'],
                        'graphics_clock (MHz)': metrics['graphics_clock'],
                        'gpu_utilization (%)': metrics['gpu_utilization'],
                        'memory_utilization (%)': metrics['memory_utilization']
                    }

                    writer.writerow(metrics_with_units)
                    csvfile.flush()  # Ensure data is written immediately
                    time.sleep(interval)
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")

    except pynvml.NVMLError as err:
        print(f"Error during monitoring: {err}")
    finally:
        # Don't close NVML here, let the main function handle it
        print(f"GPU monitoring completed. Data saved to: {output_file}")

def signal_handler(sig, frame):
    print("\nReceived termination signal. Shutting down...")
    shutdown_nvml()
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='Monitor GPU metrics (power, temperature, frequency)')
    parser.add_argument('-g', '--gpu', type=int, required=True, help='GPU ID')
    parser.add_argument('-i', '--interval', type=float, default=0.05, help='Sampling interval (seconds)')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output CSV filename')
    # Remove duration parameter, script will run until terminated

    args = parser.parse_args()

    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Get system and CUDA information
    system_info = get_system_info()
    system_info['monitoring_interval'] = args.interval

    print("System Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")

    # Initialize NVML
    nvml_initialized = initialize_nvml()
    if not nvml_initialized:
        print("Warning: Failed to initialize NVML. Monitoring may not work properly.")

    try:
        # Start monitoring
        print("Starting GPU monitoring (will run until terminated)...")
        monitor_gpu(args.gpu, args.interval, args.output, system_info)
    finally:
        # Ensure NVML is closed when program exits
        shutdown_nvml()

if __name__ == "__main__":
    main()