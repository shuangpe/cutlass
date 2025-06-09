#!/usr/bin/env python3
import pynvml
import time
import argparse
import csv
import subprocess
import threading
import platform
import os
from datetime import datetime

def get_system_info():
    """Get system and CUDA version information"""
    info = {}

    # System information
    info['hostname'] = platform.node()
    info['system'] = platform.system()
    info['architecture'] = platform.machine()
    info['python_version'] = platform.python_version()

    # NVIDIA driver version
    try:
        pynvml.nvmlInit()
        info['driver_version'] = pynvml.nvmlSystemGetDriverVersion().decode('utf-8') if isinstance(pynvml.nvmlSystemGetDriverVersion(), bytes) else pynvml.nvmlSystemGetDriverVersion()
        pynvml.nvmlShutdown()
    except Exception as e:
        info['driver_version'] = f"Error: {str(e)}"

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

    # Try to get CUDA runtime version through deviceQuery if available
    try:
        result = subprocess.run(['which', 'deviceQuery'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            device_query_path = result.stdout.decode('utf-8').strip()
            result = subprocess.run([device_query_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output = result.stdout.decode('utf-8')
            for line in output.split('\n'):
                if 'CUDA Runtime Version' in line:
                    info['cuda_runtime_version'] = line.strip()
    except Exception as e:
        info['devicequery_error'] = f"Error: {str(e)}"

    return info

def initialize_nvml():
    """Initialize NVML library"""
    try:
        pynvml.nvmlInit()
        print(f"NVML initialization successful, driver version: {pynvml.nvmlSystemGetDriverVersion()}")
        return True
    except pynvml.NVMLError as err:
        print(f"NVML initialization failed: {err}")
        return False

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

def monitor_gpu(gpu_id, interval, output_file, stop_event, system_info):
    """Continuously monitor GPU metrics and record to CSV file"""
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
            "Graphics Clock": False
        }

        # Test which metrics are available
        try:
            pynvml.nvmlDeviceGetInstantaneousPowerUsage(handle)
            capabilities["Instantaneous Power"] = True
        except (pynvml.NVMLError, AttributeError):
            pass

        try:
            pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            capabilities["SM Clock"] = True
        except pynvml.NVMLError:
            pass

        try:
            pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            capabilities["Memory Clock"] = True
        except pynvml.NVMLError:
            pass

        try:
            pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            capabilities["Graphics Clock"] = True
        except pynvml.NVMLError:
            pass

        # Print capability information
        print("GPU Monitoring Capabilities:")
        for capability, available in capabilities.items():
            print(f"  {capability}: {'Available' if available else 'Not Available'}")

        # Create CSV file and write header
        with open(output_file, 'w', newline='') as csvfile:
            # Write system information as comments at the top of the file
            csvfile.write("# GPU Monitoring Data\n")
            csvfile.write(f"# Date: {datetime.now().strftime('%Y%m%d-%H:%M:%S')}\n")
            csvfile.write(f"# GPU: {device_name}\n")
            csvfile.write(f"# warmup: {system_info.get('warmup_time', 'N/A')}\n")
            csvfile.write(f"# cooldown: {system_info.get('cooldown_time', 'N/A')}\n")

            # Write CUDA and system information
            for key, value in system_info.items():
                if key not in ['warmup_time', 'cooldown_time']:  # Already added these above
                    csvfile.write(f"# {key}: {value}\n")

            # Write capabilities
            for capability, available in capabilities.items():
                csvfile.write(f"# {capability}: {'Available' if available else 'Not Available'}\n")

            # Write note about clock frequencies
            csvfile.write("# Note: Tensor Core does not have separate frequency monitoring; it runs at SM clock frequency\n")
            csvfile.write("# Note: On many NVIDIA GPU architectures, SM clock and Graphics clock report identical values\n")

            # CSV data fields with units
            fieldnames = [
                'timestamp',
                'power_draw (W)',
                'instantaneous_power (W)',
                'temperature (°C)',
                'sm_clock (MHz)',
                'mem_clock (MHz)',
                'graphics_clock (MHz)',
                'gpu_utilization (%)',
                'memory_utilization (%)'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Continuously monitor until stop signal received
            while not stop_event.is_set():
                # Modified timestamp format: YYYYMMDD-HH:MM:SS.mmmmmm
                timestamp = datetime.now().strftime('%Y%m%d-%H:%M:%S.%f')
                metrics = get_gpu_metrics(handle)

                # Add timestamp to metrics data and rename keys to match fieldnames with units
                metrics_with_units = {
                    'timestamp': timestamp,
                    'power_draw (W)': metrics['power_draw'],
                    'instantaneous_power (W)': metrics['instantaneous_power'],
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

    except pynvml.NVMLError as err:
        print(f"Error during monitoring: {err}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass

def run_gemm_test(test_executable, test_args):
    """Run GEMM test program and capture output"""
    cmd = f"./{test_executable} {test_args}"
    print(f"Executing command: {cmd}")

    start_time = time.time()
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    end_time = time.time()

    print(f"Test completed, execution time: {end_time - start_time:.2f} seconds")
    print("Output:")
    print(stdout.decode('utf-8'))

    if stderr:
        print("Errors:")
        print(stderr.decode('utf-8'))

    return process.returncode

def main():
    parser = argparse.ArgumentParser(description='Comprehensive monitoring of GPU metrics (power, temperature, frequency) during GEMM kernel execution')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('-i', '--interval', type=float, default=0.05, help='Sampling interval (seconds)')
    parser.add_argument('-e', '--executable', type=str, required=True, help='GEMM executable to run')
    parser.add_argument('-a', '--args', type=str, default='', help='Arguments to pass to the GEMM executable')
    parser.add_argument('-o', '--output', type=str, default='', help='Output CSV filename')
    parser.add_argument('-w', '--warmup', type=int, default=3, help='Warmup time (seconds)')
    parser.add_argument('-c', '--cooldown', type=int, default=3, help='Cooldown time (seconds)')

    args = parser.parse_args()

    # Get system and CUDA information
    system_info = get_system_info()
    system_info['warmup_time'] = args.warmup
    system_info['cooldown_time'] = args.cooldown

    print("System Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")

    # Default output filename
    if not args.output:
        # Extract just the base filename without extension
        base_name = os.path.basename(args.executable)
        filename_without_ext = os.path.splitext(base_name)[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f"gpu_metrics_{filename_without_ext}_{timestamp}.csv"

    # Create and start monitoring thread
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_gpu,
        args=(args.gpu, args.interval, args.output, stop_event, system_info)
    )
    monitor_thread.daemon = True
    monitor_thread.start()

    try:
        # Wait for warmup time to get baseline metrics
        print(f"Collecting baseline metrics, warming up for {args.warmup} seconds...")
        time.sleep(args.warmup)

        # Run GEMM test
        return_code = run_gemm_test(args.executable, args.args)

        # Continue monitoring for cooldown time
        print(f"Test ended, continuing to monitor metrics cooldown for {args.cooldown} seconds...")
        time.sleep(args.cooldown)
    finally:
        # Stop monitoring thread
        stop_event.set()
        monitor_thread.join()

    print(f"GPU monitoring data saved to: {args.output}")

    # Suggestion for using the visualization script
    print("\nTo visualize the results, use the plot_gpu_metrics.py script:")
    print(f"python3 plot_gpu_metrics.py {args.output}")
    print("or")
    print(f"python3 plot_gpu_metrics.py {args.output} --show")

    return return_code

if __name__ == "__main__":
    main()