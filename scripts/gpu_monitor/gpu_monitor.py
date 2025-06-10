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

def find_closest_supported_clock(requested_freq, supported_clocks):
    """Find the closest supported clock frequency to the requested frequency."""
    if requested_freq in supported_clocks:
        return requested_freq
    closest_freq = supported_clocks[0]
    min_diff = abs(closest_freq - requested_freq)
    for clock in supported_clocks:
        diff = abs(clock - requested_freq)
        if diff < min_diff:
            min_diff = diff
            closest_freq = clock
    return closest_freq

def set_gpu_frequency(gpu_id, frequency_str):
    """Set GPU frequency (graphics clock) to the specified value in MHz

    Args:
        gpu_id: GPU ID to set frequency for
        frequency_str: String of frequencies in MHz separated by semicolons, or -1 to skip
    """
    if not frequency_str:
        return False, "No frequency specified (skipping frequency setting)"

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

        # Get supported clock frequencies
        mem_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(handle)
        if not mem_clocks:
            return False, "Could not get supported memory clocks"

        # Get current memory clock to maintain it
        current_mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)

        # Find closest supported memory clock
        closest_mem_clock = mem_clocks[0]
        for mem_clock in mem_clocks:
            if abs(mem_clock - current_mem_clock) < abs(closest_mem_clock - current_mem_clock):
                closest_mem_clock = mem_clock

        # Get supported graphics clocks for this memory clock
        graphics_clocks = pynvml.nvmlDeviceGetSupportedGraphicsClocks(handle, closest_mem_clock)
        if not graphics_clocks:
            return False, f"Could not get supported graphics clocks for memory clock {closest_mem_clock} MHz"

        # Parse frequency string into a list of frequencies
        frequency_list = [int(f.strip()) for f in frequency_str.split(';') if f.strip() and f.strip() != "-1"]

        if not frequency_list:
            return False, "No valid frequencies provided after parsing or only -1 provided (will run with default frequency)"

        # Find closest supported graphics clock to any of the requested frequencies
        requested_freq = int(frequency_str) if frequency_str.isdigit() else int(frequency_list[0])
        closest_freq = find_closest_supported_clock(requested_freq, graphics_clocks)

        if closest_freq == requested_freq:
            print(f"Requested frequency {requested_freq} MHz is directly supported")
        else:
            print(f"Requested frequency {requested_freq} MHz is not directly supported, using closest: {closest_freq} MHz")

        print(f"Setting GPU {gpu_id} frequency to {closest_freq} MHz (requested {frequency_str})")

        # Set the application clocks
        pynvml.nvmlDeviceSetApplicationsClocks(handle, closest_mem_clock, closest_freq)
        return True, f"Successfully set GPU {gpu_id} frequency to {closest_freq} MHz (requested {frequency_str})"
    except pynvml.NVMLError as err:
        return False, f"Failed to set GPU frequency: {err}"

def reset_gpu_frequency(gpu_id):
    """Reset GPU clocks to default"""
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        pynvml.nvmlDeviceResetApplicationsClocks(handle)
        print(f"Reset GPU {gpu_id} frequency to default")
        return True, "Successfully reset GPU frequency"
    except pynvml.NVMLError as err:
        return False, f"Failed to reset GPU frequency: {err}"

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

def get_next_output_directory(path, is_dir=False):
    """Find the next available output directory with pattern YYYYMMDD_gpu_metrics_name_X

    Args:
        path: Path to executable or directory
        is_dir: If True, use the directory name as base; otherwise use executable name
    """
    if is_dir:
        # Use the directory's basename
        dir_basename = os.path.basename(os.path.normpath(path))
        base_name_part = dir_basename
    else:
        # Extract base name of executable without extension
        base_executable = os.path.basename(path)
        base_name_part = os.path.splitext(base_executable)[0]

    # Get current date as timestamp
    current_date = datetime.now().strftime('%Y%m%d')

    # Create base directory name pattern
    base_name = f"{current_date}_gpu_metrics_{base_name_part}_"

    index = 0

    while True:
        dir_name = f"{base_name}{index}"
        if not os.path.exists(dir_name):
            # Create the directory
            os.makedirs(dir_name)
            return dir_name
        index += 1

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

            # Continuously monitor until stop event received
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
    parser.add_argument('-e', '--executable', type=str, required=True,
                        help='GEMM executable to run. If a directory is specified, all executable files in that directory will be tested')
    parser.add_argument('-a', '--args', type=str, default='', help='Arguments to pass to the GEMM executable')
    parser.add_argument('-o', '--output', type=str, default='', help='Output CSV filename')
    parser.add_argument('-w', '--warmup', type=int, default=3, help='Warmup time (seconds)')
    parser.add_argument('-c', '--cooldown', type=int, default=3, help='Cooldown time (seconds)')
    parser.add_argument('-f', '--frequency', type=str, default="1500;1305;1005",
                        help='Set GPU frequency in MHz for the test. Can be a single value, multiple values separated by semicolons, or -1 to skip. Default: "1500;1305;1005"')

    args = parser.parse_args()

    # Get system and CUDA information
    system_info = get_system_info()
    system_info['warmup_time'] = args.warmup
    system_info['cooldown_time'] = args.cooldown

    print("System Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")

    # Initialize NVML once at the beginning
    nvml_initialized = initialize_nvml()
    if not nvml_initialized:
        print("Warning: Failed to initialize NVML. Frequency control may not work.")

    # Prepare frequency list - always include default (no setting)
    freq_list = ["oob"]  # Out of box (default) frequency

    # Add user-specified frequencies if provided
    if args.frequency:
        user_freqs = [f.strip() for f in args.frequency.split(';') if f.strip() and f.strip() != "-1"]
        freq_list.extend(user_freqs)

    # Determine executables to test and output directory
    if os.path.isdir(args.executable):
        print(f"Directory specified: {args.executable}")
        # Get all files in the directory
        all_files = [os.path.join(args.executable, f) for f in os.listdir(args.executable)]
        # Filter to get only executable files
        executables = [f for f in all_files if os.path.isfile(f) and os.access(f, os.X_OK)]

        if not executables:
            print(f"No executable files found in directory: {args.executable}")
            return 1

        print(f"Found {len(executables)} executable files to test")
        for exe in executables:
            print(f"  - {os.path.basename(exe)}")

        # Get a single output directory for all executables based on directory name
        output_dir = get_next_output_directory(args.executable, is_dir=True)
    else:
        # Single executable file
        executables = [args.executable]
        output_dir = get_next_output_directory(args.executable)

    print(f"All output files will be saved to: {output_dir}")

    # Process each executable
    for executable in executables:
        if len(executables) > 1:
            print(f"\n{'#'*100}")
            print(f"Testing executable: {executable}")
            print(f"{'#'*100}\n")

        # Create a copy of args with the current executable
        current_args = argparse.Namespace(**vars(args))
        current_args.executable = executable

        # Run tests for all frequencies for this executable
        for freq in freq_list:
            run_single_test(current_args, system_info.copy(), freq, output_dir)

    print(f"\nAll tests completed. Results saved in {output_dir}/")
    return 0

def run_single_test(args, system_info, frequency, output_dir):
    """Run a single independent test with the specified frequency"""
    # Create frequency-specific output file
    base_name = os.path.basename(args.executable)
    filename_without_ext = os.path.splitext(base_name)[0]

    # Format frequency for filename
    freq_suffix = "oobMhz" if frequency == "oob" else f"{frequency}Mhz"

    # Create output filename
    output_file = os.path.join(output_dir, f"{filename_without_ext}_{freq_suffix}.csv")

    print(f"\n{'='*80}")
    if frequency == "oob":
        print(f"Running test with default GPU frequency (not setting frequency)")
        system_info['frequency_setting'] = "Default (not set)"
    else:
        print(f"Running test with GPU frequency set to {frequency} MHz")
        system_info['requested_gpu_frequency'] = f"{frequency} MHz"
    print(f"Output will be saved to: {output_file}")
    print(f"{'='*80}\n")

    # Set GPU frequency if needed, but don't re-initialize NVML
    frequency_set = False
    if frequency != "oob":
        # Don't call initialize_nvml() here, we did it once in main()
        success, message = set_gpu_frequency(args.gpu, frequency)
        frequency_set = success
        # Print actual set frequency for clarity
        print(f"GPU {args.gpu} frequency setting status: {message}")

    # Create monitoring thread with fresh NVML initialization just for monitoring
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_gpu,
        args=(args.gpu, args.interval, output_file, stop_event, system_info)
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
        # Reset GPU frequency if we set it
        if frequency_set:
            success, message = reset_gpu_frequency(args.gpu)
            print(message)

        # Stop monitoring thread
        stop_event.set()
        monitor_thread.join()

    print(f"GPU monitoring data saved to: {output_file}")
    return return_code

if __name__ == "__main__":
    main()