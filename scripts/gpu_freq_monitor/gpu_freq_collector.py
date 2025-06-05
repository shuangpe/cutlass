#!/usr/bin/env python3

import pynvml
import time
import csv
from datetime import datetime
import signal
import sys
import argparse

class GpuClockMonitor:
    def __init__(self, gpu_id, output_file, sample_interval=0.1):
        self.gpu_id = gpu_id
        self.output_file = output_file
        self.sample_interval = sample_interval
        self.running = True
        self.timestamps = []
        self.graphics_clocks = []
        self.memory_clocks = []
        self.sm_clocks = []

        # Set signal handlers
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

    def handle_signal(self, sig, frame):
        print(f"Received stop signal {sig}, stopping monitoring...")
        self.running = False

    def start_monitoring(self):
        # Initialize NVML
        pynvml.nvmlInit()

        # Get GPU handle
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
        device_name = pynvml.nvmlDeviceGetName(handle)

        print(f"Starting to monitor GPU {self.gpu_id} ({device_name}) frequency changes...")
        print(f"Sampling interval: {self.sample_interval} seconds")
        print(f"Data will be saved to {self.output_file}")

        # Collect data
        start_time = time.time()

        try:
            while self.running:
                current_time = datetime.now()

                # Get clock frequencies
                graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)

                # Store data
                self.timestamps.append(current_time)
                self.graphics_clocks.append(graphics_clock)
                self.memory_clocks.append(memory_clock)
                self.sm_clocks.append(sm_clock)

                time.sleep(self.sample_interval)

        except Exception as e:
            print(f"Error occurred during monitoring: {str(e)}")

        finally:
            # Shutdown NVML
            pynvml.nvmlShutdown()

            # Save data
            self.save_data()

    def save_data(self):
        # Use built-in csv module instead of pandas
        with open(self.output_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            # Write header
            csv_writer.writerow(['timestamp', 'graphics_clock', 'memory_clock', 'sm_clock'])

            # Write data rows
            for i in range(len(self.timestamps)):
                csv_writer.writerow([
                    self.timestamps[i],
                    self.graphics_clocks[i],
                    self.memory_clocks[i],
                    self.sm_clocks[i]
                ])

        print(f"Frequency data saved to {self.output_file}")

def main():
    parser = argparse.ArgumentParser(description='Monitor GPU frequency')
    parser.add_argument('--gpu-id', type=int, required=True, help='GPU ID to monitor')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--interval', type=float, default=0.1, help='Sampling interval (seconds)')

    args = parser.parse_args()

    monitor = GpuClockMonitor(args.gpu_id, args.output, args.interval)
    monitor.start_monitoring()

if __name__ == '__main__':
    main()