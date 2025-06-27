#!/usr/bin/env python3

import os
import sys
import yaml
import json
import argparse
from typing import Dict, Any, List

def parse_yaml_config(yaml_file: str) -> Dict[str, Any]:
    """
    Parse YAML configuration file

    Args:
        yaml_file: Path to YAML configuration file

    Returns:
        Dict with configuration values
    """
    try:
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
        return config or {}
    except Exception as e:
        print(f"Error parsing YAML file: {e}", file=sys.stderr)
        sys.exit(1)

def merge_options(config: Dict[str, Any], cli_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge configuration from file with command line arguments
    CLI arguments take precedence over config file settings

    Args:
        config: Configuration from YAML file
        cli_args: Command line arguments

    Returns:
        Dict with merged configuration
    """
    result = config.copy()

    # Override with CLI arguments (if not None)
    for key, value in cli_args.items():
        if value is not None:
            result[key] = value

    return result

def main():
    parser = argparse.ArgumentParser(description="Run tests and monitor GPU metrics")

    # Test configuration options
    parser.add_argument("-e", "--executable", help="Test executable or directory")
    parser.add_argument("-g", "--gpu_id", type=int, help="GPU ID")
    parser.add_argument("-i", "--interval", type=int, help="Monitoring interval (milliseconds) (default: 150)")
    parser.add_argument("-f", "--config", help="YAML configuration file")
    # Removed -a/--args option, args can only be set via config file

    args = parser.parse_args()

    # Convert namespace to dictionary, filtering out None values
    cli_args = {k: v for k, v in vars(args).items() if v is not None and k != 'config'}

    # Start with empty config
    config = {}

    # Load configuration from YAML file if specified
    if args.config and os.path.isfile(args.config):
        config = parse_yaml_config(args.config)

    # Merge configurations (CLI arguments take precedence)
    final_config = merge_options(config, cli_args)

    # Check required parameters
    if 'executable' not in final_config:
        print("Error: Must specify executable or directory (-e)", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    if 'gpu_id' not in final_config:
        print("Error: Must specify GPU ID (-g)", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    # Set default values if not provided
    defaults = {
        'interval': 150,
        'args': "",
        'frequencies': ["oob", "1500", "1305", "1005"]
    }

    for key, value in defaults.items():
        if key not in final_config:
            final_config[key] = value

    # Output final configuration as JSON
    print(json.dumps(final_config))

if __name__ == "__main__":
    main()
