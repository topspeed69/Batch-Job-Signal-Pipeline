#!/usr/bin/env python3
"""
Primetrade.ai - ML/MLOps Engineering Internship Task 0
Reproducible & Observable Batch Signal Generation Pipeline
"""

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
import yaml

# Safe Argument Parser to prevent argparse from calling sys.exit on error,
# allowing us to write the metrics.json error output as required.
class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(f"Argument parsing error: {message}")

def setup_logging(log_file_path):
    """Sets up logging to both a file and standard output."""
    logger = logging.getLogger("mlops_batch_job")
    logger.setLevel(logging.INFO)
    
    # Formatter for detailed observability
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Clear any existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # File handler
    try:
        file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        sys.stderr.write(f"Warning: Could not create log file at {log_file_path}: {e}\n")
        
    # Console handler (standard output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def main():
    start_time = time.perf_counter()
    
    # Initial fallbacks for error reporting in case early steps fail
    output_path = "metrics.json"
    version_fallback = "v1"
    seed_fallback = 42
    
    # Setup placeholder logger until arguments are parsed
    logger = logging.getLogger("mlops_batch_job")
    
    try:
        # 1. Parse Command Line Arguments Safely
        parser = SafeArgumentParser(description="MLOps Batch Job Signal Pipeline")
        parser.add_argument("--input", required=True, help="Path to the input CSV dataset")
        parser.add_argument("--config", required=True, help="Path to the YAML config file")
        parser.add_argument("--output", default="metrics.json", help="Path to save metrics.json")
        parser.add_argument("--log-file", default="run.log", help="Path to save run.log")
        
        args = parser.parse_args()
        output_path = args.output
        
        # 2. Setup Logging
        logger = setup_logging(args.log_file)
        logger.info("Job started successfully.")
        
        # 3. Load & Validate Configuration
        logger.info(f"Loading configuration from: {args.config}")
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
            
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse YAML config: {str(e)}")
            
        if not isinstance(config, dict):
            raise ValueError("Config structure is invalid: must be a YAML dictionary")
            
        # Check required fields
        required_fields = ["seed", "window", "version"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: '{field}'")
                
        seed = config["seed"]
        window = config["window"]
        version = config["version"]
        
        # Keep fallbacks updated for the error block
        version_fallback = str(version)
        if isinstance(seed, int):
            seed_fallback = seed
            
        # Validate data types
        if not isinstance(seed, int):
            raise TypeError(f"Config field 'seed' must be an integer, got {type(seed).__name__}")
        if not isinstance(window, int) or window <= 0:
            raise ValueError(f"Config field 'window' must be a positive integer, got {window}")
        if not isinstance(version, str) or not version.strip():
            raise ValueError("Config field 'version' must be a non-empty string")
            
        logger.info(f"Config loaded and validated. seed={seed}, window={window}, version={version}")
        
        # Set random seed to ensure reproducibility
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
            logger.info("Numpy random seed set successfully.")
        except ImportError:
            pass  # Numpy not installed, standard seed is sufficient
            
        # 4. Load & Validate Dataset
        logger.info(f"Loading dataset from: {args.input}")
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input dataset file not found: {args.input}")
            
        if os.path.getsize(args.input) == 0:
            raise ValueError("Input dataset file is empty")
            
        closes = []
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                # Preprocess lines to strip outer literal quotes (e.g. '"timestamp,open..."')
                processed_lines = []
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        if stripped.startswith('"') and stripped.endswith('"'):
                            stripped = stripped[1:-1]
                        processed_lines.append(stripped)
                        
                if not processed_lines:
                    raise ValueError("Input dataset contains no readable data rows")
                    
                reader = csv.DictReader(processed_lines)
                if not reader.fieldnames:
                    raise ValueError("CSV structure is invalid: no headers found")
                    
                if 'close' not in reader.fieldnames:
                    raise ValueError("Required column 'close' is missing from the dataset")
                    
                for line_num, row in enumerate(reader, start=2):
                    close_val = row.get('close')
                    if close_val is None or close_val.strip() == '':
                        raise ValueError(f"Missing 'close' value on row {line_num}")
                    try:
                        closes.append(float(close_val))
                    except ValueError:
                        raise ValueError(f"Invalid numeric 'close' value '{close_val}' on row {line_num}")
        except Exception as e:
            if isinstance(e, (ValueError, FileNotFoundError)):
                raise e
            raise ValueError(f"Failed to parse CSV dataset: {str(e)}")
            
        rows_processed = len(closes)
        logger.info(f"Successfully loaded {rows_processed} rows.")
        
        # 5. Compute Rolling Mean & Generate Signals
        logger.info(f"Computing rolling mean (window={window}) and generating signals...")
        signals = []
        valid_signals = []
        
        # Warmup period handler: first window-1 rows do not have enough history
        # We allow NaNs (represented as None in pure Python) and exclude them from signal rate computation.
        for i in range(rows_processed):
            if i < window - 1:
                signals.append(None)
            else:
                window_data = closes[i - window + 1 : i + 1]
                rolling_mean = sum(window_data) / window
                signal = 1 if closes[i] > rolling_mean else 0
                signals.append(signal)
                valid_signals.append(signal)
                
        signal_rate = sum(valid_signals) / len(valid_signals) if valid_signals else 0.0
        # Round value to 4 decimal places for consistency
        signal_rate_rounded = round(signal_rate, 4)
        
        # 6. Stop timer and compute latency
        end_time = time.perf_counter()
        latency_ms = int((end_time - start_time) * 1000)
        
        # 7. Write Success Metrics
        metrics = {
            "version": version,
            "rows_processed": rows_processed,
            "metric": "signal_rate",
            "value": signal_rate_rounded,
            "latency_ms": latency_ms,
            "seed": seed,
            "status": "success"
        }
        
        # Write to metrics.json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Metrics summary: {json.dumps(metrics)}")
        logger.info("Job completed status: success")
        
        # Print metrics to stdout as required by the Docker specification
        print(json.dumps(metrics, indent=2))
        sys.exit(0)
        
    except Exception as e:
        # Handle exceptions gracefully by logging and writing error metrics
        error_message = str(e)
        if logger.handlers:
            logger.error(f"Job failed with exception: {error_message}", exc_info=True)
        else:
            sys.stderr.write(f"Job failed: {error_message}\n")
            
        metrics_error = {
            "version": version_fallback,
            "status": "error",
            "error_message": error_message
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_error, f, indent=2)
            if logger.handlers:
                logger.info(f"Error metrics written to {output_path}")
        except Exception as write_err:
            sys.stderr.write(f"Fatal: Could not write error metrics to {output_path}: {write_err}\n")
            
        # Print error metrics to stdout
        print(json.dumps(metrics_error, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()
