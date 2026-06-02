#!/usr/bin/env python3
"""
Primetrade.ai - ML/MLOps Engineering Internship Task 0
Reproducible & Observable Batch Signal Generation Pipeline (Polars Version)
"""

import argparse
import io
import json
import logging
import os
import random
import sys
import time
import yaml
import polars as pl

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
        parser = SafeArgumentParser(description="MLOps Batch Job Signal Pipeline (Polars)")
        parser.add_argument("--input", required=True, help="Path to the input CSV dataset")
        parser.add_argument("--config", required=True, help="Path to the YAML config file")
        parser.add_argument("--output", default="metrics.json", help="Path to save metrics.json")
        parser.add_argument("--log-file", default="run.log", help="Path to save run.log")
        
        args = parser.parse_args()
        output_path = args.output
        
        # 2. Setup Logging
        logger = setup_logging(args.log_file)
        logger.info("Job started successfully using Polars engine.")
        
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
            pass
            
        # 4. Load & Validate Dataset
        logger.info(f"Loading dataset from: {args.input}")
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input dataset file not found: {args.input}")
            
        if os.path.getsize(args.input) == 0:
            raise ValueError("Input dataset file is empty")
            
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
                    
                cleaned_csv = "\n".join(processed_lines)
        except Exception as e:
            if isinstance(e, (ValueError, FileNotFoundError)):
                raise e
            raise ValueError(f"Failed to read input file: {str(e)}")
            
        try:
            df = pl.read_csv(io.StringIO(cleaned_csv))
        except Exception as e:
            raise ValueError(f"Invalid CSV format: {str(e)}")
            
        # Validate required column
        if 'close' not in df.columns:
            raise ValueError("Required column 'close' is missing from the dataset")
            
        # Cast close to Float64 to check for non-numeric values
        try:
            if df['close'].dtype == pl.String:
                df = df.with_columns(pl.col('close').cast(pl.Float64))
        except Exception as e:
            raise ValueError(f"Invalid numeric value found in 'close' column: {str(e)}")
            
        # Check for null/empty values in the close column
        if df['close'].null_count() > 0:
            raise ValueError("Column 'close' contains null or empty values")
            
        rows_processed = len(df)
        logger.info(f"Successfully loaded {rows_processed} rows.")
        
        # 5. Compute Rolling Mean & Generate Signals using Polars
        logger.info(f"Computing rolling mean (window={window}) and generating signals...")
        
        # We specify window_size = window. First window-1 rows automatically get null/NaN
        # and are excluded from the signal computation.
        df = df.with_columns(
            rolling_mean = pl.col('close').rolling_mean(window_size=window)
        )
        
        # Generate signal: 1 if close > rolling_mean else 0
        df = df.with_columns(
            signal = pl.when(pl.col('close') > pl.col('rolling_mean'))
                     .then(1)
                     .otherwise(0)
        )
        
        # Exclude warmup period (where rolling_mean is null) from the signal rate computation
        valid_df = df.filter(pl.col('rolling_mean').is_not_null())
        
        if len(valid_df) > 0:
            signal_rate = valid_df['signal'].mean()
        else:
            signal_rate = 0.0
            
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
