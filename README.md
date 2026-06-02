# MLOps Batch Job signal pipeline

This repository contains a minimal, robust, reproducible, and observable MLOps-style batch job in Python. It reads a dataset of OHLCV rows, computes a rolling mean on the close price, generates binary signals based on close vs. rolling mean, and outputs logs and structured metrics.

## Features

- **Reproducibility**: Entirely deterministic runs via configuration parameter (`seed`, `window`, `version`).
- **Observability**: Detailed job logs in `run.log` and machine-readable metrics in `metrics.json`.
- **Robust Validation**: Explicit handling and graceful failure modes for missing files, incorrect configuration schemas, empty files, and malformed CSV rows.
- **Docker Readiness**: Fully packaged and runnable inside Docker with no hardcoded paths.

## Directory Structure

```
├── Dockerfile          # Slim Python Docker packaging
├── README.md           # Documentation (this file)
├── config.yaml         # Default configuration
├── data.csv            # OHLCV dataset
├── metrics.json        # Output success/error metrics file
├── requirements.txt    # Third-party dependencies
├── run.log             # Output execution log
└── run.py              # Main execution script
```

## Setup and Local Execution

### Prerequisites
- Python 3.9 or higher
- `pip` (Python package manager)

### 1. Install Dependencies
Install the required packages locally:
```bash
pip install -r requirements.txt
```

### 2. Run the Batch Job
Run the batch signal generation using the CLI:
```bash
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

If any errors occur, the script logs the exception stack trace to `run.log` and writes a structured error JSON to `metrics.json`.

---

## Running with Docker

You can build and execute the batch job inside a Docker container without needing Python or libraries installed on your host system.

### 1. Build the Docker Image
```bash
docker build -t mlops-task .
```

### 2. Run the Docker Container
```bash
docker run --rm mlops-task
```

This runs the script with the default dataset and configuration files copied inside the container, outputs the final metrics JSON to standard output, and creates the output files inside the container context.

---

## Example Output (`metrics.json`)

### Success Output
```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 40,
  "seed": 42,
  "status": "success"
}
```

### Error Output
```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Required column 'close' is missing from the dataset"
}
```

---

## Signal Design & Implementation Details

- **Warmup Window**: For a rolling mean of size `W`, the first `W - 1` rows do not have sufficient history. We represent these values as `None` (NaN equivalent) and **exclude** them from the calculation of the final `signal_rate`.
- **Signal Logic**: For index $i \ge W - 1$, the rolling mean is computed as:
  $$\text{rolling\_mean}_i = \frac{1}{W} \sum_{k=i-W+1}^{i} \text{close}_k$$
  The signal is generated as:
  $$\text{signal}_i = \begin{cases} 1 & \text{if } \text{close}_i > \text{rolling\_mean}_i \\ 0 & \text{otherwise} \end{cases}$$
