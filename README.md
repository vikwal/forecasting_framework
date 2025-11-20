# Forecasting Framework

A comprehensive framework for time series forecasting with support for hyperparameter optimization, multi-GPU training, and efficient data caching.

## Features

- **Multi-GPU Training**: Parallel hyperparameter optimization across multiple GPUs
- **Data Caching**: Intelligent caching system to avoid redundant preprocessing
- **Hyperparameter Optimization**: Integration with Optuna for automated HPO
- **Memory Efficiency**: Lazy loading and memory-mapped data access
- **Multiple Model Architectures**: Support for FNN, LSTM, TFT, and other models
- **Cross-Validation**: K-fold validation with configurable splits

## Quick Start

### Prerequisites

- Python virtual environment (recommended path: `./frcst`)
- GPU support (CUDA-compatible)
- Required dependencies (install in virtual environment)

### Basic Usage

1. **Configure your experiment**: Create a configuration file in `configs/` directory
2. **Run hyperparameter optimization**: Use `hpo_cl.py` for single GPU or `launch_multi_gpu.sh` for multi-GPU
3. **Train final model**: Use `train_cl.py` with optimized hyperparameters

## Multi-GPU Training with `launch_multi_gpu.sh`

### Overview

The `launch_multi_gpu.sh` script orchestrates parallel hyperparameter optimization across multiple GPUs with shared data caching for maximum efficiency.

### Usage

```bash
./launch_multi_gpu.sh <config.yaml> <model_name> [options]
```

### Required Arguments

- `config.yaml`: Path to configuration file (e.g., `configs/wind_config.yaml`)
- `model_name`: Model architecture (`fnn`, `lstm`, `tft`, etc.)

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--gpus` | `1,2,3,4,5,6,7` | Comma-separated GPU IDs |
| `--trials-per-gpu` | `10` | Number of HPO trials per GPU |
| `--force-preprocess` | `false` | Force data reprocessing even if cache exists |
| `--no-cache` | `false` | Disable caching (for small datasets) |
| `--cache-dir` | `data_cache` | Directory for cached data |
| `--venv` | `./frcst` | Path to virtual environment |

### Examples

```bash
# Basic multi-GPU HPO with default settings
./launch_multi_gpu.sh configs/wind_config.yaml fnn

# Custom GPU selection and trial count
./launch_multi_gpu.sh configs/solar_config.yaml lstm --gpus 0,1,2 --trials-per-gpu 20

# Small dataset without caching
./launch_multi_gpu.sh configs/small_config.yaml fnn --no-cache

# Force data reprocessing
./launch_multi_gpu.sh configs/config.yaml tft --force-preprocess

# Custom cache directory
./launch_multi_gpu.sh configs/config.yaml model --cache-dir /custom/path
```

### How To Stop processes

```bash
sudo pkill -f "hpo_cl.py"
# or
sudo pkill -f "config.yaml"
```

### How It Works

1. **Environment Setup**: Activates virtual environment and configures TensorFlow settings
2. **Data Preprocessing**: Creates or loads cached preprocessed data (shared across all GPUs)
3. **Parallel Execution**: Launches separate HPO processes on each specified GPU
4. **Monitoring**: Tracks progress and provides real-time status updates
5. **Logging**: Individual log files for each GPU process

### Output Files

- **Log files**: `logs/hpo_cl_m-{model}_gpu{id}.log`
- **Study databases**: Optuna study files for HPO results
- **Cached data**: Preprocessed data in `data_cache/` directory

### Cache Management Commands

```bash
# Force cache refresh
./launch_multi_gpu.sh config.yaml model --force-preprocess

# Disable caching for small datasets
./launch_multi_gpu.sh config.yaml model --no-cache

# Custom cache directory
./launch_multi_gpu.sh config.yaml model --cache-dir /custom/path
```

## Single-GPU Hyperparameter Optimization with `hpo_cl.py`

### HPO Overview

Standalone hyperparameter optimization script for single GPU usage or when called by the multi-GPU launcher.

### HPO Usage

```bash
python hpo_cl.py --model <model_name> --config <config_name> [options]
```

### HPO Arguments

| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--model` | `-m` | Yes | Model architecture (fnn, lstm, tft, etc.) |
| `--config` | `-c` | Yes | Config file name (without .yaml extension) |
| `--index` | `-i` | No | Process identifier for multi-GPU setups |
| `--no-cache` | | No | Disable caching for small datasets |

### HPO Examples

```bash
# Basic HPO
python hpo_cl.py --model fnn --config wind_config

# With process index (used by multi-GPU launcher)
python hpo_cl.py --model lstm --config solar_config --index gpu2

# Small dataset without caching
python hpo_cl.py --model tft --config small_config --no-cache
```

### HPO Key Features

- **Intelligent Caching**: Automatically uses cached data when available
- **Optuna Integration**: Advanced pruning and optimization strategies
- **Cross-Validation**: K-fold validation for robust hyperparameter selection
- **Memory Management**: Efficient memory usage with lazy loading
- **Duplicate Detection**: Prevents redundant trials with identical parameters

## Model Training with `train_cl.py`

### Training Overview

Training script for final model training using optimized hyperparameters from HPO studies.

### Training Usage

```bash
python train_cl.py --model <model_name> --config <config_name> [options]
```

### Training Arguments

| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--model` | `-m` | Yes | Model architecture |
| `--config` | `-c` | Yes | Configuration file name |
| `--index` | `-i` | No | Optional identifier suffix |

### Training Examples

```bash
# Train model with best hyperparameters
python train_cl.py --model fnn --config wind_config

# Train with custom index
python train_cl.py --model lstm --config solar_config --index final_run
```

### Training Key Features

- **Automatic HPO Lookup**: Loads best hyperparameters from completed studies
- **Model Persistence**: Saves trained models and training history
- **Comprehensive Evaluation**: Tests on multiple datasets with detailed metrics
- **Result Caching**: Avoids retraining if results already exist

## Data Caching System

### Caching System Overview

The framework implements an intelligent caching system that significantly reduces preprocessing time for repeated experiments.

### How Caching Works

1. **Cache Key Generation**: Creates unique hash from config, features, and model parameters
2. **Preprocessing Detection**: Automatically determines if data needs reprocessing
3. **Memory-Mapped Storage**: Stores preprocessed data in efficient binary format
4. **Lazy Loading**: Loads only required data portions to minimize memory usage
5. **Fold Management**: Efficiently manages k-fold cross-validation data

### Cache Structure

```text
data_cache/
├── {hash_id}_metadata.pkl          # Configuration and metadata
├── {hash_id}_prepared.pkl          # Preprocessed datasets
├── {hash_id}_fold_manifest.pkl     # Fold information
└── {hash_id}_folds/                # Individual fold files
    ├── fold_000.pkl
    ├── fold_001.pkl
    └── ...
```

### Cache Benefits

- **Time Savings**: Eliminates redundant preprocessing (hours → seconds)
- **Memory Efficiency**: Shared data across multiple GPU processes
- **Consistency**: Ensures identical data across all experiments
- **Scalability**: Handles large datasets with minimal memory footprint

## Configuration

### Config File Structure

Place configuration files in `configs/` directory with `.yaml` extension:

```yaml
data:
  path: "path/to/data"
  files: ["file1.csv", "file2.csv"]
  freq: "15min"
  target_col: "power"
  test_start: "2023-01-01"

model:
  name: "fnn"
  lookback: 96
  horizon: 4
  output_dim: 1
  step_size: 1

hpo:
  trials: 100
  kfolds: 5
  val_split: 0.2
  studies_path: "studies/"

# ... additional configuration sections
```

### Environment Variables

The framework automatically sets optimal TensorFlow configurations:

- `TF_CPP_MIN_LOG_LEVEL=3`: Suppress TensorFlow logging
- `TF_ENABLE_ONEDNN_OPTS=0`: Disable oneDNN warnings

## Directory Structure

```text
forecasting_framework/
├── configs/                    # Configuration files
├── data_cache/                # Cached preprocessed data
├── logs/                      # Training and HPO logs
├── models/                    # Saved model files
├── results/                   # Evaluation results
├── studies/                   # Optuna study databases
├── utils/                     # Utility modules
│   ├── data_cache.py         # Caching implementation
│   ├── preprocessing.py      # Data preprocessing
│   ├── hpo.py               # Hyperparameter optimization
│   └── tools.py             # General utilities
├── hpo_cl.py                 # HPO script
├── train_cl.py              # Training script
└── launch_multi_gpu.sh      # Multi-GPU launcher
```

## Logging and Monitoring

### Log Files

- **HPO logs**: `logs/hpo_cl_m-{model}_{index}.log`
- **Training logs**: `logs/train_cl_m-{model}_{index}.log`

### Progress Monitoring

The multi-GPU launcher provides real-time progress monitoring:

- Process status updates every 30 seconds
- Recent log activity from all GPUs
- Completion notifications

### Study Management

Optuna studies are automatically managed:

- Persistent storage in `studies/` directory
- Automatic resume from previous runs
- Trial pruning for efficiency

## Performance Tips

### Memory Optimization

1. **Use caching**: Essential for large datasets and repeated experiments
2. **Appropriate batch sizes**: Configure based on available GPU memory
3. **Lazy loading**: Framework automatically manages memory-efficient data loading

### GPU Utilization

1. **Multi-GPU setup**: Use all available GPUs with `launch_multi_gpu.sh`
2. **GPU selection**: Exclude GPU 0 if used by other processes
3. **Trial distribution**: Balance trials across GPUs based on their capabilities

### Data Preprocessing

1. **Cache reuse**: Avoid `--force-preprocess` unless necessary
2. **Small datasets**: Use `--no-cache` for datasets that fit in memory
3. **Feature engineering**: Optimize feature selection in config files

## Troubleshooting

### Common Issues

1. **Cache conflicts**: Use `--force-preprocess` to regenerate cache
2. **Memory errors**: Reduce batch size or use `--no-cache`
3. **GPU availability**: Check `nvidia-smi` and adjust `--gpus` parameter
4. **Virtual environment**: Ensure correct environment activation

### Debug Mode

Enable detailed logging by modifying log levels in scripts or using verbose config options.

## Advanced Usage

### Custom Model Integration

1. Add model definition to appropriate module
2. Update hyperparameter search space
3. Configure model-specific preprocessing if needed

### Experiment Management

1. Use descriptive config file names
2. Organize results by dataset/model combinations
3. Maintain study databases for analysis

For more detailed information, refer to the inline documentation in each script and the utility modules.
