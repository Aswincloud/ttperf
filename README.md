# ttperf - TT-Metal Performance Profiler

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![PyPI version](https://img.shields.io/pypi/v/ttperf.svg)](https://pypi.org/project/ttperf/)
[![GitHub issues](https://img.shields.io/github/issues/Aswincloud/ttperf)](https://github.com/Aswincloud/ttperf/issues)
[![GitHub stars](https://img.shields.io/github/stars/Aswincloud/ttperf)](https://github.com/Aswincloud/ttperf/stargazers)

**A streamlined CLI tool for profiling Tenstorrent's TT-Metal tests and extracting device kernel performance metrics**

</div>

## ✨ Features

- **Automated Profiling**: Seamlessly runs Tenstorrent's TT-Metal profiler with pytest
- **CSV Analysis**: Automatically extracts and parses performance CSV files
- **Real-time Output**: Shows profiling progress in real-time
- **Performance Metrics**: Calculates total DEVICE KERNEL DURATION
- **Simple CLI**: Easy-to-use command-line interface
- **Flexible**: Supports named profiles and various test paths
- **Operation-based Profiling**: Profile specific operations by name (e.g., `ttperf add`)
- **Dynamic Configuration**: Customize tensor shape, dtype, and layout for operations
- **Config File Support**: Set defaults via `~/.ttperf.yaml` or `./.ttperf.yaml`
- **CI-friendly**: `--quiet` flag suppresses decorative output; `--verbose` enables debug logging

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI (recommended)
pip install ttperf

# With YAML config file support
pip install "ttperf[yaml]"
```

**Or install from source:**

```bash
git clone https://github.com/Aswincloud/ttperf.git
cd ttperf
pip install -e .
```

### Configuration

ttperf automatically searches for your TT-Metal installation using a simple two-step process:

```bash
# Option 1: Set PYTHONPATH to your tt-metal location
export PYTHONPATH=/path/to/your/tt-metal
ttperf add

# Option 2: Run from within tt-metal directory (or any subdirectory)
cd /path/to/your/tt-metal
ttperf relu
```

**tt-metal Path Search Order:**
1. `PYTHONPATH` environment variable (if specified)
2. Current working directory (walks up directory tree to find tt-metal root)

### Basic Usage

```bash
# Run profiling on a specific test
ttperf test_performance.py

# Run with a custom profile name
ttperf my_profile pytest test_performance.py

# Run on a specific test method
ttperf tests/test_ops.py::test_matmul

# Profile specific operations by name
ttperf add
ttperf relu
ttperf matmul

# Custom tensor configuration
ttperf add --shape 1,1,32,32 --dtype bfloat16 --layout tile
ttperf relu --shape 1,1,64,64 --dtype float32 --layout row_major

# Memory options
ttperf add --dram # Use DRAM memory (default)
ttperf relu --l1 # Use L1 memory

# CI-friendly (no emoji/decorative output)
ttperf --quiet add

# Copy CSV output to a directory
ttperf add --output-dir ./results/

# Enable verbose debug logging
ttperf --verbose add
```

## 📋 CLI Reference

```
ttperf [OPTIONS] [PROFILE_NAME] [pytest] <test_path_or_operation>

Options:
 --version Show version information
 --help, -h Show this help message
 --list-ops, -l List all supported operations
 --debug, -d Show real-time profiler output
 --verbose, -v Enable verbose logging (debug messages)
 --quiet, -q Suppress decorative/emoji output (for CI)
 --shape SHAPE         Tensor shape, e.g. 1,1,32,32 (default: 1,1,32,32)
 --dtype DTYPE         Data type: bfloat16/bf16, float32/fp32/f32, int32/i32 (default: bfloat16)
 --layout LAYOUT       Memory layout: tile, row_major/rm (default: tile)
 --memory-config       Memory configuration: dram, l1 (default: dram)
 --dram                Use DRAM memory (default)
 --l1                  Use L1 memory
 --output-dir DIR      Copy generated CSV to this directory
```

## ⚙️ Config File

Create `~/.ttperf.yaml` (global) or `./.ttperf.yaml` (project-local) to set defaults:

```yaml
# ~/.ttperf.yaml
shape: 1,1,32,32
dtype: bfloat16
layout: tile
memory_config: dram
output_dir: ./results
```

CLI flags always override config file values.

## 💡 Usage Examples

### Test File Profiling
```bash
ttperf test_conv.py
```

### Named Profile
```bash
ttperf conv_benchmark pytest test_conv.py
```

### Specific Test Method
```bash
ttperf tests/ops/test_matmul.py::test_basic_matmul
```

### Operation-based Profiling
```bash
# Basic operations
ttperf add
ttperf subtract
ttperf multiply

# Activation functions
ttperf relu
ttperf sigmoid
ttperf tanh
ttperf gelu

# Mathematical operations
ttperf sqrt
ttperf exp
ttperf log

# Comparison operations
ttperf gt
ttperf lt
ttperf eq

# Reduction operations
ttperf max
ttperf min
ttperf sum

# Backward operations
ttperf add_bw
ttperf relu_bw
```

### Dynamic Configuration
```bash
ttperf add --shape 1,1,32,32
ttperf relu --shape 2,3,64,128
ttperf add --dtype float32
ttperf add --layout row_major
ttperf add --shape 1,1,64,64 --dtype float32 --layout row_major
ttperf add --dram --shape 1,1,128,128
ttperf relu --l1 --dtype float32
```

### List All Supported Operations
```bash
ttperf --list-ops
# or
ttperf -l
```

### Output Example
```
Auto-generated profile name: temp_test_add
Running test...

============================================================
TEST SUMMARY
============================================================
Test: add
Status: PASSED
Configuration: shape=(1, 1, 32, 32), dtype=bfloat16, layout=tile, memory_config=dram (custom)
CSV Path: /path/to/profile_results.csv
DEVICE KERNEL DURATION [ns] total: 1234567.89 ns
============================================================
```

## 🔍 How It Works

1. **Command Parsing**: Analyzes input arguments to determine profile name and test path/operation
2. **Config Loading**: Reads `~/.ttperf.yaml` or `./.ttperf.yaml` for defaults (CLI flags take priority)
3. **Operation Detection**: If an operation name is provided, maps it to the corresponding test method
4. **Dynamic Configuration**: If custom configuration is provided, sets environment variables for the test
5. **Profile Execution**: Runs the Tenstorrent's TT-Metal profiler with the specified test
6. **Output Monitoring**: Streams profiling output in real-time (with `--debug`)
7. **CSV Extraction**: Parses the output to find the generated CSV file path, verifies it exists
8. **Performance Analysis**: Reads the CSV and calculates total device kernel duration
9. **Output Copy**: Optionally copies the CSV to `--output-dir` if specified

## 📊 Performance Metrics

The tool extracts the following key metrics:

- **DEVICE KERNEL DURATION [ns]**: Total time spent in device kernels
- **CSV Path**: Location of the detailed profiling results
- **Real-time Progress**: Live output during profiling (with `--debug`)

## 🔧 Configuration Options

### Shape Configuration
- **Format**: Comma-separated integers (e.g., `1,1,32,32`)
- **Default**: `1,1,32,32`
- **Example**: `--shape 2,3,64,128`

### Data Type Configuration
- **Valid Options**: `bfloat16` (or `bf16`), `float32` (or `fp32`/`f32`), `int32` (or `i32`)
- **Default**: `bfloat16`
- **Example**: `--dtype float32`

### Layout Configuration
- **Valid Options**: `tile`, `row_major` (or `rm`)
- **Default**: `tile`
- **Example**: `--layout row_major`

## 📦 Requirements

- Python 3.8+
- pandas
- Tenstorrent's TT-Metal development environment
- pytest
- PyYAML (optional, for config file support)

## 🗂️ Project Structure

```
ttperf/
├── ttperf/
│ ├── __init__.py
│ ├── ttperf.py # Main CLI implementation
│ └── data/
│ ├── operation_configs.json
│ └── test_eltwise_operations.py
├── tests/
│ └── test_ttperf.py # Unit tests
├── pyproject.toml
├── README.md
└── .gitignore
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This tool is an independent utility that interfaces with Tenstorrent's TT-Metal profiling tools. It is not affiliated with or endorsed by Tenstorrent Inc. The tool serves as a convenience wrapper around existing TT-Metal profiling infrastructure.

## Issues

If you encounter any issues, please [create an issue](https://github.com/Aswincloud/ttperf/issues) on GitHub.

## Author

**Aswin Z**
- GitHub: [@Aswincloud](https://github.com/Aswincloud)
- Portfolio: [aswincloud.com](https://aswincloud.com)

## 🙏 Acknowledgments

- Tenstorrent's TT-Metal development team for the profiling tools
- Python community for excellent libraries like pandas

---

<div align="center">
Made with care for the Tenstorrent TT-Metal community
</div>

