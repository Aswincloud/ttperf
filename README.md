# 🚀 ttperf - TT-Metal Performance Profiler

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Version](https://img.shields.io/badge/version-0.1.6-orange.svg)
[![GitHub issues](https://img.shields.io/github/issues/Aswintechie/ttperf)](https://github.com/Aswintechie/ttperf/issues)
[![GitHub stars](https://img.shields.io/github/stars/Aswintechie/ttperf)](https://github.com/Aswintechie/ttperf/stargazers)

**A streamlined CLI tool for profiling Tenstorrent's TT-Metal tests and extracting device kernel performance metrics**

</div>

## ✨ Features

- 🔍 **Automated Profiling**: Seamlessly runs Tenstorrent's TT-Metal profiler with pytest
- 📊 **CSV Analysis**: Automatically extracts and parses performance CSV files
- ⚡ **Real-time Output**: Shows profiling progress in real-time
- 📈 **Performance Metrics**: Calculates total DEVICE KERNEL DURATION
- 🎯 **Simple CLI**: Easy-to-use command-line interface
- 🛠️ **Flexible**: Supports named profiles and various test paths
- 🚀 **Operation-based Profiling**: Profile specific operations by name (e.g., `ttperf add`)
- ⚙️ **Dynamic Configuration**: Customize tensor shape, dtype, and layout for operations

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI (recommended)
pip install ttperf
```

**Or install from source:**

```bash
# Clone the repository
git clone https://github.com/Aswintechie/ttperf.git
cd ttperf

# Install the package
pip install -e .
```

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

# Profile operations with custom profile names
ttperf my_add_profile add
ttperf my_relu_profile relu

# Profile operations with custom configuration
ttperf add --shape 1,1,32,32 --dtype bfloat16 --layout tile
ttperf relu --shape 1,1,64,64 --dtype float32 --layout row_major

# Profile operations with memory configuration
ttperf add --dram                                # Use DRAM memory (default)
ttperf relu --l1                                 # Use L1 memory
ttperf add --shape 1,1,64,64 --l1                # Combined options
```

## 📋 Usage Examples

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
ttperf divide

# Activation functions
ttperf relu
ttperf sigmoid
ttperf tanh
ttperf gelu

# Mathematical operations
ttperf sqrt
ttperf exp
ttperf log
ttperf sin
ttperf cos

# Comparison operations
ttperf gt
ttperf lt
ttperf eq
ttperf ne

# Reduction operations
ttperf max
ttperf min
ttperf mean
ttperf sum

# Backward operations
ttperf add_bw
ttperf relu_bw
ttperf sigmoid_bw
```

### Dynamic Configuration
```bash
# Custom tensor shape
ttperf add --shape 1,1,32,32
ttperf relu --shape 2,3,64,128

# Custom data type
ttperf add --dtype float32
ttperf multiply --dtype int32

# Custom memory layout
ttperf add --layout row_major
ttperf relu --layout tile

# Combined configuration
ttperf add --shape 1,1,64,64 --dtype float32 --layout row_major
ttperf gelu --shape 2,1,32,32 --dtype bfloat16 --layout tile

# Memory configuration options
ttperf add --memory-config dram                  # Explicit DRAM
ttperf relu --memory-config l1                   # Explicit L1  
ttperf add --dram --shape 1,1,128,128            # DRAM with custom shape
ttperf relu --l1 --dtype float32                 # L1 with custom dtype
```

### List All Supported Operations
```bash
ttperf --list-ops
# or
ttperf -l
```

### Output Example
```
🔧 Using custom configuration:
   Shape: (1, 1, 32, 32)
   Dtype: bfloat16
   Layout: tile
🏷️ Auto-generated profile name: temp_test_add
▶️ Running: ./tools/tracy/profile_this.py -n temp_test_add -c "pytest temp_test_add.py"

... (profiling output) ...

📁 Found CSV path: /path/to/profile_results.csv
⏱️ DEVICE KERNEL DURATION [ns] total: 1234567.89 ns
```

## 🛠️ How It Works

1. **Command Parsing**: Analyzes input arguments to determine profile name and test path/operation
2. **Operation Detection**: If an operation name is provided, maps it to the corresponding test method
3. **Dynamic Configuration**: If custom configuration is provided, generates a temporary test file with the specified parameters
4. **Profile Execution**: Runs the Tenstorrent's TT-Metal profiler with the specified test
5. **Output Monitoring**: Streams profiling output in real-time
6. **CSV Extraction**: Parses the output to find the generated CSV file path
7. **Performance Analysis**: Reads the CSV and calculates total device kernel duration

## 📊 Performance Metrics

The tool extracts the following key metrics:

- **DEVICE KERNEL DURATION [ns]**: Total time spent in device kernels
- **CSV Path**: Location of the detailed profiling results
- **Real-time Progress**: Live output during profiling

## ⚙️ Configuration Options

### Shape Configuration
- **Format**: Comma-separated integers (e.g., `1,1,32,32`)
- **Default**: `(1, 1, 1024, 1024)`
- **Example**: `--shape 2,3,64,128`

### Data Type Configuration
- **Valid Options**: `bfloat16`, `float32`, `int32`
- **Default**: `bfloat16`
- **Example**: `--dtype float32`

### Layout Configuration
- **Valid Options**: `tile`, `row_major`
- **Default**: `tile`
- **Example**: `--layout row_major`

## 🔧 Requirements

- Python 3.8+
- pandas
- Tenstorrent's TT-Metal development environment
- pytest

## 📁 Project Structure

```
ttperf/
├── ttperf.py          # Main CLI implementation
├── pyproject.toml     # Project configuration
├── README.md          # This file
└── .gitignore         # Git ignore rules
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This tool is an independent utility that interfaces with Tenstorrent's TT-Metal profiling tools. It is not affiliated with or endorsed by Tenstorrent Inc. The tool serves as a convenience wrapper around existing TT-Metal profiling infrastructure.

## 🐛 Issues

If you encounter any issues, please [create an issue](https://github.com/Aswintechie/ttperf/issues) on GitHub.

## 👨‍💻 Author

**Aswin Z**
- GitHub: [@Aswintechie](https://github.com/Aswintechie)
- Portfolio: [aswinlocal.in](https://aswinlocal.in)

## 🌟 Acknowledgments

- Tenstorrent's TT-Metal development team for the profiling tools
- Python community for excellent libraries like pandas

---

<div align="center">
Made with ❤️ for the Tenstorrent TT-Metal community
</div> 