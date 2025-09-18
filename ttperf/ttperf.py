#!/usr/bin/env python3

import sys
import os
import subprocess
import pandas as pd
import re
import ast
import json
import argparse
import pkg_resources
from typing import Dict, List, Optional, Tuple

def load_operation_configs() -> Dict:
    """Load operation configurations from JSON file."""
    try:
        # Try to load from package data first
        config_path = pkg_resources.resource_filename('ttperf', 'data/operation_configs.json')
        with open(config_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, pkg_resources.DistributionNotFound):
        # Fallback to local file
        local_path = os.path.join(os.path.dirname(__file__), 'data', 'operation_configs.json')
        with open(local_path, 'r') as f:
            return json.load(f)

def get_operation_config(operation_name: str) -> Dict:
    """Get configuration for a specific operation from JSON."""
    configs = load_operation_configs()
    
    # Get operation-specific config or fall back to defaults
    op_config = configs['operations'].get(operation_name, {})
    defaults = configs['defaults'].copy()
    
    # For bitwise operations, use int32 as default dtype
    if operation_name.startswith('bitwise_'):
        defaults['dtype'] = 'int32'
    
    # Merge with defaults
    result = defaults.copy()
    result.update(op_config)
    
    return result

def get_expected_config_for_operation(operation_name: str) -> dict:
    """Get expected configuration for specific operations based on JSON config."""
    config = get_operation_config(operation_name)
    
    return {
        'shape': str(tuple(config['shape'])),
        'dtype': config['dtype'],
        'layout': config['layout']
    }


def get_test_file_path() -> str:
    """Get the path to the test_eltwise_operations.py file."""
    # Try to find the test file in the package data
    try:
        test_file = pkg_resources.resource_filename('ttperf', 'data/test_eltwise_operations.py')
        if os.path.exists(test_file):
            return test_file
    except:
        pass
    
    # Fallback: look in current directory and common locations
    possible_paths = [
        "test_eltwise_operations.py",
        "ttperf/data/test_eltwise_operations.py",
        os.path.join(os.path.dirname(__file__), "data", "test_eltwise_operations.py"),
        os.path.join(os.getcwd(), "test_eltwise_operations.py"),
        os.path.join(os.path.expanduser("~"), "ttperf", "test_eltwise_operations.py")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return "test_eltwise_operations.py"  # Default fallback


def extract_csv_path(output: str) -> str:
    match = re.search(r"OPs csv generated at: (.+?\.csv)", output)
    if not match:
        print("❌ CSV path not found in output.")
        sys.exit(1)
    return match.group(1)


def get_device_kernel_duration(csv_path: str) -> float:
    df = pd.read_csv(csv_path)
    if "DEVICE KERNEL DURATION [ns]" not in df.columns:
        print("❌ 'DEVICE KERNEL DURATION [ns]' column not found.")
        sys.exit(1)
    return df["DEVICE KERNEL DURATION [ns]"].sum()


def extract_test_methods_from_file(file_path: str) -> dict:
    """Dynamically extract test method names from the test file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse the Python file
        tree = ast.parse(content)
        
        # Find the TestEltwiseOperations class
        test_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'TestEltwiseOperations':
                test_class = node
                break
        
        if not test_class:
            return {}
        
        # Extract test method names
        operation_mapping = {}
        for node in test_class.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                # Convert test_method_name to operation_name
                operation_name = node.name[5:]  # Remove 'test_' prefix
                operation_mapping[operation_name] = node.name
        
        return operation_mapping
    except Exception as e:
        print(f"Warning: Could not parse test file: {e}")
        return {}


def get_operation_test_mapping():
    """Get mapping of operation names to test methods in test_eltwise_operations.py"""
    test_file_path = get_test_file_path()
    
    if os.path.exists(test_file_path):
        return extract_test_methods_from_file(test_file_path)
    
    # Fallback to a minimal mapping if file doesn't exist
    return {
        "add": "test_add",
        "relu": "test_relu",
        "sigmoid": "test_sigmoid",
        "tanh": "test_tanh",
        "gelu": "test_gelu",
        "sqrt": "test_sqrt",
        "exp": "test_exp",
        "log": "test_log",
        "sin": "test_sin",
        "cos": "test_cos",
    }


def is_operation_name(arg: str) -> bool:
    """Check if the argument is an operation name."""
    operation_mapping = get_operation_test_mapping()
    return arg.lower() in operation_mapping


def get_test_method_for_operation(operation_name: str) -> str:
    """Get the test method name for a given operation."""
    operation_mapping = get_operation_test_mapping()
    return operation_mapping.get(operation_name.lower())


def parse_shape(shape_str: str) -> tuple:
    """Parse shape string like '1,1,32,32' to tuple."""
    try:
        return tuple(int(x.strip()) for x in shape_str.split(','))
    except ValueError:
        print(f"❌ Invalid shape format: {shape_str}. Expected format: 1,1,32,32")
        sys.exit(1)


def validate_dtype(dtype_str: str) -> str:
    """Validate and return dtype string."""
    # Map aliases to canonical names
    dtype_aliases = {
        'bfloat16': 'bfloat16',
        'bf16': 'bfloat16',
        'float32': 'float32',
        'fp32': 'float32',
        'f32': 'float32',
        'int32': 'int32',
        'i32': 'int32'
    }
    
    dtype_lower = dtype_str.lower()
    if dtype_lower in dtype_aliases:
        return dtype_aliases[dtype_lower]
    
    valid_options = list(set(dtype_aliases.keys()))
    print(f"❌ Invalid dtype: {dtype_str}. Valid options: {', '.join(sorted(valid_options))}")
    sys.exit(1)


def validate_layout(layout_str: str) -> str:
    """Validate and return layout string."""
    # Map aliases to canonical names
    layout_aliases = {
        'tile': 'tile',
        'row_major': 'row_major',
        'rm': 'row_major',
        'rowmajor': 'row_major'
    }
    
    layout_lower = layout_str.lower()
    if layout_lower in layout_aliases:
        return layout_aliases[layout_lower]
    
    valid_options = list(set(layout_aliases.keys()))
    print(f"❌ Invalid layout: {layout_str}. Valid options: {', '.join(sorted(valid_options))}")
    sys.exit(1)


def validate_memory_config(memory_config_str: str) -> str:
    """Validate and return memory configuration string."""
    # Map aliases to canonical names
    memory_config_aliases = {
        'dram': 'dram',
        'l1': 'l1',
        'dram_interleaved': 'dram',
        'l1_memory': 'l1'
    }
    
    memory_config_lower = memory_config_str.lower()
    if memory_config_lower in memory_config_aliases:
        return memory_config_aliases[memory_config_lower]
    
    valid_options = list(set(memory_config_aliases.keys()))
    print(f"❌ Invalid memory config: {memory_config_str}. Valid options: {', '.join(sorted(valid_options))}")
    sys.exit(1)


def set_test_configuration(shape: tuple, dtype: str, layout: str, memory_config: str = None, operation_name: str = None):
    """Set environment variables for test configuration."""
    # For bitwise operations, always use int32 regardless of what's specified
    if operation_name and operation_name.startswith('bitwise_'):
        dtype = 'int32'
    
    os.environ['TTPERF_CUSTOM_SHAPE'] = str(shape)
    os.environ['TTPERF_CUSTOM_DTYPE'] = dtype
    os.environ['TTPERF_CUSTOM_LAYOUT'] = layout
    if memory_config:
        os.environ['TTPERF_CUSTOM_MEMORY_CONFIG'] = memory_config
    print(f"🔧 Using custom configuration:")
    print(f"   Shape: {shape}")
    print(f"   Dtype: {dtype}")
    print(f"   Layout: {layout}")
    if memory_config:
        print(f"   Memory Config: {memory_config}")


def print_help():
    print("""🚀 ttperf - TT-Metal Performance Profiler

Usage: ttperf [OPTIONS] [PROFILE_NAME] [pytest] <test_path_or_operation>

Examples:
  ttperf test_performance.py                    # Auto-generated profile: test_performance
  ttperf my_profile pytest test_performance.py # Custom profile name: my_profile
  ttperf tests/test_ops.py::test_matmul        # Auto-generated profile: test_matmul
  ttperf add                                    # Profile specific operation: add
  ttperf my_profile add                         # Custom profile name for operation: my_profile
  ttperf add --shape 1,1,32,32 --dtype bf16 --layout tile      # Custom configuration
  ttperf relu --dtype fp32 --layout rm                        # Using aliases
  ttperf add --dram                                            # Use DRAM memory (default)
  ttperf relu --l1                                             # Use L1 memory

Options:
  --version, -v           Show version information
  --help, -h              Show this help message
  --list-ops, -l          List all supported operations
  --debug, -d             Show real-time debug output
  --shape SHAPE           Tensor shape (e.g., 1,1,32,32)
  --dtype DTYPE           Data type (bfloat16/bf16, float32/fp32/f32, int32/i32)
  --layout LAYOUT         Memory layout (tile, row_major/rm)
  --memory-config CONFIG  Memory configuration (dram, l1)
  --dram                  Use DRAM memory (default)
  --l1                    Use L1 memory

Arguments:
  PROFILE_NAME            Optional name for the profiling session
  test_path               Path to test file or specific test method
  operation               Operation name to profile (e.g., add, relu, matmul)

For more information, visit: https://github.com/Aswintechie/ttperf""")


def print_supported_operations():
    """Print all supported operations."""
    operation_mapping = get_operation_test_mapping()
    operations = sorted(operation_mapping.keys())
    
    print("📋 Supported Operations:")
    print("=" * 50)
    
    # Group operations by category
    categories = {
        "Unary": [],
        "Binary": [],
        "Ternary": [],
        "Reduction": [],
        "Complex": [],
        "Backward": []
    }
    
    for op in operations:
        if op.endswith("_bw"):
            categories["Backward"].append(op)
        elif op in ["where", "mac", "addcdiv", "addcmul", "lerp"]:
            categories["Ternary"].append(op)
        elif op in ["max", "min", "mean", "sum", "prod", "var", "std", "cumsum", "cumprod"]:
            categories["Reduction"].append(op)
        elif op in ["complex_tensor", "real", "imag", "angle", "conj", "polar", "complex_recip"]:
            categories["Complex"].append(op)
        elif op in ["add", "subtract", "multiply", "divide", "gt", "lt", "eq", "ne", "ge", "le", 
                   "logical_and", "logical_or", "logical_xor", "atan2", "hypot", "logaddexp",
                   "logaddexp2", "maximum", "minimum", "pow", "fmod", "remainder", 
                   "squared_difference", "bitwise_and", "bitwise_or", "bitwise_xor",
                   "mul", "sub", "rpow", "rdiv", "ldexp", "xlogy", "nextafter", "bias_gelu",
                   "addalpha", "subalpha", "isclose"] or op.endswith("_"):
            categories["Binary"].append(op)
        else:
            categories["Unary"].append(op)
    
    for category, ops in categories.items():
        if ops:
            print(f"\n{category} Operations ({len(ops)}):")
            print("-" * 30)
            for i, op in enumerate(ops):
                print(f"  {op:<20}", end="")
                if (i + 1) % 3 == 0:
                    print()
            if len(ops) % 3 != 0:
                print()
    
    print(f"\n\nTotal: {len(operations)} operations supported")


def generate_profile_name(test_cmd: str) -> str:
    """Generate a profile name from the test command/path."""
    # Handle specific test method (e.g., test_ops.py::test_matmul -> test_matmul)
    if "::" in test_cmd:
        return test_cmd.split("::")[-1]
    
    # Handle file path (e.g., tests/test_conv.py -> test_conv)
    if test_cmd.endswith(".py"):
        filename = os.path.splitext(os.path.basename(test_cmd))[0]  # Gets filename without extension
        return filename
    
    # Handle directory or other cases
    return os.path.basename(test_cmd) or "profile"


def parse_args(argv):
    # Handle version and help flags
    if "--version" in argv or "-v" in argv:
        print("ttperf version 0.1.6")
        sys.exit(0)
    
    if "--help" in argv or "-h" in argv:
        print_help()
        sys.exit(1)
    
    if "--list-ops" in argv or "-l" in argv:
        print_supported_operations()
        sys.exit(0)
    
    # Parse arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--debug', '-d', action='store_true', help='Show real-time debug output')
    parser.add_argument('--shape', type=str, help='Tensor shape (e.g., 1,1,32,32)')
    parser.add_argument('--dtype', type=str, help='Data type (bfloat16/bf16, float32/fp32/f32, int32/i32)')
    parser.add_argument('--layout', type=str, help='Memory layout (tile, row_major/rm)')
    parser.add_argument('--memory-config', type=str, choices=['dram', 'l1'], default='dram', help='Memory configuration (dram, l1)')
    parser.add_argument('--dram', action='store_const', const='dram', dest='memory_config', help='Use DRAM memory (default)')
    parser.add_argument('--l1', action='store_const', const='l1', dest='memory_config', help='Use L1 memory')
    
    # Parse known args to extract configuration options
    args, remaining = parser.parse_known_args(argv)
    
    # Default values
    name = None
    test_cmd = None
    custom_config = None
    operation_name = None

    for arg in remaining:
        if arg.endswith(".py") or "::" in arg or os.path.exists(arg):
            test_cmd = arg
        elif arg.lower() == "pytest":
            continue
        elif is_operation_name(arg):
            # This is an operation name, construct the test command
            operation_name = arg
            test_method = get_test_method_for_operation(operation_name)
            test_file_path = get_test_file_path()
            test_cmd = f"{test_file_path}::TestEltwiseOperations::{test_method}"
        else:
            name = arg

    if not test_cmd:
        print("❌ Test file/path or operation name not found in arguments.")
        print_help()
        sys.exit(1)

    # Process custom configuration
    if args.shape or args.dtype or args.layout or args.memory_config:
        # Check if we're profiling an operation (not a test file)
        if test_cmd and "test_eltwise_operations.py" in test_cmd:
            # Parse configuration
            shape = parse_shape(args.shape) if args.shape else (1, 1, 32, 32)
            dtype = validate_dtype(args.dtype) if args.dtype else "bfloat16"
            layout = validate_layout(args.layout) if args.layout else "tile"
            memory_config = validate_memory_config(args.memory_config) if args.memory_config else "dram"
            
            # For bitwise operations, always use int32 regardless of what's specified
            if operation_name and operation_name.startswith('bitwise_'):
                dtype = 'int32'
            
            # Store custom configuration
            custom_config = {
                'shape': shape,
                'dtype': dtype,
                'layout': layout,
                'memory_config': memory_config
            }
            
            # Set environment variables for the test
            set_test_configuration(shape, dtype, layout, memory_config, operation_name)
        else:
            print("⚠️  Custom configuration options (--shape, --dtype, --layout, --memory-config) only work with operation names, not test files.")

    # Auto-generate profile name if not provided
    if not name:
        name = generate_profile_name(test_cmd)
        print(f"🏷️ Auto-generated profile name: {name}")

    return name, test_cmd, args.debug, custom_config


def build_profile_command(name, test_cmd):
    name_arg = f"-n {name}" if name else ""
    # Use absolute path to tt-metal directory
    tt_metal_path = "/home/ubuntu/tt-metal"
    return f"{tt_metal_path}/tools/tracy/profile_this.py {name_arg} -c \"pytest {test_cmd}\""


def extract_config_from_csv(csv_path: str) -> dict:
    """Extract test configuration from the CSV file."""
    config = {}
    
    try:
        df = pd.read_csv(csv_path)
        if len(df) > 0:
            # Get the first row (assuming single operation)
            row = df.iloc[0]
            
            # Extract shape from input dimensions
            # Format: INPUT_0_W_PAD[LOGICAL], INPUT_0_Z_PAD[LOGICAL], INPUT_0_Y_PAD[LOGICAL], INPUT_0_X_PAD[LOGICAL]
            w = row.get('INPUT_0_W_PAD[LOGICAL]', '1')
            z = row.get('INPUT_0_Z_PAD[LOGICAL]', '1')  
            y = row.get('INPUT_0_Y_PAD[LOGICAL]', '32')
            x = row.get('INPUT_0_X_PAD[LOGICAL]', '32')
            
            # Parse dimensions (they may be in format like "32[32]")
            def parse_dim(dim_str):
                if isinstance(dim_str, str) and '[' in dim_str:
                    return dim_str.split('[')[0]
                return str(dim_str)
            
            w_val = parse_dim(w)
            z_val = parse_dim(z)
            y_val = parse_dim(y)
            x_val = parse_dim(x)
            
            config['shape'] = f"{w_val}, {z_val}, {y_val}, {x_val}"
            
            # Extract dtype (prefer output datatype, fallback to input)
            output_dtype = row.get('OUTPUT_0_DATATYPE', row.get('INPUT_0_DATATYPE', 'BFLOAT16'))
            # Convert to lowercase for consistency
            config['dtype'] = output_dtype.lower() if isinstance(output_dtype, str) else 'bfloat16'
            
            # Extract layout (prefer output layout, fallback to input)
            output_layout = row.get('OUTPUT_0_LAYOUT', row.get('INPUT_0_LAYOUT', 'TILE'))
            # Convert to lowercase for consistency  
            config['layout'] = output_layout.lower() if isinstance(output_layout, str) else 'tile'
            
            # Extract memory configuration from memory columns
            output_memory = row.get('OUTPUT_0_MEMORY', row.get('INPUT_0_MEMORY', 'DEV_1_DRAM_INTERLEAVED'))
            if isinstance(output_memory, str):
                if 'L1' in output_memory.upper():
                    config['memory_config'] = 'l1'
                elif 'DRAM' in output_memory.upper():
                    config['memory_config'] = 'dram'
                else:
                    config['memory_config'] = 'dram'  # Default fallback
            else:
                config['memory_config'] = 'dram'  # Default fallback
            
    except Exception as e:
        print(f"⚠️  Warning: Could not extract config from CSV: {e}")
        # Return empty config - will fall back to other methods
        
    return config


def extract_test_config_and_status(output: str, csv_path: str = None) -> dict:
    """Extract test configuration and pass/fail status from output and CSV."""
    result = {
        'config': {},
        'status': 'unknown',
        'test_name': 'unknown'
    }
    
    # Extract test name - just the operation name, not the full class.method
    # Look for patterns like "test_eltwise_operations.py::TestEltwiseOperations::test_add"
    test_match = re.search(r'::([^:]+)::test_([a-zA-Z_]+)', output)
    if test_match:
        method_name = test_match.group(2)
        # Remove 'test_' prefix to get just the operation name
        result['test_name'] = method_name
    else:
        # Fallback: look for test method names
        test_method_match = re.search(r'test_([a-zA-Z_]+)', output)
        if test_method_match:
            result['test_name'] = test_method_match.group(1)
    
    # Try to extract configuration from CSV first (most reliable)
    if csv_path and os.path.exists(csv_path):
        csv_config = extract_config_from_csv(csv_path)
        if csv_config:
            result['config'] = csv_config
    
    # If no CSV config available, try to extract from output (fallback)
    if not result['config']:
        # Look for custom configuration patterns in output
        shape_match = re.search(r'🔧.*?Using.*?configuration.*?Shape:\s*\(([^)]+)\)', output, re.IGNORECASE)
        if shape_match:
            result['config']['shape'] = shape_match.group(1)
            
        dtype_match = re.search(r'🔧.*?Using.*?configuration.*?Dtype:\s*(bfloat16|float32|int32)', output, re.IGNORECASE)
        if dtype_match:
            result['config']['dtype'] = dtype_match.group(1)
            
        layout_match = re.search(r'🔧.*?Using.*?configuration.*?Layout:\s*(tile|row_major)', output, re.IGNORECASE)
        if layout_match:
            result['config']['layout'] = layout_match.group(1).lower()
            
        memory_config_match = re.search(r'🔧.*?Using.*?configuration.*?Memory Config:\s*(L1|DRAM|dram|l1)', output, re.IGNORECASE)
        if memory_config_match:
            result['config']['memory_config'] = memory_config_match.group(1).lower()
    
    # For bitwise operations, ensure int32 dtype if not already set from CSV
    if result['test_name'].startswith('bitwise_') and not result['config'].get('dtype'):
        result['config']['dtype'] = 'int32'
    
    # Determine test status from output
    if 'PASSED' in output or 'passed' in output:
        result['status'] = 'PASSED'
    elif 'FAILED' in output or 'failed' in output:
        result['status'] = 'FAILED'
    elif 'ERROR' in output or 'error' in output:
        result['status'] = 'ERROR'
    elif 'collected' in output and 'passed' in output:
        result['status'] = 'PASSED'
    elif 'collected' in output and 'failed' in output:
        result['status'] = 'FAILED'
    elif '1 passed' in output:
        result['status'] = 'PASSED'
    elif '1 failed' in output:
        result['status'] = 'FAILED'
    
    return result


def print_test_summary(test_info: dict, csv_path: str, duration: float, custom_config: dict = None):
    """Print a comprehensive test summary."""
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    # Test information
    print(f"🧪 Test: {test_info['test_name']}")
    print(f"📋 Status: {test_info['status']}")
    
    # Configuration - prefer custom config if available
    if custom_config:
        config_str = []
        if 'shape' in custom_config:
            config_str.append(f"shape={custom_config['shape']}")
        if 'dtype' in custom_config:
            config_str.append(f"dtype={custom_config['dtype']}")
        if 'layout' in custom_config:
            config_str.append(f"layout={custom_config['layout']}")
        if 'memory_config' in custom_config:
            config_str.append(f"memory_config={custom_config['memory_config']}")
        print(f"⚙️  Configuration: {', '.join(config_str)} (custom)")
    elif test_info['config']:
        config_str = []
        if 'shape' in test_info['config']:
            config_str.append(f"shape={test_info['config']['shape']}")
        if 'dtype' in test_info['config']:
            config_str.append(f"dtype={test_info['config']['dtype']}")
        if 'layout' in test_info['config']:
            config_str.append(f"layout={test_info['config']['layout']}")
        if 'memory_config' in test_info['config']:
            config_str.append(f"memory_config={test_info['config']['memory_config']}")
        print(f"⚙️  Configuration: {', '.join(config_str)}")
    else:
        # Try to show expected configuration based on operation name
        expected_config = get_expected_config_for_operation(test_info['test_name'])
        if expected_config:
            config_str = []
            if 'shape' in expected_config:
                config_str.append(f"shape={expected_config['shape']}")
            if 'dtype' in expected_config:
                config_str.append(f"dtype={expected_config['dtype']}")
            if 'layout' in expected_config:
                config_str.append(f"layout={expected_config['layout']}")
            if 'memory_config' in expected_config:
                config_str.append(f"memory_config={expected_config['memory_config']}")
            print(f"⚙️  Configuration: {', '.join(config_str)} (expected)")
        else:
            print("⚙️  Configuration: Not detected")
    
    # Performance metrics
    print(f"📁 CSV Path: {csv_path}")
    print(f"⏱️  DEVICE KERNEL DURATION [ns] total: {duration:.2f} ns")
    print("="*60)


def main():
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)

    name, test_cmd, debug, custom_config = parse_args(sys.argv[1:])
    profile_cmd = build_profile_command(name, test_cmd)

    if debug:
        print(f"▶️ Running: {profile_cmd}\n")
    else:
        print(f"▶️ Running test...")

    process = subprocess.Popen(
        profile_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )

    output_lines = []
    try:
        for line in process.stdout:
            if debug:
                print(line, end="")  # Real-time output only in debug mode
            output_lines.append(line)
    except KeyboardInterrupt:
        process.terminate()
        print("❌ Aborted.")
        sys.exit(1)

    process.wait()

    # Combine all output for post-analysis
    full_output = "".join(output_lines)

    # Extract CSV path and duration
    try:
        csv_path = extract_csv_path(full_output)
        duration = get_device_kernel_duration(csv_path)
        
        # Extract test configuration and status
        test_info = extract_test_config_and_status(full_output, csv_path)
        
        # Print comprehensive summary
        print_test_summary(test_info, csv_path, duration, custom_config)
        
    except Exception as e:
        print(f"\n❌ Error processing results: {e}")
        print("Raw output:")
        print(full_output)
        sys.exit(1)


if __name__ == "__main__":
    main()