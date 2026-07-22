#!/usr/bin/env python3

import sys
import os
import subprocess
import shutil
import pandas as pd
import re
import ast
import json
import argparse
import logging
from typing import Dict, List, Optional, Tuple

try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("ttperf")
    except PackageNotFoundError:
        __version__ = "dev"
except ImportError:
    # Python < 3.8 fallback
    __version__ = "dev"

logger = logging.getLogger(__name__)


def load_operation_configs() -> Dict:
    """Load operation configurations from JSON file."""
    try:
        from importlib.resources import files
        config_text = files('ttperf').joinpath('data/operation_configs.json').read_text()
        return json.loads(config_text)
    except Exception:
        pass
    try:
        import pkg_resources
        config_path = pkg_resources.resource_filename('ttperf', 'data/operation_configs.json')
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception:
        pass
    local_path = os.path.join(os.path.dirname(__file__), 'data', 'operation_configs.json')
    with open(local_path, 'r') as f:
        return json.load(f)


def get_operation_config(operation_name: str) -> Dict:
    """Get configuration for a specific operation from JSON."""
    configs = load_operation_configs()
    op_config = configs['operations'].get(operation_name, {})
    defaults = configs['defaults'].copy()
    if operation_name.startswith('bitwise_'):
        defaults['dtype'] = 'int32'
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
    try:
        import pkg_resources
        test_file = pkg_resources.resource_filename('ttperf', 'data/test_eltwise_operations.py')
        if os.path.exists(test_file):
            return test_file
    except Exception:
        pass

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

    return "test_eltwise_operations.py"


def extract_csv_path(output: str) -> str:
    """Extract the CSV file path from profiler output."""
    match = re.search(r"OPs csv generated at: (.+?\.csv)", output)
    if not match:
        logger.debug("Full output:\n%s", output)
        print("❌ CSV path not found in output.")
        sys.exit(1)
    csv_path = match.group(1).strip()
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found at path: {csv_path}")
        print(f"   Please verify the profiler completed successfully.")
        sys.exit(1)
    return csv_path


def get_device_kernel_duration(csv_path: str) -> float:
    """Read the CSV and return the total DEVICE KERNEL DURATION."""
    df = pd.read_csv(csv_path)
    target_col = "DEVICE KERNEL DURATION [ns]"
    if target_col not in df.columns:
        available = ", ".join(df.columns.tolist())
        print(f"❌ '{target_col}' column not found in CSV.")
        print(f"   Available columns: {available}")
        sys.exit(1)
    return df[target_col].sum()


def extract_test_methods_from_file(file_path: str) -> dict:
    """Dynamically extract test method names from the test file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        tree = ast.parse(content)
        test_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'TestEltwiseOperations':
                test_class = node
                break

        if not test_class:
            return {}

        operation_mapping = {}
        for node in test_class.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                operation_name = node.name[5:]
                operation_mapping[operation_name] = node.name

        return operation_mapping
    except Exception as e:
        logger.warning("Could not parse test file: %s", e)
        return {}


def get_operation_test_mapping() -> dict:
    """Get mapping of operation names to test methods in test_eltwise_operations.py."""
    test_file_path = get_test_file_path()

    if os.path.exists(test_file_path):
        return extract_test_methods_from_file(test_file_path)

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


def get_test_method_for_operation(operation_name: str) -> Optional[str]:
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
    valid_options = sorted(set(dtype_aliases.keys()))
    print(f"❌ Invalid dtype: {dtype_str}. Valid options: {', '.join(valid_options)}")
    sys.exit(1)


def validate_layout(layout_str: str) -> str:
    """Validate and return layout string."""
    layout_aliases = {
        'tile': 'tile',
        'row_major': 'row_major',
        'rm': 'row_major',
        'rowmajor': 'row_major'
    }
    layout_lower = layout_str.lower()
    if layout_lower in layout_aliases:
        return layout_aliases[layout_lower]
    valid_options = sorted(set(layout_aliases.keys()))
    print(f"❌ Invalid layout: {layout_str}. Valid options: {', '.join(valid_options)}")
    sys.exit(1)


def validate_memory_config(memory_config_str: str) -> str:
    """Validate and return memory configuration string."""
    memory_config_aliases = {
        'dram': 'dram',
        'l1': 'l1',
        'dram_interleaved': 'dram',
        'l1_memory': 'l1'
    }
    memory_config_lower = memory_config_str.lower()
    if memory_config_lower in memory_config_aliases:
        return memory_config_aliases[memory_config_lower]
    valid_options = sorted(set(memory_config_aliases.keys()))
    print(f"❌ Invalid memory config: {memory_config_str}. Valid options: {', '.join(valid_options)}")
    sys.exit(1)


def set_test_configuration(
    shape: tuple,
    dtype: str,
    layout: str,
    memory_config: Optional[str] = None,
    operation_name: Optional[str] = None,
    quiet: bool = False
) -> None:
    """Set environment variables for test configuration."""
    if operation_name and operation_name.startswith('bitwise_'):
        dtype = 'int32'

    os.environ['TTPERF_CUSTOM_SHAPE'] = str(shape)
    os.environ['TTPERF_CUSTOM_DTYPE'] = dtype
    os.environ['TTPERF_CUSTOM_LAYOUT'] = layout
    if memory_config:
        os.environ['TTPERF_CUSTOM_MEMORY_CONFIG'] = memory_config

    if not quiet:
        print(f"🔧 Using custom configuration:")
        print(f"   Shape: {shape}")
        print(f"   Dtype: {dtype}")
        print(f"   Layout: {layout}")
        if memory_config:
            print(f"   Memory Config: {memory_config}")


def load_config_file() -> dict:
    """Load defaults from ~/.ttperf.yaml or ./.ttperf.yaml if present."""
    config = {}
    candidates = [
        os.path.join(os.getcwd(), '.ttperf.yaml'),
        os.path.expanduser('~/.ttperf.yaml'),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                import yaml
                with open(path, 'r') as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        config = loaded
                        logger.debug("Loaded config from %s", path)
            except ImportError:
                # PyYAML not available; try configparser as fallback
                import configparser
                cp = configparser.ConfigParser()
                cp.read(path)
                if 'defaults' in cp:
                    config = dict(cp['defaults'])
                    logger.debug("Loaded config (ini) from %s", path)
            except Exception as e:
                logger.warning("Could not load config file %s: %s", path, e)
            break
    return config


def print_help(quiet: bool = False) -> None:
    print("""ttperf - TT-Metal Performance Profiler

Usage: ttperf [OPTIONS] [PROFILE_NAME] [pytest] <test_path_or_operation>

Examples:
  ttperf test_performance.py                    # Auto-generated profile: test_performance
  ttperf my_profile pytest test_performance.py  # Custom profile name: my_profile
  ttperf tests/test_ops.py::test_matmul         # Auto-generated profile: test_matmul
  ttperf add                                    # Profile specific operation: add
  ttperf my_profile add                         # Custom profile name for operation
  ttperf add --shape 1,1,32,32 --dtype bf16 --layout tile
  ttperf relu --dtype fp32 --layout rm
  ttperf add --dram
  ttperf relu --l1

Options:
  --version               Show version information
  --help, -h              Show this help message
  --list-ops, -l          List all supported operations
  --debug, -d             Show real-time debug output
  --verbose, -v           Enable verbose logging (logger.debug messages)
  --quiet, -q             Suppress decorative/emoji output (useful for CI)
  --shape SHAPE           Tensor shape (e.g., 1,1,32,32)
  --dtype DTYPE           Data type (bfloat16/bf16, float32/fp32/f32, int32/i32)
  --layout LAYOUT         Memory layout (tile, row_major/rm)
  --memory-config CONFIG  Memory configuration (dram, l1)
  --dram                  Use DRAM memory (default)
  --l1                    Use L1 memory
  --output-dir DIR        Copy generated CSV to this directory after profiling

Arguments:
  PROFILE_NAME            Optional name for the profiling session
  test_path               Path to test file or specific test method
  operation               Operation name to profile (e.g., add, relu, matmul)

Config File:
  ttperf reads defaults from ~/.ttperf.yaml or ./.ttperf.yaml (local takes priority).
  CLI flags always override config file values.

Environment Variables:
  PYTHONPATH             Path to tt-metal installation (optional)

For more information, visit: https://github.com/Aswincloud/ttperf""")


def print_supported_operations(quiet: bool = False) -> None:
    """Print all supported operations."""
    operation_mapping = get_operation_test_mapping()
    operations = sorted(operation_mapping.keys())

    print("Supported Operations:")
    print("=" * 50)

    categories: Dict[str, List[str]] = {
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
    if "::" in test_cmd:
        return test_cmd.split("::")[-1]
    if test_cmd.endswith(".py"):
        filename = os.path.splitext(os.path.basename(test_cmd))[0]
        return filename
    return os.path.basename(test_cmd) or "profile"


def parse_args(argv: List[str]) -> Tuple:
    """Parse CLI arguments, applying config file defaults first."""
    # Load config file defaults
    file_config = load_config_file()

    if "--version" in argv:
        print(f"ttperf version {__version__}")
        sys.exit(0)

    if "--help" in argv or "-h" in argv:
        print_help()
        sys.exit(1)

    if "--list-ops" in argv or "-l" in argv:
        print_supported_operations()
        sys.exit(0)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--quiet', '-q', action='store_true')
    parser.add_argument('--shape', type=str, default=file_config.get('shape'))
    parser.add_argument('--dtype', type=str, default=file_config.get('dtype'))
    parser.add_argument('--layout', type=str, default=file_config.get('layout'))
    parser.add_argument('--memory-config', type=str, choices=['dram', 'l1'],
                        default=file_config.get('memory_config', 'dram'))
    parser.add_argument('--dram', action='store_const', const='dram', dest='memory_config')
    parser.add_argument('--l1', action='store_const', const='l1', dest='memory_config')
    parser.add_argument('--output-dir', type=str, default=file_config.get('output_dir'))

    args, remaining = parser.parse_known_args(argv)

    # Configure logging based on --verbose
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

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

    # Only treat config as "custom" when the user actually passed a config flag
    # on the command line. We can't rely on parsed args because --memory-config
    # defaults to 'dram', which would make every op run look custom.
    config_flag_prefixes = ('--shape', '--dtype', '--layout', '--memory-config', '--dram', '--l1')
    user_passed_config = any(
        tok == flag or tok.startswith(flag + '=')
        for tok in argv for flag in config_flag_prefixes
    )

    if user_passed_config:
        if test_cmd and "test_eltwise_operations.py" in test_cmd:
            shape = parse_shape(args.shape) if args.shape else (1, 1, 32, 32)
            dtype = validate_dtype(args.dtype) if args.dtype else "bfloat16"
            layout = validate_layout(args.layout) if args.layout else "tile"
            memory_config = validate_memory_config(args.memory_config) if args.memory_config else "dram"

            if operation_name and operation_name.startswith('bitwise_'):
                dtype = 'int32'

            custom_config = {
                'shape': shape,
                'dtype': dtype,
                'layout': layout,
                'memory_config': memory_config
            }

            set_test_configuration(shape, dtype, layout, memory_config, operation_name, quiet=args.quiet)
        else:
            if not args.quiet:
                print("Warning: Custom configuration options only work with operation names, not test files.")

    if not name:
        name = generate_profile_name(test_cmd)
        if not args.quiet:
            print(f"Auto-generated profile name: {name}")

    return name, test_cmd, args.debug, custom_config, args.quiet, args.output_dir


def find_tt_metal_path() -> str:
    """Find tt-metal directory path in order of preference."""
    pythonpath = os.environ.get('PYTHONPATH', '')
    if pythonpath:
        for path in pythonpath.split(':'):
            if 'tt-metal' in path:
                if path.endswith('tt-metal') or os.path.basename(path) == 'tt-metal':
                    if os.path.exists(path) and os.path.isdir(path):
                        return path
                parent = os.path.dirname(path)
                if os.path.basename(parent) == 'tt-metal' and os.path.isdir(parent):
                    return parent

    cwd = os.getcwd()
    current = cwd
    while current != '/':
        if os.path.basename(current) == 'tt-metal':
            return current
        current = os.path.dirname(current)

    return cwd


def build_profile_command(name: str, test_cmd: str) -> str:
    """Build the tracy profile command string."""
    name_arg = f"-n {name}" if name else ""
    tt_metal_path = find_tt_metal_path()

    tracy_tool = os.path.join(tt_metal_path, "tools", "tracy", "profile_this.py")
    if not os.path.exists(tracy_tool):
        print(f"Warning: Tracy tool not found at {tracy_tool}")
        print(f"   Detected tt-metal path: {tt_metal_path}")
        print(f"   Please ensure:")
        print(f"   1. tt-metal is installed correctly, and")
        print(f"   2. Either:")
        print(f"      - Add tt-metal to PYTHONPATH: export PYTHONPATH=/path/to/tt-metal")
        print(f"      - Run from within tt-metal directory: cd /path/to/tt-metal")

    return f"{tracy_tool} {name_arg} -c \"pytest {test_cmd}\""


def extract_config_from_csv(csv_path: str) -> dict:
    """Extract test configuration from the CSV file."""
    config: dict = {}
    try:
        df = pd.read_csv(csv_path)
        if len(df) > 0:
            row = df.iloc[0]

            def parse_dim(dim_str: str) -> str:
                if isinstance(dim_str, str) and '[' in dim_str:
                    return dim_str.split('[')[0]
                return str(dim_str)

            w = parse_dim(row.get('INPUT_0_W_PAD[LOGICAL]', '1'))
            z = parse_dim(row.get('INPUT_0_Z_PAD[LOGICAL]', '1'))
            y = parse_dim(row.get('INPUT_0_Y_PAD[LOGICAL]', '32'))
            x = parse_dim(row.get('INPUT_0_X_PAD[LOGICAL]', '32'))
            config['shape'] = f"{w}, {z}, {y}, {x}"

            output_dtype = row.get('OUTPUT_0_DATATYPE', row.get('INPUT_0_DATATYPE', 'BFLOAT16'))
            config['dtype'] = output_dtype.lower() if isinstance(output_dtype, str) else 'bfloat16'

            output_layout = row.get('OUTPUT_0_LAYOUT', row.get('INPUT_0_LAYOUT', 'TILE'))
            config['layout'] = output_layout.lower() if isinstance(output_layout, str) else 'tile'

            output_memory = row.get('OUTPUT_0_MEMORY', row.get('INPUT_0_MEMORY', 'DEV_1_DRAM_INTERLEAVED'))
            if isinstance(output_memory, str):
                if 'L1' in output_memory.upper():
                    config['memory_config'] = 'l1'
                else:
                    config['memory_config'] = 'dram'
            else:
                config['memory_config'] = 'dram'

    except Exception as e:
        logger.warning("Could not extract config from CSV: %s", e)

    return config


def extract_test_config_and_status(output: str, csv_path: Optional[str] = None) -> dict:
    """Extract test configuration and pass/fail status from output and CSV."""
    result: dict = {
        'config': {},
        'status': 'unknown',
        'test_name': 'unknown'
    }

    test_match = re.search(r'::([^:]+)::test_([a-zA-Z_]+)', output)
    if test_match:
        result['test_name'] = test_match.group(2)
    else:
        test_method_match = re.search(r'test_([a-zA-Z_]+)', output)
        if test_method_match:
            result['test_name'] = test_method_match.group(1)

    if csv_path and os.path.exists(csv_path):
        csv_config = extract_config_from_csv(csv_path)
        if csv_config:
            result['config'] = csv_config

    if not result['config']:
        shape_match = re.search(r'Using.*?configuration.*?Shape:\s*\(([^)]+)\)', output, re.IGNORECASE)
        if shape_match:
            result['config']['shape'] = shape_match.group(1)

        dtype_match = re.search(r'Using.*?configuration.*?Dtype:\s*(bfloat16|float32|int32)', output, re.IGNORECASE)
        if dtype_match:
            result['config']['dtype'] = dtype_match.group(1)

        layout_match = re.search(r'Using.*?configuration.*?Layout:\s*(tile|row_major)', output, re.IGNORECASE)
        if layout_match:
            result['config']['layout'] = layout_match.group(1).lower()

    if result['test_name'].startswith('bitwise_') and not result['config'].get('dtype'):
        result['config']['dtype'] = 'int32'

    if 'PASSED' in output or '1 passed' in output:
        result['status'] = 'PASSED'
    elif 'FAILED' in output or '1 failed' in output:
        result['status'] = 'FAILED'
    elif 'ERROR' in output or 'error' in output:
        result['status'] = 'ERROR'

    return result


def print_test_summary(
    test_info: dict,
    csv_path: str,
    duration: float,
    custom_config: Optional[dict] = None,
    quiet: bool = False
) -> None:
    """Print a comprehensive test summary."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Test: {test_info['test_name']}")
    print(f"Status: {test_info['status']}")

    if custom_config:
        config_str = []
        for key in ('shape', 'dtype', 'layout', 'memory_config'):
            if key in custom_config:
                config_str.append(f"{key}={custom_config[key]}")
        print(f"Configuration: {', '.join(config_str)} (custom)")
    elif test_info['config']:
        config_str = []
        for key in ('shape', 'dtype', 'layout', 'memory_config'):
            if key in test_info['config']:
                config_str.append(f"{key}={test_info['config'][key]}")
        print(f"Configuration: {', '.join(config_str)}")
    else:
        expected_config = get_expected_config_for_operation(test_info['test_name'])
        if expected_config:
            config_str = []
            for key in ('shape', 'dtype', 'layout', 'memory_config'):
                if key in expected_config:
                    config_str.append(f"{key}={expected_config[key]}")
            print(f"Configuration: {', '.join(config_str)} (expected)")
        else:
            print("Configuration: Not detected")

    print(f"CSV Path: {csv_path}")
    print(f"DEVICE KERNEL DURATION [ns] total: {duration:.2f} ns")
    print("=" * 60)


def main() -> None:
    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    name, test_cmd, debug, custom_config, quiet, output_dir = parse_args(sys.argv[1:])
    profile_cmd = build_profile_command(name, test_cmd)

    logger.debug("Profile command: %s", profile_cmd)

    if debug:
        print(f"Running: {profile_cmd}\n")
    else:
        print(f"Running test...")

    process = subprocess.Popen(
        profile_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )

    output_lines: List[str] = []
    try:
        for line in process.stdout:
            if debug:
                print(line, end="")
            output_lines.append(line)
    except KeyboardInterrupt:
        process.terminate()
        print("❌ Aborted.")
        sys.exit(1)

    process.wait()
    full_output = "".join(output_lines)

    try:
        csv_path = extract_csv_path(full_output)
        duration = get_device_kernel_duration(csv_path)
        test_info = extract_test_config_and_status(full_output, csv_path)
        print_test_summary(test_info, csv_path, duration, custom_config, quiet=quiet)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            dest = os.path.join(output_dir, os.path.basename(csv_path))
            shutil.copy2(csv_path, dest)
            print(f"CSV copied to: {dest}")

    except Exception as e:
        logger.debug("Exception during result processing", exc_info=True)
        print(f"\n❌ Error processing results: {e}")
        print("Raw output:")
        print(full_output)
        sys.exit(1)


if __name__ == "__main__":
    main()
