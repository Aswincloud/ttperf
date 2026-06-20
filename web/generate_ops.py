#!/usr/bin/env python3
"""Generate public/ops.json for the ttperf landing page.

Reads the same source of truth as the CLI — the bundled test file
(for the full operation list) and operation_configs.json (for per-op
config) — so the website can never drift from the tool.

Run from the repo root:
    python3 web/generate_ops.py
"""

import ast
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "ttperf", "data")
TEST_FILE = os.path.join(DATA_DIR, "test_eltwise_operations.py")
CONFIGS_FILE = os.path.join(DATA_DIR, "operation_configs.json")
OUT_FILE = os.path.join(REPO_ROOT, "public", "ops.json")


def extract_operations(test_file: str) -> list:
    """Pull operation names from test_<name> methods of TestEltwiseOperations.

    Mirrors ttperf.extract_test_methods_from_file so the site lists exactly
    the operations the CLI exposes.
    """
    with open(test_file, "r") as f:
        tree = ast.parse(f.read())

    test_class = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.ClassDef) and n.name == "TestEltwiseOperations"),
        None,
    )
    if test_class is None:
        raise SystemExit("TestEltwiseOperations class not found")

    # set() dedupes the two duplicate test method names (bitwise_left_shift,
    # bitwise_right_shift) exactly as the CLI's dict-based mapping does, so the
    # site reports the same 263 operations as `ttperf --list-ops`.
    return sorted({
        node.name[len("test_"):]
        for node in test_class.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    })


# Category rules copied verbatim from ttperf.print_supported_operations so the
# grid groups operations the same way `ttperf --list-ops` does.
TERNARY = {"where", "mac", "addcdiv", "addcmul", "lerp"}
REDUCTION = {"max", "min", "mean", "sum", "prod", "var", "std", "cumsum", "cumprod"}
COMPLEX = {"complex_tensor", "real", "imag", "angle", "conj", "polar", "complex_recip"}
BINARY = {
    "add", "subtract", "multiply", "divide", "gt", "lt", "eq", "ne", "ge", "le",
    "logical_and", "logical_or", "logical_xor", "atan2", "hypot", "logaddexp",
    "logaddexp2", "maximum", "minimum", "pow", "fmod", "remainder",
    "squared_difference", "bitwise_and", "bitwise_or", "bitwise_xor",
    "mul", "sub", "rpow", "rdiv", "ldexp", "xlogy", "nextafter", "bias_gelu",
    "addalpha", "subalpha", "isclose",
}


def categorize(op: str) -> str:
    if op.endswith("_bw"):
        return "Backward"
    if op in TERNARY:
        return "Ternary"
    if op in REDUCTION:
        return "Reduction"
    if op in COMPLEX:
        return "Complex"
    if op in BINARY or op.endswith("_"):
        return "Binary"
    return "Unary"


def op_config(name: str, configs: dict) -> dict:
    """Merge defaults with the operation's overrides, matching the CLI."""
    merged = dict(configs["defaults"])
    if name.startswith("bitwise_"):
        merged["dtype"] = "int32"
    merged.update(configs["operations"].get(name, {}))
    return merged


def main() -> None:
    operations = extract_operations(TEST_FILE)
    with open(CONFIGS_FILE, "r") as f:
        configs = json.load(f)

    ops = []
    for name in operations:
        cfg = op_config(name, configs)
        shape = cfg.get("shape", [1, 1, 32, 32])
        ops.append({
            "name": name,
            "category": categorize(name),
            "dtype": cfg.get("dtype", "bfloat16"),
            "layout": cfg.get("layout", "tile"),
            "shape": "x".join(str(d) for d in shape),
        })

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    payload = {"count": len(ops), "operations": ops}
    with open(OUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(ops)} operations to {os.path.relpath(OUT_FILE, REPO_ROOT)}")


if __name__ == "__main__":
    main()
