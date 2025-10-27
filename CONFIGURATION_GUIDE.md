# ttperf Configuration Guide

## TT-Metal Path Resolution

ttperf needs to locate your TT-Metal installation to access profiling tools. It uses a simple two-step path resolution system.

### Automatic Path Detection

ttperf searches for tt-metal in the following order:

1. **PYTHONPATH environment variable**: If specified, uses path from PYTHONPATH
2. **Current working directory**: Walks up the directory tree to find tt-metal root

### Configuration Methods

#### Method 1: Using PYTHONPATH (Recommended for explicit path)

Set PYTHONPATH to point to your tt-metal installation:

```bash
# Temporary (current session only)
export PYTHONPATH=/path/to/your/tt-metal
ttperf add

# Permanent (add to ~/.bashrc or ~/.bash_profile)
echo 'export PYTHONPATH=/path/to/your/tt-metal' >> ~/.bashrc
source ~/.bashrc
ttperf relu
```

#### Method 2: Running from tt-metal Directory (Recommended for convenience)

Navigate to your tt-metal directory (or any subdirectory) and run ttperf from there:

```bash
# From tt-metal root
cd /path/to/your/tt-metal
ttperf add

# Or from any subdirectory
cd /path/to/your/tt-metal/tests
ttperf relu

cd /path/to/your/tt-metal/ttnn/ttnn/operations
ttperf matmul
```

ttperf automatically walks up the directory tree to find the tt-metal root.

### Verification

To verify ttperf can find your tt-metal installation:

```bash
# This will show where ttperf found tt-metal
ttperf --debug add 2>&1 | grep "tt-metal"
```

If tt-metal is not found, you'll see a helpful error message:

```
⚠️  Warning: Tracy tool not found at /path/to/detected/tools/tracy/profile_this.py
   Detected tt-metal path: /detected/path
   Please ensure:
   1. tt-metal is installed correctly, and
   2. Either:
      - Add tt-metal to PYTHONPATH: export PYTHONPATH=/path/to/tt-metal
      - Run from within tt-metal directory: cd /path/to/tt-metal
```

### Troubleshooting

**Problem**: ttperf can't find tt-metal

**Solutions**:
1. Check if tt-metal exists at the expected location
2. Option A: Set `PYTHONPATH` environment variable: `export PYTHONPATH=/path/to/tt-metal`
3. Option B: Run ttperf from within tt-metal directory: `cd /path/to/tt-metal && ttperf add`
4. Verify tracy tools exist: `ls /path/to/tt-metal/tools/tracy/profile_this.py`

**Problem**: Different tt-metal location for different projects

**Solution**: Use project-specific PYTHONPATH:

```bash
# Create a .env file in your project
echo "PYTHONPATH=/path/to/project/tt-metal" > .env

# Load it before running ttperf
source .env
ttperf add
```

### Best Practices

1. **For Development**: Run from within tt-metal directory (most convenient)
2. **For Multiple Users**: Each user should set `PYTHONPATH` in their `~/.bashrc`
3. **For CI/CD**: Set `PYTHONPATH` in your CI configuration
4. **For Multiple tt-metal versions**: Use different PYTHONPATH for each version

### Examples

```bash
# Example 1: Quick development (no setup needed)
cd ~/tt-metal
ttperf add --shape 1,1,128,128 --dtype bfloat16

# Example 2: Working in a subdirectory
cd ~/tt-metal/tests/ttnn/unit_tests
ttperf relu --l1

# Example 3: Using explicit PYTHONPATH
export PYTHONPATH=/opt/tenstorrent/tt-metal
ttperf matmul --dram

# Example 4: CI/CD pipeline
PYTHONPATH=/workspace/tt-metal ttperf sigmoid --shape 2,2,32,32

# Example 5: Multiple tt-metal versions
export PYTHONPATH=/home/user/tt-metal-dev
ttperf add   # Uses dev version

export PYTHONPATH=/home/user/tt-metal-prod
ttperf add   # Uses prod version
```

### Path Resolution Summary

| Method | Priority | How it Works | Example |
|--------|----------|--------------|---------|
| `PYTHONPATH` | 1 (highest) | Searches for tt-metal in PYTHONPATH | `export PYTHONPATH=/opt/tt-metal` |
| Current Directory | 2 (fallback) | Walks up directory tree to find tt-metal root | `cd /path/to/tt-metal` |

### Support

For issues or questions:
- GitHub Issues: https://github.com/Aswintechie/ttperf/issues
- Documentation: https://github.com/Aswintechie/ttperf

