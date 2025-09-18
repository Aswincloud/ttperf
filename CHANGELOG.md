# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.6] - 2025-01-14

### Added
- **Memory Configuration Support**: New command-line options for tensor memory configuration
- `--memory-config CONFIG`: General option with choices `[dram, l1]`
- `--dram`: Shortcut flag for DRAM memory (default)
- `--l1`: Shortcut flag for L1 memory
- Memory configuration extraction from CSV profiler output
- Memory config display in test result summaries

### Changed
- **Default tensor shape reduced from `[1, 1, 1024, 1024]` to `[1, 1, 32, 32]` for better performance**
- Enhanced `create_test_tensor()` function to accept memory_config parameter
- Updated all `ttnn.from_torch()` calls to use memory_config parameter
- Improved CSV extraction to read memory configuration from profiler output
- Enhanced debug output to show memory configuration

### Technical
- Added `validate_memory_config()` function with alias support
- Extended environment variable system with `TTPERF_CUSTOM_MEMORY_CONFIG`
- Updated operation_configs.json to include memory_config field
- Enhanced test file configuration parsing to handle memory settings
- Improved result reporting to include memory configuration details

## [0.1.4] - 2025-01-14

### Changed
- **Major Improvement**: Configuration extraction now reads from CSV profiler output instead of parsing text with regex
- Replaced 50+ complex regex patterns with structured CSV data parsing
- Enhanced `extract_test_config_and_status()` function to prioritize CSV data over text parsing
- Added new `extract_config_from_csv()` function for reliable configuration extraction

### Fixed
- More accurate shape, dtype, and layout detection from profiler results
- Improved reliability of configuration reporting in test summaries
- Better handling of tensor dimension parsing (e.g., "32[32]" format)

### Technical
- CSV-based extraction provides structured, consistent data vs. unreliable text parsing
- Maintains backward compatibility with text parsing as fallback
- Cleaner, more maintainable codebase with reduced complexity

## [0.1.0] - 2025-07-14

### Added
- Initial release of ttperf CLI tool
- Support for profiling TT-Metal tests with pytest
- Automatic CSV path extraction from profiler output
- Device kernel duration calculation
- Real-time output streaming
- Flexible command-line argument parsing
- Support for named profiles
- Comprehensive error handling

### Features
- Simple CLI interface: `ttperf [name] [pytest] <test_path>`
- Automatic detection of test files and paths
- Integration with TT-Metal profiler tools
- CSV parsing for performance metrics
- Real-time progress monitoring

### Dependencies
- pandas for CSV processing
- Python 3.7+ support
- TT-Metal development environment

## [Unreleased]

### Planned
- Enhanced error messages

## [0.1.3] - 2025-07-14

### Added
- Auto-generation of profile names from test paths when no name is specified
- Profile name is derived from test method name (for `::test_name`) or file name (for `.py` files)
- Visual feedback showing auto-generated profile names

### Changed
- Updated help examples to show auto-generated profile names

## [0.1.2] - 2025-07-14

### Added
- Added `--version` and `-v` flags to display version information
- Added `--help` and `-h` flags to display usage help
- Improved help message with examples and better formatting

### Changed
- Enhanced error messages to show help when no test path is provided

## [0.1.1] - 2025-07-14

### Changed
- Updated minimum Python version requirement from 3.7 to 3.8 for Ubuntu 24.04 compatibility
- Removed Python 3.7 from GitHub Actions test matrix
- Updated documentation to reflect Python 3.8+ requirement

### Fixed
- Resolved GitHub Actions failures on Ubuntu 24.04 due to Python 3.7 unavailability
- Support for multiple output formats
- Performance comparison tools
- Configuration file support
- Additional performance metrics
- Batch processing capabilities 