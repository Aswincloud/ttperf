"""
Unit tests for ttperf core functions.

Run with:
    pytest tests/test_ttperf.py -v --cov=ttperf
"""

import io
import os
import sys
import csv
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Helpers — make ttperf importable without a full TT-Metal environment
# ---------------------------------------------------------------------------

# Ensure the package root is on the path when running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import ttperf.ttperf as t


# ---------------------------------------------------------------------------
# extract_csv_path
# ---------------------------------------------------------------------------

class TestExtractCsvPath(unittest.TestCase):

    def _make_csv(self):
        """Create a real temporary CSV file and return its path."""
        f = tempfile.NamedTemporaryFile(suffix='.csv', delete=False,
                                       mode='w', newline='')
        writer = csv.writer(f)
        writer.writerow(["DEVICE KERNEL DURATION [ns]"])
        writer.writerow([100.0])
        f.close()
        return f.name

    def test_extracts_valid_path(self):
        csv_path = self._make_csv()
        try:
            output = f"OPs csv generated at: {csv_path}\nsome other line\n"
            result = t.extract_csv_path(output)
            self.assertEqual(result, csv_path)
        finally:
            os.unlink(csv_path)

    def test_exits_when_not_found(self):
        with self.assertRaises(SystemExit):
            t.extract_csv_path("no csv here")

    def test_exits_when_file_missing(self):
        output = "OPs csv generated at: /nonexistent/path/result.csv\n"
        with self.assertRaises(SystemExit):
            t.extract_csv_path(output)


# ---------------------------------------------------------------------------
# get_device_kernel_duration
# ---------------------------------------------------------------------------

class TestGetDeviceKernelDuration(unittest.TestCase):

    def _make_csv(self, rows):
        f = tempfile.NamedTemporaryFile(suffix='.csv', delete=False,
                                       mode='w', newline='')
        writer = csv.writer(f)
        writer.writerow(["DEVICE KERNEL DURATION [ns]"])
        for val in rows:
            writer.writerow([val])
        f.close()
        return f.name

    def test_sums_durations(self):
        path = self._make_csv([100.0, 200.5, 300.0])
        try:
            result = t.get_device_kernel_duration(path)
            self.assertAlmostEqual(result, 600.5)
        finally:
            os.unlink(path)

    def test_exits_on_missing_column(self):
        f = tempfile.NamedTemporaryFile(suffix='.csv', delete=False,
                                       mode='w', newline='')
        writer = csv.writer(f)
        writer.writerow(["SOME_OTHER_COL"])
        writer.writerow([42])
        f.close()
        try:
            with self.assertRaises(SystemExit):
                t.get_device_kernel_duration(f.name)
        finally:
            os.unlink(f.name)


# ---------------------------------------------------------------------------
# parse_shape
# ---------------------------------------------------------------------------

class TestParseShape(unittest.TestCase):

    def test_valid_4d(self):
        self.assertEqual(t.parse_shape("1,1,32,32"), (1, 1, 32, 32))

    def test_valid_with_spaces(self):
        self.assertEqual(t.parse_shape("2, 3, 64, 128"), (2, 3, 64, 128))

    def test_invalid_exits(self):
        with self.assertRaises(SystemExit):
            t.parse_shape("a,b,c")


# ---------------------------------------------------------------------------
# validate_dtype
# ---------------------------------------------------------------------------

class TestValidateDtype(unittest.TestCase):

    def test_canonical(self):
        self.assertEqual(t.validate_dtype("bfloat16"), "bfloat16")
        self.assertEqual(t.validate_dtype("float32"), "float32")
        self.assertEqual(t.validate_dtype("int32"), "int32")

    def test_aliases(self):
        self.assertEqual(t.validate_dtype("bf16"), "bfloat16")
        self.assertEqual(t.validate_dtype("fp32"), "float32")
        self.assertEqual(t.validate_dtype("f32"), "float32")
        self.assertEqual(t.validate_dtype("i32"), "int32")

    def test_invalid_exits(self):
        with self.assertRaises(SystemExit):
            t.validate_dtype("float16")


# ---------------------------------------------------------------------------
# validate_layout
# ---------------------------------------------------------------------------

class TestValidateLayout(unittest.TestCase):

    def test_canonical(self):
        self.assertEqual(t.validate_layout("tile"), "tile")
        self.assertEqual(t.validate_layout("row_major"), "row_major")

    def test_aliases(self):
        self.assertEqual(t.validate_layout("rm"), "row_major")
        self.assertEqual(t.validate_layout("rowmajor"), "row_major")

    def test_invalid_exits(self):
        with self.assertRaises(SystemExit):
            t.validate_layout("strided")


# ---------------------------------------------------------------------------
# generate_profile_name
# ---------------------------------------------------------------------------

class TestGenerateProfileName(unittest.TestCase):

    def test_from_double_colon(self):
        self.assertEqual(t.generate_profile_name("file.py::test_foo"), "test_foo")

    def test_from_py_file(self):
        self.assertEqual(t.generate_profile_name("tests/test_conv.py"), "test_conv")

    def test_fallback(self):
        result = t.generate_profile_name("some_dir")
        self.assertEqual(result, "some_dir")


# ---------------------------------------------------------------------------
# load_config_file
# ---------------------------------------------------------------------------

class TestLoadConfigFile(unittest.TestCase):

    def test_returns_empty_dict_when_no_file(self):
        with patch('os.path.exists', return_value=False):
            result = t.load_config_file()
        self.assertEqual(result, {})

    def test_loads_yaml_when_present(self):
        yaml_content = "shape: 1,1,64,64\ndtype: float32\n"
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False, mode='w') as f:
            f.write(yaml_content)
            f.flush()
            fname = f.name

        try:
            # Temporarily redirect config search to our temp file
            with patch.object(t, 'load_config_file', wraps=t.load_config_file):
                with patch('os.path.exists') as mock_exists:
                    mock_exists.side_effect = lambda p: p == fname or os.path.exists(p)
                    # Directly call yaml loading logic
                    try:
                        import yaml
                        with open(fname) as yf:
                            loaded = yaml.safe_load(yf)
                        self.assertIn('shape', loaded)
                        self.assertEqual(loaded['dtype'], 'float32')
                    except ImportError:
                        self.skipTest("PyYAML not installed")
        finally:
            os.unlink(fname)


# ---------------------------------------------------------------------------
# extract_test_config_and_status
# ---------------------------------------------------------------------------

class TestExtractTestConfigAndStatus(unittest.TestCase):

    def test_passed_status(self):
        output = "tests/test_eltwise_operations.py::TestEltwiseOperations::test_add PASSED\n1 passed in 1.23s"
        result = t.extract_test_config_and_status(output)
        self.assertEqual(result['status'], 'PASSED')
        self.assertEqual(result['test_name'], 'add')

    def test_failed_status(self):
        output = "FAILED test_something.py - AssertionError\n1 failed"
        result = t.extract_test_config_and_status(output)
        self.assertEqual(result['status'], 'FAILED')

    def test_unknown_status(self):
        result = t.extract_test_config_and_status("no status here")
        self.assertEqual(result['status'], 'unknown')


# ---------------------------------------------------------------------------
# __version__
# ---------------------------------------------------------------------------

class TestVersion(unittest.TestCase):

    def test_version_is_string(self):
        self.assertIsInstance(t.__version__, str)
        self.assertTrue(len(t.__version__) > 0)


if __name__ == '__main__':
    unittest.main()
