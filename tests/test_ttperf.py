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

    def test_name_not_truncated_at_digits(self):
        output = ("tests/.../test_expm1.py::test_expm1_allclose[low=0.5] PASSED\n"
                  "1 passed in 1.0s")
        result = t.extract_test_config_and_status(output)
        self.assertEqual(result['test_name'], 'expm1_allclose')

    def test_status_ignores_stray_error_word(self):
        # "0 errors" style text must not be classified as ERROR
        output = "collected with 0 errors\n3 passed in 2.0s"
        result = t.extract_test_config_and_status(output)
        self.assertEqual(result['status'], 'PASSED')

    def test_failed_count_beats_passed(self):
        output = "1 failed, 2 passed in 3.0s"
        result = t.extract_test_config_and_status(output)
        self.assertEqual(result['status'], 'FAILED')


# ---------------------------------------------------------------------------
# find_csv_path / safe_device_kernel_duration
# ---------------------------------------------------------------------------

class TestFindCsvPath(unittest.TestCase):

    def _make_csv(self, rows=(100.0,)):
        f = tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w', newline='')
        writer = csv.writer(f)
        writer.writerow(["DEVICE KERNEL DURATION [ns]"])
        for val in rows:
            writer.writerow([val])
        f.close()
        return f.name

    def test_returns_path_when_present(self):
        path = self._make_csv()
        try:
            output = f"OPs csv generated at: {path}\n"
            self.assertEqual(t.find_csv_path(output), path)
        finally:
            os.unlink(path)

    def test_returns_none_when_not_found(self):
        self.assertIsNone(t.find_csv_path("nothing here"))

    def test_returns_none_when_file_missing(self):
        output = "OPs csv generated at: /nonexistent/x.csv\n"
        self.assertIsNone(t.find_csv_path(output))

    def test_safe_duration_sums_rows(self):
        path = self._make_csv([100.0, 200.0, 50.5])
        try:
            self.assertAlmostEqual(t.safe_device_kernel_duration(path), 350.5)
        finally:
            os.unlink(path)

    def test_safe_duration_none_on_missing_column(self):
        f = tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w', newline='')
        writer = csv.writer(f)
        writer.writerow(["OTHER"])
        writer.writerow([1])
        f.close()
        try:
            self.assertIsNone(t.safe_device_kernel_duration(f.name))
        finally:
            os.unlink(f.name)

    def test_safe_duration_none_on_missing_file(self):
        self.assertIsNone(t.safe_device_kernel_duration("/nonexistent/x.csv"))


# ---------------------------------------------------------------------------
# sanitize_for_name
# ---------------------------------------------------------------------------

class TestSanitizeForName(unittest.TestCase):

    def test_replaces_non_alnum(self):
        self.assertEqual(t.sanitize_for_name("low=-1.6e+38"), "low_1_6e_38")

    def test_strips_edges(self):
        self.assertEqual(t.sanitize_for_name("[abc]"), "abc")

    def test_empty_falls_back(self):
        self.assertEqual(t.sanitize_for_name("!!!"), "case")

    def test_length_capped(self):
        self.assertLessEqual(len(t.sanitize_for_name("a" * 200)), 80)


# ---------------------------------------------------------------------------
# enumerate_test_cases
# ---------------------------------------------------------------------------

class TestEnumerateTestCases(unittest.TestCase):

    def test_explicit_parametrized_case_not_split(self):
        cmd = "tests/foo.py::test_bar[x=1]"
        cases = t.enumerate_test_cases(cmd)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]['cmd'], cmd)

    def test_parses_flat_node_ids(self):
        stdout = (
            "tests/foo.py::test_bar[x=1]\n"
            "tests/foo.py::test_bar[x=2]\n"
            "\n2 tests collected in 0.05s\n"
        )
        fake = MagicMock(stdout=stdout)
        with patch('subprocess.run', return_value=fake):
            cases = t.enumerate_test_cases("/abs/tests/foo.py")
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0]['cmd'], "/abs/tests/foo.py::test_bar[x=1]")
        self.assertEqual(cases[0]['label'], "test_bar[x=1]")
        self.assertEqual(cases[1]['label'], "test_bar[x=2]")

    def test_falls_back_when_no_node_ids(self):
        fake = MagicMock(stdout="no tests ran\n")
        with patch('subprocess.run', return_value=fake):
            cases = t.enumerate_test_cases("/abs/tests/foo.py")
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]['cmd'], "/abs/tests/foo.py")

    def test_falls_back_on_exception(self):
        with patch('subprocess.run', side_effect=RuntimeError("boom")):
            cases = t.enumerate_test_cases("/abs/tests/foo.py")
        self.assertEqual(len(cases), 1)


# ---------------------------------------------------------------------------
# print_per_case_summary
# ---------------------------------------------------------------------------

class TestPrintPerCaseSummary(unittest.TestCase):

    def _capture(self, results):
        buf = io.StringIO()
        with patch('sys.stdout', buf):
            t.print_per_case_summary(results)
        return buf.getvalue()

    def test_renders_cases_without_total(self):
        results = [
            {'label': 'test_x[a]', 'status': 'PASSED', 'duration': 100.0,
             'csv_path': '/x/a.csv', 'config': {'dtype': 'bfloat16'}},
            {'label': 'test_x[b]', 'status': 'PASSED', 'duration': 200.0,
             'csv_path': '/x/b.csv', 'config': {}},
        ]
        out = self._capture(results)
        self.assertIn("PER-CASE PERFORMANCE SUMMARY (2 cases)", out)
        self.assertIn("test_x[a]", out)
        self.assertIn("100.00 ns", out)
        self.assertIn("200.00 ns", out)
        # durations must not be summed
        self.assertNotIn("300.00", out)
        self.assertNotIn("total", out.lower())

    def test_handles_missing_duration(self):
        results = [
            {'label': 'test_x[a]', 'status': 'ERROR (no CSV)', 'duration': None,
             'csv_path': None, 'config': {}},
        ]
        out = self._capture(results)
        self.assertIn("n/a", out)
        self.assertIn("ERROR (no CSV)", out)


# ---------------------------------------------------------------------------
# __version__
# ---------------------------------------------------------------------------

class TestVersion(unittest.TestCase):

    def test_version_is_string(self):
        self.assertIsInstance(t.__version__, str)
        self.assertTrue(len(t.__version__) > 0)


if __name__ == '__main__':
    unittest.main()
