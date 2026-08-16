"""
Tests for toolkit/utils.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from toolkit.utils import get_flag, ensure_output_dir


def test_get_flag_critical():
    assert get_flag(p_value=0.001, disparity_pp=35.0) == "CRITICAL"

def test_get_flag_warning_high_disparity():
    assert get_flag(p_value=0.01, disparity_pp=15.0) == "WARNING"

def test_get_flag_warning_low_disparity():
    assert get_flag(p_value=0.03, disparity_pp=5.0) == "WARNING"

def test_get_flag_info_not_significant():
    assert get_flag(p_value=0.20, disparity_pp=40.0) == "INFO"

def test_ensure_output_dir_creates_folder(tmp_path):
    new_dir = str(tmp_path / "test_output")
    ensure_output_dir(new_dir)
    assert os.path.isdir(new_dir)


if __name__ == "__main__":
    passed = 0
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                if name == "test_ensure_output_dir_creates_folder":
                    import tempfile, pathlib
                    with tempfile.TemporaryDirectory() as tmp:
                        fn(pathlib.Path(tmp))
                else:
                    fn()
                print(f"  PASS  {name}")
                passed += 1
            except AssertionError as e:
                print(f"  FAIL  {name}: {e}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")