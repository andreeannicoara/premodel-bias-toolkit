"""
Tests for the private helper functions in toolkit/missingness.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from toolkit.missingness import _missingness_indicator, _chi2_independence


# _missingness_indicator 

def test_indicator_marks_nan_as_one():
    s = pd.Series([1.0, np.nan, 3.0, np.nan])
    result = _missingness_indicator(s)
    assert list(result) == [0, 1, 0, 1]

def test_indicator_no_missing_returns_all_zeros():
    s = pd.Series([1, 2, 3])
    result = _missingness_indicator(s)
    assert list(result) == [0, 0, 0]

def test_indicator_all_missing_returns_all_ones():
    s = pd.Series([np.nan, np.nan])
    result = _missingness_indicator(s)
    assert list(result) == [1, 1]


# _chi2_independence

def test_chi2_returns_three_values():
    df = pd.DataFrame({
        "race":   ["A", "A", "B", "B", "A", "B"],
        "weight": [70,  np.nan, 80, np.nan, 65, np.nan],
    })
    chi2, p, dof = _chi2_independence(df, "weight", "race")
    assert chi2 is not None
    assert 0 <= p <= 1
    assert dof > 0

def test_chi2_no_variation_returns_none():
    # All values missing for all groups - contingency table has only one column
    df = pd.DataFrame({
        "race":   ["A", "A", "B", "B"],
        "weight": [np.nan, np.nan, np.nan, np.nan],
    })
    chi2, p, dof = _chi2_independence(df, "weight", "race")
    assert chi2 is None

def test_chi2_excludes_missing_group_rows():
    df = pd.DataFrame({
        "race":   ["A", None, "B", "A"],
        "weight": [1.0, 2.0, np.nan, np.nan],
    })
    # Row with None race should be excluded without raising an error
    chi2, p, dof = _chi2_independence(df, "weight", "race")
    assert chi2 is not None


if __name__ == "__main__":
    passed = 0
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
                passed += 1
            except AssertionError as e:
                print(f"  FAIL  {name}: {e}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")