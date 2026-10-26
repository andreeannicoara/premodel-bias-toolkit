"""
Shared utility functions used across all toolkit modules.
"""

import os

import numpy as np

from scipy import stats


def get_flag(p_value, disparity_pp, p_threshold=0.05,
             warning_threshold=10.0, critical_threshold=30.0):
    """
    Map a p-value and disparity to a severity flag. Returns 'CRITICAL', 'WARNING', or 'INFO'.
    """
    if p_value < p_threshold and disparity_pp >= critical_threshold:
        return "CRITICAL"
    elif p_value < p_threshold and disparity_pp >= warning_threshold:
        return "WARNING"
    elif p_value < p_threshold:
        return "WARNING"
    return "INFO"


def ensure_output_dir(output_dir):
    """Create output directory if it does not already exist."""
    os.makedirs(output_dir, exist_ok=True)


def contingency_association(contingency):
    """Choose an independence test based on expected cell frequencies.

    Returns (chi2, p_value, degrees_of_freedom, test_method).
    Sparse 2x2 tables use Fisher's exact test.
    Sparse larger tables are left untested.
    """
    contingency = np.asarray(contingency)

    if min(contingency.shape) < 2:
        return None, None, None, "not_testable"

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    if (expected < 5).any():
        if contingency.shape == (2, 2):
            _, p_value = stats.fisher_exact(
                contingency,
                alternative="two-sided",
            )
            return None, float(p_value), None, "fisher_exact"

        return None, None, None, "not_tested_sparse"

    return float(chi2), float(p_value), int(dof), "chi_square"
