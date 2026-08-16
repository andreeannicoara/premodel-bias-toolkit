"""
Shared utility functions used across all toolkit modules.
"""

import os


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