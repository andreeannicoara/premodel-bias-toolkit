"""
Reporting utilities for the pre-modelling bias audit toolkit.

This module will later collect audit outputs and write them to CSV, HTML, or
other report formats.
"""


def save_audit_table(table, output_path):
    """
    Save an audit result table to CSV.

    Parameters
    ----------
    table:
        A pandas DataFrame containing audit results.

    output_path:
        Path where the CSV file should be saved.
    """
    table.to_csv(output_path, index=False)