"""
Integrated runner for the pre-modelling bias audit toolkit.

This module coordinates the four audit dimensions:
representation, missingness, measurement, and label bias-risk detection.
"""

import pandas as pd

from toolkit.representation import run_representation_audit
from toolkit.missingness import run_missingness_audit
from toolkit.measurement import run_measurement_audit
from toolkit.label import run_label_audit


def run_full_audit(
    df: pd.DataFrame,
    group_cols: list[str],
    measure_cols: list[str],
    label_col: str,
    positive_label,
    output_dir: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run all four pre-modelling bias-risk audits.

    Parameters
    ----------
    df:
        Input dataset.

    group_cols:
        Demographic columns used across the audits.

    measure_cols:
        Numeric variables selected for measurement analysis.

    label_col:
        Binary outcome variable used for label analysis.

    positive_label:
        Value treated as the positive outcome in the label audit.

    output_dir:
        Optional directory where individual audit CSV files are saved.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Results for representation, missingness, measurement, and label audits.
    """
    representation = run_representation_audit(
        df=df,
        group_cols=group_cols,
        output_dir=output_dir,
    )

    missingness = run_missingness_audit(
        df=df,
        group_cols=group_cols,
        output_dir=output_dir,
    )

    measurement = run_measurement_audit(
        df=df,
        group_cols=group_cols,
        measure_cols=measure_cols,
        output_dir=output_dir,
    )

    label = run_label_audit(
        df=df,
        group_cols=group_cols,
        label_col=label_col,
        positive_label=positive_label,
        output_dir=output_dir,
    )

    return {
        "representation": representation,
        "missingness": missingness,
        "measurement": measurement,
        "label": label,
    }