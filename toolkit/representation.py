"""
Module 1: Representation Bias Detection

This module audits whether demographic groups are sufficiently visible in a
dataset before model training. It does not prove unfairness. It flags
representation bias-risk indicators that may require closer interpretation.

Main checks:
    - Count and percentage of each subgroup
    - Small subgroup flags based on configurable thresholds
    - Optional CSV output for documentation and reporting
"""

import os
import pandas as pd

from toolkit.utils import ensure_output_dir


DEFAULT_MIN_COUNT = 30
DEFAULT_MIN_PERCENTAGE = 5.0


def _validate_group_columns(df: pd.DataFrame, group_cols: list[str]) -> None:
    """
    Check that all requested demographic columns exist in the dataframe.

    Parameters
    ----------
    df:
        Input dataset.

    group_cols:
        Demographic columns to audit.

    Raises
    ------
    ValueError:
        If one or more requested group columns are missing.
    """
    missing_cols = [col for col in group_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"The following group columns are missing from the dataframe: {missing_cols}"
        )


def _flag_representation(
    count: int,
    percentage: float,
    min_count: int = DEFAULT_MIN_COUNT,
    min_percentage: float = DEFAULT_MIN_PERCENTAGE,
) -> str:
    """
    Assign a representation bias-risk flag.

    A subgroup is flagged when it has too few rows or represents too small a
    share of the dataset. This is a bias-risk indicator, not proof of bias.
    """
    if count < min_count and percentage < min_percentage:
        return "CRITICAL"

    if count < min_count or percentage < min_percentage:
        return "WARNING"

    return "INFO"


def _summarise_group_column(
    df: pd.DataFrame,
    group_col: str,
    min_count: int = DEFAULT_MIN_COUNT,
    min_percentage: float = DEFAULT_MIN_PERCENTAGE,
) -> pd.DataFrame:
    """
    Summarise subgroup representation for one demographic column.

    Missing values in the demographic column are retained as 'MISSING' so that
    the audit does not silently remove incomplete demographic data.
    """
    total_rows = len(df)

    if total_rows == 0:
        raise ValueError("The dataframe is empty.")

    group_values = df[group_col].fillna("MISSING").astype(str)

    counts = (
        group_values
        .value_counts(dropna=False)
        .reset_index()
    )

    counts.columns = ["group_value", "count"]
    counts["group_variable"] = group_col
    counts["percentage"] = (counts["count"] / total_rows * 100).round(2)

    counts["flag"] = counts.apply(
        lambda row: _flag_representation(
            count=int(row["count"]),
            percentage=float(row["percentage"]),
            min_count=min_count,
            min_percentage=min_percentage,
        ),
        axis=1,
    )

    counts["reason"] = counts.apply(
        lambda row: _representation_reason(
            count=int(row["count"]),
            percentage=float(row["percentage"]),
            min_count=min_count,
            min_percentage=min_percentage,
        ),
        axis=1,
    )

    return counts[
        [
            "group_variable",
            "group_value",
            "count",
            "percentage",
            "flag",
            "reason",
        ]
    ]


def _representation_reason(
    count: int,
    percentage: float,
    min_count: int = DEFAULT_MIN_COUNT,
    min_percentage: float = DEFAULT_MIN_PERCENTAGE,
) -> str:
    """
    Provide a short explanation for the assigned representation flag.
    """
    if count < min_count and percentage < min_percentage:
        return (
            f"Subgroup has fewer than {min_count} rows and represents less than "
            f"{min_percentage}% of the dataset."
        )

    if count < min_count:
        return f"Subgroup has fewer than {min_count} rows."

    if percentage < min_percentage:
        return f"Subgroup represents less than {min_percentage}% of the dataset."

    return "Subgroup meets the configured representation thresholds."


def run_representation_audit(
    df: pd.DataFrame,
    group_cols: list[str],
    min_count: int = DEFAULT_MIN_COUNT,
    min_percentage: float = DEFAULT_MIN_PERCENTAGE,
    output_dir: str | None = None,
) -> pd.DataFrame:
    """
    Run the representation bias-risk audit.

    Parameters
    ----------
    df:
        Input dataset.

    group_cols:
        Demographic columns to audit, for example ['race', 'gender', 'age'].

    min_count:
        Minimum row count expected for a subgroup before it is flagged.

    min_percentage:
        Minimum dataset percentage expected for a subgroup before it is flagged.

    output_dir:
        Optional directory where the audit table should be saved as CSV.

    Returns
    -------
    pandas.DataFrame
        Representation audit table.
    """
    _validate_group_columns(df, group_cols)

    audit_tables = [
        _summarise_group_column(
            df=df,
            group_col=group_col,
            min_count=min_count,
            min_percentage=min_percentage,
        )
        for group_col in group_cols
    ]

    result = pd.concat(audit_tables, ignore_index=True)

    if output_dir is not None:
        ensure_output_dir(output_dir)
        output_path = os.path.join(output_dir, "representation_audit.csv")
        result.to_csv(output_path, index=False)

    return result