"""
Module 3: Measurement Bias Detection

Audits whether numeric variables show different distributions across
demographic groups before model training. Distributional differences are
treated as measurement bias-risk indicators that require contextual
interpretation, not as proof of measurement bias.

Main checks:
    - Group-wise numeric summaries
    - Pairwise distribution comparisons across demographic groups
    - Two-sample Kolmogorov-Smirnov tests
    - Bonferroni correction for multiple group comparisons
    - Minimum group-size checks
"""

import os
from itertools import combinations

import pandas as pd
from scipy import stats

from toolkit.utils import ensure_output_dir


PVALUE_THRESHOLD = 0.05
MIN_GROUP_SIZE = 20


def _validate_columns(
    df: pd.DataFrame,
    group_cols: list[str],
    measure_cols: list[str],
) -> None:
    """
    Check that requested group and measurement columns exist and that
    measurement columns are numeric.
    """
    requested_cols = group_cols + measure_cols
    missing_cols = [col for col in requested_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"The following columns are missing from the dataframe: {missing_cols}"
        )

    non_numeric_cols = [
        col
        for col in measure_cols
        if not pd.api.types.is_numeric_dtype(df[col])
    ]

    if non_numeric_cols:
        raise TypeError(
            "Measurement columns must be numeric. "
            f"Non-numeric columns: {non_numeric_cols}"
        )


def _group_summary(
    df: pd.DataFrame,
    measure_col: str,
    group_col: str,
    group_value,
) -> dict:
    """
    Calculate descriptive statistics for one measurement within one group.
    """
    values = df.loc[
        df[group_col] == group_value,
        measure_col,
    ].dropna()

    return {
        "n": len(values),
        "mean": values.mean() if not values.empty else None,
        "median": values.median() if not values.empty else None,
        "std": values.std() if len(values) > 1 else None,
    }


def _measurement_flag(
    adjusted_p_value: float | None,
    p_threshold: float = PVALUE_THRESHOLD,
) -> tuple[str, str]:
    """
    Assign a measurement bias-risk flag.

    A statistically significant distributional difference is flagged for
    contextual review. It is not treated as proof of measurement bias.
    """
    if adjusted_p_value is None:
        return (
            "INFO",
            "Insufficient data for distribution comparison.",
        )

    if adjusted_p_value < p_threshold:
        return (
            "WARNING",
            "Group distributions differ significantly; contextual review is required.",
        )

    return (
        "INFO",
        "No statistically significant distribution difference detected.",
    )


def run_measurement_audit(
    df: pd.DataFrame,
    group_cols: list[str],
    measure_cols: list[str],
    p_threshold: float = PVALUE_THRESHOLD,
    min_group_size: int = MIN_GROUP_SIZE,
    output_dir: str | None = None,
) -> pd.DataFrame:
    """
    Run the measurement bias-risk audit.

    Parameters
    ----------
    df:
        Input dataset.

    group_cols:
        Demographic columns used for subgroup comparisons.

    measure_cols:
        Numeric variables selected for measurement analysis.

    p_threshold:
        Significance threshold applied after Bonferroni correction.

    min_group_size:
        Minimum number of non-missing observations required in each group.

    output_dir:
        Optional directory where the audit table should be saved as CSV.

    Returns
    -------
    pandas.DataFrame
        Pairwise group comparisons containing descriptive statistics,
        Kolmogorov-Smirnov results, and bias-risk flags.
    """
    if df.empty:
        raise ValueError("The dataframe is empty.")

    if not 0 < p_threshold < 1:
        raise ValueError("p_threshold must be between 0 and 1.")

    if min_group_size < 1:
        raise ValueError("min_group_size must be at least 1.")

    _validate_columns(
        df=df,
        group_cols=group_cols,
        measure_cols=measure_cols,
    )

    audit_rows = []

    for measure_col in measure_cols:
        for group_col in group_cols:
            if measure_col == group_col:
                continue

            group_values = df[group_col].dropna().unique()
            group_pairs = list(combinations(group_values, 2))

            number_of_comparisons = len(group_pairs)

            for group_1, group_2 in group_pairs:
                summary_1 = _group_summary(
                    df,
                    measure_col=measure_col,
                    group_col=group_col,
                    group_value=group_1,
                )

                summary_2 = _group_summary(
                    df,
                    measure_col=measure_col,
                    group_col=group_col,
                    group_value=group_2,
                )

                values_1 = df.loc[
                    df[group_col] == group_1,
                    measure_col,
                ].dropna()

                values_2 = df.loc[
                    df[group_col] == group_2,
                    measure_col,
                ].dropna()

                if (
                    len(values_1) < min_group_size
                    or len(values_2) < min_group_size
                ):
                    ks_statistic = None
                    p_value = None
                    adjusted_p_value = None

                else:
                    ks_result = stats.ks_2samp(
                        values_1,
                        values_2,
                        alternative="two-sided",
                        method="auto",
                    )

                    ks_statistic = float(ks_result.statistic)
                    p_value = float(ks_result.pvalue)

                    adjusted_p_value = min(
                        p_value * number_of_comparisons,
                        1.0,
                    )

                flag, reason = _measurement_flag(
                    adjusted_p_value=adjusted_p_value,
                    p_threshold=p_threshold,
                )

                audit_rows.append(
                    {
                        "variable": measure_col,
                        "group_variable": group_col,
                        "group_1": str(group_1),
                        "group_2": str(group_2),
                        "n_1": summary_1["n"],
                        "n_2": summary_2["n"],
                        "mean_1": summary_1["mean"],
                        "mean_2": summary_2["mean"],
                        "median_1": summary_1["median"],
                        "median_2": summary_2["median"],
                        "std_1": summary_1["std"],
                        "std_2": summary_2["std"],
                        "ks_statistic": ks_statistic,
                        "p_value": p_value,
                        "adjusted_p_value": adjusted_p_value,
                        "flag": flag,
                        "reason": reason,
                    }
                )

    result = pd.DataFrame(
        audit_rows,
        columns=[
            "variable",
            "group_variable",
            "group_1",
            "group_2",
            "n_1",
            "n_2",
            "mean_1",
            "mean_2",
            "median_1",
            "median_2",
            "std_1",
            "std_2",
            "ks_statistic",
            "p_value",
            "adjusted_p_value",
            "flag",
            "reason",
        ],
    )

    numeric_result_cols = [
        "mean_1",
        "mean_2",
        "median_1",
        "median_2",
        "std_1",
        "std_2",
        "ks_statistic",
        "p_value",
        "adjusted_p_value",
    ]

    if not result.empty:
        result[numeric_result_cols] = result[numeric_result_cols].round(6)

    if output_dir is not None:
        ensure_output_dir(output_dir)
        output_path = os.path.join(
            output_dir,
            "measurement_audit.csv",
        )
        result.to_csv(output_path, index=False)

    return result