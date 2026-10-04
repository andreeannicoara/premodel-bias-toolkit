"""
Module 2: Missingness Bias Detection

Audits whether missing data patterns differ across demographic groups before
model training. The module identifies missingness bias-risk indicators rather
than determining whether data are MCAR, MAR, or MNAR.

Main checks:
    - Overall missingness rate per variable
    - Group-conditional missingness rates
    - Chi-square test of independence: missingness indicator vs. group
    - Missingness disparity across demographic groups
    - Configurable handling of explicit missing-value markers

Reference:
    Rubin, D.B. (1976). Inference and missing data.
    Biometrika, 63(3), pp. 581-592. DOI: 10.1093/biomet/63.3.581

    Shi, J. et al. (2025). Implicit bias in ICU electronic health record data.
    BMC Medical Informatics and Decision Making, 25, Article 260.
    DOI: 10.1186/s12911-025-03058-9
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

from toolkit.utils import get_flag, ensure_output_dir


# Thresholds
MISSINGNESS_MIN_THRESHOLD = 0.01   # only analyse variables with >1% overall missing
PVALUE_THRESHOLD          = 0.05   # chi-square significance level
DISPARITY_WARNING         = 10.0   # pp difference across groups: WARNING
DISPARITY_CRITICAL        = 30.0   # pp difference across groups: CRITICAL


def _missingness_indicator(series: pd.Series) -> pd.Series:
    """
    Convert a column to a binary missingness indicator.
    Returns 1 where value is missing and 0 where present.
    """
    return series.isna().astype(int)


def _chi2_independence(df: pd.DataFrame, var_col: str, group_col: str):
    """
    Chi-square test of independence between the missingness indicator
    for var_col and the demographic group in group_col.
    Rows where group_col is itself missing are excluded.

    Returns:
    (chi2_stat, p_value, degrees_of_freedom) or (None, None, None)
    if the contingency table cannot be computed.
    """
    miss_indicator = _missingness_indicator(df[var_col])
    mask = df[group_col].notna()
    contingency = pd.crosstab(df.loc[mask, group_col], miss_indicator[mask])

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return None, None, None

    chi2, p, dof, _ = stats.chi2_contingency(contingency)
    return chi2, p, dof


def _validate_group_columns(df: pd.DataFrame, group_cols: list[str]) -> None:
    """
    Check that all requested demographic columns exist in the dataframe.
    """
    missing_cols = [col for col in group_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"The following group columns are missing from the dataframe: {missing_cols}"
        )


def _group_missingness_rates(
    df: pd.DataFrame,
    var_col: str,
    group_col: str,
) -> pd.DataFrame:
    """
    Calculate the missingness rate of one variable for each demographic group.

    Rows where the demographic group itself is missing are excluded.
    """
    valid_groups = df[group_col].dropna().unique()

    rows = []

    for group_value in valid_groups:
        group_data = df.loc[df[group_col] == group_value, var_col]

        group_count = len(group_data)
        missing_count = int(group_data.isna().sum())
        missing_percentage = (
            missing_count / group_count * 100
            if group_count > 0
            else 0.0
        )

        rows.append(
            {
                "group_value": str(group_value),
                "group_count": group_count,
                "missing_count": missing_count,
                "missing_percentage": round(missing_percentage, 2),
            }
        )

    return pd.DataFrame(rows)


def run_missingness_audit(
    df: pd.DataFrame,
    group_cols: list[str],
    min_missing_rate: float = MISSINGNESS_MIN_THRESHOLD,
    missing_values: list[str] | None = None,
    output_dir: str | None = None,
) -> pd.DataFrame:
    """
    Run the missingness bias-risk audit.

    Parameters
    ----------
    df:
        Input dataset.

    group_cols:
        Demographic columns used for subgroup comparisons.

    min_missing_rate:
        Minimum overall missingness rate required for a variable to be audited.
        Expressed as a proportion between 0 and 1.

    missing_values:
        Optional explicit values that should be treated as missing, for example
        ['Unknown', '?'].

    output_dir:
        Optional directory where the audit table should be saved as CSV.

    Returns
    -------
    pandas.DataFrame
        Missingness audit table containing subgroup missingness rates,
        disparities, statistical test results, and bias-risk flags.
    """
    if df.empty:
        raise ValueError("The dataframe is empty.")

    if not 0 <= min_missing_rate <= 1:
        raise ValueError("min_missing_rate must be between 0 and 1.")

    _validate_group_columns(df, group_cols)

    audit_df = df.copy()

    if missing_values:
        audit_df = audit_df.replace(missing_values, np.nan)

    audit_rows = []

    for var_col in audit_df.columns:
        overall_missing_rate = audit_df[var_col].isna().mean()

        if overall_missing_rate < min_missing_rate:
            continue

        overall_missing_percentage = round(
            overall_missing_rate * 100,
            2,
        )

        for group_col in group_cols:
            if var_col == group_col:
                continue

            group_rates = _group_missingness_rates(
                audit_df,
                var_col=var_col,
                group_col=group_col,
            )

            if group_rates.empty:
                continue

            disparity_pp = (
                group_rates["missing_percentage"].max()
                - group_rates["missing_percentage"].min()
            )

            chi2, p_value, dof = _chi2_independence(
                audit_df,
                var_col=var_col,
                group_col=group_col,
            )

            if p_value is None:
                flag = "INFO"
            else:
                flag = get_flag(
                    p_value=p_value,
                    disparity_pp=disparity_pp,
                    p_threshold=PVALUE_THRESHOLD,
                    warning_threshold=DISPARITY_WARNING,
                    critical_threshold=DISPARITY_CRITICAL,
                )

            for _, group_row in group_rates.iterrows():
                audit_rows.append(
                    {
                        "variable": var_col,
                        "group_variable": group_col,
                        "group_value": group_row["group_value"],
                        "group_count": int(group_row["group_count"]),
                        "missing_count": int(group_row["missing_count"]),
                        "missing_percentage": group_row["missing_percentage"],
                        "overall_missing_percentage": overall_missing_percentage,
                        "disparity_pp": round(disparity_pp, 2),
                        "chi2": None if chi2 is None else round(chi2, 4),
                        "p_value": None if p_value is None else round(p_value, 6),
                        "degrees_of_freedom": dof,
                        "flag": flag,
                    }
                )

    result = pd.DataFrame(
        audit_rows,
        columns=[
            "variable",
            "group_variable",
            "group_value",
            "group_count",
            "missing_count",
            "missing_percentage",
            "overall_missing_percentage",
            "disparity_pp",
            "chi2",
            "p_value",
            "degrees_of_freedom",
            "flag",
        ],
    )

    if output_dir is not None:
        ensure_output_dir(output_dir)
        output_path = os.path.join(output_dir, "missingness_audit.csv")
        result.to_csv(output_path, index=False)

    return result