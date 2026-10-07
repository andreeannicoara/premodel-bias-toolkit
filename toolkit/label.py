"""
Module 4: Label Bias Detection

Audits whether an outcome variable differs across demographic groups before
model training. The module identifies label bias-risk indicators, especially
where an outcome may act as an imperfect proxy for the clinical construct of
interest.

Main checks:
    - Outcome rates by demographic group
    - Pairwise outcome-rate disparities
    - Chi-square association between group membership and outcome
    - Bias-risk flags requiring contextual interpretation
"""

from itertools import combinations

import pandas as pd
from scipy import stats


PVALUE_THRESHOLD = 0.05
DISPARITY_WARNING = 10.0
DISPARITY_CRITICAL = 30.0


def _validate_columns(
    df: pd.DataFrame,
    group_cols: list[str],
    label_col: str,
) -> None:
    """
    Check that the requested group and label columns exist.
    """
    requested_cols = group_cols + [label_col]
    missing_cols = [col for col in requested_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"The following columns are missing from the dataframe: {missing_cols}"
        )


def _validate_binary_label(
    df: pd.DataFrame,
    label_col: str,
    positive_label,
) -> None:
    """
    Check that the label contains exactly two non-missing values and that the
    requested positive label exists.
    """
    label_values = df[label_col].dropna().unique()

    if len(label_values) != 2:
        raise ValueError(
            "Label audit currently requires a binary outcome variable."
        )

    if positive_label not in label_values:
        raise ValueError(
            f"Positive label '{positive_label}' was not found in '{label_col}'."
        )


def _group_outcome_rate(
    df: pd.DataFrame,
    group_col: str,
    group_value,
    label_col: str,
    positive_label,
) -> dict:
    """
    Calculate the positive outcome rate for one demographic group.
    """
    group_data = df.loc[
        df[group_col] == group_value,
        label_col,
    ].dropna()

    group_count = len(group_data)

    if group_count == 0:
        positive_count = 0
        positive_rate = 0.0
    else:
        positive_count = int((group_data == positive_label).sum())
        positive_rate = positive_count / group_count * 100

    return {
        "group_count": group_count,
        "positive_count": positive_count,
        "positive_rate": round(positive_rate, 2),
    }


def _chi2_label_association(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
):
    """
    Test whether demographic group membership and label values are associated.

    Rows with missing group or label values are excluded.
    """
    valid_data = df[[group_col, label_col]].dropna()

    contingency = pd.crosstab(
        valid_data[group_col],
        valid_data[label_col],
    )

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return None, None, None

    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)

    return chi2, p_value, dof


def _label_flag(
    p_value: float | None,
    disparity_pp: float,
    p_threshold: float = PVALUE_THRESHOLD,
    warning_threshold: float = DISPARITY_WARNING,
    critical_threshold: float = DISPARITY_CRITICAL,
) -> tuple[str, str]:
    """
    Assign a label bias-risk flag.

    The flag indicates that outcome-rate disparities require investigation.
    It does not establish that the label itself is discriminatory or invalid.
    """
    if p_value is None:
        return (
            "INFO",
            "Insufficient variation for association testing.",
        )

    if p_value < p_threshold and disparity_pp >= critical_threshold:
        return (
            "CRITICAL",
            "Large and statistically significant outcome-rate disparity detected.",
        )

    if p_value < p_threshold and disparity_pp >= warning_threshold:
        return (
            "WARNING",
            "Statistically significant outcome-rate disparity detected.",
        )

    if p_value < p_threshold:
        return (
            "WARNING",
            "Statistical association detected, but the outcome-rate disparity is small.",
        )

    return (
        "INFO",
        "No statistically significant association detected.",
    )


def run_label_audit(
    df: pd.DataFrame,
    group_cols: list[str],
    label_col: str,
    positive_label,
    p_threshold: float = PVALUE_THRESHOLD,
    warning_threshold: float = DISPARITY_WARNING,
    critical_threshold: float = DISPARITY_CRITICAL,
) -> pd.DataFrame:
    """
    Run the label bias-risk audit.

    Parameters
    ----------
    df:
        Input dataset.

    group_cols:
        Demographic columns used for subgroup comparisons.

    label_col:
        Binary outcome variable being audited.

    positive_label:
        Label value treated as the positive outcome.

    p_threshold:
        Significance threshold for the chi-square association test.

    warning_threshold:
        Outcome-rate disparity in percentage points required for WARNING.

    critical_threshold:
        Outcome-rate disparity in percentage points required for CRITICAL.

    Returns
    -------
    pandas.DataFrame
        Pairwise demographic-group comparisons with outcome rates,
        disparities, statistical results, and bias-risk flags.
    """
    if df.empty:
        raise ValueError("The dataframe is empty.")

    _validate_columns(
        df=df,
        group_cols=group_cols,
        label_col=label_col,
    )

    _validate_binary_label(
        df=df,
        label_col=label_col,
        positive_label=positive_label,
    )

    audit_rows = []

    for group_col in group_cols:
        group_values = df[group_col].dropna().unique()

        chi2, p_value, dof = _chi2_label_association(
            df=df,
            group_col=group_col,
            label_col=label_col,
        )

        for group_1, group_2 in combinations(group_values, 2):
            outcome_1 = _group_outcome_rate(
                df=df,
                group_col=group_col,
                group_value=group_1,
                label_col=label_col,
                positive_label=positive_label,
            )

            outcome_2 = _group_outcome_rate(
                df=df,
                group_col=group_col,
                group_value=group_2,
                label_col=label_col,
                positive_label=positive_label,
            )

            disparity_pp = abs(
                outcome_1["positive_rate"]
                - outcome_2["positive_rate"]
            )

            flag, reason = _label_flag(
                p_value=p_value,
                disparity_pp=disparity_pp,
                p_threshold=p_threshold,
                warning_threshold=warning_threshold,
                critical_threshold=critical_threshold,
            )

            audit_rows.append(
                {
                    "label_variable": label_col,
                    "positive_label": positive_label,
                    "group_variable": group_col,
                    "group_1": str(group_1),
                    "group_2": str(group_2),
                    "group_1_count": outcome_1["group_count"],
                    "group_2_count": outcome_2["group_count"],
                    "group_1_positive_count": outcome_1["positive_count"],
                    "group_2_positive_count": outcome_2["positive_count"],
                    "group_1_positive_rate": outcome_1["positive_rate"],
                    "group_2_positive_rate": outcome_2["positive_rate"],
                    "disparity_pp": round(disparity_pp, 2),
                    "chi2": None if chi2 is None else round(chi2, 4),
                    "p_value": None if p_value is None else round(p_value, 6),
                    "degrees_of_freedom": dof,
                    "flag": flag,
                    "reason": reason,
                }
            )

    return pd.DataFrame(audit_rows)