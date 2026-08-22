"""
Module 2: Missingness Bias Detection

Detects whether missing data follows non-random patterns across demographic
groups (Missing Not At Random — MNAR), which can introduce systematic bias
into any downstream model trained on the dataset.

Planned tests:
    - Group-conditional missingness rate per variable per demographic group
    - Chi-square test of independence: missingness indicator vs. group
    - Missingness disparity flag: variables where rate varies >10pp across groups
    - Missingness heatmap and disparity bar chart

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
TOP_N_VARIABLES           = 10     # variables to show in visualisations


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