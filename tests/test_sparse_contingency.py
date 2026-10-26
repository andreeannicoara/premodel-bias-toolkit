"""Tests for sparse contingency-table handling."""

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from toolkit.label import run_label_audit
from toolkit.missingness import run_missingness_audit


def test_label_uses_fisher_for_sparse_two_group_table():
    df = pd.DataFrame({
        "group": ["A"] * 5 + ["B"] * 5,
        "outcome": [1] * 5 + [0] * 5,
    })

    row = run_label_audit(df, ["group"], "outcome", 1).iloc[0]
    expected = fisher_exact([[5, 0], [0, 5]])[1]

    assert row["omnibus_test"] == "fisher_exact"
    assert row["pairwise_test"] == "fisher_exact"
    assert pd.isna(row["chi2"])
    assert np.isclose(row["p_value"], expected)
    assert np.isclose(row["pairwise_adjusted_p_value"], expected)
    assert row["flag"] == "CRITICAL"


def test_missingness_uses_fisher_for_sparse_two_group_table():
    df = pd.DataFrame({
        "group": ["A"] * 5 + ["B"] * 5,
        "lab_value": [np.nan] * 5 + [1.0] * 5,
    })

    results = run_missingness_audit(df, ["group"])
    rows = results[results["variable"] == "lab_value"]

    assert set(rows["test_method"]) == {"fisher_exact"}
    assert np.isclose(
        rows["p_value"].iloc[0],
        fisher_exact([[5, 0], [0, 5]])[1],
    )
    assert set(rows["flag"]) == {"CRITICAL"}


def test_sparse_three_group_omnibus_is_not_tested():
    df = pd.DataFrame({
        "group": ["A"] * 6 + ["B"] * 6 + ["C"] * 6,
        "outcome": [1] * 12 + [0] * 6,
    })

    results = run_label_audit(df, ["group"], "outcome", 1)

    assert set(results["omnibus_test"]) == {"not_tested_sparse"}
    assert results["p_value"].isna().all()

    ac = results.loc[
        (results["group_1"] == "A")
        & (results["group_2"] == "C")
    ].iloc[0]

    assert ac["pairwise_test"] == "fisher_exact"
    assert ac["pairwise_adjusted_p_value"] < 0.05
    assert ac["flag"] == "CRITICAL"


def test_missingness_sparse_three_group_is_not_tested():
    df = pd.DataFrame({
        "group": ["A"] * 6 + ["B"] * 6 + ["C"] * 6,
        "lab": [np.nan] * 12 + [1.0] * 6,
    })

    results = run_missingness_audit(df, ["group"])
    rows = results[results["variable"] == "lab"]

    assert set(rows["test_method"]) == {"not_tested_sparse"}
    assert rows["p_value"].isna().all()
    assert set(rows["flag"]) == {"INFO"}


def test_large_tables_still_use_chi_square():
    df = pd.DataFrame({
        "group": ["A"] * 50 + ["B"] * 50,
        "outcome": [1] * 40 + [0] * 10
                   + [1] * 10 + [0] * 40,
        "lab": [np.nan] * 40 + [1.0] * 10
               + [np.nan] * 10 + [1.0] * 40,
    })

    label = run_label_audit(df, ["group"], "outcome", 1).iloc[0]

    assert label["omnibus_test"] == "chi_square"
    assert label["pairwise_test"] == "chi_square"
    assert label["chi2"] > 0

    missingness = run_missingness_audit(df, ["group"])

    assert set(missingness["test_method"]) == {"chi_square"}
