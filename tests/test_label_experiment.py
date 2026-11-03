"""Tests for controlled outcome-rate sensitivity experiments."""

import numpy as np
import pandas as pd

from experiments.label_sensitivity import SCENARIOS, run_experiment


def test_all_label_scenarios_match_known_results(tmp_path):
    path = tmp_path / "label_sensitivity.csv"
    results = run_experiment(path)

    assert len(SCENARIOS) == 22
    assert len(results) == 26
    assert path.is_file()

    for quantity in ("rate_1", "rate_2", "disparity_pp"):
        assert np.allclose(
            results[f"expected_{quantity}"],
            results[f"observed_{quantity}"],
        )

    assert np.allclose(
        results["expected_adjusted_p"],
        results["adjusted_p_value"],
        equal_nan=True,
    )
    assert (results["expected_flag"] == results["flag"]).all()
    pd.testing.assert_frame_equal(results, pd.read_csv(path))


def test_equal_outcome_rates_are_negative_controls(tmp_path):
    results = run_experiment(tmp_path / "label.csv")
    controls = results.loc[
        results["scenario"].isin(
            ["n50_positive10", "n100_positive10", "n200_positive10"]
        )
    ]

    assert len(controls) == 3
    assert (controls["observed_disparity_pp"] == 0).all()
    assert (controls["pairwise_p_value"] == 1).all()
    assert set(controls["flag"]) == {"INFO"}

    gap_10pp = results.loc[
        results["scenario"].isin(
            ["n50_positive20", "n100_positive20", "n200_positive20"]
        )
    ].set_index("n_1")
    assert gap_10pp.loc[50, "flag"] == "INFO"
    assert gap_10pp.loc[200, "flag"] == "WARNING"


def test_warning_and_critical_boundaries(tmp_path):
    results = run_experiment(tmp_path / "label.csv").set_index("scenario")

    assert results.loc["below_warning_gap", "observed_disparity_pp"] == 9
    assert results.loc["below_warning_gap", "flag"] == "WARNING"
    assert results.loc["at_warning_gap", "observed_disparity_pp"] == 10
    assert results.loc["at_warning_gap", "flag"] == "WARNING"
    assert results.loc["at_critical_gap", "observed_disparity_pp"] == 30
    assert results.loc["at_critical_gap", "flag"] == "CRITICAL"


def test_three_groups_and_sparse_exact_tests(tmp_path):
    results = run_experiment(tmp_path / "label.csv")
    three = results.loc[results["scenario"] == "three_groups"]

    assert len(three) == 3
    ab = three.loc[
        (three["group_1"] == "A") & (three["group_2"] == "B")
    ].iloc[0]
    ac = three.loc[
        (three["group_1"] == "A") & (three["group_2"] == "C")
    ].iloc[0]

    assert ab["flag"] == "INFO"
    assert ab["pairwise_p_value"] == 1.0
    assert ac["flag"] == "CRITICAL"
    assert np.isclose(
        ac["adjusted_p_value"], 3 * ac["pairwise_p_value"]
    )

    sparse2 = results.loc[
        results["scenario"] == "sparse_two_groups"
    ].iloc[0]
    assert sparse2["pairwise_test"] == "fisher_exact"
    assert sparse2["flag"] == "CRITICAL"

    sparse3 = results.loc[
        results["scenario"] == "sparse_three_groups"
    ]
    assert set(sparse3["omnibus_test"]) == {"not_tested_sparse"}

    same = sparse3.loc[
        (sparse3["group_1"] == "A") & (sparse3["group_2"] == "B")
    ].iloc[0]
    assert same["pairwise_test"] == "not_testable"
    assert pd.isna(same["pairwise_p_value"])
    assert same["flag"] == "INFO"

    different = sparse3.loc[
        (sparse3["group_1"] == "A") & (sparse3["group_2"] == "C")
    ].iloc[0]
    assert different["pairwise_test"] == "fisher_exact"
    assert different["flag"] == "CRITICAL"


def test_unrounded_disparity_controls_classification(tmp_path):
    results = run_experiment(tmp_path / "label.csv").set_index("scenario")
    row = results.loc["just_below_critical_gap"]

    assert row["observed_disparity_pp"] == 29.9958
    assert row["observed_disparity_pp"] < 30
    assert row["expected_flag"] == "WARNING"
    assert row["flag"] == "WARNING"