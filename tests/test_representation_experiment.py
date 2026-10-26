"""Tests for controlled representation sensitivity experiments."""

import pandas as pd

from experiments.representation_sensitivity import (
    SCENARIOS,
    make_dataset,
    run_experiment,
)
from toolkit.representation import run_representation_audit


def test_representation_scenarios_match_predefined_expectations(tmp_path):
    output = tmp_path / "representation_sensitivity.csv"
    results = run_experiment(output)

    assert len(results) == len(SCENARIOS) == 12
    assert output.is_file()

    assert (results["expected_count"] == results["observed_count"]).all()
    assert (
        results["expected_percentage"] == results["observed_percentage"]
    ).all()
    assert (results["expected_flag"] == results["observed_flag"]).all()
    assert set(results["other_group_flag"]) == {"INFO"}

    pd.testing.assert_frame_equal(results, pd.read_csv(output))


def test_representation_threshold_equality_is_not_flagged():
    df = make_dataset(total_rows=600, group_a_count=30)
    audit = run_representation_audit(df, ["group"])

    row = audit.loc[audit["group_value"] == "A"].iloc[0]

    assert row["count"] == 30
    assert row["percentage"] == 5.0
    assert row["flag"] == "INFO"


def test_missing_demographic_values_are_reported_separately():
    df = make_dataset(total_rows=1000, group_a_count=50)
    df.loc[:9, "group"] = pd.NA

    audit = run_representation_audit(df, ["group"])

    missing = audit.loc[
        audit["group_value"] == "MISSING"
    ].iloc[0]

    assert missing["count"] == 10
    assert missing["percentage"] == 1.0
    assert missing["flag"] == "CRITICAL"
    assert audit["count"].sum() == 1000


def test_percentage_flag_uses_unrounded_value():
    df = make_dataset(total_rows=20001, group_a_count=1000)
    audit = run_representation_audit(df, ["group"])

    row = audit.loc[audit["group_value"] == "A"].iloc[0]

    assert 4.999 < row["percentage"] < 5.0
    assert row["flag"] == "WARNING"
