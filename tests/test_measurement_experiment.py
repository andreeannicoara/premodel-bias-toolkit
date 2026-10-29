"""Checks of the reproducible measurement sensitivity experiment."""

import numpy as np
import pandas as pd

from experiments.measurement_sensitivity import (
    make_dataset,
    empirical_ks,
    run_experiment,
)


def test_measurement_sensitivity_matches_expected_statistics(tmp_path):
    path = tmp_path / "measurement_sensitivity.csv"
    result = run_experiment(path)

    assert path.exists()
    assert len(result) == 17

    assert np.allclose(
        result["expected_ks"],
        result["observed_ks"],
        equal_nan=True,
    )

    assert np.allclose(
        result["expected_adjusted_p"],
        result["adjusted_p_value"],
        equal_nan=True,
    )

    assert (result["expected_flag"] == result["flag"]).all()

    pd.testing.assert_frame_equal(result, pd.read_csv(path))


def test_fixed_shift_changes_significance_with_sample_size(tmp_path):
    result = run_experiment(tmp_path / "measure.csv")

    shift = result.loc[
        (result["scenario_type"] == "quantile_shift")
        & (result["shift"] == 0.5)
    ].set_index("n_per_group")

    assert shift.loc[20, "flag"] == "INFO"
    assert shift.loc[100, "flag"] == "WARNING"

    assert np.isclose(shift.loc[20, "observed_ks"], 0.2)
    assert np.isclose(shift.loc[100, "observed_ks"], 0.2)

    controls = result.loc[
        result["scenario"].str.startswith("shift_0_n")
    ]

    assert len(controls) == 4
    assert (controls["observed_ks"] == 0).all()
    assert (controls["p_value"] == 1).all()
    assert set(controls["flag"]) == {"INFO"}


def test_three_group_bonferroni_and_pairwise_null(tmp_path):
    result = run_experiment(tmp_path / "measure.csv")
    rows = result.loc[result["scenario"] == "three_groups"]

    assert len(rows) == 3
    assert set(rows["number_of_comparisons"]) == {3}

    ab = rows.loc[
        (rows["group_1"] == "A")
        & (rows["group_2"] == "B")
    ].iloc[0]

    ac = rows.loc[
        (rows["group_1"] == "A")
        & (rows["group_2"] == "C")
    ].iloc[0]

    assert ab["shift"] == 0.0
    assert ab["observed_ks"] == 0.0
    assert ab["flag"] == "INFO"

    assert ac["shift"] == 1.0
    assert ac["flag"] == "WARNING"

    assert np.isclose(
        ac["adjusted_p_value"],
        min(ac["p_value"] * 3, 1.0),
    )


def test_insufficient_group_and_discrete_values_are_identified(tmp_path):
    result = run_experiment(tmp_path / "measure.csv")

    small = result.loc[
        result["scenario"] == "below_minimum"
    ].iloc[0]

    assert pd.isna(small["observed_ks"])
    assert pd.isna(small["adjusted_p_value"])
    assert small["flag"] == "INFO"

    discrete = result.loc[
        result["scenario"] == "discrete_counts"
    ].iloc[0]

    assert np.isclose(discrete["observed_ks"], 0.1)
    assert "exploratory" in discrete["interpretation_note"]

    sample = make_dataset(20, 0.5)

    a = sample.loc[
        sample["group"] == "A", "measurement"
    ].to_numpy()

    b = sample.loc[
        sample["group"] == "B", "measurement"
    ].to_numpy()

    assert np.isclose(empirical_ks(a, b), 0.2)