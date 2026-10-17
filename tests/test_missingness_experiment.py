import pandas as pd

from experiments.missingness_sensitivity import (
    make_dataset,
    run_experiment,
)


def test_missingness_experiment_matches_known_rates(tmp_path):

    output = tmp_path / "missingness_sensitivity.csv"
    results = run_experiment(output)

    assert len(results) == 15
    assert output.exists()

    assert set(results["n_per_group"]) == {50, 100, 200}

    assert (
        results["expected_a_pct"]
        == results["observed_a_pct"]
    ).all()

    assert (
        results["expected_b_pct"]
        == results["observed_b_pct"]
    ).all()

    assert (
        results["expected_disparity_pp"].round(6)
        == results["observed_disparity_pp"].round(6)
    ).all()

    controls = results.loc[
        results["expected_disparity_pp"] == 0
    ]

    assert set(controls["flag"]) == {"INFO"}

    smallest_p = results.loc[
        (results["n_per_group"] == 200)
        & (results["expected_a_pct"] == 50),
        "p_value",
    ].iloc[0]

    assert 0 < smallest_p < 1e-6


def test_synthetic_missingness_is_reproducible():

    first = make_dataset(100, 0.4, 0.1, seed=2026)
    second = make_dataset(100, 0.4, 0.1, seed=2026)

    pd.testing.assert_frame_equal(first, second)