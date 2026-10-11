
import numpy as np
import pandas as pd

from run_audit import run_full_audit


def _run(df, output_dir=None):
    return run_full_audit(
        df=df,
        group_cols=["group"],
        measure_cols=["measurement"],
        label_col="outcome",
        positive_label=1,
        output_dir=output_dir,
    )


def _injected_dataset():
    return pd.DataFrame({
        "group": ["A"] * 50 + ["B"] * 50,
        "measurement": list(range(50)) + list(range(100, 150)),
        "missing_var": [np.nan] * 20 + [1.0] * 80,
        "outcome": [1] * 40 + [0] * 10 + [1] * 10 + [0] * 40,
    })


def test_runner_returns_all_four_results_without_changing_data():
    df = _injected_dataset()
    original = df.copy(deep=True)

    results = _run(df)

    assert set(results) == {
        "representation", "missingness", "measurement", "label"
    }
    assert all(
        isinstance(table, pd.DataFrame)
        for table in results.values()
    )
    pd.testing.assert_frame_equal(df, original)


def test_runner_detects_expected_injected_patterns():
    results = _run(_injected_dataset())

    representation = results["representation"]
    assert dict(zip(
        representation["group_value"],
        representation["percentage"],
    )) == {"A": 50.0, "B": 50.0}
    assert set(representation["flag"]) == {"INFO"}

    missingness = results["missingness"]
    missing_var = missingness.loc[
        missingness["variable"] == "missing_var"
    ]
    assert dict(zip(
        missing_var["group_value"],
        missing_var["missing_percentage"],
    )) == {"A": 40.0, "B": 0.0}
    assert set(missing_var["disparity_pp"]) == {40.0}
    assert set(missing_var["flag"]) == {"CRITICAL"}

    measurement = results["measurement"].iloc[0]
    assert measurement["ks_statistic"] == 1.0
    assert measurement["flag"] == "WARNING"

    label = results["label"].iloc[0]
    assert label["group_1_positive_rate"] == 80.0
    assert label["group_2_positive_rate"] == 20.0
    assert label["disparity_pp"] == 60.0
    assert label["flag"] == "CRITICAL"


def test_runner_negative_control_remains_unflagged():
    df = pd.DataFrame({
        "group": ["A"] * 50 + ["B"] * 50,
        "measurement": list(range(50)) * 2,
        "missing_var": ([np.nan] * 10 + [1.0] * 40) * 2,
        "outcome": ([1] * 25 + [0] * 25) * 2,
    })

    results = _run(df)

    assert set(results["representation"]["flag"]) == {"INFO"}
    assert set(results["missingness"]["flag"]) == {"INFO"}
    assert results["measurement"].iloc[0]["ks_statistic"] == 0.0
    assert results["measurement"].iloc[0]["flag"] == "INFO"
    assert results["label"].iloc[0]["disparity_pp"] == 0.0
    assert results["label"].iloc[0]["flag"] == "INFO"


def test_runner_writes_four_audit_tables(tmp_path):
    results = _run(
        _injected_dataset(),
        output_dir=str(tmp_path),
    )

    for name, result in results.items():
        csv_path = tmp_path / f"{name}_audit.csv"

        assert csv_path.is_file(), f"Missing CSV: {csv_path.name}"

        saved = pd.read_csv(csv_path)

        assert len(saved) == len(result)
        assert list(saved.columns) == list(result.columns)
