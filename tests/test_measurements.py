import numpy as np
import pandas as pd
import pytest

from toolkit.measurement import run_measurement_audit


def test_measurement_audit_detects_distribution_difference():
    df = pd.DataFrame(
        {
            "race": ["A"] * 30 + ["B"] * 30,
            "measurement": list(range(30)) + list(range(100, 130)),
        }
    )

    result = run_measurement_audit(
        df,
        group_cols=["race"],
        measure_cols=["measurement"],
        min_group_size=20,
    )

    assert len(result) == 1

    row = result.iloc[0]

    assert row["ks_statistic"] == 1.0
    assert row["adjusted_p_value"] < 0.05
    assert row["flag"] == "WARNING"


def test_measurement_audit_identical_groups_return_info():
    values = list(range(30))

    df = pd.DataFrame(
        {
            "race": ["A"] * 30 + ["B"] * 30,
            "measurement": values + values,
        }
    )

    result = run_measurement_audit(
        df,
        group_cols=["race"],
        measure_cols=["measurement"],
        min_group_size=20,
    )

    row = result.iloc[0]

    assert row["ks_statistic"] == 0.0
    assert row["adjusted_p_value"] == 1.0
    assert row["flag"] == "INFO"


def test_measurement_audit_reports_insufficient_group_size():
    df = pd.DataFrame(
        {
            "race": ["A"] * 10 + ["B"] * 30,
            "measurement": list(range(10)) + list(range(30)),
        }
    )

    result = run_measurement_audit(
        df,
        group_cols=["race"],
        measure_cols=["measurement"],
        min_group_size=20,
    )

    row = result.iloc[0]

    assert pd.isna(row["ks_statistic"])
    assert pd.isna(row["p_value"])
    assert row["flag"] == "INFO"
    assert "Insufficient data" in row["reason"]


def test_measurement_audit_rejects_non_numeric_measurement():
    df = pd.DataFrame(
        {
            "race": ["A", "A", "B", "B"],
            "measurement": ["low", "high", "low", "high"],
        }
    )

    with pytest.raises(TypeError, match="must be numeric"):
        run_measurement_audit(
            df,
            group_cols=["race"],
            measure_cols=["measurement"],
        )


def test_measurement_audit_raises_for_missing_column():
    df = pd.DataFrame(
        {
            "race": ["A", "B"],
            "measurement": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="columns are missing"):
        run_measurement_audit(
            df,
            group_cols=["race"],
            measure_cols=["missing_measurement"],
        )


def test_measurement_audit_writes_csv(tmp_path):
    values = list(range(30))

    df = pd.DataFrame(
        {
            "race": ["A"] * 30 + ["B"] * 30,
            "measurement": values + values,
        }
    )

    run_measurement_audit(
        df,
        group_cols=["race"],
        measure_cols=["measurement"],
        output_dir=str(tmp_path),
    )

    output_file = tmp_path / "measurement_audit.csv"

    assert output_file.exists()