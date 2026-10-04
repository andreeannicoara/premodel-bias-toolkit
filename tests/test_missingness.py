import pandas as pd
import numpy as np
import pytest

from toolkit.missingness import run_missingness_audit


def test_missingness_audit_detects_group_disparity():
    df = pd.DataFrame(
        {
            "race": ["A"] * 20 + ["B"] * 20,
            "weight": [np.nan] * 15 + [70.0] * 5 + [70.0] * 20,
        }
    )

    result = run_missingness_audit(
        df,
        group_cols=["race"],
        min_missing_rate=0.01,
    )

    weight_rows = result[result["variable"] == "weight"]

    assert not weight_rows.empty
    assert weight_rows["disparity_pp"].iloc[0] == 75.0
    assert set(weight_rows["group_value"]) == {"A", "B"}


def test_missingness_audit_ignores_complete_variables():
    df = pd.DataFrame(
        {
            "race": ["A", "A", "B", "B"],
            "age": [20, 30, 40, 50],
        }
    )

    result = run_missingness_audit(
        df,
        group_cols=["race"],
        min_missing_rate=0.01,
    )

    assert result.empty


def test_missingness_audit_handles_explicit_missing_values():
    df = pd.DataFrame(
        {
            "sex": ["F", "F", "M", "M"],
            "race": ["Caucasian", "Unknown", "Unknown", "Asian"],
        }
    )

    result = run_missingness_audit(
        df,
        group_cols=["sex"],
        missing_values=["Unknown"],
        min_missing_rate=0.01,
    )

    race_rows = result[result["variable"] == "race"]

    assert not race_rows.empty
    assert race_rows["overall_missing_percentage"].iloc[0] == 50.0


def test_missingness_audit_raises_for_missing_group_column():
    df = pd.DataFrame(
        {
            "race": ["A", "B"],
            "weight": [70.0, np.nan],
        }
    )

    with pytest.raises(ValueError, match="group columns are missing"):
        run_missingness_audit(
            df,
            group_cols=["gender"],
        )


def test_missingness_audit_writes_csv(tmp_path):
    df = pd.DataFrame(
        {
            "race": ["A", "A", "B", "B"],
            "weight": [70.0, np.nan, 80.0, 75.0],
        }
    )

    run_missingness_audit(
        df,
        group_cols=["race"],
        min_missing_rate=0.01,
        output_dir=str(tmp_path),
    )

    output_file = tmp_path / "missingness_audit.csv"

    assert output_file.exists()