import pandas as pd
import pytest

from toolkit.representation import (
    _flag_representation,
    _summarise_group_column,
    run_representation_audit,
)


def test_flag_representation_returns_info_for_sufficient_group():
    flag = _flag_representation(
        count=100,
        percentage=25.0,
        min_count=30,
        min_percentage=5.0,
    )

    assert flag == "INFO"


def test_flag_representation_returns_warning_for_low_count_only():
    flag = _flag_representation(
        count=20,
        percentage=10.0,
        min_count=30,
        min_percentage=5.0,
    )

    assert flag == "WARNING"


def test_flag_representation_returns_critical_for_low_count_and_low_percentage():
    flag = _flag_representation(
        count=5,
        percentage=1.0,
        min_count=30,
        min_percentage=5.0,
    )

    assert flag == "CRITICAL"


def test_summarise_group_column_returns_counts_and_percentages():
    df = pd.DataFrame(
        {
            "race": [
                "Caucasian",
                "Caucasian",
                "African American",
                "Unknown",
            ]
        }
    )

    result = _summarise_group_column(
        df,
        group_col="race",
        min_count=1,
        min_percentage=1.0,
    )

    assert set(result["group_value"]) == {
        "Caucasian",
        "African American",
        "Unknown",
    }

    caucasian_row = result[result["group_value"] == "Caucasian"].iloc[0]

    assert caucasian_row["count"] == 2
    assert caucasian_row["percentage"] == 50.0


def test_summarise_group_column_keeps_missing_values_as_missing():
    df = pd.DataFrame(
        {
            "race": [
                "Caucasian",
                None,
                "African American",
                None,
            ]
        }
    )

    result = _summarise_group_column(
        df,
        group_col="race",
        min_count=1,
        min_percentage=1.0,
    )

    assert "MISSING" in set(result["group_value"])


def test_run_representation_audit_multiple_columns():
    df = pd.DataFrame(
        {
            "race": ["A", "A", "B", "B"],
            "gender": ["F", "M", "F", "F"],
        }
    )

    result = run_representation_audit(
        df,
        group_cols=["race", "gender"],
        min_count=1,
        min_percentage=1.0,
    )

    assert set(result["group_variable"]) == {"race", "gender"}


def test_run_representation_audit_raises_for_missing_column():
    df = pd.DataFrame(
        {
            "race": ["A", "B", "C"],
        }
    )

    with pytest.raises(ValueError, match="group columns are missing"):
        run_representation_audit(
            df,
            group_cols=["race", "gender"],
        )


def test_run_representation_audit_writes_csv(tmp_path):
    df = pd.DataFrame(
        {
            "race": ["A", "A", "B", "C"],
        }
    )

    run_representation_audit(
        df,
        group_cols=["race"],
        min_count=1,
        min_percentage=1.0,
        output_dir=str(tmp_path),
    )

    output_file = tmp_path / "representation_audit.csv"

    assert output_file.exists()