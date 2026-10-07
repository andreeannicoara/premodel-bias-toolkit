import pandas as pd
import numpy as np
import pytest

from toolkit.label import run_label_audit


def test_label_audit_detects_large_outcome_disparity():
    df = pd.DataFrame(
        {
            "race": ["A"] * 50 + ["B"] * 50,
            "readmitted": [1] * 40 + [0] * 10 + [1] * 10 + [0] * 40,
        }
    )

    result = run_label_audit(
        df,
        group_cols=["race"],
        label_col="readmitted",
        positive_label=1,
    )

    assert len(result) == 1

    row = result.iloc[0]

    assert row["group_1_positive_rate"] == 80.0
    assert row["group_2_positive_rate"] == 20.0
    assert row["disparity_pp"] == 60.0
    assert row["p_value"] < 0.05
    assert row["flag"] == "CRITICAL"


def test_label_audit_similar_groups_return_info():
    df = pd.DataFrame(
        {
            "race": ["A"] * 50 + ["B"] * 50,
            "readmitted": (
                [1] * 25
                + [0] * 25
                + [1] * 25
                + [0] * 25
            ),
        }
    )

    result = run_label_audit(
        df,
        group_cols=["race"],
        label_col="readmitted",
        positive_label=1,
    )

    row = result.iloc[0]

    assert row["group_1_positive_rate"] == 50.0
    assert row["group_2_positive_rate"] == 50.0
    assert row["disparity_pp"] == 0.0
    assert row["flag"] == "INFO"


def test_label_audit_handles_missing_labels():
    df = pd.DataFrame(
        {
            "sex": ["F", "F", "F", "M", "M", "M"],
            "outcome": [1, 0, np.nan, 1, 1, 0],
        }
    )

    result = run_label_audit(
        df,
        group_cols=["sex"],
        label_col="outcome",
        positive_label=1,
    )

    row = result.iloc[0]

    assert row["group_1_count"] == 2
    assert row["group_2_count"] == 3


def test_label_audit_rejects_non_binary_label():
    df = pd.DataFrame(
        {
            "race": ["A", "A", "B", "B", "B", "A"],
            "outcome": [0, 1, 2, 0, 1, 2],
        }
    )

    with pytest.raises(
        ValueError,
        match="requires a binary outcome variable",
    ):
        run_label_audit(
            df,
            group_cols=["race"],
            label_col="outcome",
            positive_label=1,
        )


def test_label_audit_rejects_unknown_positive_label():
    df = pd.DataFrame(
        {
            "race": ["A", "A", "B", "B"],
            "outcome": [0, 1, 0, 1],
        }
    )

    with pytest.raises(
        ValueError,
        match="Positive label",
    ):
        run_label_audit(
            df,
            group_cols=["race"],
            label_col="outcome",
            positive_label=2,
        )


def test_label_audit_raises_for_missing_column():
    df = pd.DataFrame(
        {
            "race": ["A", "B", "A", "B"],
            "outcome": [0, 1, 0, 1],
        }
    )

    with pytest.raises(
        ValueError,
        match="columns are missing",
    ):
        run_label_audit(
            df,
            group_cols=["gender"],
            label_col="outcome",
            positive_label=1,
        )


def test_label_audit_rejects_invalid_p_threshold():
    df = pd.DataFrame(
        {
            "race": ["A", "A", "B", "B"],
            "outcome": [0, 1, 0, 1],
        }
    )

    with pytest.raises(
        ValueError,
        match="p_threshold must be between 0 and 1",
    ):
        run_label_audit(
            df,
            group_cols=["race"],
            label_col="outcome",
            positive_label=1,
            p_threshold=1.5,
        )


def test_label_audit_rejects_invalid_threshold_order():
    df = pd.DataFrame(
        {
            "race": ["A", "A", "B", "B"],
            "outcome": [0, 1, 0, 1],
        }
    )

    with pytest.raises(
        ValueError,
        match="critical_threshold",
    ):
        run_label_audit(
            df,
            group_cols=["race"],
            label_col="outcome",
            positive_label=1,
            warning_threshold=30.0,
            critical_threshold=10.0,
        )


def test_label_audit_writes_csv(tmp_path):
    df = pd.DataFrame(
        {
            "race": ["A"] * 20 + ["B"] * 20,
            "outcome": [1] * 10 + [0] * 10 + [1] * 10 + [0] * 10,
        }
    )

    result = run_label_audit(
        df,
        group_cols=["race"],
        label_col="outcome",
        positive_label=1,
        output_dir=str(tmp_path),
    )

    output_file = tmp_path / "label_audit.csv"

    assert output_file.exists()

    saved = pd.read_csv(output_file)

    assert len(saved) == len(result)
    assert list(saved.columns) == list(result.columns)