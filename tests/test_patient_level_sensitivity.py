"""Tests for patient-level selection and audit result summaries."""

import pandas as pd
import pytest

from experiments.diabetes_case_study import prepare_diabetes_data
from experiments.patient_level_sensitivity import (
    select_one_encounter_per_patient,
    summarise_results,
)


def test_patient_selection_uses_smallest_encounter_id():
    data = pd.DataFrame({
        "encounter_id": [30, 20, 40, 10],
        "patient_nbr": [1, 2, 2, 1],
    })

    selected = select_one_encounter_per_patient(data)

    assert selected["encounter_id"].tolist() == [10, 20]
    assert selected["patient_nbr"].tolist() == [1, 2]
    assert data["encounter_id"].tolist() == [30, 20, 40, 10]


def test_selection_rejects_ambiguous_identifiers():
    data = pd.DataFrame({
        "encounter_id": [10, 10],
        "patient_nbr": [1, 2],
    })

    with pytest.raises(ValueError, match="unique"):
        select_one_encounter_per_patient(data)


def test_diabetes_selection_matches_unique_patient_count():
    df = prepare_diabetes_data()
    selected = select_one_encounter_per_patient(df)

    assert len(df) == 101_766
    assert len(selected) == 71_518
    assert selected["patient_nbr"].is_unique
    assert selected["readmitted_30d"].sum() == 6_293


def test_summary_counts_missingness_test_only_once():
    missing = pd.DataFrame({
        "variable": ["lab", "lab"],
        "group_variable": ["race", "race"],
        "group_value": ["A", "B"],
        "flag": ["INFO", "INFO"],
        "test_method": ["not_tested_sparse", "not_tested_sparse"],
    })

    results = {
        "representation": pd.DataFrame({"flag": ["WARNING"]}),
        "missingness": missing,
        "measurement": pd.DataFrame({
            "flag": ["INFO"],
            "p_value": [None],
        }),
        "label": pd.DataFrame({
            "flag": ["CRITICAL"],
            "pairwise_test": ["chi_square"],
        }),
    }

    summary = summarise_results(
        results, "patient", 10
    ).set_index("audit")

    assert summary.loc["missingness", "findings"] == 1
    assert summary.loc["missingness", "info"] == 1
    assert summary.loc["missingness", "tests_unavailable"] == 1
    assert summary.loc["measurement", "tests_unavailable"] == 1
    assert summary.loc["label", "critical"] == 1