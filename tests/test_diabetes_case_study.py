from experiments.diabetes_case_study import prepare_diabetes_data


def test_diabetes_preprocessing_matches_source_data():
    df = prepare_diabetes_data()

    assert len(df) == 101_766
    assert df["patient_nbr"].nunique() == 71_518
    assert df["race"].isna().sum() == 2_273
    assert df["gender"].isna().sum() == 3
    assert df["readmitted_30d"].sum() == 11_357
    assert set(df["readmitted_30d"].unique()) == {0, 1}
