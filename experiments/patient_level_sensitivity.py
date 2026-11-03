"""Compare encounter-level and one-encounter-per-patient Diabetes audits."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from experiments.diabetes_case_study import (
    DATA_PATH,
    GROUP_COLS,
    LABEL_COL,
    MEASURE_COLS,
    PROJECT_ROOT,
    prepare_diabetes_data,
)
from run_audit import run_full_audit


OUTPUT_DIR = PROJECT_ROOT / "output" / "patient_sensitivity"
AUDIT_NAMES = ("representation", "missingness", "measurement", "label")
UNTESTED = {"not_tested_sparse", "not_testable"}


def select_one_encounter_per_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the smallest encounter_id per patient, without assuming dates."""
    required = {"patient_nbr", "encounter_id"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing identifier columns: {sorted(missing)}")

    if df[list(required)].isna().any().any():
        raise ValueError("Encounter and patient identifiers cannot be missing")

    if df["encounter_id"].duplicated().any():
        raise ValueError("Encounter identifiers must be unique")

    return (
        df.sort_values(["encounter_id", "patient_nbr"], kind="stable")
        .drop_duplicates(subset="patient_nbr", keep="first")
        .reset_index(drop=True)
    )


def summarise_results(
    results: dict[str, pd.DataFrame],
    analysis: str,
    n_rows: int,
) -> pd.DataFrame:
    """Count findings and record unavailable tests for each audit."""
    rows = []

    for name in AUDIT_NAMES:
        table = results[name]

        findings = (
            table.drop_duplicates(["variable", "group_variable"])
            if name == "missingness" and not table.empty
            else table
        )

        flags = (
            findings["flag"].value_counts()
            if "flag" in findings
            else pd.Series(dtype=int)
        )

        unavailable = 0

        if name == "missingness" and "test_method" in findings:
            unavailable = int(
                findings["test_method"].isin(UNTESTED).sum()
            )
        elif name == "label" and "pairwise_test" in findings:
            unavailable = int(
                findings["pairwise_test"].isin(UNTESTED).sum()
            )
        elif name == "measurement" and "p_value" in findings:
            unavailable = int(findings["p_value"].isna().sum())

        rows.append({
            "analysis": analysis,
            "audit": name,
            "n_rows": n_rows,
            "findings": len(findings),
            "critical": int(flags.get("CRITICAL", 0)),
            "warning": int(flags.get("WARNING", 0)),
            "info": int(flags.get("INFO", 0)),
            "tests_unavailable": unavailable,
        })

    return pd.DataFrame(rows)


def subgroup_outcome_rates(
    df: pd.DataFrame,
    analysis: str,
) -> pd.DataFrame:
    """Describe binary outcome rates for observed demographic groups."""
    rows = []

    for col in GROUP_COLS:
        known = df.dropna(subset=[col])

        for group, subset in known.groupby(col):
            rows.append({
                "analysis": analysis,
                "group_variable": col,
                "group_value": str(group),
                "observations": len(subset),
                "positive_count": int(subset[LABEL_COL].sum()),
                "positive_rate_pct": float(
                    subset[LABEL_COL].mean() * 100
                ),
            })

    return pd.DataFrame(rows)


def run_sensitivity(
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, pd.DataFrame]:
    """Run both analyses and export their results and comparison tables."""
    encounters = prepare_diabetes_data()
    patients = select_one_encounter_per_patient(encounters)

    output_dir = Path(output_dir)
    summaries = []
    rates = []

    for name, data in (
        ("encounter", encounters),
        ("patient", patients),
    ):
        results = run_full_audit(
            df=data,
            group_cols=GROUP_COLS,
            measure_cols=MEASURE_COLS,
            label_col=LABEL_COL,
            positive_label=1,
            output_dir=str(output_dir / name),
        )

        summaries.append(summarise_results(results, name, len(data)))
        rates.append(subgroup_outcome_rates(data, name))

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(
        output_dir / "audit_summary_comparison.csv",
        index=False,
    )

    rates_df = pd.concat(rates, ignore_index=True)

    encounter = rates_df.loc[
        rates_df["analysis"] == "encounter"
    ].drop(columns="analysis")

    patient = rates_df.loc[
        rates_df["analysis"] == "patient"
    ].drop(columns="analysis")

    compared = encounter.merge(
        patient,
        on=["group_variable", "group_value"],
        how="outer",
        suffixes=("_encounter", "_patient"),
        validate="one_to_one",
    )

    compared["positive_rate_change_pp"] = (
        compared["positive_rate_pct_patient"]
        - compared["positive_rate_pct_encounter"]
    )

    compared.to_csv(
        output_dir / "subgroup_outcome_comparison.csv",
        index=False,
    )

    config = {
        "source": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "source_sha256": hashlib.sha256(
            DATA_PATH.read_bytes()
        ).hexdigest(),
        "selection_rule": (
            "Lowest encounter_id per patient_nbr; "
            "chronological order not verified"
        ),
        "encounters": len(encounters),
        "patients": len(patients),
        "encounters_removed": len(encounters) - len(patients),
        "group_cols": GROUP_COLS,
        "measure_cols": MEASURE_COLS,
        "positive_label": "readmitted == '<30'",
        "notes": [
            "The subset changes both sample size and patient composition.",
            "Encounter-level and patient-level results are not independent experiments.",
            "Flag differences do not establish bias or causal effects.",
        ],
    }

    (output_dir / "sensitivity_configuration.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    return {
        "summary": summary,
        "subgroup_outcomes": compared,
    }


if __name__ == "__main__":
    results = run_sensitivity()

    print(results["summary"].to_string(index=False))
    print(f"\nResults saved to {OUTPUT_DIR}")