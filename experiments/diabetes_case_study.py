"""Reproducible encounter-level audit of the Diabetes 130-US Hospitals data."""

from pathlib import Path
import hashlib
import json

import pandas as pd

from run_audit import run_full_audit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Data" / "diabetic_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "diabetes"

GROUP_COLS = ["race", "gender", "age"]
MEASURE_COLS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_medications",
]
LABEL_COL = "readmitted_30d"


def prepare_diabetes_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load encounters and define a binary 30-day readmission outcome."""

    df = pd.read_csv(path, na_values=["?"], low_memory=False)

    df["gender"] = df["gender"].replace("Unknown/Invalid", pd.NA)

    valid_labels = {"NO", ">30", "<30"}
    found_labels = set(df["readmitted"].dropna().unique())

    if found_labels != valid_labels or df["readmitted"].isna().any():
        raise ValueError(
            "Unexpected or missing readmission categories in source data."
        )

    df[LABEL_COL] = (df["readmitted"] == "<30").astype(int)

    return df


def run_case_study(
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, pd.DataFrame]:
    """Execute the case study and record its configuration."""

    df = prepare_diabetes_data()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_full_audit(
        df=df,
        group_cols=GROUP_COLS,
        measure_cols=MEASURE_COLS,
        label_col=LABEL_COL,
        positive_label=1,
        output_dir=str(output_dir),
    )

    config = {
        "source": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "source_sha256": hashlib.sha256(
            DATA_PATH.read_bytes()
        ).hexdigest(),
        "observational_unit": "hospital encounter",
        "encounters": len(df),
        "unique_patients": int(df["patient_nbr"].nunique()),
        "missing_value_markers": [
            "?",
            "None (pandas default)",
        ],
        "invalid_gender_treated_as_missing": "Unknown/Invalid",
        "group_cols": GROUP_COLS,
        "measure_cols": MEASURE_COLS,
        "label_col": LABEL_COL,
        "positive_label": 1,
        "positive_label_definition": "readmitted == '<30'",
        "negative_label_definition": "readmitted in ['NO', '>30']",
        "analysis_notes": [
            "Encounter-level observations include repeat patients.",
            "KS test p-values for discrete count features are exploratory.",
            "Audit flags are indicators for contextual investigation, not proof of bias.",
        ],
    }

    (output_dir / "run_configuration.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    return results


if __name__ == "__main__":
    results = run_case_study()

    print(f"HTML report: {OUTPUT_DIR / 'audit_report.html'}")
    print(f"Audit tables: {', '.join(results)}")
    print(f"Configuration: {OUTPUT_DIR / 'run_configuration.json'}")
