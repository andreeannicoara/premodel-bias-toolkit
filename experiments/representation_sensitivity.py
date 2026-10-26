"""Check representation-audit counts and flags at predefined thresholds."""

from pathlib import Path

import pandas as pd

from toolkit.representation import run_representation_audit


MIN_COUNT = 30
MIN_PERCENTAGE = 5.0

# Each expected flag is specified before running the audit.
SCENARIOS = (
    ("below_both", 1000, 29, "CRITICAL"),
    ("below_count_only", 500, 29, "WARNING"),
    ("at_count_below_percentage", 1000, 30, "WARNING"),
    ("at_count_above_percentage", 500, 30, "INFO"),
    ("above_count", 500, 31, "INFO"),
    ("below_percentage_only", 1000, 49, "WARNING"),
    ("at_percentage", 1000, 50, "INFO"),
    ("above_percentage", 1000, 51, "INFO"),
    ("at_both_thresholds", 600, 30, "INFO"),
    ("below_count_above_percentage", 500, 28, "WARNING"),
    ("above_count_below_percentage", 1000, 31, "WARNING"),
    ("just_below_percentage_at_precision_limit", 20001, 1000, "WARNING"),
)


def make_dataset(total_rows: int, group_a_count: int) -> pd.DataFrame:
    """Construct two demographic groups with exact predefined sizes."""
    if not 0 < group_a_count < total_rows:
        raise ValueError(
            "Group A must contain between 1 and total_rows - 1 rows."
        )

    return pd.DataFrame({
        "group": ["A"] * group_a_count + ["B"] * (total_rows - group_a_count),
    })


def run_experiment(output_path: str | Path) -> pd.DataFrame:
    """Check counts, proportions and flags against predefined expectations."""
    records = []

    for scenario, total_rows, group_a_count, expected_flag in SCENARIOS:
        data = make_dataset(total_rows, group_a_count)

        audit = run_representation_audit(
            data,
            group_cols=["group"],
            min_count=MIN_COUNT,
            min_percentage=MIN_PERCENTAGE,
        ).set_index("group_value")

        expected_percentage = round(group_a_count / total_rows * 100, 4)
        observed_count = int(audit.loc["A", "count"])
        observed_percentage = float(audit.loc["A", "percentage"])
        observed_flag = str(audit.loc["A", "flag"])
        other_flag = str(audit.loc["B", "flag"])

        if (
            observed_count != group_a_count
            or observed_percentage != expected_percentage
            or observed_flag != expected_flag
            or other_flag != "INFO"
        ):
            raise AssertionError(
                f"Unexpected representation result: {scenario}"
            )

        records.append({
            "scenario": scenario,
            "total_rows": total_rows,
            "expected_count": group_a_count,
            "observed_count": observed_count,
            "expected_percentage": expected_percentage,
            "observed_percentage": observed_percentage,
            "expected_flag": expected_flag,
            "observed_flag": observed_flag,
            "other_group_flag": other_flag,
        })

    result = pd.DataFrame(records)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)

    return result


if __name__ == "__main__":
    output = Path("output/synthetic/representation_sensitivity.csv")

    results = run_experiment(output)

    print(results.to_string(index=False))
    print(f"\nSaved results to {output}")
