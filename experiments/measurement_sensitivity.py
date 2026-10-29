"""Controlled checks of the measurement audit's KS comparisons."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from toolkit.measurement import run_measurement_audit


SAMPLE_SIZES = (20, 50, 100, 200)
SHIFTS = (0.0, 0.5, 1.0)


def make_dataset(n_per_group: int, shift: float) -> pd.DataFrame:
    """Create two groups with known quantiles and a fixed location shift."""
    if n_per_group < 1:
        raise ValueError("n_per_group must be positive")

    quantiles = (np.arange(n_per_group) + 0.5) / n_per_group
    values = norm.ppf(quantiles)

    return pd.DataFrame({
        "group": ["A"] * n_per_group + ["B"] * n_per_group,
        "measurement": np.concatenate([values, values + shift]),
    })


def empirical_ks(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Calculate KS distance directly from two empirical CDFs."""
    points = np.unique(np.concatenate([values_a, values_b]))

    left = (
        np.searchsorted(np.sort(values_a), points, side="right")
        / len(values_a)
    )
    right = (
        np.searchsorted(np.sort(values_b), points, side="right")
        / len(values_b)
    )

    return float(np.max(np.abs(left - right)))


def run_experiment(output_path: str | Path) -> pd.DataFrame:
    """Save known KS distances, adjusted p-values and diagnostic flags."""
    cases = []

    for n in SAMPLE_SIZES:
        for shift in SHIFTS:
            cases.append((
                f"shift_{shift:g}_n{n}",
                "quantile_shift",
                shift,
                make_dataset(n, shift),
            ))

    base = norm.ppf((np.arange(100) + 0.5) / 100)

    cases.append((
        "three_groups",
        "multiple_comparisons",
        1.0,
        pd.DataFrame({
            "group": ["A"] * 100 + ["B"] * 100 + ["C"] * 100,
            "measurement": np.concatenate([base, base, base + 1.0]),
        }),
    ))

    cases.append((
        "below_minimum",
        "minimum_group_size",
        1.0,
        make_dataset(19, 1.0),
    ))

    discrete = np.tile(np.arange(10), 10)

    cases.append((
        "discrete_counts",
        "discrete_ties",
        1.0,
        pd.DataFrame({
            "group": ["A"] * 100 + ["B"] * 100,
            "measurement": np.concatenate([discrete, discrete + 1]),
        }),
    ))

    rows = []

    for case_name, case_type, shift, df in cases:
        audit = run_measurement_audit(
            df,
            group_cols=["group"],
            measure_cols=["measurement"],
        )
        comparisons = len(audit)

        for _, result in audit.iterrows():
            group_1 = result["group_1"]
            group_2 = result["group_2"]

            a = df.loc[
                df["group"] == group_1, "measurement"
            ].to_numpy()
            b = df.loc[
                df["group"] == group_2, "measurement"
            ].to_numpy()

            enough_data = min(len(a), len(b)) >= 20

            expected_ks = empirical_ks(a, b) if enough_data else None
            observed_ks = result["ks_statistic"]

            p_value = result["p_value"]
            adjusted = result["adjusted_p_value"]

            expected_adjusted = (
                min(float(p_value) * comparisons, 1.0)
                if pd.notna(p_value)
                else None
            )

            expected_flag = (
                "WARNING"
                if expected_adjusted is not None
                and expected_adjusted < 0.05
                else "INFO"
            )

            if enough_data:
                if not np.isclose(observed_ks, expected_ks):
                    raise AssertionError(
                        f"KS distance differs in {case_name}"
                    )

                if not np.isclose(adjusted, expected_adjusted):
                    raise AssertionError(
                        f"Adjusted p-value differs in {case_name}"
                    )

            elif pd.notna(observed_ks) or pd.notna(p_value):
                raise AssertionError(
                    f"Small group unexpectedly tested in {case_name}"
                )

            if result["flag"] != expected_flag:
                raise AssertionError(
                    f"Unexpected flag in {case_name}"
                )

            pair_shift = (
                0.0
                if case_name == "three_groups"
                and {group_1, group_2} == {"A", "B"}
                else shift
            )

            rows.append({
                "scenario": case_name,
                "scenario_type": case_type,
                "n_per_group": len(a),
                "shift": pair_shift,
                "group_1": group_1,
                "group_2": group_2,
                "number_of_comparisons": comparisons,
                "expected_ks": expected_ks,
                "observed_ks": observed_ks,
                "p_value": p_value,
                "expected_adjusted_p": expected_adjusted,
                "adjusted_p_value": adjusted,
                "expected_flag": expected_flag,
                "flag": result["flag"],
                "interpretation_note": (
                    "KS p-value exploratory for tied discrete counts"
                    if case_type == "discrete_ties"
                    else "Deterministic scenario; not a power estimate"
                ),
            })

    output = pd.DataFrame(rows)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)

    return output


if __name__ == "__main__":
    destination = Path(
        "output/synthetic/measurement_sensitivity.csv"
    )

    results = run_experiment(destination)

    print(results.to_string(index=False))
    print(f"\nSaved results to {destination}")