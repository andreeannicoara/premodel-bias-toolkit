"""Assess missingness-audit behaviour under controlled subgroup disparities."""

from pathlib import Path

import numpy as np
import pandas as pd

from toolkit.missingness import run_missingness_audit


SAMPLE_SIZES = (50, 100, 200)
GROUP_A_RATES = (0.10, 0.20, 0.30, 0.40, 0.50)
GROUP_B_RATE = 0.10
SEED = 2026


def make_dataset(
    n_per_group: int,
    rate_a: float,
    rate_b: float,
    seed: int,
) -> pd.DataFrame:
    """Generate two groups with predefined missingness rates."""

    rng = np.random.default_rng(seed)
    measurements = []

    for rate in (rate_a, rate_b):
        missing_count = round(n_per_group * rate)

        values = rng.normal(size=n_per_group)

        missing_indices = rng.choice(
            n_per_group,
            size=missing_count,
            replace=False,
        )

        values[missing_indices] = np.nan
        measurements.append(values)

    return pd.DataFrame({
        "group": ["A"] * n_per_group + ["B"] * n_per_group,
        "lab_value": np.concatenate(measurements),
    })


def run_experiment(output_path: str | Path) -> pd.DataFrame:
    """Compare expected and observed missingness across scenarios."""

    records = []

    scenarios = (
        (n, rate)
        for n in SAMPLE_SIZES
        for rate in GROUP_A_RATES
    )

    for scenario_id, (n, rate_a) in enumerate(scenarios, start=1):

        data = make_dataset(
            n,
            rate_a,
            GROUP_B_RATE,
            SEED + scenario_id,
        )

        audit = run_missingness_audit(
            data,
            group_cols=["group"],
            min_missing_rate=0.0,
        )

        lab_results = audit.loc[
            audit["variable"] == "lab_value"
        ]

        rates = lab_results.set_index("group_value")[
            "missing_percentage"
        ]

        expected_a = rate_a * 100
        expected_b = GROUP_B_RATE * 100

        actual_a = float(rates.loc["A"])
        actual_b = float(rates.loc["B"])

        expected_gap = abs(expected_a - expected_b)
        actual_gap = float(lab_results["disparity_pp"].iloc[0])

        if not np.allclose(
            [actual_a, actual_b, actual_gap],
            [expected_a, expected_b, expected_gap],
            atol=0.01,
        ):
            raise AssertionError(
                f"Scenario {scenario_id} failed the known-rate checks"
            )

        records.append({
            "scenario": scenario_id,
            "seed": SEED + scenario_id,
            "n_per_group": n,
            "expected_a_pct": expected_a,
            "observed_a_pct": actual_a,
            "expected_b_pct": expected_b,
            "observed_b_pct": actual_b,
            "expected_disparity_pp": expected_gap,
            "observed_disparity_pp": actual_gap,
            "p_value": lab_results["p_value"].iloc[0],
            "flag": lab_results["flag"].iloc[0],
        })

    result = pd.DataFrame(records)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(path, index=False)

    return result


if __name__ == "__main__":

    output = Path(
        "output/synthetic/missingness_sensitivity.csv"
    )

    results = run_experiment(output)

    print(results.to_string(index=False))
    print(f"\nSaved results to {output}")