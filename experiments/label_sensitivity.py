"""Controlled checks of outcome-rate disparities and label-audit tests."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from toolkit.label import run_label_audit


# Each group is (name, number of observations, number of positive outcomes).
SCENARIOS = tuple(
    (
        f"n{n}_positive{rate}",
        "sample_size_and_disparity",
        (("A", n, n * rate // 100), ("B", n, n // 10)),
        "chi_square",
    )
    for n in (50, 100, 200)
    for rate in (10, 20, 30, 40, 50)
) + (
    ("below_warning_gap", "threshold",
     (("A", 1000, 190), ("B", 1000, 100)), "chi_square"),
    ("at_warning_gap", "threshold",
     (("A", 1000, 200), ("B", 1000, 100)), "chi_square"),
    ("at_critical_gap", "threshold",
     (("A", 1000, 400), ("B", 1000, 100)), "chi_square"),
    ("just_below_critical_gap", "precision_boundary",
     (("A", 51, 20), ("B", 141, 13)), "chi_square"),
    ("three_groups", "multiple_comparisons",
     (("A", 100, 50), ("B", 100, 50), ("C", 100, 90)), "chi_square"),
    ("sparse_two_groups", "sparse",
     (("A", 5, 5), ("B", 5, 0)), "fisher_exact"),
    ("sparse_three_groups", "sparse",
     (("A", 6, 6), ("B", 6, 6), ("C", 6, 0)), "not_tested_sparse"),
)


def make_dataset(groups: tuple[tuple[str, int, int], ...]) -> pd.DataFrame:
    """Construct group outcomes from exact positive and negative counts."""
    rows = []

    for group, count, positives in groups:
        if count <= 0 or not 0 <= positives <= count:
            raise ValueError("Invalid group size or positive outcome count.")

        rows.extend((group, 1) for _ in range(positives))
        rows.extend((group, 0) for _ in range(count - positives))

    return pd.DataFrame(rows, columns=["group", "outcome"])


def reference_pairwise_p(
    n_1: int,
    positives_1: int,
    n_2: int,
    positives_2: int,
) -> tuple[float | None, str]:
    """Calculate a reference p-value directly from known contingency counts."""
    counts = np.array([
        [positives_1, n_1 - positives_1],
        [positives_2, n_2 - positives_2],
    ])

    if (counts.sum(axis=0) == 0).any():
        return None, "not_testable"

    _, chi_p, _, expected = stats.chi2_contingency(counts)

    if (expected < 5).any():
        return float(stats.fisher_exact(counts)[1]), "fisher_exact"

    return float(chi_p), "chi_square"


def expected_flag(adjusted_p: float | None, disparity: float) -> str:
    """Apply the predefined diagnostic decision rules to reference results."""
    if adjusted_p is None or adjusted_p >= 0.05:
        return "INFO"
    return "CRITICAL" if disparity >= 30 else "WARNING"


def run_experiment(output_path: str | Path) -> pd.DataFrame:
    """Validate outcome rates, pairwise tests, correction and flags."""
    records = []

    for scenario, case_type, groups, expected_omnibus in SCENARIOS:
        df = make_dataset(groups)
        audit = run_label_audit(
            df, group_cols=["group"], label_col="outcome", positive_label=1,
        )
        group_counts = {name: (n, positive) for name, n, positive in groups}
        comparison_count = len(list(combinations(groups, 2)))

        if len(audit) != comparison_count:
            raise AssertionError(f"Wrong comparison count: {scenario}")
        if set(audit["omnibus_test"]) != {expected_omnibus}:
            raise AssertionError(f"Wrong omnibus method: {scenario}")

        for _, result in audit.iterrows():
            group_1, group_2 = result["group_1"], result["group_2"]
            n_1, positives_1 = group_counts[group_1]
            n_2, positives_2 = group_counts[group_2]
            rate_1, rate_2 = 100 * positives_1 / n_1, 100 * positives_2 / n_2
            disparity = abs(rate_1 - rate_2)

            reference_p, reference_method = reference_pairwise_p(
                n_1, positives_1, n_2, positives_2,
            )
            adjusted_reference = (
                min(reference_p * comparison_count, 1.0)
                if reference_p is not None else None
            )
            reference_flag = expected_flag(adjusted_reference, disparity)

            if (
                result["group_1_count"] != n_1
                or result["group_2_count"] != n_2
                or result["group_1_positive_count"] != positives_1
                or result["group_2_positive_count"] != positives_2
                or not np.isclose(result["group_1_positive_rate"], rate_1)
                or not np.isclose(result["group_2_positive_rate"], rate_2)
                or not np.isclose(result["disparity_pp"], disparity)
                or result["pairwise_test"] != reference_method
                or result["flag"] != reference_flag
            ):
                raise AssertionError(
                    f"Incorrect outcome rates or flags: {scenario}, "
                    f"{group_1}-{group_2}"
                )

            for field, reference in (
                ("pairwise_p_value", reference_p),
                ("pairwise_adjusted_p_value", adjusted_reference),
            ):
                observed = result[field]
                if reference is None:
                    correct = pd.isna(observed)
                else:
                    correct = np.isclose(
                        observed, reference, rtol=1e-10, atol=0,
                    )
                if not correct:
                    raise AssertionError(
                        f"Incorrect {field}: {scenario}, {group_1}-{group_2}"
                    )

            records.append({
                "scenario": scenario,
                "scenario_type": case_type,
                "group_1": group_1,
                "group_2": group_2,
                "n_1": n_1,
                "n_2": n_2,
                "expected_rate_1": round(rate_1, 4),
                "observed_rate_1": result["group_1_positive_rate"],
                "expected_rate_2": round(rate_2, 4),
                "observed_rate_2": result["group_2_positive_rate"],
                "expected_disparity_pp": round(disparity, 4),
                "observed_disparity_pp": result["disparity_pp"],
                "omnibus_test": result["omnibus_test"],
                "pairwise_test": result["pairwise_test"],
                "pairwise_p_value": result["pairwise_p_value"],
                "expected_adjusted_p": adjusted_reference,
                "adjusted_p_value": result["pairwise_adjusted_p_value"],
                "expected_flag": reference_flag,
                "flag": result["flag"],
            })

    output = pd.DataFrame(records)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    return output


if __name__ == "__main__":
    destination = Path("output/synthetic/label_sensitivity.csv")
    results = run_experiment(destination)
    print(results.to_string(index=False))
    print(f"\nSaved results to {destination}")