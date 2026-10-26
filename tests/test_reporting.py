import re

import numpy as np
import pandas as pd

from run_audit import run_full_audit


def _test_data(group_a="A"):
    return pd.DataFrame({
        "group": [group_a] * 50 + ["B"] * 50,
        "measurement": list(range(50)) + list(range(100, 150)),
        "missing_var": [np.nan] * 20 + [1.0] * 80,
        "outcome": [1] * 40 + [0] * 10 + [1] * 10 + [0] * 40,
    })


def _run(df, tmp_path):
    return run_full_audit(
        df=df,
        group_cols=["group"],
        measure_cols=["measurement"],
        label_col="outcome",
        positive_label=1,
        output_dir=str(tmp_path),
    )


def test_consolidated_report_contains_all_audits(tmp_path):
    results = _run(_test_data(), tmp_path)
    report = tmp_path / "audit_report.html"

    assert report.is_file()

    page = report.read_text(encoding="utf-8")

    for heading in (
        "Representation",
        "Missingness",
        "Measurement",
        "Label",
    ):
        assert f"<h2>{heading}</h2>" in page

    assert "Rows audited:</strong> 100" in page
    assert "Subgroup proportions alone" in page
    assert "same association repeatedly" in page
    assert len(results) == 4

    # One missingness association, despite two subgroup rows.
    assert re.search(
        r"<td>Missingness</td>\s*<td>1</td>\s*<td>1</td>",
        page,
    )


def test_report_escapes_values_from_data(tmp_path):
    _run(_test_data(group_a="<script>"), tmp_path)

    page = (
        tmp_path / "audit_report.html"
    ).read_text(encoding="utf-8")

    assert "&lt;script&gt;" in page
    assert "<script>" not in page
