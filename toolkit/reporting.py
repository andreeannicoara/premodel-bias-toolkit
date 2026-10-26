"""CSV and consolidated HTML reporting for the pre-modelling audit toolkit."""

from html import escape
from pathlib import Path

import pandas as pd


AUDIT_NAMES = ("representation", "missingness", "measurement", "label")

AUDIT_NOTES = {
    "representation": (
        "Subgroup proportions alone cannot establish representativeness "
        "without a reference population."
    ),
    "missingness": (
        "Associations in missingness do not identify MCAR, MAR or MNAR "
        "mechanisms. An INFO flag may mean that no valid statistical "
        "test was available; inspect the test_method column."
    ),
    "measurement": (
        "KS comparisons indicate distributional differences, not measurement "
        "error. Conventional KS p-values require caution for discrete values "
        "and ties."
    ),
    "label": (
        "Outcome-rate disparities do not establish an invalid label, "
        "discrimination or causation. Pairwise flags use adjusted "
        "pairwise p-values. Sparse multi-group tables may lack a "
        "valid omnibus test."
    ),
}


def save_audit_table(table, output_path):
    """Save an audit result table to CSV."""
    table.to_csv(output_path, index=False)


def generate_audit_report(
    results: dict[str, pd.DataFrame],
    n_rows: int,
    group_cols: list[str],
    measure_cols: list[str],
    label_col: str,
    positive_label,
    output_path: str | Path,
) -> Path:
    """Write a self-contained HTML summary of the four audit results."""

    missing = set(AUDIT_NAMES) - set(results)
    if missing:
        raise ValueError(f"Missing audit results: {sorted(missing)}")

    summary_rows = []
    sections = []

    for name in AUDIT_NAMES:
        table = results[name]

        # Missingness repeats the same test for each demographic subgroup.
        findings = (
            table.drop_duplicates(["variable", "group_variable"])
            if name == "missingness" and not table.empty
            else table
        )

        counts = (
            findings["flag"].value_counts()
            if "flag" in findings
            else pd.Series(dtype=int)
        )

        summary_rows.append({
            "Audit": name.title(),
            "Findings": len(findings),
            "Critical": int(counts.get("CRITICAL", 0)),
            "Warning": int(counts.get("WARNING", 0)),
            "Info": int(counts.get("INFO", 0)),
        })

        table_markup = (
            table.to_html(
                index=False,
                border=0,
                escape=True,
                na_rep="—",
                classes="audit-table",
            )
            if not table.empty
            else "<p>No eligible findings were produced by this audit.</p>"
        )

        sections.append(
            f"<section><h2>{name.title()}</h2>"
            f"<p class='note'>{escape(AUDIT_NOTES[name])}</p>"
            f"<div class='table-wrap'>{table_markup}</div></section>"
        )

    summary = pd.DataFrame(summary_rows).to_html(
        index=False,
        border=0,
        escape=True,
        classes="audit-table",
    )

    metadata = (
        f"<p><strong>Rows audited:</strong> {n_rows:,}</p>"
        f"<p><strong>Demographic attributes:</strong> "
        f"{escape(', '.join(group_cols))}</p>"
        f"<p><strong>Numerical features:</strong> "
        f"{escape(', '.join(measure_cols))}</p>"
        f"<p><strong>Binary outcome:</strong> {escape(label_col)} "
        f"(positive class: {escape(str(positive_label))})</p>"
    )

    html = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Pre-modelling Dataset Audit</title>"
        "<style>body{font:15px Arial,sans-serif;max-width:1100px;"
        "margin:32px auto;padding:0 20px;color:#233043;line-height:1.5}"
        "h1,h2{color:#153b56}section{margin-top:36px}"
        ".table-wrap{overflow-x:auto}"
        ".audit-table{border-collapse:collapse;width:100%;font-size:13px}"
        ".audit-table th,.audit-table td{border:1px solid #d4dce4;"
        "padding:8px;text-align:left}"
        ".audit-table th{background:#edf3f6}"
        ".audit-table tr:nth-child(even){background:#f9fbfc}"
        ".note{color:#506171}"
        "footer{margin:30px 0;color:#506171}</style></head><body>"
        "<h1>Pre-modelling Dataset Audit</h1>"
        + metadata
        + "<h2>Summary</h2>"
        + "<p>Each table row is one finding, except missingness counts "
          "are deduplicated by variable and demographic attribute to avoid "
          "counting the same association repeatedly.</p>"
        + summary
        + "".join(sections)
        + "<footer>Diagnostic flags are screening indicators, not proof of "
          "unfairness. Inferential tests require appropriate sample sizes "
          "and independent observations.</footer>"
        + "</body></html>"
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")

    return path
