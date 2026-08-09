# healthcare-bias-audit

A Python-based diagnostic toolkit for detecting pre-modelling bias-risk indicators in healthcare datasets, developed as part of an MSc Data Science dissertation at the University of Essex Online.

## Project Overview

Most fairness research in healthcare AI focuses on model outputs. This toolkit intervenes earlier — at the dataset stage, before any model is trained. It audits healthcare datasets across four bias dimensions:

- **Representation bias** — are demographic groups proportionally represented?
- **Missingness bias** — is missing data patterned across demographic groups (MNAR)?
- **Measurement bias** — is the same variable collected consistently across groups?
- **Label bias** — does the outcome variable mean the same thing for all groups?

The toolkit accepts a tabular healthcare dataset (CSV) as input and produces a structured diagnostic report with quantitative metrics, severity flags, and visualisations.

## Research Context

**Title:** Diagnosing Bias Before the Model: A Multi-Dimensional Audit Toolkit for Healthcare Datasets

**Supervisor:** Dr Bakhtiyar Ahmed, University of Essex Online

**Dataset used for validation:** Diabetes 130-US Hospitals for Years 1999–2008 (Strack et al., 2014), available via the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008).

## Data Setup

Download the dataset from the UCI repository and place `diabetic_data.csv` in the `data/` folder. The data folder is gitignored to avoid redistributing the dataset.

## Usage

```bash
python run_audit.py --data data/diabetic_data.csv --group race
```

Output is saved to the `outputs/` folder.
