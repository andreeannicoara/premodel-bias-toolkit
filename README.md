# Aidit-HD

**A Toolkit for Auditing Potential Bias in Healthcare Data**

Aidit-HD is a Python-based diagnostic toolkit developed for an MSc Data Science dissertation at the University of Essex Online. It examines healthcare datasets before predictive model development and identifies statistical indicators of potential bias.

## Audit Dimensions

- **Representation:** Demographic subgroup counts, proportions and threshold-based flags.
- **Missingness:** Missing-data rates and statistical associations across demographic groups.
- **Measurement:** Pairwise comparisons of numeric feature distributions using Kolmogorov-Smirnov tests.
- **Label:** Binary outcome-rate disparities and statistical comparisons between demographic groups.

The toolkit produces individual CSV audit tables and a consolidated HTML report. Findings are classified as INFO, WARNING or CRITICAL.

These classifications are diagnostic indicators. They do not establish discrimination, causation or clinical unfairness.

## Research Context

**Dissertation:** Aidit-HD: A Toolkit for Auditing Potential Bias in Healthcare Data

**Supervisor:** Dr Bakhtiyar Ahmed, University of Essex Online

**Case study:** Diabetes 130-US Hospitals for Years 1999–2008 (Strack et al., 2014).

[Dataset source: UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008)

## Installation

```bash
pip install -r requirements.txt
```

## Running the Case Study

Download the dataset and place `diabetic_data.csv` in the `Data/` directory.

Run the encounter-level audit:

```bash
python -m experiments.diabetes_case_study
```

The HTML report is saved to:

`output/diabetes/audit_report.html`

To run the one-encounter-per-patient sensitivity analysis:

```bash
python -m experiments.patient_level_sensitivity
```

## Tests

```bash
python -m pytest
```

## Limitations

Aidit-HD uses established statistical methods to screen datasets for potential concerns. Its results require contextual interpretation and do not establish whether an observed difference is unfair.

The controlled experiments verify expected computational behaviour. They do not establish the toolkit's ability to identify all genuine forms of bias.