# NHS Dorset — Outpatient DNA Prediction

## Data Scientist Technical Assessment

### Project overview

Missed outpatient appointments, commonly called Did Not Attend (DNA), waste clinical capacity and can contribute to longer waiting lists.

This project investigates whether machine learning can help identify appointments at increased risk of DNA and support targeted, patient-centred outreach.

The assessment addresses five objectives:

1. Explore and clean the outpatient appointment dataset.
2. Develop and evaluate a model predicting missed appointments.
3. Assess fairness across demographic groups.
4. Prioritise up to 150 patients for telephone outreach each week and estimate potential prevented DNAs.
5. Recommend how the approach could be evaluated and implemented safely in an NHS setting.

The dataset is synthetic and contains no real patients.

---

## 1. Dataset exploration and data quality

### Original dataset

The supplied file, `Outpatient_appointments_synthetic.csv`, contains:

- 40,180 appointment records.
- 20 original columns.
- Appointment dates covering approximately April 2025 to March 2026.
- Patient, appointment, historical attendance and outcome information.

The target field is `attendance_status`.

For binary classification, the outcome is defined as:

- DNA: appointment not attended.
- Attended: appointment attended.

Cancelled appointments are excluded from this binary modelling task rather than treated as DNAs.

### Data quality assessment

The profiling process investigated:

- Missing values and inconsistent data types.
- Duplicate records and appointment identifiers.
- Inconsistent categorical values, including recorded sex.
- Invalid booking and appointment chronology.
- Numerical ranges, negative values and outliers.
- Target distribution and potential class imbalance.

### Cleaning decisions

The cleaning and modelling preparation identified:

| Stage | Records |
|---|---:|
| Original dataset | 40,180 |
| Exact duplicate records removed | 180 |
| Records after duplicate removal | 40,000 |
| Cancelled appointments excluded | 3,636 |
| Invalid booking chronology excluded | 100 |
| Eligible binary modelling records | 36,264 |

Short-notice bookings were retained where their chronology was valid.

Cleaning decisions, profiling outputs and processed datasets are documented in the notebooks and reports.

---

## 2. Feature engineering and model development

### Feature engineering

The final model uses 15 predictors.

**Patient and accessibility characteristics**

- `age`
- `age_missing_flag`
- `distance_km`
- `distance_missing_flag`
- `long_term_conditions`
- `interpreter_required`

**Appointment characteristics**

- `specialty`
- `appointment_type`
- `consultation_mode`
- `referral_source`

**Booking and appointment timing**

- `booking_lead_days_exact`
- `is_short_notice`
- `appointment_day_of_week`
- `appointment_month`
- `appointment_hour`

Numerical and categorical preprocessing is included in a scikit-learn pipeline.

Missing-value imputation and categorical encoding are fitted using training data rather than the full dataset.

### Feature selection and leakage prevention

Several potentially useful fields were not included as final predictors:

- `previous_dnas_12m`
- `rescheduled_count`
- `sms_reminder_sent`

These fields could improve prediction, but their values could not be reliably verified as available at the weekly prediction cutoff.

For example, an SMS reminder may be recorded after the Monday morning prediction point. Using information that becomes available later would introduce data leakage.

The variables should be reconsidered when timestamped operational records can establish their availability before prediction.

Sex, ethnicity and deprivation were not included among the final model predictors but were retained for fairness auditing.

### Chronological validation

The dataset was separated chronologically to reduce the risk of information leakage.

The model-development process used:

- Training data for fitting models and preprocessing.
- Validation data for comparing approaches and selecting the operational strategy.
- A held-out test set for final evaluation.

The prepared split contained:

| Split | Records |
|---|---:|
| Training | 24,612 |
| Validation | 4,773 |
| Test | 5,488 |

An additional 1,391 boundary records were excluded from these splits as a temporal leakage safeguard.

### Models evaluated

Three classification algorithms were compared:

1. Logistic Regression.
2. Random Forest.
3. XGBoost.

The development process considered baseline models, class weighting, hyperparameter tuning and validation-based operational performance.

The selected model was **Tuned XGBoost**, chosen using validation results before the held-out test evaluation.

---

## 3. Model evaluation

### Evaluation metrics

The assessment uses several complementary metrics:

- ROC-AUC: discrimination between DNA and attended appointments.
- PR-AUC: performance when the DNA class is relatively uncommon.
- Precision: proportion of predicted DNAs that were actual DNAs.
- Recall: proportion of actual DNAs identified.
- Brier score: accuracy of predicted probabilities.
- Precision@150: effectiveness of the weekly outreach shortlist.

Accuracy alone is not sufficient because most appointments are attended.

### Held-out test results

| Metric | Result |
|---|---:|
| Selected model | Tuned XGBoost |
| Test appointments | 5,488 |
| Actual DNAs | 524 |
| DNA prevalence | 9.55% |
| ROC-AUC | 0.6042 |
| PR-AUC | 0.1328 |
| Brier score | 0.0854 |

At the validation-selected classification threshold of 0.10:

| Metric | Result |
|---|---:|
| True positives | 213 |
| False positives | 1,389 |
| True negatives | 3,575 |
| False negatives | 311 |
| Precision | 13.30% |
| Recall | 40.65% |

### Interpretation

The model shows modest predictive discrimination.

Its performance does not justify autonomous clinical decision-making or restricting access to appointments.

The main potential application is to support a limited, carefully evaluated patient outreach service.

The operational top-150 ranking is more relevant to the clinic's stated staffing constraint than a general binary classification threshold.

---

## 4. Fairness and bias assessment

The model was evaluated across demographic groups, including age, recorded sex, ethnicity and deprivation.

The analysis identified disparities requiring further investigation.

Age-related recall differences were particularly notable, with some age groups receiving substantially lower recall than others.

The fairness assessment considered:

- Group-level performance differences.
- Sample sizes and uncertainty.
- Precision and recall differences.
- Potential consequences of uneven outreach selection.
- The limitations of evaluating fairness on synthetic data.

### Proposed safeguards

Before implementation:

1. Investigate performance differences across relevant groups.
2. Review whether model errors disadvantage particular populations.
3. Evaluate accessibility, communication and transport barriers.
4. Monitor fairness throughout a prospective pilot.
5. Maintain human oversight and an accessible alternative to automated selection.

Removing demographic variables from the predictors does not guarantee fairness, because other features may act as proxies.

---

## 5. Weekly outreach decision

### Operational objective

The clinic can telephone up to 150 patients each week before their appointments.

The proposed process is to:

1. Establish a weekly prediction cutoff, such as Monday at 09:00.
2. Identify appointments that are eligible for outreach at that point.
3. Generate predicted DNA probabilities.
4. Rank eligible patients by predicted risk.
5. Select up to 150 unique patients per week.
6. Use supportive telephone contact to address barriers to attendance.

Patient-level deduplication prevents the same patient from occupying multiple places on the weekly shortlist.

### Retrospective outreach results

Across nine evaluated weeks:

| Measure | Result |
|---|---:|
| Weekly outreach capacity | 150 patients |
| Evaluated weeks | 9 |
| Total patient-week selections | 1,350 |
| DNAs among selected appointments | 175 |
| Precision@150 | 12.96% |
| Lift relative to random selection | 1.37x |
| Average captured DNAs per week | 19.44 |

These are retrospective ranking results, not observed intervention outcomes.

### Estimated prevented DNAs

The number of missed appointments potentially prevented depends on two assumptions:

- Contact rate: proportion of selected patients successfully contacted.
- Intervention effectiveness: proportion of contacted patients who would otherwise miss their appointment but attend because of the intervention.

The illustrative calculation is:

Potential prevented DNAs =
Captured DNAs x Contact rate x Intervention effectiveness

| Scenario | Contact rate | Effectiveness | Estimated prevented DNAs/week |
|---|---:|---:|---:|
| Conservative | 50% | 10% | 0.97 |
| Central | 70% | 20% | 2.72 |
| Optimistic | 85% | 30% | 4.96 |

These estimates are hypothetical. The synthetic dataset does not demonstrate that telephone outreach actually prevents DNAs.

A prospective controlled pilot would be needed to estimate real intervention effectiveness.

---

## 6. Additional NHS data that could improve predictions

Potentially useful additional information includes:

- Timestamped historical attendance and cancellation records.
- Verified previous DNAs available before the prediction date.
- Appointment rescheduling history and timing.
- Reminder delivery and response information.
- Patient communication preferences.
- Interpreter and accessibility requirements.
- Travel accessibility and transport barriers.
- Clinic-level scheduling patterns and historical demand.

Each new data source would need assessment for availability at prediction time, quality, clinical relevance, privacy and fairness.

Sensitive information should only be used where justified and appropriately governed.

---

## 7. NHS implementation and governance

The model should initially be treated as an exploratory decision-support tool rather than a production-ready system.

### Suggested implementation

1. Validate the approach using appropriately governed local NHS data.
2. Reconstruct historical features as they were known at each prediction cutoff.
3. Test model calibration, operational precision and fairness.
4. Conduct a prospective pilot of model-guided outreach.
5. Compare outcomes against existing outreach practice.
6. Monitor effectiveness, patient experience and unintended consequences.
7. Review and retrain the model when appropriate.

### Governance considerations

Implementation would require:

- Compliance with UK GDPR and the Data Protection Act 2018.
- Appropriate lawful basis and purpose limitation.
- Data minimisation and retention controls.
- Role-based access and least privilege.
- Secure storage and encryption.
- Audit logging and monitoring.
- Information governance and data protection review.
- Human oversight and clear accountability.
- Review of fairness, accessibility and patient safety.

Model predictions should support patients, not penalise them or restrict access to NHS care.

### Future Technical Deployment Considerations

Following successful validation, a controlled pilot and appropriate NHS governance approval, the following technical enhancements could be considered.

**1. Scheduled batch predictions**

Automate weekly risk scoring using the trained XGBoost model to generate a prioritised outreach list of up to 150 unique patients every Monday.

The proposed workflow would:

- Retrieve eligible upcoming appointments.
- Validate and preprocess the required features.
- Generate predicted DNA probabilities.
- Rank patients by predicted risk.
- Select up to 150 unique patients for supportive outreach.
- Produce an auditable weekly output.

Scheduled batch scoring is the preferred initial approach because the operational requirement is weekly, not real-time.

2. FastAPI integration

Develop a secure prediction API if future integration with NHS clinical or operational systems requires on-demand predictions.

FastAPI could expose the validated prediction pipeline through controlled endpoints with authentication, input validation, logging and access restrictions.

An API is not essential for the initial weekly batch pilot.

3. Docker containerisation

Containerise the prediction pipeline and its dependencies using Docker to improve consistency and reproducibility across development, testing and approved deployment environments.

Containerisation would simplify packaging and deployment but would not, by itself, establish clinical safety or information governance compliance.

4. Automated testing and CI/CD

Introduce Git-based version control and controlled CI/CD processes covering:

- Unit and integration tests.
- Data quality and schema validation.
- Model validation and regression tests.
- Controlled promotion between environments.
- Approval gates and rollback procedures.

5. Model monitoring

Monitor:

- Data quality and missing values.
- Feature and prediction drift.
- Predictive performance and probability calibration.
- Fairness across demographic groups.
- Weekly outreach effectiveness and workload.
- Pipeline failures and operational alerts.

6. Security and NHS governance

Apply appropriate access controls, audit logging, encryption, data minimisation and NHS information governance requirements.

Any use of identifiable patient data would require appropriate authorisation, secure handling and clinical oversight.

Deployment recommendation

Begin with a carefully governed weekly batch-scoring pilot rather than immediate production deployment.

Consider FastAPI integration, Docker containerisation and wider deployment only after the model demonstrates sufficient clinical and operational value.

These enhancements are proposed future improvements. FastAPI integration, Docker deployment and production CI/CD have not been implemented as part of this technical assessment.

---

## 8. Project structure

The main project components are:

```text
nhs-dorset-dna-prediction/
|-- data/
|   |-- raw/
|   `-- processed/
|-- notebooks/
|   |-- 01_data_profiling.ipynb
|   |-- 02_data_cleaning.ipynb
|   |-- 03_eda.ipynb
|   |-- 04_Feature_Engineering.ipynb
|   `-- 05_model_development.ipynb
|-- artifacts/
|-- reports/
|   |-- profiling/
|   |-- figures/
|   `-- model_development/
|       `-- saved_model/
|           |-- nhs_dorset_xgboost_tuned_pipeline.joblib
|           `-- model_metadata.joblib
|-- src/
|-- tests/
|-- presentation/
|-- requirements.txt
`-- README.md
```

Some directories contain additional analysis outputs and intermediate artifacts.

---

## 9. Reproducibility

### Python environment

The project was developed using Python 3.12 and Jupyter notebooks.

Create a virtual environment from the project root.

On Windows using Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

### Dataset

Place the original synthetic dataset at:

`data/raw/Outpatient_appointments_synthetic.csv`

The dataset is excluded from Git tracking by the project's `.gitignore` rules. It must be supplied separately or included in an appropriately authorised assessment package for full reproduction.

### Notebook execution order

Execute the notebooks in this order:

1. `notebooks/01_data_profiling.ipynb`
2. `notebooks/02_data_cleaning.ipynb`
3. `notebooks/03_eda.ipynb`
4. `notebooks/04_Feature_Engineering.ipynb`
5. `notebooks/05_model_development.ipynb`

Run from the project environment and check any notebook-specific file paths.

### Loading the saved model

The final fitted pipeline can be loaded with:

```python
import joblib

model = joblib.load(
    "reports/model_development/saved_model/"
    "nhs_dorset_xgboost_tuned_pipeline.joblib"
)
```

The pipeline contains the fitted preprocessing and XGBoost classifier.

It expects the engineered predictor columns used during training, rather than the untouched original CSV columns.

Only load joblib files from trusted sources.

---

## 10. Limitations and conclusions

This assessment uses synthetic data, so findings may not generalise to real NHS Dorset patients.

Important limitations include:

- Modest model discrimination.
- Potentially useful historical predictors excluded because their timing could not be verified.
- Uncertainty in subgroup fairness estimates.
- Retrospective rather than prospective outreach evaluation.
- Hypothetical assumptions about telephone contact and intervention effectiveness.
- Need for local validation, operational testing and governance approval.

### Conclusion

The analysis demonstrates an end-to-end approach to data quality assessment, feature engineering, predictive modelling, fairness evaluation and operational decision support.

Although the model has limited predictive power, a capacity-constrained outreach strategy may be worth evaluating in a controlled pilot.

The priority should be improving reliable historical data, validating performance and fairness, and measuring whether supportive patient contact produces a meaningful reduction in missed appointments.
