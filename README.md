# KNEELSA: estimated probability of prevalent radiographic knee osteoarthritis

Source for the web application implementing the two logistic models developed in the ELSA-Brasil Musculoskeletal Study.

**Live app:** [ADD URL once deployed]

## What it does, and what it does not

For a knee assessed now, the app estimates the probability that radiographic knee
osteoarthritis is **already present** — Kellgren–Lawrence grade 2 or higher in
the tibiofemoral or the patellofemoral joint.

It does **not** estimate the chance of developing osteoarthritis later. The study
behind it is cross-sectional, so the model describes present structural status
only. It is a diagnostic prediction model in the TRIPOD sense, not a prognostic
one.

## Research use only

- Developed and validated **internally only**, in a single cohort of Brazilian
  civil servants aged 38–79. Never tested in another population; performance
  elsewhere is unknown.
- Radiographic osteoarthritis and symptoms are frequently discordant. A high
  estimated probability does not mean the knee is painful and does not by itself
  indicate any treatment.
- No threshold has been established at which a knee radiograph should be
  obtained, so the app defines no decision rule and reports no cut-off.
- Not a substitute for clinical assessment or professional medical advice.

## The two models

**Constitutional** (default, eight variables) uses characteristics that do not
depend on current symptom status. This is the primary model of the paper and the
one to use when symptoms are unknown.

- Person-level: age, body mass index, waist-hip ratio, occupational nature
  (non-routine non-manual versus other), race and skin color (self-reported
  White versus other), and squatting for 30 minutes or more in a single day in
  the last 30 days
- Knee-level: history of knee surgery, history of knee trauma

**Symptom-Augmented** (ten variables) was selected with three self-reported
symptom items also available. It keeps seven of the Constitutional variables (all
but squatting) and adds the three items, recorded per knee: frequent knee symptoms, symptoms in the last seven days, and
knee-related activity limitation. It answers a different question, namely how
well structural disease is identified once the clinical presentation is already
known. Discrimination was 0.821 against 0.811 for the Constitutional model.

Parameters live in `final_model.csv` and `symptom_augmented_model.csv`, which carry, per variable, the imputation
median, the standardization mean and scale, and the coefficient on the
standardised scale. The prediction is

    z      = (x - scaler_mean) / scaler_scale
    logit  = intercept + Σ coef_on_scaled × z
    p      = 1 / (1 + exp(−logit))

Both are generated from the fitted models by `scripts/17_export_calculator_model.py`
and `scripts/18_symptom_augmented_model.py` in the analysis repository, so the app
and the paper cannot drift apart. Do not edit it by hand.

## Provenance

Analysis code and results: https://github.com/juliogdomingues/clinical-rKOAprev

## Running locally

```
pip install -r requirements.txt
streamlit run app.py
```
