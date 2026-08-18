# KNEELSA: estimated probability of prevalent radiographic knee osteoarthritis

Source for the web application implementing the seven-variable Constitutional
logistic model developed in the ELSA-Brasil Musculoskeletal Study.

**Live app:** https://kneelsa-clinical.streamlit.app/

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

## The model

Logistic regression on seven variables.

Person-level: age, body mass index, waist–hip ratio, occupational nature
(non-routine non-manual versus other), race and skin colour (self-reported White
versus other).

Knee-level: history of knee surgery, history of knee trauma.

Parameters live in `final_model.csv`, which carries, per variable, the imputation
median, the standardisation mean and scale, and the coefficient on the
standardised scale. The prediction is

    z      = (x - scaler_mean) / scaler_scale
    logit  = intercept + Σ coef_on_scaled × z
    p      = 1 / (1 + exp(−logit))

`final_model.csv` is generated from the fitted model by
`scripts/17_export_calculator_model.py` in the analysis repository, so the app
and the paper cannot drift apart. Do not edit it by hand.

## Provenance

Analysis code and results: https://github.com/juliogdomingues/clinical-rKOAprev

## Running locally

```
pip install -r requirements.txt
streamlit run app.py
```
