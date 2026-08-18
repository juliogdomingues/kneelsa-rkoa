import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURAÇÃO DO MODELO (carregado de CSV)
# ==========================================
SINGLE_CSV = Path("final_model.csv")
FIGURE_1_PATH = Path("fig1.png")


@st.cache_data
def load_model_params():
    """
    Loads the 7-variable Constitutional model parameters from a single CSV:
    results_final_analysis/final_model.csv
    """
    if not SINGLE_CSV.exists():
        raise FileNotFoundError(
            "Model CSV not found. Please run complete.py to generate:\n"
            f"- {SINGLE_CSV}"
        )

    df = pd.read_csv(SINGLE_CSV)

    intercept_row = df.loc[df["feature"] == "__INTERCEPT__"].iloc[0]
    intercept = float(intercept_row["intercept"])

    features_df = df.loc[
        df["param_type"] == "feature",
        ["feature", "imputer_median", "scaler_mean", "scaler_scale", "coef_on_scaled"],
    ].copy()

    return intercept, features_df


INTERCEPT, MODEL_FEATURES = load_model_params()

# ==========================================
# INTERFACE DO USUÁRIO
# ==========================================
st.set_page_config(page_title="Kneelsa-Clinical", page_icon="🦵")

st.title("KNEELSA: estimated probability of prevalent radiographic knee osteoarthritis")
st.markdown("""
This tool implements the seven-variable Constitutional logistic model developed in the
ELSA-Brasil Musculoskeletal Study. For a knee assessed **now**, it estimates the probability that
radiographic knee osteoarthritis is **already present** (Kellgren-Lawrence grade 2 or higher in the
tibiofemoral or patellofemoral joint).

It does not estimate the chance of developing osteoarthritis in the future: the study behind it is
cross-sectional, so the model describes present structural status only.
""")

st.markdown("---")

# Patient Demographics (same for both knees)
st.subheader("Patient characteristics")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=30, max_value=100, value=55)

with col2:
    bmi = st.number_input("BMI (kg/m²)", min_value=15.0, max_value=60.0, value=25.0, format="%.1f", step=1.0)

col3, col4 = st.columns(2)
with col3:
    whr = st.number_input(
        "Waist-hip ratio", min_value=0.50, max_value=1.60, value=0.91, format="%.2f", step=0.01,
        help="Waist circumference divided by hip circumference.",
    )
with col4:
    occupation = st.selectbox(
        "Occupational nature",
        ["Non-routine non-manual", "Routine non-manual", "Routine manual", "Non-routine manual"],
        help="Classified by the tasks performed, as recorded in ELSA-Brasil.",
    )

race = st.selectbox(
    "Race and skin colour (self-reported)",
    ["White", "Brown/Mixed", "Black", "Asian", "Indigenous"],
    help="Self-reported using the Brazilian census (IBGE) categories.",
)

st.markdown("---")

# Knee-specific Clinical Features
st.subheader("Clinical Features per Knee")

# Select which knees to assess
knee_selection = st.radio(
    "Which knee(s) would you like to assess?",
    options=["Left Knee Only", "Right Knee Only", "Both Knees"],
    index=2
)


def knee_svg(knee: str) -> str:
    label = "RIGHT (R)" if knee == "Right" else "LEFT (L)"
    accent = "#1f77b4" if knee == "Right" else "#d62728"
    return "".join(
        [
            f'<svg width="54" height="54" viewBox="0 0 54 54" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{label} knee">',
            f'<rect x="18" y="4" width="18" height="18" rx="6" fill="{accent}" opacity="0.85"/>',
            '<circle cx="27" cy="27" r="9" fill="#f2f2f2" stroke="#666" stroke-width="2"/>',
            f'<rect x="18" y="32" width="18" height="18" rx="6" fill="{accent}" opacity="0.35"/>',
            '<path d="M18 23 C22 17, 32 17, 36 23" fill="none" stroke="#666" stroke-width="2"/>',
            '<path d="M18 31 C22 37, 32 37, 36 31" fill="none" stroke="#666" stroke-width="2"/>',
            "</svg>",
        ]
    )


def knee_badge_html(knee: str, *, show_subtitle: bool = True) -> str:
    # Simple, inline SVG to make side obvious.
    label = "RIGHT (R)" if knee == "Right" else "LEFT (L)"
    subtitle = (
        '<div style="font-size:12px;color:#666;line-height:1.2;">Radiographic view</div>'
        if show_subtitle
        else ""
    )
    return "".join(
        [
            '<div style="display:flex;align-items:center;gap:10px;">',
            knee_svg(knee),
            "<div>",
            f'<div style="font-weight:700;line-height:1;">{label}</div>',
            subtitle,
            "</div>",
            "</div>",
        ]
    )


def radiographic_view_html() -> str:
    return '<div style="text-align:center;font-size:12px;color:#666;margin:6px 0 10px 0;">Radiographic view</div>'


def both_knees_icons_line_html() -> str:
    # Centered line: RIGHT label + right icon + left icon + LEFT label
    return "".join(
        [
            '<div style="display:flex;justify-content:center;align-items:center;gap:18px;margin-bottom:10px;">',
            '<div style="font-weight:700;">Right knee (R)</div>',
            knee_svg("Right"),
            knee_svg("Left"),
            '<div style="font-weight:700;">Left knee (L)</div>',
            "</div>",
        ]
    )


def knee_column_header_html(knee: str, *, justify: str) -> str:
    label = "RIGHT (R)" if knee == "Right" else "LEFT (L)"
    return (
        f'<div style="display:flex;justify-content:{justify};align-items:center;gap:10px;">'
        f'<div style="font-weight:700;">{label}</div>'
        f'{knee_svg(knee)}'
        "</div>"
    )


def get_knees_to_assess(selection: str) -> list[str]:
    if selection == "Both Knees":
        return ["Left", "Right"]
    if selection == "Left Knee Only":
        return ["Left"]
    if selection == "Right Knee Only":
        return ["Right"]
    return []


knees_to_assess = get_knees_to_assess(knee_selection)

SURGERY_LABEL = "History of Surgery?"
TRAUMA_LABEL = "History of Trauma/Injury?"

# Create inputs for knee-specific variables
if len(knees_to_assess) == 1:
    knee = knees_to_assess[0]
    st.markdown(knee_badge_html(knee).strip(), unsafe_allow_html=True)
    st.checkbox(
        SURGERY_LABEL,
        key=f"surgery_{knee}",
        help="Ever undergone any type of surgery, including arthroscopy, meniscal or ligament repair?",
    )
    st.checkbox(
        TRAUMA_LABEL,
        key=f"trauma_{knee}",
        help="Ever injured or suffered trauma that caused difficulty walking for at least one week?",
    )
else:
    # Radiographic convention: RIGHT knee on the LEFT side of the screen
    st.markdown(radiographic_view_html(), unsafe_allow_html=True)
    
    # Simple two-column layout (fully responsive, Streamlit-native)
    col_right, col_left = st.columns(2)
    
    with col_right:
        st.markdown(
            f'<div style="text-align:center;margin-bottom:10px;">'
            f'<div style="font-weight:700;">Right knee (R)</div>'
            f'{knee_svg("Right")}'
            f'</div>',
            unsafe_allow_html=True
        )
        st.checkbox(
            SURGERY_LABEL,
            key="surgery_Right",
            help="Ever undergone any type of surgery, including arthroscopy, meniscal or ligament repair?"
        )
        st.checkbox(
            TRAUMA_LABEL,
            key="trauma_Right",
            help="Ever injured or suffered trauma that caused difficulty walking for at least one week?"
        )
    
    with col_left:
        st.markdown(
            f'<div style="text-align:center;margin-bottom:10px;">'
            f'<div style="font-weight:700;">Left knee (L)</div>'
            f'{knee_svg("Left")}'
            f'</div>',
            unsafe_allow_html=True
        )
        st.checkbox(
            SURGERY_LABEL,
            key="surgery_Left",
            help="Ever undergone any type of surgery, including arthroscopy, meniscal or ligament repair?"
        )
        st.checkbox(
            TRAUMA_LABEL,
            key="trauma_Left",
            help="Ever injured or suffered trauma that caused difficulty walking for at least one week?"
        )


st.markdown("---")

# ==========================================
# CÁLCULO
# ==========================================
def calculate_probability(age, bmi, whr, occupation, race, surgery, trauma):
    """Calculate the probability of KOA for a single knee using the saved preprocessing + LR params."""
    x = {
        "age": float(age),
        "bmi": float(bmi),
        "waist_hip_ratio": float(whr),
        "history_surgery": 1.0 if surgery else 0.0,
        "history_trauma": 1.0 if trauma else 0.0,
        "occupation_4": 1.0 if occupation == "Non-routine non-manual" else 0.0,
        "race_raw_3": 1.0 if race == "White" else 0.0,
    }

    logit = float(INTERCEPT)
    for _, row in MODEL_FEATURES.iterrows():
        f = row["feature"]
        val = x.get(f, None)

        # median imputation (only used if val is missing)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = float(row["imputer_median"])

        z = (float(val) - float(row["scaler_mean"])) / float(row["scaler_scale"])
        logit += float(row["coef_on_scaled"]) * z

    probability = 1 / (1 + np.exp(-logit))
    return float(probability)

if st.button("Calculate Probability", type="primary"):
    st.markdown("---")
    st.subheader("Results")

    knees_to_display = get_knees_to_assess(knee_selection)

    if not knees_to_display:
        st.error("No knees selected. Please select at least one knee.")
    else:
        def render_knee_result(container, knee: str, *, show_badge: bool) -> None:
            knee_surgery = st.session_state.get(f"surgery_{knee}", False)
            knee_trauma = st.session_state.get(f"trauma_{knee}", False)

            prob = calculate_probability(
                age=age,
                bmi=bmi,
                whr=whr,
                occupation=occupation,
                race=race,
                surgery=knee_surgery,
                trauma=knee_trauma,
            )

            with container:
                if show_badge:
                    st.markdown(
                        f"<div style=\"display:flex;justify-content:center;\">{knee_badge_html(knee)}</div>",
                        unsafe_allow_html=True,
                    )
                st.metric(
                    label=(
                        "Probability of KOA"
                        if show_badge
                        else f"{knee} Knee - Probability of KOA"
                    ),
                    value=f"{prob:.1%}",
                    help="Probability of radiographic KOA (KL >= 2)",
                )
                st.write(
                    f"**Input data:** Age {age}y | BMI {bmi:.1f} | "
                    f"Surgery: {'Yes' if knee_surgery else 'No'} | "
                    f"Trauma: {'Yes' if knee_trauma else 'No'}"
                )

        # Side-by-side when both knees are selected (RIGHT shown on LEFT side)
        if set(knees_to_display) == {"Left", "Right"}:
            st.markdown(radiographic_view_html(), unsafe_allow_html=True)
            col_r, col_l = st.columns(2)
            with col_r:
                st.markdown(knee_column_header_html("Right", justify="flex-end"), unsafe_allow_html=True)
                spacer, results = st.columns([1, 3])
                render_knee_result(results, "Right", show_badge=False)

            with col_l:
                st.markdown(knee_column_header_html("Left", justify="flex-start"), unsafe_allow_html=True)
                results, spacer = st.columns([3, 1])
                render_knee_result(results, "Left", show_badge=False)
        else:
            render_knee_result(st.container(), knees_to_display[0], show_badge=True)
        
st.markdown("---")
st.warning(
    """**Research use only. Not for clinical decision-making.**

- The model was developed and validated **internally only**, in a single cohort of Brazilian civil
  servants aged 38-79. It has never been tested in another population, and its performance elsewhere
  is unknown.
- It estimates whether radiographic osteoarthritis is **present now**. It says nothing about whether
  osteoarthritis will develop, nor about progression.
- Radiographic osteoarthritis and symptoms are frequently discordant. A high estimated probability
  does not mean a knee is painful, and does not by itself indicate any treatment.
- No threshold has been established at which a knee radiograph should be obtained, so this tool
  defines no decision rule.
- It does not replace clinical assessment or professional medical advice.
"""
)

st.markdown("---")
st.subheader("OARSI 2026 Late-Breaking Abstract")

st.markdown("""
**Title:**
A CLINICAL-EPIDEMIOLOGICAL TOOL FOR IDENTIFYING PREVALENT RADIOGRAPHIC KNEE OSTEOARTHRITIS: DATA FROM ELSA-BRASIL MSK

**Authors:** Domingues JG, Veloso AA, Telles RW, Barreto SM.

**Category:** Epidemiology, Clinical Aspects & Outcomes
""")

with st.expander("Purpose", expanded=False):
    st.write(
        "In certain settings, evaluation of patients' knee symptoms by clinical parameters may be preferable to or "
        "more feasible than obtaining radiographs to diagnose knee OA (KOA) or estimate its prevalence in "
        "epidemiologic studies. We aimed to develop a parsimonious clinical-epidemiological tool to estimate the "
        "probability of prevalent radiographic KOA (rKOA) using routinely obtainable variables in a large "
        "population-based cohort."
    )

with st.expander("Methods", expanded=False):
    st.write(
        "We used data from the ELSA-Brasil Musculoskeletal Study (2012-2014), a cohort of civil servants, ages "
        "38-74 unselected for any medical conditions. All participants underwent bilateral fixed-flexion PA and "
        "lateral knee radiography. The outcome was prevalent rKOA, defined as Kellgren-Lawrence grade >= 2 in the "
        "tibiofemoral compartment and/or definite patellofemoral OA."
    )
    st.write(
        "We evaluated candidate variables spanning demographic, anthropometric, clinical, mechanical, metabolic, "
        "and lifestyle domains. Variable selection followed a staged, interpretability-oriented strategy in "
        "knee-level analyses with participant-level clustering."
    )
    st.write(
        "We first used L1-penalized logistic regression (LASSO) to rank variables by importance. We then identified "
        "the optimal number of variables by maximizing the AUC with a penalty for model complexity (0.1% per "
        "variable). From among this reduced set of variables, we performed a forward stepwise procedure until the "
        "subsequent addition of the next variable did not increase the AUC gain by a further 0.5% to identify a "
        "final parsimonious model. We also evaluated tree-based algorithms (Random Forest and XGBoost)."
    )
    st.write(
        "Internal validation was undertaken using 5-fold cross-validation with participant-level grouping to account "
        "for bilateral dependency. Discrimination was quantified by the area under the ROC curve (AUC), with "
        "confidence intervals derived from bootstrapping pooled predictions. Calibration was additionally assessed "
        "using the Brier score."
    )

with st.expander("Results", expanded=False):
    st.write(
        "Of the 2830 participants (mean age 56, 52% female, 18% KOA), there were 5,652 knee radiographs available "
        "for analysis (8 excluded due to arthroplasty or technical artifacts). From the initial 55 variables, 42 were "
        "selected by the LASSO model. Subsequently, the complexity-penalized selection followed by the forward "
        "selection process identified a 5-variable model as both parsimonious and optimal in terms of performance "
        "with good discrimination (pooled AUC 0.815; 95% CI 0.799-0.829) and calibration (Brier score 0.093) "
        "(Figure 1A); adding more variables did not increase AUC by >= 0.5% (Figure 1B). This final model included "
        "age, body mass index (BMI), frequent knee symptoms, history of knee surgery, and history of knee trauma. "
        "Complex machine learning approaches did not improve performance, as Random Forest (AUC 0.796) and XGBoost "
        "(AUC 0.776) did not outperform the simple logistic model in this dataset."
    )
    st.write(
        "All selected variables were independently associated with rKOA (p < 0.0001). History of knee surgery had "
        "the strongest association (OR 7.11; 95% CI 4.83-10.43), followed by frequent symptoms (OR 2.46; 95% CI "
        "1.98-3.04) and past trauma (OR 2.08; 95% CI 1.64-2.64). In terms of contribution to discriminative power, "
        "Age (standardized OR (sOR) 2.20; 95% CI 1.97-2.46) and BMI (sOR 1.79; 95% CI 1.63-1.97) were the strongest "
        "drivers of AUC (Figure 1B)."
    )

with st.expander("Conclusions", expanded=False):
    st.write(
        "A simple clinical-epidemiological tool using 5 routinely available variables effectively identifies knees "
        "with a high probability of prevalent rKOA and outperformed more complex (including machine learning) "
        "models. Of note, sex was not retained in the final model, likely reflecting its effects being captured "
        "within the clinical parameters. This tool may support efficient triage and prioritization of imaging in "
        "epidemiological studies and healthcare settings with limited access to radiography."
    )

st.markdown("**Figures and Tables**")

table_df = pd.DataFrame(
    {
        "Characteristic": [
            "Age (years) (Mean +/- SD)",
            "Sex (Female) (%)",
            "BMI (kg/m^2) (Mean +/- SD)",
            "Frequent Knee Symptoms (%)",
            "History of Knee Trauma (%)",
            "History of Knee Surgery (%)",
        ],
        "Total (N=2830)": ["56 +/- 9", "53", "27 +/- 5", "23", "24", "7"],
        "No rKOA (N=2318)": ["55 +/- 9", "52", "27 +/- 5", "19", "20", "3"],
        "With rKOA (N=512)": ["61 +/- 8", "55", "29 +/- 5", "45", "43", "22"],
        "P-value": ["<0.001", "0.4", "<0.001", "<0.001", "<0.001", "<0.001"],
    }
)

st.caption("Table 1: Demographic and clinical characteristics of participants by rKOA status")
st.dataframe(table_df, use_container_width=True, hide_index=True)
st.caption("Note: P-values derived from independent t-tests (continuous) and Chi-square tests (categorical).")

if FIGURE_1_PATH.exists():
    st.image(
        str(FIGURE_1_PATH),
        caption=(
            "Figure 1: (A) Discriminative performance of the models (B) Incremental gain in AUC for selected "
            "variables. Filled circles: variables included in the final model. Open circles: variables not included."
        ),
        use_container_width=True,
    )
else:
    st.caption(
        "Figure 1: (A) Discriminative performance of the models (B) Incremental gain in AUC for selected "
        "variables. Filled circles: variables included in the final model. Open circles: variables not included."
    )

st.markdown("---")
st.markdown(
    """
    <div style="display:flex; justify-content:center; margin-top:4px; margin-bottom:2px;">
        <a href="https://www.linkedin.com/in/juliogd" target="_blank" style="
            display:inline-flex;
            align-items:center;
            gap:8px;
            text-decoration:none;
            padding:9px 14px;
            border-radius:999px;
            background:#0a66c2;
            color:#ffffff;
            font-weight:600;
            letter-spacing:0.1px;
            box-shadow:0 6px 18px rgba(10, 102, 194, 0.18);
        ">
            <span style="
                display:inline-flex;
                align-items:center;
                justify-content:center;
                width:20px;
                height:20px;
                border-radius:4px;
                background:#ffffff;
                color:#0a66c2;
                font-weight:800;
                font-size:12px;
                line-height:1;
            ">in</span>
            <span>Júlio Domingues on LinkedIn</span>
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)
