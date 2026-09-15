import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURAÇÃO DO MODELO (carregado de CSV)
# ==========================================
MODEL_FILES = {
    "Constitutional": Path("final_model.csv"),
    "Symptom-Augmented": Path("symptom_augmented_model.csv"),
}

# Knee-level variables that only the Symptom-Augmented model uses.
SYMPTOM_FEATURES = ["frequent_symptoms", "knee_disability", "recent_pain_7d"]


@st.cache_data
def load_model_params(path_str: str):
    """Load one model specification: intercept plus per-variable parameters."""
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Model CSV not found: {path}")

    df = pd.read_csv(path)
    intercept = float(df.loc[df["feature"] == "__INTERCEPT__", "intercept"].iloc[0])
    features_df = df.loc[
        df["param_type"] == "feature",
        ["feature", "imputer_median", "scaler_mean", "scaler_scale", "coef_on_scaled"],
    ].copy()
    return intercept, features_df


MODELS = {name: load_model_params(str(p)) for name, p in MODEL_FILES.items()}

# ==========================================
# INTERFACE DO USUÁRIO
# ==========================================
st.set_page_config(page_title="Kneelsa-Clinical", page_icon="🦵")

st.title("KNEELSA: estimated probability of prevalent radiographic knee osteoarthritis")
st.markdown("""
This tool implements the eight-variable Constitutional logistic model developed in the
ELSA-Brasil Musculoskeletal Study. For a knee assessed **now**, it estimates the probability that
radiographic knee osteoarthritis is **already present** (Kellgren-Lawrence grade 2 or higher in the
tibiofemoral or patellofemoral joint).

It does not estimate the chance of developing osteoarthritis in the future: the study behind it is
cross-sectional, so the model describes present structural status only.
""")

st.markdown("---")

# Patient Demographics (same for both knees)
model_name = st.radio(
    "Model",
    list(MODEL_FILES),
    horizontal=True,
    help=(
        "Constitutional uses eight characteristics that do not depend on current symptoms. "
        "Symptom-Augmented was selected with three self-reported knee symptom items also "
        "available: it keeps seven of the Constitutional variables and adds those items, and so answers a "
        "different question: how well structural disease is identified once the clinical "
        "presentation is already known."
    ),
)
if model_name == "Symptom-Augmented":
    st.caption(
        "The Symptom-Augmented model requires symptom information for each knee. Its "
        "discrimination was 0.821 against 0.811 for the Constitutional model."
    )
INTERCEPT, MODEL_FEATURES = MODELS[model_name]
USES_SYMPTOMS = model_name == "Symptom-Augmented"

st.markdown("---")

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
    "Race and skin color (self-reported)",
    ["White", "Brown/Mixed", "Black", "Asian", "Indigenous"],
    help="Self-reported using the Brazilian census (IBGE) categories.",
)

# Person-level variable that only the Constitutional model uses.
squatting = False
if not USES_SYMPTOMS:
    squatting = st.checkbox(
        "Squatting for 30 minutes or more in a single day?",
        help="In the last 30 days, squatted for 30 minutes or more in a single day.",
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

SYMPTOM_LABELS = {
    "frequent_symptoms": ("Frequent knee symptoms?",
                          "Pain, discomfort or stiffness on most days for at least one month "
                          "in the last 12 months"),
    "recent_pain_7d": ("Knee symptoms in the last 7 days?",
                       "Pain, discomfort or stiffness in this knee in the last 7 days"),
    "knee_disability": ("Activity limitation from this knee?",
                        "Knee pain, discomfort or stiffness that prevented normal activities "
                        "in the last 12 months"),
}


def symptom_checkboxes(knee: str) -> None:
    """Render the three symptom items for one knee, Symptom-Augmented model only."""
    if not USES_SYMPTOMS:
        return
    for key, (label, help_text) in SYMPTOM_LABELS.items():
        st.checkbox(label, key=f"{key}_{knee}", help=help_text)


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
    symptom_checkboxes(knee)
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
        symptom_checkboxes("Right")
    
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
        symptom_checkboxes("Left")


st.markdown("---")

# ==========================================
# CÁLCULO
# ==========================================
def calculate_probability(age, bmi, whr, occupation, race, surgery, trauma,
                          squatting=False,
                          frequent_symptoms=False, knee_disability=False, recent_pain=False):
    """Calculate the probability of KOA for a single knee using the saved preprocessing + LR params."""
    x = {
        "age": float(age),
        "bmi": float(bmi),
        "waist_hip_ratio": float(whr),
        "history_surgery": 1.0 if surgery else 0.0,
        "history_trauma": 1.0 if trauma else 0.0,
        "occupation_4": 1.0 if occupation == "Non-routine non-manual" else 0.0,
        "race_raw_3": 1.0 if race == "White" else 0.0,
        "occ_squatting": 1.0 if squatting else 0.0,
        "frequent_symptoms": 1.0 if frequent_symptoms else 0.0,
        "knee_disability": 1.0 if knee_disability else 0.0,
        "recent_pain_7d": 1.0 if recent_pain else 0.0,
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
                squatting=squatting,
                frequent_symptoms=st.session_state.get(f"frequent_symptoms_{knee}", False),
                knee_disability=st.session_state.get(f"knee_disability_{knee}", False),
                recent_pain=st.session_state.get(f"recent_pain_7d_{knee}", False),
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
