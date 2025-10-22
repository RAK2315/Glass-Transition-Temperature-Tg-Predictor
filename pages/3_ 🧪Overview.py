import streamlit as st
import pandas as pd

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Tg Prediction Research Homepage",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. HEADER ---
st.title("Predicting Glass Transition Temperature (Tg)")

# --- 3. ABSTRACT ---
st.header("Abstract")
st.markdown(
    """
    <p style='font-size:18px;'>
    The Glass Transition Temperature (Tg) is a key property for polymers, dictating mechanical behavior and thermal stability. 
    Experimental determination of Tg is costly and time-consuming. This project compares two Machine Learning methodologies for rapid Tg prediction:
    </p>
    <ul style='font-size:16px;'>
        <li><b>Functional Group Mode:</b> Uses manually engineered, chemically meaningful descriptors such as -OH counts, Molecular Weight (M), Tm, and O:C ratio.</li>
        <li><b>SMILES Mode:</b> Uses automated textual features (2–3 character n-grams from SMILES strings) as a feature-free baseline.</li>
    </ul>
    <p style='font-size:18px;'>
    Using a dataset of ~700 unique polymers, the Functional Group Mode achieved superior performance (R² ≈ 0.986, RMSE ≈ 11 K) with Gradient Boosting Regressor, demonstrating the power of physics-informed features for accurate and interpretable predictions.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# --- 4. INTRODUCTION AND MOTIVATION ---
st.header("1. Introduction and Motivation")

st.subheader("1.1 What is Tg?")
st.markdown(
    """
    <p style='font-size:18px;'>
    The Glass Transition Temperature (Tg) is the temperature at which polymers transition from a rigid, glassy state to a flexible, rubbery state. 
    This reversible transition determines the operational temperature range and mechanical behavior of polymer-based materials, influencing their suitability for engineering and commercial applications.
    </p>
    """,
    unsafe_allow_html=True
)

st.subheader("1.2 Why Predict Tg?")
st.markdown(
    """
    <p style='font-size:18px;'>
    Synthesizing and testing polymers to measure Tg is slow and resource-intensive. Machine Learning allows rapid prediction of Tg from molecular structure, enabling high-throughput screening and prioritization of promising candidates, which accelerates materials discovery and design.
    </p>
    """,
    unsafe_allow_html=True
)

# --- 5. METHODOLOGY ---
st.header("2. Methodology: Feature Engineering Approaches")
st.markdown(
    """
    <p style='font-size:18px;'>
    We investigated whether physics-informed features outperform purely data-driven features. Two separate feature modes were compared:
    </p>
    """,
    unsafe_allow_html=True
)

st.subheader("2.1 Functional Group Mode (Physics-Informed)")
st.markdown(
    """
    <p style='font-size:18px;'>
    Uses domain knowledge from polymer chemistry to construct interpretable features:
    </p>
    <ul style='font-size:16px;'>
        <li><b>Features:</b> Counts of 11 functional groups (#CH3, #CH2, #CH, #C, #OH, #C-O-C, #O=C, DBA, #N, #Hal) plus Molecular Weight (M), Oxygen-to-Carbon ratio (O:C), and Melting Temperature (Tm).</li>
        <li><b>Rationale:</b> Features such as #OH count capture hydrogen bonding effects; Tm and M reflect chain stiffness and flexibility, which influence Tg.</li>
    </ul>
    """,
    unsafe_allow_html=True
)

st.subheader("2.2 SMILES Mode (Data-Driven Baseline)")
st.markdown(
    """
    <p style='font-size:18px;'>
    Uses purely data-driven character-level features extracted from SMILES strings:
    </p>
    <ul style='font-size:16px;'>
        <li><b>Features:</b> Character n-grams (2–3 characters) from SMILES strings converted to count vectors.</li>
        <li><b>Rationale:</b> No manual chemical input is needed. Model learns structural patterns directly from the string, but features lack direct physical interpretability.</li>
    </ul>
    """,
    unsafe_allow_html=True
)

# --- 6. KEY RESULTS ---
st.header("3. Key Results Overview")
st.markdown(
    "<p style='font-size:18px;'>Regression performance was evaluated using R², MSE, and RMSE.</p>",
    unsafe_allow_html=True
)

st.subheader("3.1 Comparative Performance")
comparison_data = {
    "Mode": ["Functional Group", "SMILES"],
    "Best Model": ["Gradient Boosting Regressor", "Extra Trees Regressor"],
    "R² Score": ["0.9859", "0.9463"],
    "RMSE (K)": ["10.97", "21.43"]
}
df_comparison = pd.DataFrame(comparison_data).set_index("Mode")
st.dataframe(df_comparison)
 
st.subheader("3.2 Understanding Error Metrics")
st.markdown(
    """
    <p style='font-size:18px;'>
    To interpret the model performance, the following metrics are used:
    </p>
    <ul style='font-size:16px;'>
        <li><b>Mean Squared Error (MSE):</b> The average of squared differences between predicted and actual Tg values. Lower MSE indicates predictions are closer to the experimental values. Highly sensitive to large errors.</li>
        <li><b>Root Mean Squared Error (RMSE):</b> Square root of MSE, expressed in the same units as Tg (Kelvin). Provides a direct sense of average prediction error. Lower is better.</li>
        <li><b>R² Score (Coefficient of Determination):</b> Indicates the proportion of variance in Tg explained by the model. R² closer to 1 means the model explains more of the variance.</li>
        <li><b>Mean Absolute Error (MAE, optional for SMILES mode):</b> Average absolute difference between predicted and actual Tg. Less sensitive to outliers than RMSE, provides an intuitive error magnitude.</li>
    </ul>
    <p style='font-size:18px;'>
    These metrics together provide a full picture of prediction accuracy, precision, and reliability of the models.
    </p>
    """,
    unsafe_allow_html=True
)


st.subheader("3.3 Importance of Interpretable Features")
st.markdown(
    """
    <p style='font-size:18px;'>
    Feature Importance from the Gradient Boosting model highlights the physics behind the predictions:
    </p>
    <ul style='font-size:16px;'>
        <li><b>#OH count:</b> Highest importance, confirming the role of hydrogen bonding in restricting chain mobility and raising Tg.</li>
        <li><b>Melting Temperature (Tm):</b> Strong indicator of chain stiffness and thermal energy barrier for phase transition.</li>
    </ul>
    <p style='font-size:18px;'>
    This interpretability allows rational design of new polymeric materials with target Tg values.
    </p>
    """,
    unsafe_allow_html=True
)

# --- 7. CONCLUSION ---
st.header("4. Project Conclusion")
st.markdown(
    """
    <p style='font-size:18px;'>
    Machine Learning is a powerful tool for polymer physics prediction. The Functional Group Mode (Gradient Boosting) outperforms SMILES-only models, achieving roughly twice the accuracy (RMSE ≈ 11 K vs 21 K). 
    High feature importance of chemical descriptors like #OH count and Tm confirms that physics-informed feature engineering is crucial for reliable and interpretable Tg predictions.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")
