import streamlit as st
import pandas as pd
import altair as alt

# -------------------------------
# Setup and Configuration
# -------------------------------
st.set_page_config(page_title="ML Explanation", layout="wide", initial_sidebar_state="expanded")
st.title("Machine Learning Analysis")
st.markdown(
    "<p style='font-size:20px;'>This technical document compares the performance of various Machine Learning regression models developed for accurately predicting the Glass Transition Temperature (<b>Tg</b>) of polymers.</p>",
    unsafe_allow_html=True
)

# -------------------------------
# Functional Group Mode Overview
# -------------------------------
st.header("1. Functional Group Mode (Chemically Engineered Features)")
st.markdown(
    "<p style='font-size:20px;'>The <b>Functional Group Mode</b> uses features manually created from chemistry knowledge. This makes the models highly interpretable and directly connects the results to fundamental polymer chemistry.</p>",
    unsafe_allow_html=True
)

st.subheader("1.1 Input Feature Space")
st.markdown(
    """
    <p style='font-size:20px;'>
    The input features are based on polymer structure and thermal behavior: counts of specific functional groups (#CH3, #CH2, #CH, #C, #OH, #C-O-C, #O=C, DBA, #N, #Hal),
    along with global properties like molecular weight (M), melting temperature (Tm), and the Oxygen-to-Carbon ratio (O:C).
    </p>
    """,
    unsafe_allow_html=True
)

# Functional Group Mode metrics
fg_metrics = pd.DataFrame({
    "Model": ["RandomForest", "ExtraTrees", "GradientBoosting"],
    "MSE": [150.16, 141.68, 120.16],
    "R2": [0.9825, 0.9834, 0.9859],
    "RMSE": [12.25, 11.91, 10.97]
})

st.subheader("1.2 Model Performance Metrics")
metric_choice = st.selectbox("Choose Metric to Visualize:", ('MSE', 'R2', 'RMSE'), key='fg_metric_select')

st.markdown(
    """
    <p style='font-size:20px;'>
    The performance of these models is evaluated using three standard regression metrics:
    <ul>
        <li><b>R2 (Coefficient of Determination):</b> Shows the fraction of the Tg variance explained by the model's features. A value close to 1.0 means a very strong fit.</li>
        <li><b>MSE (Mean Squared Error):</b> The average of the squared prediction errors. Lower scores mean better accuracy.</li>
        <li><b>RMSE (Root Mean Squared Error):</b> The standard deviation of the prediction errors, reported in the same units as Tg (Kelvin). It indicates the typical size of a prediction error. Lower is better.</li>
    </ul>
    </p>
    """,
    unsafe_allow_html=True
)
st.dataframe(fg_metrics)

# Altair chart — rotate x-axis labels 90°
chart_color = "#2a9d8f" if metric_choice in ('MSE', 'RMSE') else "#e76f51"
fg_chart = alt.Chart(fg_metrics).mark_bar(color=chart_color).encode(
    x=alt.X('Model', axis=alt.Axis(labelAngle=0), title='Model'),
    y=alt.Y(metric_choice, title=metric_choice),
    tooltip=['Model', 'MSE', 'R2', 'RMSE']
).properties(title=f"Functional Group Mode — {metric_choice} Comparison", width=700, height=400)
st.altair_chart(fg_chart, use_container_width=True)

st.markdown(
    """
    <p style='font-size:20px;'>
    <b>Interpretation:</b> The Gradient Boosting Regressor performed the best, achieving the lowest MSE (120.16) 
    and a very high R<sup>2</sup> (0.9859). The RMSE of 10.97 K shows that, on average, the model's Tg predictions 
    are within &plusmn; 11 K of the measured values. This confirms that using chemically defined features is highly 
    effective for predicting Tg.
    </p>
    """,
    unsafe_allow_html=True
)

st.header("1.3 Feature Importance Analysis (Gradient Boosting)")
feature_importance = pd.DataFrame({
    "Feature": ['#CH3', '#CH2', '#CH', '#C', '#OH', '#C-O-C', '#O=C', 'DBA', '#N', '#Hal', 'O:C', 'M', 'Tm'],
    "Importance": [0.10,0.08,0.06,0.05,0.12,0.07,0.09,0.08,0.05,0.04,0.06,0.10,0.10]
})

importance_chart = alt.Chart(feature_importance).mark_bar(color="#2a9d8f").encode(
    x=alt.X('Importance', title='Feature Importance Score (Normalized)'),
    y=alt.Y('Feature', sort='-x'),
    tooltip=['Feature', 'Importance']
).properties(width=700, height=500, title="Gradient Boosting — Feature Importance for Tg Prediction")
st.altair_chart(importance_chart, use_container_width=True)

st.markdown(
    """
    <p style='font-size:20px;'>
    <b>Key Influencers:</b> The analysis highlights the count of OH groups and Melting Temperature (Tm) as the most important predictors. This makes sense in polymer science: hydrogen bonding (OH) and chain stiffness (Tm) are the main drivers of glass transition. Molecular Weight (M) and CH3 counts are also relevant, reflecting their impact on chain length and end-group movement.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------------------
# SMILES Mode Overview
# -------------------------------
st.header("2. SMILES Mode (Textual/Representation Learning)")
st.markdown(
    "<p style='font-size:20px;'>The <b>SMILES Mode</b> uses a fully automated feature approach by extracting 2–3 character n-grams directly from the polymer's SMILES strings. This method requires no manual input, serving as a powerful, feature-free baseline.</p>",
    unsafe_allow_html=True
)

st.subheader("2.1 Model and Input")
st.markdown(
    "<p style='font-size:20px;'>Input features: A large vector of character n-grams from SMILES strings. The best performing model was the ExtraTreesRegressor.</p>",
    unsafe_allow_html=True
)

# SMILES Mode metrics
smiles_metrics = pd.DataFrame({
    "Model": ["RandomForest", "ExtraTrees", "GradientBoosting"],
    "MSE": [562.33, 459.25, 532.33],
    "R2": [0.9343, 0.9463, 0.9378],
    "RMSE": [23.72, 21.43, 23.07],
    "MAE": [13.45, 12.11, 12.90]
})

st.subheader("2.2 Model Metrics")

st.markdown(
    """
    <div style='font-size:20px;'>
        The SMILES mode also uses the <b>Mean Absolute Error (MAE)</b>:
        <ul style='font-size:20px;'>
            <li><b>MAE (Mean Absolute Error):</b> The average absolute difference between predicted and actual values. Like RMSE, it is in Kelvin, but MAE is less sensitive to extreme errors (outliers). Lower is better.</li>
            <li><b>R2, MSE, and RMSE:</b> These metrics are used as previously defined to quantify model variance explained and error magnitude.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)


st.dataframe(smiles_metrics)

# --- SMILES Mode — Charts Side-by-Side ---
smiles_mse_chart = alt.Chart(smiles_metrics).mark_bar(color="#2a9d8f").encode(
    x=alt.X("Model", axis=alt.Axis(labelAngle=0), title='Model'),
    y=alt.Y("MSE", title="Mean Squared Error (MSE)"),
    tooltip=["Model", "MSE", "R2", "RMSE", "MAE"]
).properties(title="SMILES Mode — MSE Comparison")

smiles_r2_chart = alt.Chart(smiles_metrics).mark_bar(color="#e76f51").encode(
    x=alt.X("Model", axis=alt.Axis(labelAngle=0), title='Model'),
    y=alt.Y("R2", title="R² Score (Variance Explained)"),
    tooltip=["Model", "MSE", "R2", "RMSE", "MAE"]
).properties(title="SMILES Mode — R² Comparison")

col1, col2 = st.columns(2)

with col1:
    st.altair_chart(smiles_mse_chart, use_container_width=True)

with col2:
    st.altair_chart(smiles_r2_chart, use_container_width=True)
# ----------------------------------------

st.markdown(
    """
    <p style='font-size:20px; line-height:1.5;'>
        <b>Interpretation:</b> The ExtraTrees Regressor performed best here 
        (R² = 0.9463, RMSE = 21.43 K). This shows that simple SMILES features 
        can explain about 94.6% of the Tg variability.<br><br>
        Although the RMSE is higher than the Functional Group Mode, this 
        automatic method is still accurate and highly scalable for analyzing 
        new polymers quickly.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------------------
# Comparative Analysis
st.header("3. Comparative Analysis: Best Models")
st.markdown(
    "<p style='font-size:20px;'>A direct comparison between the most accurate model from each approach.</p>", unsafe_allow_html=True
)

comparison_df = pd.DataFrame({
    "Mode": ["Functional Group (GradientBoosting)", "SMILES (ExtraTrees)"],
    "Best R2": [fg_metrics.loc[fg_metrics['Model'] == 'GradientBoosting', 'R2'].iloc[0],
                smiles_metrics.loc[smiles_metrics['Model'] == 'ExtraTrees', 'R2'].iloc[0]],
    "Best RMSE (K)": [fg_metrics.loc[fg_metrics['Model'] == 'GradientBoosting', 'RMSE'].iloc[0],
                      smiles_metrics.loc[smiles_metrics['Model'] == 'ExtraTrees', 'RMSE'].iloc[0]]
})
st.dataframe(comparison_df)

st.markdown(
    """
    <p style='font-size:20px;'>
    <b>Key Findings:</b> The Functional Group Mode provides significantly better accuracy (RMSE &asymp; 11 K) 
    compared to the SMILES Mode (RMSE &asymp; 21 K). This difference highlights the importance of including 
    fundamental chemical knowledge and thermal properties (Tm) in the model's feature set. The Functional Group 
    approach offers both high precision and clear interpretability, which are vital for directing future polymer 
    design work.
    </p>
    """,
    unsafe_allow_html=True
)


st.markdown("---")

# -------------------------------
# Conclusion and Future Work
# -------------------------------
st.header("4. Project Conclusion")
st.success(
    "The Functional Group Mode, using the Gradient Boosting Regressor, is the top recommendation for high-precision Tg prediction in this research. Its strong performance (R2 = 0.9859, RMSE = 10.97 K) establishes a high benchmark, and its feature importance analysis provides direct evidence for the relationship between chemical structure and polymer properties."
)

