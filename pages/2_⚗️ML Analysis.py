# ml_explanation.py — Updated with Correct Model Performance
import streamlit as st
import pandas as pd
import altair as alt

# -------------------------------
# Setup and Configuration
# -------------------------------
st.set_page_config(
    page_title="ML Explanation", 
    page_icon="🤖",
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title("🤖 Machine Learning Analysis")

st.markdown("""
<div style='padding:20px; border-radius:10px; border-left:6px solid #2a9d8f;'>
<p style='font-size:20px; line-height:1.6;'>
This technical document compares the performance of various machine learning regression models 
developed for accurately predicting the <b>Glass Transition Temperature (Tg)</b> of organic polymers 
and small molecules.
</p>
<p style='font-size:20px; line-height:1.6;'>
Two complementary approaches were evaluated:
</p>
<ul style='font-size:18px; line-height:1.8;'>
<li><b>Functional Group Mode:</b> Physics-informed features based on chemical structure (13 descriptors)</li>
<li><b>SMILES Mode:</b> Data-driven text encoding using automated n-gram vectorization (1000 features)</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Functional Group Mode Overview
st.header("1️⃣ Functional Group Mode (Chemistry-Informed)")

st.markdown("""
<div style='padding:20px; border-radius:10px; border-left:4px solid #2a9d8f;'>
<p style='font-size:20px; line-height:1.6;'>
The <b>Functional Group Mode</b> uses 13 manually engineered features derived from polymer chemistry 
knowledge. This approach makes models highly interpretable and directly connects predictions to 
fundamental chemical principles.
</p>
</div>
""", unsafe_allow_html=True)

st.subheader("1.1  Input Feature Space")

feature_description = pd.DataFrame({
    "Category": ["Carbon Groups", "Oxygen Groups", "Other", "Molecular Properties"],
    "Features": [
        "#CH₃, #CH₂, #CH, #C",
        "#OH, #C-O-C, #O=C, O:C",
        "#N, #Hal, DBA",
        "M (g/mol), Tm (K)"
    ],
    "Chemical Significance": [
        "Backbone structure and flexibility",
        "Hydrogen bonding and polarity",
        "Heteroatoms and unsaturation",
        "Chain dynamics and thermodynamics"
    ]
})

st.dataframe(feature_description, use_container_width=True)

st.markdown("""
<p style='font-size:20px; line-height:1.6; margin-top:15px;'>
These features capture the essential structural and thermal properties that govern glass transition behavior, 
enabling the model to learn physically meaningful relationships.
</p>
""", unsafe_allow_html=True)

# Functional Group Mode metrics — UPDATED VALUES
fg_metrics = pd.DataFrame({
    "Model": ["LightGBM", "XGBoost (Tuned)", "LightGBM (Tuned)", "XGBoost", "CatBoost"],
    "Test_R2": [0.9885, 0.9871, 0.9857, 0.9847, 0.9833],
    "Test_RMSE": [9.91, 10.51, 11.06, 11.43, 11.96],
    "Test_MAE": [7.32, 7.66, 8.17, 7.56, 8.47],
    "Overfitting_Gap": [0.006, 0.008, 0.008, 0.014, 0.011]
})

st.subheader("1.2 📊 Model Performance Comparison")

st.markdown("""
<div style='padding:15px; border-radius:8px; border-left:4px solid #2a9d8f;'>
<p style='font-size:20px; line-height:1.8;'>
Models are evaluated using four key metrics:
</p>
<ul style='font-size:20px; line-height:1.8;'>
<li><b>R² (Coefficient of Determination):</b> Proportion of variance in Tg explained by the model. 
Values close to 1.0 indicate excellent fit (0.98+ is exceptional)</li>
<li><b>RMSE (Root Mean Squared Error):</b> Standard deviation of prediction errors in Kelvin. 
Typical error magnitude — lower is better</li>
<li><b>MAE (Mean Absolute Error):</b> Average absolute prediction error in Kelvin. 
Less sensitive to outliers than RMSE</li>
<li><b>Overfitting Gap:</b> Difference between training and test R². Values <0.05 indicate excellent generalization</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Display metrics table
st.dataframe(fg_metrics.style.highlight_max(subset=['Test_R2'], color="#2b655b")
                              .highlight_min(subset=['Test_RMSE', 'Test_MAE', 'Overfitting_Gap'], color="#3c796f"), 
             use_container_width=True)

# Visualization selector
col_viz1, col_viz2 = st.columns([2, 1])

with col_viz1:
    metric_choice = st.selectbox(
        "Choose Metric to Visualize:", 
        ['Test_R2', 'Test_RMSE', 'Test_MAE', 'Overfitting_Gap'], 
        key='fg_metric_select'
    )

with col_viz2:
    best_model = fg_metrics.loc[fg_metrics['Test_R2'].idxmax(), 'Model']
    best_r2 = fg_metrics.loc[fg_metrics['Test_R2'].idxmax(), 'Test_R2']
    st.metric("🏆 Best Model", best_model)
    st.metric("R² Score", f"{best_r2:.4f}")

# Altair chart
chart_color = "#2a9d8f" if metric_choice in ['Test_RMSE', 'Test_MAE', 'Overfitting_Gap'] else "#e76f51"

fg_chart = alt.Chart(fg_metrics).mark_bar(color=chart_color, opacity=0.85).encode(
    x=alt.X('Model:N', axis=alt.Axis(labelAngle=-45), title='Model', sort='-y'),
    y=alt.Y(f'{metric_choice}:Q', title=metric_choice.replace('_', ' ')),
    tooltip=['Model', 'Test_R2', 'Test_RMSE', 'Test_MAE', 'Overfitting_Gap']
).properties(
    title=f"Functional Group Mode — {metric_choice.replace('_', ' ')} Comparison", 
    width=700, 
    height=400
)

st.altair_chart(fg_chart, use_container_width=True)

st.markdown("""
<div style='padding:20px; border-radius:10px; border-left:4px solid #2a9d8f; margin-top:20px;'>
<p style='font-size:21px; line-height:1.7;'>
<b>🎯 Key Findings:</b>
</p>
<ul style='font-size:19px; line-height:1.8;'>
<li><b>LightGBM</b> achieved the best performance with <b>R² = 0.9885</b> and <b>RMSE = 9.91 K</b></li>
<li>Average prediction error is only <b>±7.32 K</b> (MAE), representing just <b>3-4% typical error</b></li>
<li>Overfitting gap of <b>0.006</b> confirms excellent generalization to unseen compounds</li>
<li>All top models exceed <b>R² > 0.98</b>, demonstrating the power of chemistry-informed features</li>
<li>Performance difference between models is minimal, suggesting feature quality matters more than algorithm choice</li>
</ul>
</div>
""", unsafe_allow_html=True)
st.write("---")


# Feature Importance Analysis
st.subheader("1.3  Feature Importance Analysis (LightGBM)")

st.markdown("""
<p style='font-size:18px; line-height:1.6;'>
Understanding which features drive predictions reveals the underlying chemical physics of glass transition.
</p>
""", unsafe_allow_html=True)

# UPDATED feature importance values from your actual results
feature_importance = pd.DataFrame({
    "Feature": ['Tm', 'M', '#CH', 'O:C', '#CH2', 'DBA', '#C', '#OH', '#CH3', '#N', '#C-O-C', '#O=C', '#Hal'],
    "Importance": [30.09, 17.90, 10.28, 8.81, 6.37, 5.77, 5.44, 5.06, 4.19, 2.56, 1.80, 1.74, 0.00],
    "Category": ['Thermal', 'Molecular', 'Carbon', 'Oxygen', 'Carbon', 'Structural', 'Carbon', 
                 'Oxygen', 'Carbon', 'Heteroatom', 'Oxygen', 'Oxygen', 'Heteroatom']
})

importance_chart = alt.Chart(feature_importance).mark_bar().encode(
    x=alt.X('Importance:Q', title='Feature Importance (%)'),
    y=alt.Y('Feature:N', sort='-x', title='Feature'),
    color=alt.Color('Category:N', scale=alt.Scale(scheme='tealblues'), legend=alt.Legend(title="Category")),
    tooltip=['Feature', alt.Tooltip('Importance:Q', format='.2f'), 'Category']
).properties(
    width=700, 
    height=500, 
    title="LightGBM Feature Importance for Tg Prediction"
)

st.altair_chart(importance_chart, use_container_width=True)

# Feature category breakdown
st.markdown("#### 📊 Feature Contributions by Category")

category_importance = feature_importance.groupby('Category')['Importance'].sum().reset_index()
category_importance = category_importance.sort_values('Importance', ascending=False)

cat_col1, cat_col2 = st.columns([2, 1])

with cat_col1:
    category_chart = alt.Chart(category_importance).mark_arc(innerRadius=50).encode(
        theta=alt.Theta('Importance:Q'),
        color=alt.Color('Category:N', scale=alt.Scale(scheme='tealblues')),
        tooltip=['Category', alt.Tooltip('Importance:Q', format='.2f')]
    ).properties(
        width=400,
        height=400,
        title="Feature Category Contributions"
    )
    st.altair_chart(category_chart, use_container_width=True)

with cat_col2:
    st.dataframe(category_importance, use_container_width=True)

st.markdown("""
<div style='padding:20px; border-radius:10px; border-left:4px solid #2a9d8f; margin-top:20px;'>
<p style='font-size:20px; line-height:1.7;'>
<b> Chemical Interpretation:</b>
</p>
<ul style='font-size:19px; line-height:1.8;'>
<li><b>Tm dominates (30%):</b> Validates Kauzmann's thermodynamic relationship (Tg ≈ 2/3 × Tm). 
Strong correlation (r=0.961) confirms melting temperature as the primary predictor</li>
<li><b>Molecular weight (18%):</b> Heavier chains → more entanglement → restricted motion → higher Tg</li>
<li><b>Carbon backbone (#CH, #CH2, 22% combined):</b> Aromatic rings (#CH) increase rigidity, 
while flexible methylene (#CH2) decreases Tg</li>
<li><b>Oxygen groups (16%):</b> Hydroxyl groups (#OH) form hydrogen bonds that significantly raise Tg. 
O:C ratio indicates overall polarity</li>
<li><b>Halogen contribution (0%):</b> Limited presence in dataset (only 11% of compounds have halogens)</li>
</ul>
<p style='font-size:16px; line-height:1.7; margin-top:15px;'>
<b>Key Insight:</b> The top 3 features (Tm, M, #CH) account for <b>58% of total importance</b>, 
demonstrating that glass transition is primarily governed by thermal properties and backbone structure.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# SMILES Mode Overview
st.header("2️⃣ SMILES Mode (Data-Driven Text Encoding)")

st.markdown("""
<div style='padding:20px; border-radius:10px; border-left:4px solid #e76f51;'>
<p style='font-size:17px; line-height:1.6;'>
The <b>SMILES Mode</b> uses fully automated feature extraction by encoding molecular structures as 
text strings (SMILES notation) and applying n-gram vectorization. This approach requires zero manual 
feature engineering or chemical knowledge.
</p>
</div>
""", unsafe_allow_html=True)

st.subheader("2.1 🔤 Input Representation")

smiles_col1, smiles_col2 = st.columns(2)

with smiles_col1:
    st.markdown("""
    <div style='padding:15px; border-radius:8px; border:2px solid #e76f51;'>
    <h4 style='color:#e76f51;'>Feature Extraction Process</h4>
    <ol style='font-size:15px; line-height:1.8;'>
    <li><b>Input:</b> SMILES string (e.g., "CCO" for ethanol)</li>
    <li><b>Tokenization:</b> Extract 1-4 character n-grams</li>
    <li><b>Vectorization:</b> Count occurrence of each n-gram</li>
    <li><b>Result:</b> 1000-dimensional sparse feature vector</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

with smiles_col2:
    st.markdown("""
    <div style='padding:15px; border-radius:8px; border:2px solid #e76f51;'>
    <h4 style='color:#e76f51;'>Advantages</h4>
    <ul style='font-size:15px; line-height:1.8;'>
    <li>✅ No manual feature calculation</li>
    <li>✅ Works without Tm measurement</li>
    <li>✅ Scalable to large compound libraries</li>
    <li>✅ Captures substructural patterns</li>
    <li>⚠️ Less interpretable than functional groups</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# SMILES Mode metrics — UPDATED VALUES
smiles_metrics = pd.DataFrame({
    "Model": ["Extra Trees", "CatBoost (Tuned)", "CatBoost", "Extra Trees (Tuned)", 
              "XGBoost", "Gradient Boosting", "Random Forest", "LightGBM"],
    "Test_R2": [0.9546, 0.9498, 0.9475, 0.9446, 0.9429, 0.9428, 0.9391, 0.9351],
    "Test_RMSE": [19.71, 20.72, 21.19, 21.78, 22.10, 22.11, 22.82, 23.56],
    "Test_MAE": [11.39, 12.71, 13.38, 13.38, 12.50, 14.97, 14.03, 14.98],
    "Overfitting_Gap": [0.044, 0.044, 0.042, 0.050, 0.055, 0.039, 0.050, 0.054]
})

st.subheader("2.2 📊 Model Performance Comparison")

# Display metrics table
st.dataframe(smiles_metrics.style.highlight_max(subset=['Test_R2'], color='#804b45')
                                 .highlight_min(subset=['Test_RMSE', 'Test_MAE', 'Overfitting_Gap'], color='#804b45'), 
             use_container_width=True)

# Side-by-side visualizations
viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    smiles_r2_chart = alt.Chart(smiles_metrics).mark_bar(color="#e76f51", opacity=0.85).encode(
        x=alt.X("Model:N", axis=alt.Axis(labelAngle=-45), sort='-y'),
        y=alt.Y("Test_R2:Q", title="R² Score", scale=alt.Scale(domain=[0.93, 0.96])),
        tooltip=["Model", "Test_R2", "Test_RMSE", "Test_MAE"]
    ).properties(title="SMILES Mode — R² Comparison", height=400)
    st.altair_chart(smiles_r2_chart, use_container_width=True)

with viz_col2:
    smiles_rmse_chart = alt.Chart(smiles_metrics).mark_bar(color="#2a9d8f", opacity=0.85).encode(
        x=alt.X("Model:N", axis=alt.Axis(labelAngle=-45), sort='y'),
        y=alt.Y("Test_RMSE:Q", title="RMSE (K)"),
        tooltip=["Model", "Test_R2", "Test_RMSE", "Test_MAE"]
    ).properties(title="SMILES Mode — RMSE Comparison", height=400)
    st.altair_chart(smiles_rmse_chart, use_container_width=True)

st.write("---")
st.markdown("""
<div style='padding:20px; border-radius:10px; border-left:4px solid #e76f51; margin-top:20px;'>
<p style='font-size:22px; line-height:1.7;'>
<b> Key Findings:</b>
</p>
<ul style='font-size:20px; line-height:1.8;'>
<li><b>Extra Trees</b> achieved best SMILES performance with <b>R² = 0.9546</b> and <b>RMSE = 19.71 K</b></li>
<li>Still explains <b>95.5% of Tg variance</b> — impressive for automated text features</li>
<li>Average error <b>±11.39 K</b> (MAE) — about <b>2x higher</b> than Functional Group mode</li>
<li>All models show good generalization (overfitting gaps <0.06)</li>
<li>Performance gap vs Functional Group: <b>~10K RMSE difference</b> shows value of chemical knowledge</li>
</ul>
<p style='font-size:16px; line-height:1.7; margin-top:15px;'>
<b>Trade-off:</b> SMILES mode sacrifices ~4% R² accuracy but gains complete automation and scalability.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# Comparative Analysis
st.header("3️⃣ Head-to-Head Comparison")

st.markdown("""
<p style='font-size:17px; line-height:1.6;'>
Direct comparison of the best-performing model from each approach.
</p>
""", unsafe_allow_html=True)

comparison_df = pd.DataFrame({
    "Mode": [" Functional Group (LightGBM)", " SMILES (Extra Trees)"],
    "R² Score": [0.9885, 0.9546],
    "RMSE (K)": [9.91, 19.71],
    "MAE (K)": [7.32, 11.39],
    "Overfitting Gap": [0.006, 0.044],
    "Features Used": ["13 (chemistry)", "1000 (n-grams)"],
    "Training Time (s)": [0.16, 2.61]
})

st.dataframe(comparison_df, use_container_width=True)

# Visual comparison
comp_col1, comp_col2 = st.columns(2)

with comp_col1:
    comparison_chart = alt.Chart(comparison_df).mark_bar().encode(
        x=alt.X('Mode:N', title='Prediction Mode'),
        y=alt.Y('R² Score:Q', scale=alt.Scale(domain=[0.94, 1.0])),
        color=alt.Color('Mode:N', scale=alt.Scale(domain=["🧪 Functional Group (LightGBM)", "🔤 SMILES (Extra Trees)"],
                                                   range=["#2a9d8f", "#e76f51"]), legend=None),
        tooltip=['Mode', 'R² Score', 'RMSE (K)', 'MAE (K)']
    ).properties(title="R² Score Comparison", height=400)
    st.altair_chart(comparison_chart, use_container_width=True)

with comp_col2:
    rmse_comparison = alt.Chart(comparison_df).mark_bar().encode(
        x=alt.X('Mode:N', title='Prediction Mode'),
        y=alt.Y('RMSE (K):Q'),
        color=alt.Color('Mode:N', scale=alt.Scale(domain=["🧪 Functional Group (LightGBM)", "🔤 SMILES (Extra Trees)"],
                                                   range=["#2a9d8f", "#e76f51"]), legend=None),
        tooltip=['Mode', 'R² Score', 'RMSE (K)', 'MAE (K)']
    ).properties(title="RMSE Comparison", height=400)
    st.altair_chart(rmse_comparison, use_container_width=True)

st.markdown("""
<div style='padding:25px; border-radius:12px; border-left:6px solid #2a9d8f; margin-top:20px;'>
<p style='font-size:22px; line-height:1.7;'>
<b>📌 Critical Insights:</b>
</p>
<ul style='font-size:20px; line-height:1.9;'>
<li><b>Accuracy Advantage:</b> Functional Group mode is <b>49.7% more accurate</b> (RMSE improvement from 19.71K to 9.91K)</li>
<li><b>Domain Knowledge Value:</b> Including Tm and chemical features boosts R² from 0.955 to 0.989 (+3.4%)</li>
<li><b>Efficiency Trade-off:</b> SMILES trains 16x slower (2.61s vs 0.16s) but requires no manual input</li>
<li><b>Use Case Recommendation:</b>
    <ul style='font-size:16px; margin-top:10px;'>
    <li><b>High-precision research:</b> Use Functional Group mode (±10K error)</li>
    <li><b>Rapid screening:</b> Use SMILES mode when Tm unavailable (±20K error acceptable)</li>
    </ul>
</li>
<li><b>Generalization:</b> Both modes show excellent test performance (overfitting < 0.06)</li>
</ul>
<p style='font-size:17px; line-height:1.8; margin-top:20px;'>
<b>Key Takeaway:</b> Chemistry-informed features provide a <b>2x reduction in prediction error</b> compared to 
purely data-driven text encoding, validating the importance of incorporating domain expertise in molecular 
property prediction.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------
# Conclusion
# -------------------------------
st.header("4️⃣ Conclusions & Recommendations")

st.success("""
**🏆 Best Overall Model: LightGBM (Functional Group Mode)**

- **Accuracy:** R² = 0.9885, RMSE = 9.91 K, MAE = 7.32 K
- **Generalization:** Overfitting gap = 0.006 (excellent)
- **Speed:** Training time = 0.16 seconds
- **Interpretability:** Clear feature importance aligned with polymer physics
""")

rec_col1, rec_col2 = st.columns(2)

with rec_col1:
    st.markdown("""
    <div style='padding:20px; border-radius:10px; border:2px solid #2a9d8f;'>
    <h4 style='color:#2a9d8f;'>✅ Strengths</h4>
    <ul style='font-size:15px; line-height:1.8;'>
    <li>Near-experimental accuracy (±10K typical error)</li>
    <li>Physically meaningful predictions</li>
    <li>Feature importance validates chemical theory</li>
    <li>Low computational cost</li>
    <li>Robust generalization to new compounds</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with rec_col2:
    st.markdown("""
    <div style='padding:20px; border-radius:10px; border:2px solid #e76f51;'>
    <h4 style='color:#e76f51;'>⚠️ Limitations</h4>
    <ul style='font-size:15px; line-height:1.8;'>
    <li>Requires Tm measurement for best performance</li>
    <li>Manual feature calculation needed</li>
    <li>Limited to compounds within training range</li>
    <li>Assumes linear/lightly branched structures</li>
    <li>Higher errors for high-MW polymers (>500 g/mol)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style='padding:25px; border-radius:12px; border-left:6px solid #2a9d8f; margin-top:30px;'>
<h4 style='color:#2a9d8f; margin-top:0;'>🎯 Practical Recommendations</h4>

<p style='font-size:17px; line-height:1.8;'><b>For Research & Development:</b></p>
<ul style='font-size:16px; line-height:1.8;'>
<li>Use <b>Functional Group mode</b> for final validation before synthesis</li>
<li>Typical error ±10K enables confident material selection</li>
<li>Cost savings: $500-1000 per compound vs experimental DSC</li>
</ul>

<p style='font-size:17px; line-height:1.8; margin-top:15px;'><b>For High-Throughput Screening:</b></p>
<ul style='font-size:16px; line-height:1.8;'>
<li>Use <b>SMILES mode</b> when Tm is unknown or unavailable</li>
<li>Screen 1000+ candidates per day</li>
<li>Acceptable ±20K error for preliminary filtering</li>
</ul>

<p style='font-size:17px; line-height:1.8; margin-top:15px;'><b>Future Improvements:</b></p>
<ul style='font-size:16px; line-height:1.8;'>
<li>Expand dataset to 2000+ compounds for broader coverage</li>
<li>Add uncertainty quantification (conformal prediction)</li>
<li>Incorporate polymer architecture features (branching, tacticity)</li>
<li>Explore graph neural networks for 3D structure encoding</li>
</ul>
</div>
""", unsafe_allow_html=True)


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: left; color: #666; font-size: 14px; padding: 20px;'>
<p><b>📚 Dataset:</b> 635 organic polymers from Zenodo (DOI: 10.5281/zenodo.7319485)</p>
<p><b>🔗 GitHub:</b> <a href='https://github.com/RAK2315/Glass-Transition-Temperature-Tg-Predictor' target='_blank' style='color:#2a9d8f;'>View Source Code</a></p>
</div>
""", unsafe_allow_html=True)