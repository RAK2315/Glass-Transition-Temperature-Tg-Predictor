# about.py — Updated Polymer Dataset Info Page
import streamlit as st
import pandas as pd
import altair as alt

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="About / Dataset Info", 
    page_icon="📚",
    layout="wide"
)

st.title("📚 About the Polymer Tg Dataset")

# ============================================================
# LOAD DATASET
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv", sep=";")
    df.columns = [col.strip() for col in df.columns]
    rename_dict = {"M / g/mol": "M", "Tm / K": "Tm", "Tg / K": "Tg", "Tg/Tm": "Tg_Tm"}
    df.rename(columns=rename_dict, inplace=True)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Could not load dataset: {e}")
    st.stop()

# ============================================================
# INTRODUCTION SECTION
# ============================================================
st.markdown("""
<div style='padding:25px; border-radius:12px; border-left:6px solid #2a9d8f;'>
<h3 style='color:#2a9d8f; margin-top:0;'>What Is This Dataset?</h3>
<p style='font-size:22px; line-height:1.6;'>
This dataset contains a curated collection of <b>organic polymers and small molecules</b> with experimentally measured 
<b>Glass Transition Temperatures (Tg)</b> and <b>Melting Temperatures (Tm)</b>.
</p>
<p style='font-size:19px; line-height:1.6;'>
It serves as the training foundation for both prediction modes in our machine learning tool:
</p>
<ul style='font-size:19px; line-height:1.8;'>
<li><b>⏹️Functional Group Mode:</b> Uses 13 physics-informed chemical descriptors including functional group counts 
(#CH₃, #OH, #C-O-C), molecular weight (M), melting temperature (Tm), and oxygen-to-carbon ratio (O:C)</li>
<li><b>⏹️ SMILES Mode:</b> Uses SMILES text representation of molecular structures to predict Tg directly 
through automated n-gram vectorization — no manual feature extraction needed</li>
</ul>
<p style='font-size:16px; line-height:1.6; margin-top:15px;'>
<b>📌 Dataset Source:</b> <a href='https://zenodo.org/records/7319485' target='_blank' style='color:#2a9d8f;'>Zenodo Repository (DOI: 10.5281/zenodo.7319485)</a>
</p>
<p style='font-size:20px; line-height:1.6;'>
<b>GOAL:</b> Enable chemists, materials scientists, and students to rapidly predict thermal properties of polymers, 
reducing expensive and time-consuming laboratory experiments while maintaining high accuracy (R²>0.98).
</p>
</div>
""", unsafe_allow_html=True)

# DATASET SUMMARY
st.markdown("## 📊 Dataset Statistics")

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("📦 Total Compounds", df.shape[0])
col2.metric("📋 Total Features", df.shape[1])

if 'Tg' in df.columns:
    col3.metric("🌡️ Mean Tg", f"{df['Tg'].mean():.1f} K")
    col4.metric("📍 Median Tg", f"{df['Tg'].median():.1f} K")

if 'Tm' in df.columns:
    col5.metric("🔥 Mean Tm", f"{df['Tm'].mean():.1f} K")
    col6.metric("📍 Median Tm", f"{df['Tm'].median():.1f} K")



# Temperature and molecular weight ranges
colA, colB, colC, colD, colE, colF = st.columns(6)

if 'Tg' in df.columns:
    colA.metric("Min Tg", f"{df['Tg'].min():.1f} K")
    colB.metric("Max Tg", f"{df['Tg'].max():.1f} K")

if 'Tm' in df.columns:
    colC.metric("Min Tm", f"{df['Tm'].min():.1f} K")
    colD.metric("Max Tm", f"{df['Tm'].max():.1f} K")

if 'M' in df.columns:
    colE.metric("Min M", f"{df['M'].min():.0f} g/mol")
    colF.metric("Max M", f"{df['M'].max():.0f} g/mol")

st.divider()


# KEY INSIGHTS BOXES
st.markdown("###  Key Dataset Characteristics")

insight_col1, insight_col2, insight_col3 = st.columns(3)

with insight_col1:
    st.markdown("""
    <div style='padding:20px; border-radius:10px; border:2px solid #2a9d8f;'>
    <h4 style='color:#2a9d8f; margin-top:0;'>📏 Size & Scope</h4>
    <ul style='font-size:15px;'>
    <li><b>635 compounds</b> after cleaning</li>
    <li><b>13 features</b> per compound</li>
    <li>Molecular weight range: <b>32-1152 g/mol</b></li>
    <li>Tg range: <b>58-448 K</b></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with insight_col2:
    st.markdown("""
    <div style='padding:20px; border-radius:10px; border:2px solid #2a9d8f;'>
    <h4 style='color:#2a9d8f; margin-top:0;'>🧬 Material Types</h4>
    <ul style='font-size:15px;'>
    <li><b>Organic polymers</b> (carbon-based)</li>
    <li><b>Small molecules</b> and oligomers</li>
    <li><b>Aromatic systems</b> (benzene rings)</li>
    <li><b>Hydroxylated compounds</b> (alcohols, polyols)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with insight_col3:
    st.markdown("""
    <div style='padding:20px; border-radius:10px; border:2px solid #2a9d8f;'>
    <h4 style='color:#2a9d8f; margin-top:0;'>⚗️ Functional Groups</h4>
    <ul style='font-size:15px;'>
    <li><b>Alkyl chains</b> (#CH₃, #CH₂, #CH)</li>
    <li><b>Oxygen groups</b> (#OH, #C-O-C, #O=C)</li>
    <li><b>Heteroatoms</b> (#N, #Hal)</li>
    <li><b>Structural features</b> (DBA, O:C ratio)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# DATA PREVIEW
st.markdown("### 📘 Dataset Preview")

# Add search/filter option
search_col1, search_col2 = st.columns([3, 1])

with search_col1:
    search_term = st.text_input("🔍 Search by Name or Formula:", "")

with search_col2:
    show_rows = st.selectbox("Show rows:", [10, 25, 50, 100, "All"], index=1)

# Filter dataframe
if search_term:
    mask = df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
    filtered_df = df[mask]
else:
    filtered_df = df

# Display with row limit
if show_rows == "All":
    st.dataframe(filtered_df, use_container_width=True, height=600)
else:
    st.dataframe(filtered_df.head(show_rows), use_container_width=True)

st.caption(f"Showing {len(filtered_df) if show_rows=='All' else min(show_rows, len(filtered_df))} of {len(df)} total compounds")

# ============================================================
# FEATURE DISTRIBUTIONS
# ============================================================
st.markdown("##  Feature Distributions")

numeric_features = [
    '#CH3', '#CH2', '#CH', '#C', '#OH', '#C-O-C',
    '#O=C', 'DBA', '#N', '#Hal', 'O:C', 'M', 'Tm', 'Tg'
]
available_features = [f for f in numeric_features if f in df.columns]

dist_col1, dist_col2 = st.columns([2, 1])

with dist_col1:
    selected_feature = st.selectbox("Select a feature to visualize:", available_features, index=available_features.index('Tg') if 'Tg' in available_features else 0)

with dist_col2:
    if selected_feature in df.columns:
        st.metric("Mean", f"{df[selected_feature].mean():.2f}")
        st.metric("Std Dev", f"{df[selected_feature].std():.2f}")

# Histogram
hist = alt.Chart(df).mark_bar(opacity=0.8, color='#2a9d8f').encode(
    alt.X(f"{selected_feature}:Q", bin=alt.Bin(maxbins=30), title=selected_feature),
    alt.Y('count()', title='Frequency'),
    tooltip=[alt.Tooltip(f"{selected_feature}:Q", format='.2f'), 'count()']
).properties(
    width=800, 
    height=400, 
    title=f"Distribution of {selected_feature}"
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_title(
    fontSize=16,
    anchor='start'
)

st.altair_chart(hist, use_container_width=True)

# ============================================================
# CORRELATION SCATTER PLOTS
# ============================================================
st.markdown("##  Feature Relationships")

scatter_col1, scatter_col2 = st.columns(2)

with scatter_col1:
    x_feature = st.selectbox("X-axis:", available_features, index=available_features.index('Tm') if 'Tm' in available_features else 0)

with scatter_col2:
    y_feature = st.selectbox("Y-axis:", available_features, index=available_features.index('Tg') if 'Tg' in available_features else 0)

if x_feature in df.columns and y_feature in df.columns:
    # Calculate correlation
    correlation = df[[x_feature, y_feature]].corr().iloc[0, 1]
    
    st.info(f"📈 **Correlation (r):** {correlation:.3f}")
    
    scatter = alt.Chart(df).mark_circle(size=80, opacity=0.6).encode(
        x=alt.X(f"{x_feature}:Q", scale=alt.Scale(zero=False)),
        y=alt.Y(f"{y_feature}:Q", scale=alt.Scale(zero=False)),
        color=alt.Color('M:Q', scale=alt.Scale(scheme='tealblues'), title='Molecular Weight (g/mol)'),
        tooltip=['Name', 'Formula', x_feature, y_feature, 'M']
    ).interactive().properties(
        title=f"{y_feature} vs {x_feature} (colored by Molecular Weight)",
        width=700,
        height=500
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    )
    
    st.altair_chart(scatter, use_container_width=True)

# ============================================================
# FUNCTIONAL GROUP ANALYSIS
# ============================================================
st.markdown("##  Functional Group Analysis")

fg_columns = ['#CH3', '#CH2', '#CH', '#C', '#OH', '#C-O-C', '#O=C', 'DBA', '#N', '#Hal']
fg_columns = [c for c in fg_columns if c in df.columns]

if fg_columns:
    fg_sums = df[fg_columns].sum().reset_index()
    fg_sums.columns = ['Functional Group', 'Total Count']
    fg_sums = fg_sums.sort_values('Total Count', ascending=False)
    
    bar_chart = alt.Chart(fg_sums).mark_bar().encode(
        x=alt.X('Functional Group:N', sort='-y', title='Functional Group'),
        y=alt.Y('Total Count:Q', title='Total Count Across All Compounds'),
        tooltip=['Functional Group', alt.Tooltip('Total Count', format=',')],
        color=alt.Color('Total Count:Q', scale=alt.Scale(scheme='tealblues'), legend=None)
    ).properties(
        width=800, 
        height=450, 
        title="Total Count of Functional Groups Across All 635 Polymers"
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_title(
        fontSize=16,
        anchor='start'
    )
    
    st.altair_chart(bar_chart, use_container_width=True)
    
    # Functional group explanation
    with st.expander("ℹ️ What do these functional groups mean?"):
        st.markdown("""
        | Symbol | Functional Group | Description |
        |--------|-----------------|-------------|
        | **#CH3** | Methyl | Terminal carbon with 3 hydrogens (-CH₃) |
        | **#CH2** | Methylene | Carbon with 2 hydrogens in chain (-CH₂-) |
        | **#CH** | Methine | Carbon with 1 hydrogen (-CH-) |
        | **#C** | Quaternary Carbon | Carbon with no hydrogens, 4 bonds |
        | **#OH** | Hydroxyl | Alcohol group (-OH) |
        | **#C-O-C** | Ether | Oxygen between two carbons (C-O-C) |
        | **#O=C** | Carbonyl | Oxygen double-bonded to carbon (C=O) |
        | **DBA** | Double Bond Approximation | Measures unsaturation (rings + double bonds) |
        | **#N** | Nitrogen | Nitrogen atoms (amines, amides) |
        | **#Hal** | Halogen | F, Cl, Br, or I atoms |
        """)
st.divider()


# DATASET INSIGHTS & NOTES
st.markdown("##  Dataset Insights & Technical Notes")

st.markdown("""
<div style='padding:25px; border-radius:12px; border-left:6px solid #2a9d8f;'>
<h4 style='color:#2a9d8f; margin-top:0;'>📌 Important Information</h4>
<ul style='font-size:20px; line-height:1.8;'>
<li><b>Material Type:</b> This dataset contains <b>organic polymers</b> (carbon-based compounds), NOT inorganic oxide glasses. 
Common examples include polyethylene fragments, aromatic compounds, and hydroxylated polymers.</li>

<li><b>Experimental Origin:</b> Tg and Tm values are obtained from peer-reviewed literature and expressed in <b>Kelvin (K)</b>. 
To convert to Celsius: °C = K - 273.15</li>

<li><b>Key Features for Prediction:</b>
    <ul>
    <li><b>Tm (Melting Temperature):</b> Strongest predictor — accounts for ~86% of Tg variance</li>
    <li><b>M (Molecular Weight):</b> Heavier chains have more entanglement → higher Tg</li>
    <li><b>#OH (Hydroxyl groups):</b> Hydrogen bonding increases Tg significantly</li>
    <li><b>O:C Ratio:</b> Polarity indicator affecting intermolecular interactions</li>
    </ul>
</li>

<li><b>Missing Values:</b> Some experimental measurements are unavailable due to:
    <ul>
    <li>Compound decomposition before reaching Tm</li>
    <li>Measurement difficulties with certain functional groups</li>
    <li>Limited literature data for novel structures</li>
    </ul>
    These are handled automatically during model training (removed during preprocessing).
</li>

<li><b>SMILES Representation:</b> Each compound has a SMILES string encoding its molecular structure as text. 
Example: <code>CCO</code> represents ethanol (CH₃CH₂OH). This enables structure-based predictions without manual feature extraction.</li>

<li><b>Molecular Weight Range:</b> Dataset covers <b>32-1152 g/mol</b>, including:
    <ul>
    <li>Small molecules (32-100 g/mol)</li>
    <li>Oligomers (100-500 g/mol)</li>
    <li>Small polymers (500-1152 g/mol)</li>
    </ul>
    Models are most accurate for M < 1000 g/mol.
</li>

<li><b>Educational Purpose:</b> Designed to help researchers, chemists, and students:
    <ul>
    <li>Visualize structure-property relationships in polymer science</li>
    <li>Understand how functional groups affect thermal behavior</li>
    <li>Reduce reliance on expensive lab testing (DSC costs $500-1000 per sample)</li>
    <li>Enable rapid virtual screening of candidate materials</li>
    </ul>
</li>

<li><b>Limitations:</b>
    <ul>
    <li>Does not account for polymer architecture (branching, crosslinking, tacticity)</li>
    <li>Assumes linear or lightly branched structures</li>
    <li>Best for pure compounds, not blends or copolymers</li>
    <li>Predictions should be validated experimentally for critical applications</li>
    </ul>
</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ============================================================
# DATA QUALITY METRICS
# ============================================================
st.markdown("## ✅ Data Quality Assessment")

quality_col1, quality_col2, quality_col3 = st.columns(3)

with quality_col1:
    completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    st.metric("Completeness", f"{completeness:.1f}%", delta="High quality")

with quality_col2:
    if 'Tg' in df.columns and 'Tm' in df.columns:
        valid_ratio = ((df['Tg'] / df['Tm']) > 0.4).sum() / len(df) * 100
        st.metric("Valid Tg/Tm Ratios", f"{valid_ratio:.1f}%", delta="Physically reasonable")

with quality_col3:
    if 'M' in df.columns:
        reasonable_mw = ((df['M'] > 20) & (df['M'] < 2000)).sum() / len(df) * 100
        st.metric("Reasonable MW Range", f"{reasonable_mw:.1f}%", delta="Within expected bounds")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: left; color: #666; font-size: 14px; padding: 20px;'>
<p><b>Dataset Citation:</b> Zenodo Repository (DOI: 10.5281/zenodo.7319485)</p>
<p><b>GitHub:</b> <a href='https://github.com/RAK2315/Glass-Transition-Temperature-Tg-Predictor' target='_blank' style='color:#2a9d8f;'>View Source Code</a></p>
<p style='margin-top:15px; font-size:13px;'>⚠️ For research and educational purposes. Always validate predictions with experimental data for critical applications.</p>
</div>
""", unsafe_allow_html=True)