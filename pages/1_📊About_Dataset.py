import streamlit as st
import pandas as pd
import altair as alt
from rdkit import Chem

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="About / Dataset Info", layout="wide")
st.title("About the Tg Dataset")

# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv", sep=";")
    df.columns = [col.strip() for col in df.columns]
    rename_dict = {"M / g/mol": "M", "Tm / K": "Tm", "Tg / K": "Tg", "Tg/Tm": "Tg_Tm"}
    df.rename(columns=rename_dict, inplace=True)
    return df

df = load_data()

# -------------------------------
# Functional Group Detection (RDKit)
# -------------------------------
SMARTS_GROUPS = {
    "Oxide": ["[OX2H]", "C-O-C", "C=O"],
    "Amine": ["[NX3;H0,H1,H2;!$(NC=O)]"],
    "Acid/Derivatives": ["C(=O)[OX2H1]", "C(=O)O", "C(=O)N"],
    "Halide": ["[F,Cl,Br,I]"],
    "Nitrile": ["C#N"],
    "Aromatic": ["a1aaaaa1"],
    "Hydrocarbon / Unsaturated HC": ["C=C", "C#C"]  # previously "Other"
}

def detect_main_groups(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ["Unrecognized"]
    detected = set()
    for fam, patterns in SMARTS_GROUPS.items():
        for smarts in patterns:
            patt = Chem.MolFromSmarts(smarts)
            if patt and mol.HasSubstructMatch(patt):
                detected.add(fam)
    if not detected:
        detected.add("Hydrocarbon / Unsaturated HC")
    return list(detected)

if "Detected Groups" not in df.columns and "SMILES" in df.columns:
    df["Detected Groups"] = df["SMILES"].apply(lambda s: detect_main_groups(str(s)))
    df["Main Group"] = df["Detected Groups"].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "Hydrocarbon / Unsaturated HC"
    )

# -------------------------------
# Intro Section
# -------------------------------
st.markdown("""
<h2 style="color: #2a9d8f;">Introduction</h2>
<p style="font-size:20px;">
This page provides a comprehensive overview of the dataset used to train our <b>Tg prediction models</b>.
It includes a wide range of polymer compounds and small molecules, with detailed molecular and structural features.
</p>
<ul style="font-size:19px;">
<li><b>Functional groups:</b> Counts of methyl (-CH3), methylene (-CH2), hydroxyl (-OH), ethers (-O-), carbonyls (=O), nitrogen atoms, halogens, etc.</li>
<li><b>Molecular properties:</b> Includes Molecular Weight (M), Melting Temperature (Tm), Glass Transition Temperature (Tg), and derived ratios like O:C.</li>
<li><b>Derived features:</b> Double Bond Equivalents (DBA), and other structure-related descriptors.</li>
<li><b>References:</b> Original literature sources for Tg and Tm values are included for traceability.</li>
</ul>
<p style="font-size:19px;">
This dataset is the foundation for both <b>SMILES mode</b> and <b>Functional Group Mode</b> predictions, helping scientists estimate polymer properties based on structure.
</p>
""", unsafe_allow_html=True)

# -------------------------------
# Filter by Detected Group
# -------------------------------
st.subheader("🔍 Filter by Functional Group")
ignore_unrecognized = st.checkbox("Ignore unrecognized/NaN SMILES", value=True)
if "Main Group" in df.columns:
    unique_groups = sorted(df["Main Group"].explode().dropna().unique().tolist())
    if ignore_unrecognized and "Unrecognized" in unique_groups:
        unique_groups.remove("Unrecognized")
    selected_groups = st.multiselect(
        "Select one or more functional groups to view (default shows all):",
        unique_groups,
        default=[]
    )
    if selected_groups:
        df_filtered = df[df["Main Group"].isin(selected_groups)]
    else:
        df_filtered = df.copy()
    if ignore_unrecognized:
        df_filtered = df_filtered[df_filtered["Main Group"] != "Unrecognized"]
else:
    df_filtered = df.copy()

# -------------------------------
# Dataset Summary Metrics
# -------------------------------
st.subheader("Dataset Summary")

# First row for general metrics and Mean/Std Dev of Tg and Tm
col1, col2, col3, col4, col5, col6 = st.columns(6) 

col1.metric("Total Compounds", df_filtered.shape[0])
col2.metric("Total Features", df_filtered.shape[1])

# Tg Metrics (Mean and Median)
if 'Tg' in df_filtered.columns:
    col3.metric("Mean Tg [K]", round(df_filtered['Tg'].mean(), 2))
    col4.metric("Median Tg [K]", round(df_filtered['Tg'].median(), 2))

# Tm Metrics (Mean and Median)
if 'Tm' in df_filtered.columns:
    col5.metric("Mean Tm [K]", round(df_filtered['Tm'].mean(), 2))
    col6.metric("Median Tm [K]", round(df_filtered['Tm'].median(), 2))

st.subheader("Property Range (Min/Max)")
colA, colB, colC, colD, colE, colF = st.columns(6)

# Tg Metrics (Min and Max)
if 'Tg' in df_filtered.columns:
    colA.metric("Min Tg [K]", round(df_filtered['Tg'].min(), 2))
    colB.metric("Max Tg [K]", round(df_filtered['Tg'].max(), 2))

# Tm Metrics (Min and Max)
if 'Tm' in df_filtered.columns:
    colC.metric("Min Tm [K]", round(df_filtered['Tm'].min(), 2))
    colD.metric("Max Tm [K]", round(df_filtered['Tm'].max(), 2))

# Molecular Weight (M) Metrics (Mean, Median, Min, Max)
if 'M' in df_filtered.columns:
    colE.metric("Min M [g/mol]", round(df_filtered['M'].min(), 2))
    colF.metric("Max M [g/mol]", round(df_filtered['M'].max(), 2))
    

# -------------------------------
# Data Table
st.subheader("Dataset Preview")
st.dataframe(df_filtered)

# -------------------------------
# Interactive Feature Distributions
st.subheader("📊 Feature Distributions (Interactive)")
numeric_features = ['#CH3', '#CH2', '#CH', '#C', '#OH', '#C-O-C',
                    '#O=C', 'DBA', '#N', '#Hal', 'O:C', 'M', 'Tm', 'Tg']
available_features = [f for f in numeric_features if f in df_filtered.columns]
selected_feature = st.selectbox("Select a feature to visualize:", available_features)

hist = alt.Chart(df_filtered).mark_bar(opacity=0.7, color='#2a9d8f').encode(
    alt.X(f"{selected_feature}", bin=alt.Bin(maxbins=30)),
    y='count()',
    tooltip=[f"{selected_feature}", 'count()']
).properties(width=800, height=400, title=f"Distribution of {selected_feature}")
st.altair_chart(hist, use_container_width=True)

# -------------------------------
# Scatter plots
st.subheader("🔬 Scatter Plots")
if all(x in df_filtered.columns for x in ['Tg', 'Tm', 'M']):
    scatter1 = alt.Chart(df_filtered).mark_circle(size=60).encode(
        x='Tm', y='Tg',
        color=alt.Color('M', scale=alt.Scale(scheme='blues'), title='Molecular Weight (M)'),
        tooltip=['Name', 'Formula', 'Tm', 'Tg', 'M', 'Main Group']
    ).interactive().properties(title="Tg vs Tm colored by Molecular Weight")
    st.altair_chart(scatter1, use_container_width=True)

# -------------------------------
# Functional Group Counts
st.subheader("🧪 Total Counts of Functional Groups")
fg_columns = ['#CH3', '#CH2', '#CH', '#C', '#OH', '#C-O-C', '#O=C', 'DBA', '#N', '#Hal']
fg_columns = [c for c in fg_columns if c in df_filtered.columns]

fg_sums = df_filtered[fg_columns].sum().reset_index()
fg_sums.columns = ['Functional Group', 'Total Count']

bar_chart = alt.Chart(fg_sums).mark_bar().encode(
    x=alt.X('Functional Group', sort='-y'),
    y='Total Count',
    tooltip=['Functional Group', 'Total Count'],
    color=alt.Color('Total Count', scale=alt.Scale(scheme='tealblues'))
).properties(width=800, height=400)
st.altair_chart(bar_chart, use_container_width=True)

# -------------------------------
# Additional insights
st.markdown("""
<h3 style="color: #2a9d8f;">Dataset Insights & Notes</h3>
<ul style="font-size:20px;">
<li><b>Composition:</b> The dataset contains polymers and small molecules with a variety of functional groups, including methyl (-CH3), methylene (-CH2), hydroxyl (-OH), ethers (-O-), carbonyls (=O), nitrogen atoms, halogens, and more.</li>
<li><b>Experimental Data:</b> Tg (glass transition temperature) and Tm (melting temperature) values are experimentally measured and recorded in Kelvin.</li>
<li><b>Derived Features:</b> Some features, like <code>DBA</code> (double bond equivalent) and <code>O:C</code> ratio, are computed based on molecular structure.</li>
<li><b>Missing Values:</b> Some columns may have missing entries due to unavailable measurements or undefined values.</li>
<li><b>Data Usage:</b> This dataset supports both the SMILES mode and Functional Group Mode prediction models, forming the core training data.</li>
<li><b>Diversity:</b> Includes small molecules, oligomers, and polymer fragments with a range of molecular weights and functional group counts.</li>
<li><b>References:</b> Original experimental references for Tg and Tm values are included in the dataset for traceability.</li>
<li><b>Applications:</b> Enables prediction of polymer glass transition temperatures based on structure, aiding material design and research.</li>
</ul>
""", unsafe_allow_html=True)

