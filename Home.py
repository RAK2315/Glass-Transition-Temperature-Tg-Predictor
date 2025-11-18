import streamlit as st
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# PAGE CONFIGURATION
st.set_page_config(
    page_title="Glass Transition Temperature Predictor",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# LOAD MODELS 
@st.cache_resource
def load_models():
    """Load all models and preprocessors with error handling."""
    try:
        models = {
            'fg_model': joblib.load("models/functional_group_model.pkl"),
            'fg_scaler': joblib.load("models/functional_group_scaler.pkl"),
            'smiles_model': joblib.load("models/ExtraTrees_SMILES_Model.pkl"),
            'smiles_vectorizer': joblib.load("models/smiles_vectorizer.pkl"),
            'metadata': joblib.load("models/model_metadata.pkl")
        }
        return models, None
    except Exception as e:
        return None, str(e)

models, error = load_models()

if error:
    st.error(f"❌ Error loading models: {error}")
    st.info("Please ensure all model files are in the 'models/' folder.")
    st.stop()

# Extract models
fg_model = models['fg_model']
fg_scaler = models['fg_scaler']
smiles_model = models['smiles_model']
smiles_vectorizer = models['smiles_vectorizer']
metadata = models['metadata']

# LOAD DATASET (with error handling)
@st.cache_data
def load_dataset():
    """Load and clean the dataset."""
    try:
        df = pd.read_csv("dataset.csv", sep=";")
        df.columns = [col.strip() for col in df.columns]
        
        # Rename columns
        rename_dict = {
            'M / g/mol': 'M',
            'Tm / K': 'Tm',
            'Tg / K': 'Tg',
            '#C ': '#C'
        }
        df.rename(columns=rename_dict, inplace=True)
        return df, None
    except Exception as e:
        return None, str(e)

df, df_error = load_dataset()

if df_error:
    st.warning(f"⚠️ Could not load dataset: {df_error}")
    st.info("Formula matching will be disabled.")
    df = None


# TITLE AND INTRODUCTION
st.title("Glass Transition Temperature (Tg) Predictor")

st.markdown("""
<div style=padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h3 style='color: #2a9d8f; margin-top: 0;'>About This Tool</h3>
    <p style='font-size: 20px;'>
        Predict the <b>glass transition temperature (Tg)</b> of polymers and organic compounds using 
        machine learning. Choose between two prediction modes:
    </p>
    <ul style='font-size: 20px;'>
        <li><b>Functional Group Mode:</b> High accuracy (R² = 0.989, RMSE ≈ 10 K) using chemical features</li>
        <li><b>SMILES Mode:</b> Quick predictions (R² = 0.955, RMSE ≈ 20 K) from molecular text</li>
    </ul>
</div>
""", unsafe_allow_html=True)

 
# SIDEBAR - MODE SELECTION
st.sidebar.title("⚙️ Prediction Mode")
mode = st.sidebar.selectbox(
    "Select Input Method:",
    ["Functional Group Mode", "SMILES Mode"],
    help="Choose how you want to input your molecule"
)

# Display model info in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Information")

if mode == "Functional Group Mode":
    st.sidebar.metric("Best Model", metadata['best_fg_model'])
    st.sidebar.metric("Accuracy (R²)", f"{metadata['best_fg_r2']:.4f}")
    st.sidebar.metric("Error (RMSE)", f"{metadata['best_fg_rmse']:.2f} K")
    st.sidebar.metric("Typical Error (MAE)", f"±{metadata['best_fg_mae']:.2f} K")
else:
    st.sidebar.metric("Best Model", metadata['best_smiles_model'])
    st.sidebar.metric("Accuracy (R²)", f"{metadata['best_smiles_r2']:.4f}")
    st.sidebar.metric("Error (RMSE)", f"{metadata['best_smiles_rmse']:.2f} K")
    st.sidebar.metric("Typical Error (MAE)", f"±{metadata['best_smiles_mae']:.2f} K")

st.sidebar.markdown("---")
st.sidebar.info(f"**Dataset Size:** {metadata['dataset_size']} compounds\n\n**Training Date:** {metadata['training_date']}")

# FUNCTIONAL GROUP MODE
if mode == "Functional Group Mode":
    
    st.header("🧪 Functional Group Mode")
    
    st.markdown("""
    <div style='padding: 15px; border-radius: 8px; border-left: 5px solid #2a9d8f;'>
        <p style='margin: 0; font-size: 21px;'>
            <b>How to use:</b> Enter the count of each functional group in your molecule. 
            If a value is zero, leave it as is. If melting temperature (Tm) is unknown, 
            consider using SMILES Mode instead.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Input Molecular Features")
    
    # Feature list
    fg_features = metadata['functional_group_features']
    
    # Initialize with defaults (zeros except for realistic defaults)
    default_values = {
        '#CH3': 2.0, '#CH2': 4.0, '#CH': 1.0, '#C': 0.0,
        '#OH': 1.0, '#C-O-C': 0.0, '#O=C': 0.0, 'DBA': 2.0,
        '#N': 0.0, '#Hal': 0.0, 'O:C': 0.2, 'M': 150.0, 'Tm': 300.0
    }
    
    # Use session state to persist values
    if 'fg_inputs' not in st.session_state:
        st.session_state.fg_inputs = default_values.copy()
    
    # Layout in 3 columns
    col1, col2, col3 = st.columns(3)
    
    col_mapping = {
        col1: ['#CH3', '#CH2', '#CH', '#C', '#OH'],
        col2: ['#C-O-C', '#O=C', 'DBA', '#N', '#Hal'],
        col3: ['O:C', 'M', 'Tm']
    }
    
    input_values = {}
    
    for col, features in col_mapping.items():
        with col:
            for feature in features:
                # Add help text for each feature
                help_text = {
                    '#CH3': "Methyl groups (-CH₃)",
                    '#CH2': "Methylene groups (-CH₂-)",
                    '#CH': "Methine groups (-CH-)",
                    '#C': "Quaternary carbons",
                    '#OH': "Hydroxyl groups (-OH)",
                    '#C-O-C': "Ether oxygen atoms",
                    '#O=C': "Carbonyl oxygen atoms",
                    'DBA': "Double bond equivalent (unsaturation)",
                    '#N': "Nitrogen atoms",
                    '#Hal': "Halogen atoms (F, Cl, Br, I)",
                    'O:C': "Oxygen to carbon ratio",
                    'M': "Molecular weight (g/mol)",
                    'Tm': "Melting temperature (K)"
                }.get(feature, "")
                
                input_values[feature] = st.number_input(
                    feature,
                    min_value=0.0,
                    value=st.session_state.fg_inputs.get(feature, default_values.get(feature, 0.0)),
                    step=0.1 if feature in ['O:C', 'DBA'] else 1.0,
                    help=help_text,
                    key=f"fg_{feature}"
                )
                st.session_state.fg_inputs[feature] = input_values[feature]
    
    # Predict button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    
    with col_btn2:
        predict_button = st.button("🔮 Predict Tg", type="primary", use_container_width=True)
    
    if predict_button:
        try:
            # Prepare DataFrame
            fg_df = pd.DataFrame([input_values])
            fg_df = fg_df[fg_features]  # Ensure correct order
            
            # Validate inputs
            if input_values['M'] <= 0:
                st.error("❌ Molecular weight (M) must be greater than 0")
                st.stop()
            
            if input_values['Tm'] <= 0:
                st.warning("⚠️ Melting temperature (Tm) should be greater than 0. Consider using SMILES mode if Tm is unknown.")
            
            # Scale and predict
            fg_scaled = fg_scaler.transform(fg_df)
            predicted_tg = fg_model.predict(fg_scaled)[0]
            
            # Display results
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.metric("Predicted Tg", f"{predicted_tg:.2f} K", delta=None)
            
            with col_res2:
                st.metric("Predicted Tg (°C)", f"{predicted_tg - 273.15:.2f} °C", delta=None)
            
            with col_res3:
                st.metric("Typical Error", f"±{metadata['best_fg_mae']:.1f} K", delta=None)
            
            # Check for matching compound
            if df is not None:
                match = None
                for _, row in df.iterrows():
                    is_match = True
                    for feature in fg_features:
                        val_input = round(input_values[feature], 2)
                        val_data = row.get(feature, None)
                        if pd.isna(val_data):
                            is_match = False
                            break
                        if abs(round(val_data, 2) - val_input) > 0.1:
                            is_match = False
                            break
                    if is_match:
                        match = row
                        break
                
                if match is not None:
                    formula = match.get("Formula", "Unknown")
                    name = match.get("Name", "Unknown")
                    actual_tg = match.get("Tg", None)
                    
                    st.success(f"✅ **Match Found in Database!**")
                    st.info(f"""
                    **Compound Name:** {name}  
                    **Formula:** {formula}  
                    **Actual Tg:** {actual_tg:.2f} K (experimental)  
                    **Prediction Error:** {abs(predicted_tg - actual_tg):.2f} K
                    """)
            
            st.markdown(
                """
            <div style='border-left:5px solid #2a9d8f; padding:15px; border-radius:8px; margin-top:15px;'>
            <p style='margin:0; font-size:18px;'>
                <b>ℹ️ Prediction Confidence:</b><br>
                • Average prediction error: ±{mae:.1f} K (Mean Absolute Error)<br>
                • 95% of predictions fall within: ±{rmse_range:.1f} K (Typical range)<br>
                • Model used: {model_name}  (R² = {r2:.4f})
            </p>
            <hr style='border:0; border-top:1px solid #ccc; margin:10px 0;'>
            <p style='margin:0; font-size:20px;'>
                <b>In simple terms:</b><br>
                On average, the model’s predictions differ from real experimental values by about 
                <b>±{mae:.1f} K</b> (that’s the usual small error).  
                In rare cases, it can be off by up to <b>±{rmse_range:.1f} K</b> — this covers most outliers (≈95% of all predictions).  
                The <b>R² value</b> shows how well the model captures patterns — the closer to 1.0, the more reliable it is.
            </p>
            </div>
            """.format(
                    mae=metadata['best_fg_mae'],
                    rmse_range=metadata['best_fg_rmse'] * 2,
                    model_name=metadata['best_fg_model'],
                    r2=metadata['best_fg_r2']
                ),
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")
            st.info("Please check your input values and try again.")
    st.write("---")
    # Feature explanations
    with st.expander("📖 Feature Definitions & Guidelines"):
        st.markdown("""
        ### Functional Group Counting Rules
        
        **Carbon Groups:**
        - **#CH3**: Methyl groups (-CH₃) bonded to one other atom
        - **#CH2**: Methylene groups (-CH₂-) bonded to two other atoms
        - **#CH**: Methine groups (-CH-) bonded to three other atoms
        - **#C**: Quaternary carbons bonded to four other atoms (no hydrogens)
        
        **Oxygen Groups:**
        - **#OH**: Hydroxyl groups (-OH)
        - **#C-O-C**: Oxygen in ether linkages (count oxygen only, not carbons)
        - **#O=C**: Carbonyl oxygen (doubly bonded, count oxygen only)
        
        **Other Features:**
        - **DBA**: Double bond equivalent = 1 + Σ[nᵢ(vᵢ - 2)], measures rings + double bonds
        - **#N**: Total nitrogen atoms
        - **#Hal**: Total halogen atoms (F, Cl, Br, I)
        - **O:C**: Ratio of oxygen to carbon atoms
        - **M**: Molecular weight in g/mol
        - **Tm**: Melting temperature in Kelvin
        
        **Important Rules:**
        - Each atom is counted in only ONE category
        - Example: Oxygen in -OH is counted in #OH, NOT in #C-O-C
        - All counts must be non-negative integers (or close to it)
        
        **Not sure?** Use SMILES Mode for automatic feature extraction!
        """)

# SMILES MODE
elif mode == "SMILES Mode":
    
    st.header("🔤 SMILES Mode")
    
    st.markdown("""
    <div style=padding: 15px; border-radius: 8px; border-left: 5px solid #2a9d8f;'>
        <p style='margin: 0; font-size: 18px;'>
            <b>How to use:</b> Enter a SMILES string representing your molecule. 
            This mode automatically extracts features and makes a prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Enter SMILES String")
    
    # SMILES generator link
    st.markdown("""
    **Don't have a SMILES string?** Use this free tool:  
    [🔗 SMILES Generator](https://www.cheminfo.org/flavor/malaria/Utilities/SMILES_generator___checker/index.html)
    """)
    
    # Example SMILES
    with st.expander("📝 Example SMILES Strings"):
        st.markdown("""
        | Compound | SMILES | Formula |
        |----------|--------|---------|
        | Ethanol | `CCO` | C₂H₆O |
        | Acetone | `CC(=O)C` | C₃H₆O |
        | Benzene | `c1ccccc1` | C₆H₆ |
        | Glucose | `C(C1C(C(C(C(O1)O)O)O)O)O` | C₆H₁₂O₆ |
        | Phenol | `Oc1ccccc1` | C₆H₆O |
        """)
    
    # Input field
    smiles_input = st.text_input(
        "SMILES String:",
        value="CCO",
        help="Enter the SMILES representation of your molecule",
        placeholder="Example: CCO for ethanol"
    )
    
    # Predict button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    
    with col_btn2:
        predict_smiles_button = st.button("🔮 Predict Tg", type="primary", use_container_width=True)
    
    if predict_smiles_button:
        try:
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors, Descriptors
            
            # Validate SMILES
            if not smiles_input or smiles_input.strip() == "":
                st.error("❌ Please enter a SMILES string")
                st.stop()
            
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles_input.strip())
            
            if mol is None:
                st.error("❌ Invalid SMILES string. Please check your input.")
                st.info("💡 Tip: Use the SMILES generator tool linked above to create a valid SMILES.")
                st.stop()
            
            # Get molecular properties
            formula = rdMolDescriptors.CalcMolFormula(mol)
            mol_weight = Descriptors.MolWt(mol)
            num_atoms = mol.GetNumAtoms()
            
            # Vectorize SMILES
            X_input = smiles_vectorizer.transform([smiles_input.strip()])
            
            # Predict
            predicted_tg = smiles_model.predict(X_input)[0]
            
            # Display results
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.metric("Predicted Tg", f"{predicted_tg:.2f} K", delta=None)
            
            with col_res2:
                st.metric("Predicted Tg", f"{predicted_tg - 273.15:.2f} °C", delta=None)
            
            with col_res3:
                st.metric("Typical Error", f"±{metadata['best_smiles_mae']:.1f} K", delta=None)
            
            # Molecular info
            st.markdown("### 🧬 Molecular Information")
            
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                st.metric("Formula", formula)
            
            with col_info2:
                st.metric("Molecular Weight", f"{mol_weight:.2f} g/mol")
            
            with col_info3:
                st.metric("Number of Atoms", num_atoms)
            
            # Check database match
            if df is not None and 'SMILES' in df.columns:
                match = df[df['SMILES'] == smiles_input.strip()]
                
                if not match.empty:
                    match_row = match.iloc[0]
                    name = match_row.get("Name", "Unknown")
                    actual_tg = match_row.get("Tg", None)
                    
                    st.success(f"✅ **Match Found in Database!**")
                    st.info(f"""
                    **Compound Name:** {name}  
                    **Actual Tg:** {actual_tg:.2f} K (experimental)  
                    **Prediction Error:** {abs(predicted_tg - actual_tg):.2f} K
                    """)
            
            # Uncertainty info
            st.markdown(
                """
            <div style='border-left:5px solid #2a9d8f; padding:15px; border-radius:8px; margin-top:15px'>
            <p style='margin:0; font-size:18px;'>
                <b>ℹ️ Prediction Confidence:</b><br>
                • Average prediction error: ±{mae:.1f} K (Mean Absolute Error)<br>
                • 95% of predictions fall within: ±{rmse_range:.1f} K (Typical range)<br>
                • Model used: {model_name}  (R² = {r2:.4f})<br>
                • Note: SMILES mode is slightly less accurate than Functional Group mode
            </p>
            <hr style='border:0; border-top:1px solid #ccc; margin:10px 0;'>
            <p style='margin:0; font-size:18px;'>
                <b>In simple terms:</b><br>
                On average, predictions differ from experimental Tg values by about <b>±{mae:.1f} K</b>.  
                In most real-world cases, the difference stays within <b>±{rmse_range:.1f} K</b> (covers 95% of compounds).  
                The <b>R²</b> value tells how well the model understands the data — closer to 1.0 means higher accuracy.
            </p>
            </div>
            """.format(
                    mae=metadata['best_smiles_mae'],
                    rmse_range=metadata['best_smiles_rmse'] * 2,
                    model_name=metadata['best_smiles_model'],
                    r2=metadata['best_smiles_r2']
                ),
                unsafe_allow_html=True,
            )

        except ImportError:
            st.error("❌ RDKit library not available. Please install: `pip install rdkit`")
        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")
            st.info("Please check your SMILES string and try again.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")

col_foot1, col_foot2, col_foot3 = st.columns(3)


with col_foot2:
    st.markdown("**📊 Dataset**")
    st.markdown(f"{metadata['dataset_size']} compounds")

with col_foot3:
    st.markdown("**🔗 Links**")
    st.markdown("[GitHub](https://github.com/RAK2315/Glass-Transition-Temperature-Tg-Predictor)")

st.markdown("""
<div style='text-align: center; color: #666; font-size: 13px; margin-top: 20px;'>
    <p>Predict glass transition temperatures for polymers and organic compounds using machine learning</p>
    <p>⚠️ For research and educational purposes. Always validate with experimental data for critical applications.</p>
</div>
""", unsafe_allow_html=True)