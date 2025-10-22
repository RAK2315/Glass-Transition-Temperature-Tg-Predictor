# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load models, vectorizers and scalers
fg_model = joblib.load("models/functional_group_model.pkl")
fg_scaler = joblib.load("models/functional_group_scaler.pkl")

smiles_model = joblib.load("models/ExtraTrees_SMILES_Model.pkl")
smiles_vectorizer = joblib.load("models/smiles_vectorizer.pkl")


# -------------------------------
# Streamlit app configuration
st.set_page_config(page_title="Glass Transition Temperature Predictor", layout="centered")
st.title("Glass Transition Temperature (Tg) Predictor")
# Sidebar for mode selection
mode = st.sidebar.selectbox("Select Input Mode", ["Functional Group Mode", "SMILES Mode"])


# -------------------------------
# Functional Group Mode Page
# -------------------------------
if mode == "Functional Group Mode":
    import pandas as pd
    import streamlit as st

    st.header("Functional Group Mode")
    st.markdown(
        """
    This mode allows you to manually input **counts of functional groups** in your molecule to predict its **glass transition temperature (Tg)**.  

    - If a field is **zero**, you can leave it as is.  
    - If the **melting temperature (Tm)** is unknown, leave it blank (we recommend using **SMILES Mode** in that case).  
    - Scroll down to see **detailed explanations of each input variable**.  

    💡 **Hint:** This mode is useful if you know your molecule's structure and want precise control over input.
    """
    )



    # -------------------------------
    # Load and clean dataset
    dataset = pd.read_csv("dataset.csv", sep=";")
    # Remove leading/trailing spaces in column names
    dataset.columns = [col.strip() for col in dataset.columns]

    # Rename columns to match feature names
    dataset.rename(columns={
        'M / g/mol': 'M',
        'Tm / K': 'Tm',
        '#C ': '#C'  # remove trailing space
    }, inplace=True)

    # Working copy
    df = dataset.copy()

    # -------------------------------
    # Feature list
    fg_features = ['#CH3', '#CH2', '#CH', '#C', '#OH', '#C-O-C',
                   '#O=C', 'DBA', '#N', '#Hal', 'O:C', 'M', 'Tm']

    # -------------------------------
    # Initialize input values
    # Use first row from dataset as defaults
    input_values = {}
    first_row = df.iloc[0]
    for feature in fg_features:
        input_values[feature] = float(first_row.get(feature, 0.0))

    # -------------------------------
    # Layout input fields in 3 columns
    col1, col2, col3 = st.columns(3)
    col_mapping = {
        col1: ['#CH3', '#CH2', '#CH', '#C', '#OH'],
        col2: ['#C-O-C', '#O=C', 'DBA', '#N', '#Hal'],
        col3: ['O:C', 'M', 'Tm']
    }

    for col, features in col_mapping.items():
        with col:
            for feature in features:
                input_values[feature] = st.number_input(
                    feature, min_value=0.0, value=input_values[feature]
                )

    # Prepare DataFrame for prediction
    fg_df = pd.DataFrame([input_values])
    fg_df = fg_df.reindex(columns=fg_features)

    # Predict button
    if st.button("Predict Tg (Functional Group Mode)"):
        # Scale and predict
        fg_scaled = fg_scaler.transform(fg_df)
        predicted_tg = fg_model.predict(fg_scaled)[0]

        # Check for matching compound in dataset
        formula_display = ""
        match = None
        for _, row in df.iterrows():
            is_match = True
            for feature in fg_features:
                val_input = int(round(input_values[feature]))
                val_data = row.get(feature, None)
                if pd.isna(val_data):
                    is_match = False
                    break
                # Compare as integers
                if int(round(val_data)) != val_input:
                    is_match = False
                    break
            if is_match:
                match = row
                break

        if match is not None:
            formula_display = match.get("Formula", "")

        # Display results
        st.success(f"✅ Predicted Tg: {predicted_tg:.2f} K")
        if formula_display:
            st.info(f"**Sum formula of chemical compound:** {formula_display}")
        st.write(f"**Mean absolute error (cross validation) [K]:** ± 8.01")

    # Input variable descriptions
    st.markdown("---")
    st.subheader("Meaning of the input variables (features)")
    st.write("""
        - **#CH3:** Number of methyl groups (-CH3) in the molecule. Count only carbon atoms bonded to three hydrogens and one other atom.
        - **#CH2:** Number of methylene groups (-CH2-). These are carbons bonded to two hydrogens and two other atoms.
        - **#CH:** Number of methine groups (-CH-). Carbons bonded to one hydrogen and three other atoms.
        - **#C:** Number of quaternary carbons (C) that are not bonded to any hydrogens.
        - **#OH:** Number of hydroxyl groups (-OH) in the molecule.
        - **#C-O-C:** Number of oxygen atoms in ether linkages. Only counts oxygen atoms; the carbons are counted separately in #CHx/#C.
        - **#O=C:** Number of doubly bonded oxygen atoms (carbonyls). Only counts the oxygen atoms.
        - **DBA (Double Bond Equivalent):** A measure of unsaturation. Calculated as DBE = 1 + sum[ni(vi - 2)], where ni is number of atoms with valence vi. Represents total number of rings and pi bonds.
        - **#N:** Number of nitrogen atoms in the molecule.
        - **#Hal:** Number of halogen atoms (F, Cl, Br, I).
        - **O:C:** Ratio of oxygen atoms to carbon atoms in the molecule.
        - **M:** Molecular weight of the molecule in g/mol.
        - **Tm:** Melting temperature of the compound in Kelvin.
            
        **Notes:**
        - When counting functional groups, ensure **each atom is only counted in one feature**. For example, oxygen in an -OH counts for #OH, not for #C-O-C.
        - These features are designed to uniquely describe the molecule so that the model can predict Tg accurately.
        - If you are unsure about some counts, using **SMILES Mode** might give better accuracy.
        """)







# -------------------------------
# SMILES Mode Page
if mode == "SMILES Mode":
    st.header("SMILES Mode")
    st.markdown(
        """
    This mode allows you to input a **SMILES string** representing your molecule to predict its **glass transition temperature (Tg)**.  

    - Use the **SMILES Generator** linked below to create your molecule's SMILES string.
    - This mode is ideal if you want a **quick prediction** without manually entering functional group counts.

    [📘 Generate SMILES here](https://www.cheminfo.org/flavor/malaria/Utilities/SMILES_generator___checker/index.html)
    """
    )
    st.info("""
    **Example inputs:**

    1. Glucose → `C(C1C(C(C(C(O1)O)O)O)O)O`  Formula: C6H12O6  
    2. Ethanol → `CCO`                       Formula: C2H6O  
    3. Acetone → `CC(=O)C`                   Formula: C3H6O  
    """)

    # Input field
    smiles_input = st.text_input("Enter SMILES string", value="C(C1C(C(C(C(O1)O)O)O)O)O")

    if st.button("Predict Tg (SMILES Mode)"):
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors

        # Convert SMILES to RDKit molecule to get molecular formula
        mol = Chem.MolFromSmiles(smiles_input)
        if mol:
            formula = rdMolDescriptors.CalcMolFormula(mol)
        else:
            st.error("❌ Invalid SMILES string")
            st.stop()

        # Transform SMILES using the saved vectorizer
        X_input = smiles_vectorizer.transform([smiles_input])

        # Predict Tg
        try:
            predicted_tg = smiles_model.predict(X_input)[0]
            st.success(f"✅ Predicted Tg [K]: {predicted_tg:.2f}")
            st.info(f"**Sum formula of chemical compound:** {formula}")
            st.write("**Mean absolute error (cross validation) [K]:** ± 15.1")
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")



# -------------------------------
# Footer
st.markdown("---")
st.markdown("Project by **Rehaan & Krishna** | Predict Tg for Functional Groups or SMILES compounds")
