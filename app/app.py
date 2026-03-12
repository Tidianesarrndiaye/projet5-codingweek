# app/app.py
import streamlit as st
import pandas as pd
import shap
import joblib
import numpy as np

st.set_page_config(page_title="Pediatric Appendicitis - CDS", layout="centered")
st.title("Aide à la décision : Appendicite pédiatrique (XAI)")

model_path = "artifacts/lgbm.joblib"  # OU meilleur modèle
model = joblib.load(model_path)

st.sidebar.header("Saisir les variables cliniques")
# TODO: créer les champs (âge, douleur, fièvre, CRP, leucocytes, etc.) selon dataset
# Exemple:
age = st.sidebar.number_input("Âge (années)", min_value=0, max_value=18, value=10)
crp = st.sidebar.number_input("CRP (mg/L)", min_value=0.0, value=5.0)
wbc = st.sidebar.number_input("Leucocytes (10^9/L)", min_value=0.0, value=8.0)
# ... ajouter tous les champs nécessaires

if st.button("Prédire"):
    # construire DataFrame X_user aligné sur features d'entraînement
    # -> il faut reproduire exactement le pipeline (mêmes colonnes, encodages)
    X_user = pd.DataFrame([{
        "age": age, "crp": crp, "wbc": wbc
        # ...
    }])

    # ATTENTION: si modèle est un Pipeline (préproc + model), SHAP doit utiliser l’étape finale
    # Voici une approche générique :
    try:
        clf = model.named_steps["model"]
        preproc = model.named_steps["preproc"]
        X_trans = preproc.transform(X_user)
    except Exception:
        clf = model
        X_trans = X_user  # si déjà prétraité

    proba = clf.predict_proba(X_trans)[:, 1][0]
    st.metric("Probabilité d'appendicite", f"{proba*100:.1f}%")

    # SHAP
    explainer = shap.Explainer(clf, X_trans)
    shap_values = explainer(X_trans)
    st.subheader("Explicabilité SHAP (bar plot)")
    st.pyplot(shap.plots.bar(shap_values[0], show=False))