"""
Interface web Streamlit pour le diagnostic d'appendicite pédiatrique.

Application de clinical decision support utilisant:
- Machine Learning (modèle Random Forest)
- Explicabilité SHAP pour l'interprétabilité
- Interface simple et intuitive pour les cliniciens
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
from datetime import datetime

# Ajouter le dossier parent au sys.path pour importer src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.explainability import create_shap_explainer, explain_prediction, plot_shap_summary
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("⚠️ Module explicabilité SHAP non disponible")

# Configuration Streamlit
st.set_page_config(
    page_title="Appendicite Pédiatrique - Aide à la Décision",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement du CSS personnalisé
try:
    with open('app/style.css', 'r', encoding='utf-8') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    pass


# ==================== FONCTIONS UTILITAIRES ====================

@st.cache_resource
def load_model(model_path: str):
    """Charge le modèle entraîné avec caching."""
    if not os.path.exists(model_path):
        st.error(f"❌ Modèle non trouvé: {model_path}")
        st.info("Assurez-vous d'exécuter d'abord: `python src/train_model.py`")
        st.stop()
    return joblib.load(model_path)


@st.cache_resource
def create_explainer(_model, X_background):
    """Crée l'explainer SHAP avec caching."""
    if not SHAP_AVAILABLE:
        return None

    try:
        # Prépare le data background pour SHAP
        if hasattr(_model, 'named_steps'):  # Pipeline
            preproc = _model.named_steps['preproc']
            X_bg_transformed = preproc.transform(X_background)

            # Préserve les noms de features transformées pour les modèles qui les utilisent.
            try:
                feature_names = preproc.get_feature_names_out()
                if hasattr(X_bg_transformed, "toarray"):
                    X_bg_transformed = X_bg_transformed.toarray()
                X_bg_transformed = pd.DataFrame(
                    np.asarray(X_bg_transformed),
                    columns=feature_names
                )
            except Exception:
                pass
        else:
            X_bg_transformed = X_background
        
        clf = _model.named_steps.get('model', _model) if hasattr(_model, 'named_steps') else _model
        explainer = create_shap_explainer(clf, X_bg_transformed)
        return explainer
    except Exception as e:
        st.warning(f"⚠️ Impossible de créer l'explainer SHAP: {e}")
        return None


def prepare_model_input(_model, X_input: pd.DataFrame):
    """Applique le preprocessing et conserve les noms de features si possible."""
    if not hasattr(_model, "named_steps"):
        return X_input

    preproc = _model.named_steps["preproc"]
    X_transformed = preproc.transform(X_input)

    try:
        feature_names = preproc.get_feature_names_out()
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()
        return pd.DataFrame(np.asarray(X_transformed), columns=feature_names)
    except Exception:
        return X_transformed


def build_user_input() -> pd.DataFrame:
    """Construit un DataFrame à partir des inputs utilisateur."""
    user_data = {}
    
    # Crée les champs d'entrée en colonnes
    col1, col2, col3 = st.columns(3)
    
    # Feature 1: WBC_Count
    with col1:
        user_data['WBC_Count'] = st.number_input(
            "🩸 Leucocytes (10^9/L)",
            min_value=0.0,
            max_value=30.0,
            value=8.0,
            step=0.1,
            help="Nombre de globules blancs"
        )
    
    # Feature 2: Management_primary_surgical
    with col2:
        user_data['Management_primary surgical'] = st.checkbox(
            "🏥 Gestion primaire chirurgicale",
            value=False,
            help="Intervention chirurgicale primaire réalisée"
        )
    
    # Feature 3: Management_secondary_surgical
    with col3:
        user_data['Management_secondary surgical'] = st.checkbox(
            "🏥 Gestion secondaire chirurgicale",
            value=False,
            help="Intervention chirurgicale secondaire réalisée"
        )
    
    # Deuxième ligne
    col1, col2, col3 = st.columns(3)
    
    # Feature 4: Diagnosis_Presumptive_no appendicitis
    with col1:
        user_data['Diagnosis_Presumptive_no appendicitis'] = st.checkbox(
            "📋 Diagnostic présomptif (pas appendicite)",
            value=False,
            help="Diagnostic initial d'absence d'appendicite"
        )
    
    # Feature 5: Appendix_on_US_yes
    with col2:
        user_data['Appendix_on_US_yes'] = st.checkbox(
            "🔍 Appendice visible à l'échographie",
            value=False,
            help="Appendice identifié à l'imagerie ultrasonore"
        )
    
    # Crée le DataFrame
    return pd.DataFrame([user_data])


# ==================== INTERFACE PRINCIPALE ====================

# Titre principal
st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1>🏥 Aide à la Décision : Appendicite Pédiatrique</h1>
        <p style="font-size: 16px; color: #666;">
            Système d'Intelligence Artificielle expliquée (XAI) pour le diagnostic de l'appendicite chez l'enfant
        </p>
    </div>
""", unsafe_allow_html=True)

# Barre latérale
st.sidebar.title("⚙️ Configuration")

# Sélection du modèle
model_choice = st.sidebar.selectbox(
    "Sélectionner le modèle",
    ["Random Forest (meilleur)", "LightGBM", "SVM-RBF"],
    index=0,
    help="Choisir le modèle pour les prédictions"
)

model_map = {
    "Random Forest (meilleur)": "artifacts/rf.joblib",
    "LightGBM": "artifacts/lgbm.joblib",
    "SVM-RBF": "artifacts/svm_rbf.joblib"
}

model_path = model_map[model_choice]

st.sidebar.markdown("---")
st.sidebar.subheader("🔁 Rafraîchissement")

if "last_model_refresh" not in st.session_state:
    st.session_state["last_model_refresh"] = None

# Rafraîchissement manuel du modèle/explainer
if st.sidebar.button("🔄 Rafraîchir le modèle", use_container_width=True):
    load_model.clear()
    create_explainer.clear()
    st.session_state["last_model_refresh"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    st.rerun()

if st.session_state["last_model_refresh"] is not None:
    st.sidebar.caption(f"Dernier rafraîchissement: {st.session_state['last_model_refresh']}")

# Charger le modèle
try:
    model = load_model(model_path)
    st.sidebar.success(f"✅ Modèle chargé: {model_choice}")
    
except Exception as e:
    st.error(f"❌ Erreur lors du chargement: {e}")
    st.stop()

# Charger les données de background pour SHAP
explainer = None
try:
    X_background = pd.read_csv("data/processed/features_and_target.csv")
    if 'Diagnosis_no appendicitis' in X_background.columns:
        X_background = X_background.drop(columns=['Diagnosis_no appendicitis'])
    
    # Crée l'explainer
    explainer = create_explainer(model, X_background.iloc[:100])
except Exception as e:
    st.warning(f"⚠️ Impossible de charger les données background: {e}")

# ==================== SAISIE DES DONNÉES ====================

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Données cliniques du patient")

with st.container():
    st.markdown("### 📋 Saisir les données cliniques du patient:")
    
    X_user = build_user_input()

# ==================== PRÉDICTION ET INTERPRÉTABILITÉ ====================

col_predict, col_explain = st.columns(2)

with col_predict:
    if st.button("🎯 Prédire", key="predict_button", use_container_width=True):
        st.markdown("---")
        
        with st.spinner("⏳ Calcul en cours..."):
            try:
                # Prépare les données pour le modèle
                if hasattr(model, 'named_steps'):
                    clf = model.named_steps['model']
                    X_user_processed = prepare_model_input(model, X_user)
                else:
                    X_user_processed = X_user
                    clf = model
                
                # Prédiction
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*feature names.*"
                    )
                    prediction_proba = clf.predict_proba(X_user_processed)[0, 1]
                    prediction = int(clf.predict(X_user_processed)[0])
                
                # Affichage du résultat
                st.markdown("### 📊 Résultats de la prédiction")
                
                # Probabilité
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Probabilité d'appendicite",
                        f"{prediction_proba*100:.1f}%",
                        delta=None
                    )
                
                with col2:
                    if prediction_proba > 0.7:
                        risk_level = "🔴 ÉLEVÉE"
                    elif prediction_proba > 0.3:
                        risk_level = "🟡 MODÉRÉE"
                    else:
                        risk_level = "🟢 FAIBLE"
                    st.markdown(f"#### Niveau de risque: {risk_level}")
                
                # Recommandation clinique
                st.markdown("---")
                if prediction == 1:
                    st.success("✅ **Appendicite probable** - Recommandation: Consultation chirurgicale recommandée")
                else:
                    st.info("ℹ️ **Appendicite improbable** - Recommandation: Surveillance clinique possible")
                
            except Exception as e:
                st.error(f"❌ Erreur lors de la prédiction: {e}")
                st.info("Vérifiez que toutes les données sont correctement saisies.")

# ==================== EXPLICABILITÉ SHAP ====================

with st.container():
    st.markdown("---")
    st.markdown("### 🧠 Explicabilité (SHAP - XAI)")
    
    if st.button("📈 Générer l'explication SHAP", key="shap_button", use_container_width=True):
        if not SHAP_AVAILABLE:
            st.warning("⚠️ SHAP n'est pas installé dans l'environnement actuel.")
        elif explainer is None:
            st.warning("⚠️ L'explainer SHAP n'est pas disponible.")
        else:
            with st.spinner("⏳ Génération de l'explication..."):
                try:
                    # Prépare les données
                    if hasattr(model, 'named_steps'):
                        X_user_processed = prepare_model_input(model, X_user)
                    else:
                        X_user_processed = X_user
                    
                    # Calcule les SHAP values
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*feature names.*")
                        shap_values, _ = explain_prediction(explainer, X_user_processed)
                    
                    # Génère le graphique
                    st.markdown("#### 📊 Importance des features (SHAP Bar Plot)")
                    st.info(
                        "Ce graphique montre comment chaque variable clinique contribue à la prédiction."
                    )
                    
                    try:
                        fig = plot_shap_summary(shap_values, X_user_processed, plot_type="bar")
                        st.pyplot(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"⚠️ Impossible d'afficher le graphique SHAP: {e}")
                        
                except Exception as e:
                    st.error(f"❌ Erreur lors de la génération de l'explication: {e}")

# ==================== FOOTER ====================

st.markdown("""
    ---
    <div style="text-align: center; color: #888; margin-top: 30px;">
        <small>
            💡 <b>Information:</b> Cet outil utilise un modèle de Machine Learning entraîné sur des données pédiatriques.
            Il est conçu pour <u>assister</u> le clinicien, <u>pas pour remplacer</u> son jugement clinique.
            <br/>
            🔒 Données patient: Aucun stockage en base de données. Utilisation locale uniquement.
            <br/>
            📧 Pour toute question: contactez l'équipe médicale
        </small>
    </div>
""", unsafe_allow_html=True)
