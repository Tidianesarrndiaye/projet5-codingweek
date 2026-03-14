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
from typing import Any

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


def infer_target_column(df: pd.DataFrame) -> str | None:
    """Détecte la colonne cible dans le CSV prétraité."""
    candidates = [
        "Diagnosis",
        "Diagnosis_no appendicitis",
        "Diagnosis_no_appendicitis",
        "target",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    for col in df.columns:
        low = col.lower()
        if "diagnosis" in low and "presumptive" not in low:
            return col
    return None


def compute_feature_defaults(df: pd.DataFrame) -> dict[str, Any]:
    """Calcule une valeur par défaut stable pour chaque feature."""
    defaults: dict[str, Any] = {}
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()

        if non_null.empty:
            defaults[col] = 0
            continue

        if pd.api.types.is_bool_dtype(series):
            mode = non_null.mode(dropna=True)
            defaults[col] = bool(mode.iloc[0]) if not mode.empty else bool(non_null.iloc[0])
        elif pd.api.types.is_numeric_dtype(series):
            defaults[col] = float(non_null.median())
        else:
            mode = non_null.mode(dropna=True)
            defaults[col] = mode.iloc[0] if not mode.empty else non_null.iloc[0]
    return defaults


def get_expected_raw_features(_model, fallback_features: list[str]) -> list[str]:
    """Retourne les colonnes brutes attendues par le pipeline de preprocessing."""
    if hasattr(_model, "named_steps") and "preproc" in _model.named_steps:
        preproc = _model.named_steps["preproc"]
        if hasattr(preproc, "feature_names_in_"):
            return list(preproc.feature_names_in_)
    return fallback_features


def align_input_to_model(
    X_input: pd.DataFrame,
    expected_features: list[str],
    defaults: dict[str, Any],
) -> pd.DataFrame:
    """Complète et ordonne X_input pour matcher exactement les colonnes attendues."""
    if not expected_features:
        return X_input.copy()

    row = {}
    source = X_input.iloc[0].to_dict() if not X_input.empty else {}

    for col in expected_features:
        if col in source:
            row[col] = source[col]
        else:
            row[col] = defaults.get(col, 0)

    return pd.DataFrame([row], columns=expected_features)


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
    """Construit un DataFrame à partir des inputs utilisateur (15 features)."""
    user_data = {}

    # ------------------------------------------------------------------
    # Section 1 : Données démographiques et biologie
    # ------------------------------------------------------------------
    st.markdown("#### 🔬 Données biologiques et démographiques")
    col1, col2, col3 = st.columns(3)

    with col1:
        user_data['Age'] = st.number_input(
            "🧒 Âge (années)",
            min_value=0.0,
            max_value=20.0,
            value=11.4,
            step=0.1,
            help="Âge du patient en années"
        )
    with col2:
        user_data['Appendix_Diameter'] = st.number_input(
            "📏 Diamètre appendice (mm)",
            min_value=0.0,
            max_value=30.0,
            value=7.6,
            step=0.1,
            help="Diamètre de l'appendice mesuré à l'échographie (mm)"
        )
    with col3:
        user_data['WBC_Count'] = st.number_input(
            "🩸 Leucocytes (10⁹/L)",
            min_value=0.0,
            max_value=35.0,
            value=11.9,
            step=0.1,
            help="Numération des globules blancs"
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        user_data['Segmented_Neutrophils'] = st.number_input(
            "🔬 Neutrophiles segmentés (%)",
            min_value=0.0,
            max_value=100.0,
            value=67.0,
            step=1.0,
            help="Pourcentage de neutrophiles segmentés"
        )
    with col2:
        user_data['US_Number'] = st.number_input(
            "🔢 Numéro d'examen US",
            min_value=1.0,
            max_value=1000.0,
            value=408.0,
            step=1.0,
            help="Numéro séquentiel de l'examen échographique"
        )

    st.markdown("---")

    # ------------------------------------------------------------------
    # Section 2 : Prise en charge et diagnostic présomptif
    # ------------------------------------------------------------------
    st.markdown("#### 🏥 Prise en charge et diagnostic présomptif")
    col1, col2, col3 = st.columns(3)

    with col1:
        user_data['Management_primary surgical'] = st.checkbox(
            "🔪 Traitement chirurgical primaire",
            value=False,
            help="Intervention chirurgicale réalisée en première intention"
        )
    with col2:
        user_data['Management_secondary surgical'] = st.checkbox(
            "🔪 Traitement chirurgical secondaire",
            value=False,
            help="Intervention chirurgicale réalisée en deuxième intention"
        )
    with col3:
        user_data['Diagnosis_Presumptive_appendicitis'] = st.checkbox(
            "📋 Diagnostic présomptif : appendicite",
            value=True,
            help="Diagnostic clinique initial d'appendicite"
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        user_data['Diagnosis_Presumptive_no appendicitis'] = st.checkbox(
            "📋 Diagnostic présomptif : pas d'appendicite",
            value=False,
            help="Diagnostic clinique initial d'absence d'appendicite"
        )
    with col2:
        user_data['Appendix_on_US_yes'] = st.checkbox(
            "🔍 Appendice visible à l'échographie",
            value=True,
            help="Appendice identifié lors de l'examen échographique"
        )

    st.markdown("---")

    # ------------------------------------------------------------------
    # Section 3 : Signes cliniques
    # ------------------------------------------------------------------
    st.markdown("#### 🩺 Signes cliniques")
    col1, col2, col3 = st.columns(3)

    with col1:
        user_data['Contralateral_Rebound_Tenderness_yes'] = st.checkbox(
            "↔️ Douleur rebond controlatérale",
            value=False,
            help="Douleur à la décompression du côté opposé à la fosse iliaque droite"
        )
    with col2:
        user_data['Coughing_Pain_yes'] = st.checkbox(
            "💨 Douleur à la toux",
            value=False,
            help="Douleur provoquée par la toux"
        )
    with col3:
        user_data['Lymph_Nodes_Location_MB'] = st.checkbox(
            "🔵 Ganglions mésentériques / bord mésentère",
            value=False,
            help="Présence de ganglions lymphatiques au niveau mésentérique"
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        user_data['Ileus_yes'] = st.checkbox(
            "🚫 Iléus",
            value=False,
            help="Présence d'un iléus (occlusion intestinale fonctionnelle)"
        )
    with col2:
        user_data['Coprostasis_yes'] = st.checkbox(
            "💊 Coprostase",
            value=True,
            help="Présence de matières fécales dans le côlon (coprostase)"
        )

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
feature_defaults: dict[str, Any] = {}
expected_features: list[str] = []
try:
    background_df = pd.read_csv("data/processed/features_and_target.csv")
    target_col = infer_target_column(background_df)

    X_background = background_df.drop(columns=[target_col]) if target_col in background_df.columns else background_df.copy()
    feature_defaults = compute_feature_defaults(X_background)
    expected_features = get_expected_raw_features(model, list(X_background.columns))
    X_background = X_background.reindex(columns=expected_features)

    for col in X_background.columns:
        X_background[col] = X_background[col].fillna(feature_defaults.get(col, 0))
    
    # Crée l'explainer
    explainer = create_explainer(model, X_background.iloc[:100])
except Exception as e:
    st.warning(f"⚠️ Impossible de charger les données background: {e}")

# ==================== SAISIE DES DONNÉES ====================

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Données cliniques du patient")

with st.container():
    st.markdown("### 📋 Saisir les données cliniques du patient (15 variables)")
    
    X_user = build_user_input()

if not expected_features:
    expected_features = list(X_user.columns)

# ==================== PRÉDICTION ET INTERPRÉTABILITÉ ====================

if st.button("🎯 Prédire", key="predict_button", use_container_width=True):
    st.markdown("---")
    
    with st.spinner("⏳ Calcul en cours..."):
        try:
            X_user_aligned = align_input_to_model(X_user, expected_features, feature_defaults)
            
            # Prédiction
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*feature names.*"
                )
                if hasattr(model, "predict_proba"):
                    prediction_proba = float(model.predict_proba(X_user_aligned)[0, 1])
                else:
                    prediction_proba = float(model.predict(X_user_aligned)[0])
                prediction = int(prediction_proba >= 0.5)
            
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
                    X_user_aligned = align_input_to_model(X_user, expected_features, feature_defaults)
                    if hasattr(model, 'named_steps'):
                        X_user_processed = prepare_model_input(model, X_user_aligned)
                    else:
                        X_user_processed = X_user_aligned
                    
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
