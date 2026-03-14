# src/explainability.py
"""
Module d'explicabilité SHAP pour l'interprétabilité des prédictions.

Fournit des fonctions pour générer des visualisations SHAP
(SHapley Additive exPlanations) montrant comment les features
contribuent à chaque prédiction.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Tuple, Optional
from sklearn.pipeline import Pipeline


CLINICAL_FEATURE_GROUPS = {
    "Diagnosis_Presumptive_appendicitis": "Diagnosis_Presumptive",
    "Diagnosis_Presumptive_no appendicitis": "Diagnosis_Presumptive",
}


def _strip_transformer_prefix(feature_name: str) -> str:
    """Retire le préfixe de ColumnTransformer (ex: num__ / cat__)."""
    if "__" in feature_name:
        return feature_name.split("__", 1)[1]
    return feature_name


def _map_to_selected_feature(
    transformed_name: str,
    selected_features: Optional[list[str]]
) -> str:
    """
    Mappe un nom transformé vers une feature clinique d'origine.

    Pour OneHotEncoder, on associe des colonnes comme
    `cat__FeatureA_valeur` à `FeatureA` si la correspondance existe.
    """
    clean_name = _strip_transformer_prefix(transformed_name)
    if not selected_features:
        return clean_name

    # Trie par longueur décroissante pour éviter les collisions de préfixes.
    for feat in sorted(selected_features, key=len, reverse=True):
        if clean_name == feat or clean_name.startswith(f"{feat}_"):
            return CLINICAL_FEATURE_GROUPS.get(feat, feat)
    return CLINICAL_FEATURE_GROUPS.get(clean_name, clean_name)


def create_shap_explainer(
    model: Union[Pipeline, object],
    X_background: pd.DataFrame,
    model_type: str = "auto"
) -> shap.Explainer:
    """
    Crée un explainer SHAP pour un modèle donné.
    
    Parameters
    ----------
    model : Pipeline ou modèle sklearn
        Le modèle entraîné (peut être une Pipeline ou un estimateur)
    X_background : pd.DataFrame
        Données d'arrière-plan (sample) pour l'approximation SHAP
        Généralement un petit sous-ensemble des données de training
    model_type : str
        Type de modèle ("auto", "tree", "linear", etc.)
    
    Returns
    -------
    shap.Explainer
        L'explainer SHAP configuré
    
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> import pandas as pd
    >>> X_bg = pd.DataFrame({'feat1': [1, 2], 'feat2': [3, 4]})
    >>> clf = RandomForestClassifier().fit(X_bg, [0, 1])
    >>> explainer = create_shap_explainer(clf, X_bg)
    """
    try:
        # Si c'est une Pipeline, extrait du modèle final
        if isinstance(model, Pipeline):
            clf = model.named_steps.get("model", model)
        else:
            clf = model
        
        # Crée l'explainer avec le type spécifié
        if model_type == "auto":
            explainer = shap.Explainer(clf, X_background)
        else:
            explainer = shap.Explainer(clf, X_background, model_type=model_type)
        
        return explainer
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la création du SHAP Explainer: {e}")


def explain_prediction(
    explainer: shap.Explainer,
    X: pd.DataFrame,
    max_display: int = 20
) -> Tuple[shap.Explanation, np.ndarray]:
    """
    Calcule les SHAP values pour une prédiction.
    
    Parameters
    ----------
    explainer : shap.Explainer
        L'explainer SHAP pré-configuré
    X : pd.DataFrame
        Les données pour lesquelles expliquer les prédictions
    max_display : int
        Nombre maximum de features à afficher dans les visualisations
    
    Returns
    -------
    Tuple[shap.Explanation, np.ndarray]
        - SHAP values
        - Predictions probabilities (si disponibles)
    
    Examples
    --------
    >>> X_explain = pd.DataFrame({'feat1': [1, 2], 'feat2': [3, 4]})
    >>> shap_values, proba = explain_prediction(explainer, X_explain)
    """
    try:
        # Calcule les SHAP values
        shap_values = explainer(X)
        
        # Récupère les probabilités (si possible)
        try:
            if hasattr(explainer.model, 'predict_proba'):
                proba = explainer.model.predict_proba(X)[:, 1]
            else:
                proba = explainer.model.predict(X)
        except Exception:
            proba = np.array([np.nan] * len(X))
        
        return shap_values, proba
    except Exception as e:
        raise RuntimeError(f"Erreur lors du calcul des SHAP values: {e}")


def plot_shap_summary(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    plot_type: str = "bar",
    max_display: int = 20,
    save_path: Optional[str] = None,
    selected_features: Optional[list[str]] = None
) -> plt.Figure:
    """
    Crée un graphique de résumé SHAP montrant l'importance moyenne des features.
    
    Parameters
    ----------
    shap_values : shap.Explanation
        Les SHAP values calculées
    X : pd.DataFrame
        Les données utilisées pour le calcul
    plot_type : str
        Type de graphique ("bar", "violin", "beeswarm")
    max_display : int
        Nombre de features à afficher
    save_path : str, optional
        Chemin où sauvegarder la figure
    
    Returns
    -------
    plt.Figure
        La figure matplotlib créée
    
    Examples
    --------
    >>> fig = plot_shap_summary(shap_values, X, plot_type="bar")
    >>> plt.show()
    """
    try:
        # Extraire les valeurs numpy brutes
        if hasattr(shap_values, 'values'):
            vals = np.asarray(shap_values.values)
            if hasattr(shap_values, 'feature_names') and shap_values.feature_names is not None:
                feature_names = list(shap_values.feature_names)
            elif isinstance(X, pd.DataFrame):
                feature_names = list(X.columns)
            else:
                n_feats = vals.shape[1] if vals.ndim >= 2 else vals.shape[0]
                feature_names = [f"Feature {i}" for i in range(n_feats)]
        else:
            vals = np.asarray(shap_values)
            feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else [
                f"Feature {i}" for i in range(vals.shape[-1])
            ]

        # Classification binaire : shape (n_samples, n_features, 2) → classe 1
        if vals.ndim == 3:
            vals = vals[:, :, 1]

        # Garantir 2D
        if vals.ndim == 1:
            vals = vals.reshape(1, -1)

        # Importance moyenne |SHAP|
        mean_abs = np.abs(vals).mean(axis=0)

        # Agrège les colonnes transformées par nom de feature clinique unique.
        aggregated: dict[str, float] = {}
        for idx, importance in enumerate(mean_abs):
            mapped_name = _map_to_selected_feature(feature_names[idx], selected_features)
            aggregated[mapped_name] = aggregated.get(mapped_name, 0.0) + float(importance)

        if not aggregated:
            raise RuntimeError("Aucune importance SHAP calculable pour l'affichage.")

        sorted_items = sorted(aggregated.items(), key=lambda kv: kv[1])
        n_display = min(max_display, len(sorted_items))
        top_items = sorted_items[-n_display:]
        sorted_names = [name for name, _ in top_items]
        sorted_vals = [value for _, value in top_items]

        fig, ax = plt.subplots(figsize=(8, max(3.5, n_display * 0.5)))
        ax.barh(range(n_display), sorted_vals, color="#e05c5c", alpha=0.85)
        ax.set_yticks(range(n_display))
        ax.set_yticklabels(sorted_names, fontsize=10)
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title("Importance des variables (SHAP)", fontsize=13)
        ax.margins(y=0.05)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
            print(f"✅ Graphique SHAP sauvegardé: {save_path}")

        return fig
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la création du graphique SHAP: {e}")


def plot_shap_local_contributions(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    max_display: int = 15,
    selected_features: Optional[list[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Affiche les contributions SHAP locales (signées) pour une instance."""
    try:
        if hasattr(shap_values, "values"):
            vals = np.asarray(shap_values.values)
            if hasattr(shap_values, "feature_names") and shap_values.feature_names is not None:
                feature_names = list(shap_values.feature_names)
            elif isinstance(X, pd.DataFrame):
                feature_names = list(X.columns)
            else:
                n_feats = vals.shape[1] if vals.ndim >= 2 else vals.shape[0]
                feature_names = [f"Feature {i}" for i in range(n_feats)]
        else:
            vals = np.asarray(shap_values)
            feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else [
                f"Feature {i}" for i in range(vals.shape[-1])
            ]

        # Classification binaire : shape (n_samples, n_features, 2) -> classe 1
        if vals.ndim == 3:
            vals = vals[:, :, 1]

        # On affiche une explication locale d'une seule observation.
        if vals.ndim == 2:
            vals = vals[0]
        elif vals.ndim != 1:
            vals = vals.reshape(-1)

        aggregated: dict[str, float] = {}
        for idx, contribution in enumerate(vals):
            mapped_name = _map_to_selected_feature(feature_names[idx], selected_features)
            aggregated[mapped_name] = aggregated.get(mapped_name, 0.0) + float(contribution)

        if not aggregated:
            raise RuntimeError("Aucune contribution SHAP locale disponible.")

        ordered = sorted(aggregated.items(), key=lambda kv: abs(kv[1]))
        n_display = min(max_display, len(ordered))
        top_items = ordered[-n_display:]
        names = [name for name, _ in top_items]
        values = [value for _, value in top_items]
        colors = ["#2e8b57" if v >= 0 else "#c94c4c" for v in values]

        fig, ax = plt.subplots(figsize=(8, max(3.5, n_display * 0.5)))
        ax.barh(range(n_display), values, color=colors, alpha=0.9)
        ax.axvline(0.0, color="#444", linewidth=1)
        ax.set_yticks(range(n_display))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel("Contribution SHAP (signée)", fontsize=11)
        ax.set_title("Contributions locales des variables (SHAP)", fontsize=13)
        ax.margins(y=0.05)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
            print(f"✅ Graphique SHAP local sauvegardé: {save_path}")

        return fig
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la création du graphique SHAP local: {e}")


def plot_shap_waterfall(
    shap_values: shap.Explanation,
    instance_index: int = 0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Crée un graphique Waterfall SHAP pour une seule prédiction.
    
    Le graphique Waterfall montre comment chaque feature pousse
    la prédiction vers la haut ou vers le bas depuis la valeur de base.
    
    Parameters
    ----------
    shap_values : shap.Explanation
        Les SHAP values calculées
    instance_index : int
        Index de l'instance à visualiser
    save_path : str, optional
        Chemin où sauvegarder la figure
    
    Returns
    -------
    plt.Figure
        La figure matplotlib créée
    
    Examples
    --------
    >>> fig = plot_shap_waterfall(shap_values, instance_index=0)
    >>> plt.show()
    """
    try:
        fig = shap.plots.waterfall(shap_values[instance_index])
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"✅ Graphique Waterfall SHAP sauvegardé: {save_path}")
        
        return fig
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la création du graphique Waterfall: {e}")


def plot_shap_force(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    instance_index: int = 0,
    save_path: Optional[str] = None
) -> None:
    """
    Crée un graphique Force SHAP pour visualiser une prédiction.
    
    Le graphique Force montre comment les features poussent
    la prédiction dans une direction (rouge positif, bleu négatif).
    
    Parameters
    ----------
    shap_values : shap.Explanation
        Les SHAP values calculées
    X : pd.DataFrame
        Les données utilisées pour le calcul
    instance_index : int
        Index de l'instance à visualiser
    save_path : str, optional
        Chemin où sauvegarder en HTML
    
    Returns
    -------
    None
    
    Examples
    --------
    >>> plot_shap_force(shap_values, X, instance_index=0)
    """
    try:
        shap.plots.force(shap_values[instance_index], save_path=save_path)
        if save_path:
            print(f"✅ Graphique Force SHAP sauvegardé: {save_path}")
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la création du graphique Force: {e}")


def get_feature_importance(
    shap_values: shap.Explanation,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Extrait l'importance moyenne de chaque feature à partir des SHAP values.
    
    Parameters
    ----------
    shap_values : shap.Explanation
        Les SHAP values calculées
    top_k : int
        Nombre de top features à retourner
    
    Returns
    -------
    pd.DataFrame
        DataFrame avec colonnes ['feature', 'mean_abs_shap_value', 'count']
    
    Examples
    --------
    >>> importance_df = get_feature_importance(shap_values, top_k=15)
    >>> print(importance_df)
    """
    try:
        # Si shap_values est une Explanation, utilise .base_values et .values
        if hasattr(shap_values, 'values'):
            values = np.asarray(shap_values.values)
        else:
            values = np.asarray(shap_values)

        # Classification binaire/multiclasse: (n_samples, n_features, n_classes)
        # -> on agrège sur les classes pour obtenir une importance par feature.
        if values.ndim == 3:
            values = np.abs(values).mean(axis=2)
        else:
            values = np.abs(values)

        if values.ndim == 1:
            values = values.reshape(1, -1)
        
        # Moyenne par feature
        mean_abs_shap = np.mean(values, axis=0)
        
        # Crée DataFrame
        feature_names = getattr(shap_values, "feature_names", None)
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(mean_abs_shap))]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap_value': mean_abs_shap,
            'count': [len(values)] * len(mean_abs_shap)
        }).sort_values('mean_abs_shap_value', ascending=False)
        
        return importance_df.head(top_k)
    except Exception as e:
        raise RuntimeError(f"Erreur lors de l'extraction d'importance: {e}")


def generate_shap_report(
    model: Union[Pipeline, object],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    output_dir: str = "reports"
) -> dict:
    """
    Génère un rapport complet SHAP avec visualisations et métadonnées.
    
    Parameters
    ----------
    model : Pipeline ou modèle sklearn
        Le modèle entraîné
    X_train : pd.DataFrame
        Données d'entraînement (pour background)
    X_test : pd.DataFrame
        Données de test à expliquer
    output_dir : str
        Répertoire où sauvegarder les graphiques
    
    Returns
    -------
    dict
        Dictionnaire contenant
        - "explainer": l'explainer SHAP
        - "shap_values": les SHAP values
        - "importance": l'importance des features
        - "report_path": le chemin du rapport
    
    Examples
    --------
    >>> report = generate_shap_report(model, X_train, X_test)
    >>> print(report['importance'])
    """
    import os
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Crée l'explainer
        print("📊 Création du SHAP Explainer...")
        explainer = create_shap_explainer(model, X_train.iloc[:100])  # Background sample
        
        # Calcule les SHAP values
        print("📈 Calcul des SHAP values...")
        shap_values, proba = explain_prediction(explainer, X_test)
        
        # Génère les graphiques
        print("🎨 Génération des visualisations...")
        
        # Bar plot
        plot_shap_summary(
            shap_values,
            X_test,
            plot_type="bar",
            save_path=f"{output_dir}/shap_summary_bar.png"
        )
        
        # Waterfall pour la première prédiction
        if len(X_test) > 0:
            plot_shap_waterfall(
                shap_values,
                instance_index=0,
                save_path=f"{output_dir}/shap_waterfall_0.png"
            )
        
        # Feature importance
        importance = get_feature_importance(shap_values, top_k=15)
        print("\n📊 Top 10 features les plus importantes:")
        print(importance.head(10))
        
        # Save importance to CSV
        importance.to_csv(f"{output_dir}/shap_importance.csv", index=False)
        
        print(f"\n✅ Rapport SHAP généré dans {output_dir}/")
        
        return {
            "explainer": explainer,
            "shap_values": shap_values,
            "importance": importance,
            "report_path": output_dir
        }
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la génération du rapport SHAP: {e}")
