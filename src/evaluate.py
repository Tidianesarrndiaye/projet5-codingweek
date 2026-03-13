import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

def compute_metrics(y_true, y_prob, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

def find_best_model(results_df, metric=None, weights=None):
    """
    Trouve le meilleur modèle selon un ou plusieurs critères.
    
    Args:
        results_df: DataFrame contenant les résultats des modèles (avec colonnes 'model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc')
        metric: str - Métrique unique pour évaluer ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')
               Si None et weights est None, utilise 'roc_auc' par défaut
        weights: dict - Dictionnaire avec poids pour combiner plusieurs métriques
               Exemple: {'roc_auc': 0.4, 'f1': 0.3, 'accuracy': 0.3}
    
    Returns:
        tuple: (best_model_name, best_score)
    """
    
    if weights is None:
        # Utiliser une metrique simple
        if metric is None:
            metric = 'roc_auc'  # Par défaut
        
        best_idx = results_df[metric].idxmax()
        best_name = results_df.loc[best_idx, 'model']
        best_score = results_df.loc[best_idx, metric]
        return best_name, best_score
    
    else:
        # Combiner plusieurs métriques avec des poids
        # Normaliser les poids
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calculer le score pondéré
        results_df = results_df.copy()
        results_df['weighted_score'] = sum(
            results_df[metric] * weight for metric, weight in weights.items()
        )
        
        best_idx = results_df['weighted_score'].idxmax()
        best_name = results_df.loc[best_idx, 'model']
        best_score = results_df.loc[best_idx, 'weighted_score']
        return best_name, best_score

def plot_roc_curves(roc_items, title="Courbes ROC"):
    # roc_items: list of tuples (name, y_true, y_prob)
    plt.title(title)
    for name, y_true, y_prob in roc_items:
        RocCurveDisplay.from_predictions(y_true, y_prob, name=name)
        plt.savefig(f"artifacts/roc_curves_{name}.png")