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

def plot_roc_curves(roc_items, title="Courbes ROC"):
    # roc_items: list of tuples (name, y_true, y_prob)
    for name, y_true, y_prob in roc_items:
        RocCurveDisplay.from_predictions(y_true, y_prob, name=name)
    plt.title(title)
    plt.show()