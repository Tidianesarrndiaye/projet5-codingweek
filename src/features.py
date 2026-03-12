import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


"""_summary_
    Stratégie pratique :

    Supprimer les paires > 0.9,
    Prendre le top‑k MI,
    Optionnel : RFE pour stabiliser le sous‑ensemble.
"""
def rank_features_by_mi(X: pd.DataFrame, y: pd.Series, top_k: int = 20):
    X_enc = pd.get_dummies(X, drop_first=True)
    num_cols = X_enc.select_dtypes(include=[np.number]).columns
    mi = mutual_info_classif(X_enc[num_cols], y, random_state=42)
    mi_series = pd.Series(mi, index=num_cols).sort_values(ascending=False)
    return mi_series.head(top_k).index.tolist(), mi_series

def rfe_logreg(X: pd.DataFrame, y: pd.Series, n_features: int = 15):
    X_enc = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler(with_mean=False)  # sparse-friendly
    X_scaled = scaler.fit_transform(X_enc)
    est = LogisticRegression(max_iter=500, n_jobs=None)
    selector = RFE(est, n_features_to_select=n_features, step=1)
    selector.fit(X_scaled, y)
    selected_cols = X_enc.columns[selector.get_support()]
    return list(selected_cols)

def drop_highly_correlated(X: pd.DataFrame, threshold: float = 0.9):
    X_enc = pd.get_dummies(X, drop_first=True)
    corr = X_enc.corr().abs()
    upper = corr.where(~np.tril(np.ones(corr.shape)).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return X_enc.drop(columns=to_drop), to_drop

   