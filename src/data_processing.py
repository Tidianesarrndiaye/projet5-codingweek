# src/data_processing.py

from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd


def drop_high_missing_columns(
    df: pd.DataFrame,
    threshold: float = 0.05
) -> pd.DataFrame:
    """
    Supprime les colonnes dont la proportion de valeurs manquantes dépasse `threshold`.
    threshold=0.05 => >5% de NaN -> drop.
    """
    df_local = df.copy()
    missing_ratio = df_local.isna().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    if cols_to_drop:
        df_local = df_local.drop(columns=cols_to_drop)
    return df_local


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute les valeurs manquantes:
      - numériques: médiane
      - catégorielles/objet: mode
    """
    df_local = df.copy()

    num_cols = df_local.select_dtypes(include=[np.number]).columns
    cat_cols = df_local.select_dtypes(exclude=[np.number]).columns

    # Numériques -> médiane
    for c in num_cols:
        if df_local[c].isna().any():
            df_local[c] = df_local[c].fillna(df_local[c].median())

    # Catégorielles -> mode
    for c in cat_cols:
        if df_local[c].isna().any():
            mode_vals = df_local[c].mode(dropna=True)
            if not mode_vals.empty:
                df_local[c] = df_local[c].fillna(mode_vals.iloc[0])
            else:
                # si tout est NaN, remplace par chaîne vide
                df_local[c] = df_local[c].fillna("")

    return df_local


def handle_outliers(
    df: pd.DataFrame,
    iqr_factor: float = 1.5
) -> pd.DataFrame:
    """
    IQR clipping: pour chaque colonne numérique,
    clip les valeurs en dehors de [Q1 - k*IQR, Q3 + k*IQR].
    Contrairement à un filtrage de lignes (qui peut vider le dataset),
    on clippe les valeurs extrêmes aux bornes calculées.
    """
    df_local = df.copy()
    num_cols = df_local.select_dtypes(include=[np.number]).columns

    for c in num_cols:
        q1 = df_local[c].quantile(0.25)
        q3 = df_local[c].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr
        df_local[c] = df_local[c].clip(lower, upper)

    return df_local


def encode_features(
    df: pd.DataFrame,
    drop_first: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encodage one-hot des colonnes non numériques (sécurisé pour les non ordinales).
    Retourne le DataFrame encodé + un méta-dictionnaire avec les colonnes finales.
    """
    df_local = df.copy()
    cat_cols = df_local.select_dtypes(exclude=[np.number]).columns.tolist()

    encoded = pd.get_dummies(df_local, columns=cat_cols, drop_first=drop_first)
    meta = {
        "categorical_columns": cat_cols,
        "encoded_columns": list(encoded.columns),
        "drop_first": drop_first
    }
    return encoded, meta


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Réduction de la mémoire:
      - downcast int/float
      - conversion object -> category si faible cardinalité
    """
    df_opt = df.copy()

    for col in df_opt.columns:
        s = df_opt[col]

        # Downcast numériques
        if pd.api.types.is_integer_dtype(s):
            df_opt[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_float_dtype(s):
            df_opt[col] = pd.to_numeric(s, downcast="float")

        # Object -> Category si faible cardinalité
        elif pd.api.types.is_object_dtype(s):
            nunique = s.nunique(dropna=True)
            total = len(s)
            if total > 0 and 0 < nunique < 0.5 * total:
                df_opt[col] = s.astype("category")

    return df_opt


def preprocess_pipeline(
    df: pd.DataFrame,
    missing_threshold: float = 0.05,
    iqr_factor: float = 1.5,
    drop_first: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Pipeline complet:
      1) drop colonnes avec > missing_threshold de NaN
      2) imputation des NaN (num: médiane, cat: mode)
      3) clipping des outliers (IQR)
      4) one-hot encoding des colonnes catégorielles
      5) optimization mémoire
    Retourne: (X_preprocessed, meta)
    """
    step1 = drop_high_missing_columns(df, threshold=missing_threshold)
    step2 = handle_missing_values(step1)
    step3 = handle_outliers(step2, iqr_factor=iqr_factor)
    step4, meta = encode_features(step3, drop_first=drop_first)
    step5 = optimize_memory(step4)
    return step5, meta