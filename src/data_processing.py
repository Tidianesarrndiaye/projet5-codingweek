# src/data_processing.py

from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def drop_high_missing_columns(
    df: pd.DataFrame,
    threshold: float = 0.9
) -> pd.DataFrame:
    """
    Supprime les colonnes dont la proportion de valeurs manquantes dépasse `threshold`.
    threshold=0.9 => >90% de NaN -> drop.
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
    # downcast int/float; convertir object -> category si cardinalité raisonnable
    for col in df.select_dtypes(include=["int64","int32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64","float32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["object"]).columns:
        nunique = df[col].nunique(dropna=False)
        if 1 < nunique < (0.5 * len(df)):  # heuristique
            df[col] = df[col].astype("category")
    return df


def train_test_prepare(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42):
    df = df.copy()
    y = df[target].astype(int)
    X = df.drop(columns=[target])

    num_cols = X.select_dtypes(include=["int16","int32","int64","float16","float32","float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()

    # Pipelines
    num_pipe = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    preproc = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    return preproc, X_train, X_test, y_train, y_test


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


def fit_preprocessor_on_train(
    X_train: pd.DataFrame,
    missing_threshold: float = 0.8,
    iqr_factor: float = 1.5,
    drop_first: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ajuste un préprocesseur leakage-safe sur les données d'entraînement.

    Étapes calculées sur X_train uniquement :
      1) Suppression des colonnes avec trop de NaN (> missing_threshold)
      2) Imputation : médiane pour les numériques, mode pour les catégorielles
      3) IQR clipping des colonnes numériques
      4) One-hot encoding des colonnes catégorielles
      5) Optimisation mémoire

    Retourne:
      - X_train_encoded : DataFrame préprocessé
      - meta : dictionnaire contenant toutes les valeurs/paramètres ajustés sur
        le train, à passer ensuite à `transform_with_train_preprocessor`.
    """
    X_train_local = X_train.copy()

    missing_ratio = X_train_local.isna().mean()
    cols_to_drop = missing_ratio[missing_ratio > missing_threshold].index.tolist()
    X_train_local = X_train_local.drop(columns=cols_to_drop)

    numeric_columns = X_train_local.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [col for col in X_train_local.columns if col not in numeric_columns]

    numeric_fill_values = (
        X_train_local[numeric_columns].median()
        if numeric_columns
        else pd.Series(dtype="float64")
    )
    categorical_fill_values: Dict[str, Any] = {}
    for col in categorical_columns:
        modes = X_train_local[col].mode(dropna=True)
        categorical_fill_values[col] = modes.iloc[0] if not modes.empty else "Missing"

    for col in numeric_columns:
        X_train_local[col] = X_train_local[col].fillna(numeric_fill_values[col])
    for col in categorical_columns:
        X_train_local[col] = X_train_local[col].fillna(categorical_fill_values[col])

    clipping_bounds: Dict[str, Any] = {}
    for col in numeric_columns:
        q1 = X_train_local[col].quantile(0.25)
        q3 = X_train_local[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            clipping_bounds[col] = None
            continue
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr
        clipping_bounds[col] = (lower, upper)
        X_train_local[col] = X_train_local[col].clip(lower, upper)

    X_train_encoded = pd.get_dummies(
        X_train_local,
        columns=categorical_columns,
        drop_first=drop_first,
    )
    X_train_encoded = optimize_memory(X_train_encoded)

    meta: Dict[str, Any] = {
        "feature_columns_before_encoding": X_train_local.columns.tolist(),
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "columns_dropped_for_missingness": cols_to_drop,
        "numeric_fill_values": numeric_fill_values.to_dict(),
        "categorical_fill_values": categorical_fill_values,
        "clipping_bounds": clipping_bounds,
        "encoded_columns": X_train_encoded.columns.tolist(),
        "drop_first": drop_first,
    }
    return X_train_encoded, meta


def transform_with_train_preprocessor(
    X: pd.DataFrame,
    meta: Dict[str, Any],
) -> pd.DataFrame:
    """
    Applique sur X les transformations calculées lors de `fit_preprocessor_on_train`.

    Utilise strictement les paramètres du dictionnaire `meta` (valeurs d'imputation,
    bornes IQR, colonnes encodées…) sans recalculer quoi que ce soit sur X,
    ce qui garantit l'absence de data leakage train/test.
    """
    X_local = X.copy().drop(columns=meta["columns_dropped_for_missingness"], errors="ignore")

    for col in meta["feature_columns_before_encoding"]:
        if col not in X_local.columns:
            X_local[col] = np.nan if col in meta["numeric_columns"] else "Missing"
    X_local = X_local[meta["feature_columns_before_encoding"]]

    for col, value in meta["numeric_fill_values"].items():
        X_local[col] = X_local[col].fillna(value)
    for col, value in meta["categorical_fill_values"].items():
        X_local[col] = X_local[col].fillna(value)

    for col, bounds in meta["clipping_bounds"].items():
        if bounds is None:
            continue
        lower, upper = bounds
        X_local[col] = X_local[col].clip(lower, upper)

    X_local = pd.get_dummies(
        X_local,
        columns=meta["categorical_columns"],
        drop_first=meta["drop_first"],
    )
    X_local = X_local.reindex(columns=meta["encoded_columns"], fill_value=0)
    X_local = optimize_memory(X_local)
    return X_local