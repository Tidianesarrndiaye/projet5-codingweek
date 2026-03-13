# src/data_processing.py
from __future__ import annotations
import pandas as pd
import numpy as np
import re
from typing import Tuple, Dict, Any
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. OPTIMISATION DE LA MÉMOIRE
##################################################################
def optimize_memory(df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Réduit l'empreinte mémoire du DataFrame en ajustant les types de données.
    Convertit les entiers et flottants vers des formats plus légers (downcasting).
    """
    for col in df_original.columns:
        col_type = df_original[col].dtype
        if col_type != object and not pd.api.types.is_categorical_dtype(df_original[col]):
            c_min = df_original[col].min()
            c_max = df_original[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df_original[col] = df_original[col].astype(np.int8)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df_original[col] = df_original[col].astype(np.float32)
        elif col_type == object:
            df_original[col] = df_original[col].astype('category')
    return df_original

# 2. LOGIQUE MÉDICALE ET GESTION DU LEAKAGE
##################################################################
def classifier_gravite(row):
    """
    Définit les 3 classes de diagnostic : 
    0: Sain, 1: Appendicite simple, 2: Appendicite grave.
    """
    if str(row['Diagnosis']).lower() == 'no appendicitis':
        return 0
    elif str(row['Severity']).lower() == 'uncomplicated':
        return 1
    else:
        return 2

def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les colonnes contenant des informations post-diagnostic
    ou des identifiants administratifs inutiles pour l'apprentissage.
    """
    cols_to_exclude = [
        'Diagnosis', 'Diagnosis_Presumptive', 'Severity', 'Management', 
        'Length_of_Stay', 'Admitted', 'us_number', 'Patient_ID', 'Date'
    ]
    return df.drop(columns=cols_to_exclude, errors='ignore')

# 3. ÉTAPES DE PRÉTRAITEMENT INDIVIDUELLES
##################################################################
def handle_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    """
    Imputation des valeurs manquantes : médiane pour le numérique, 
    mode pour les variables catégorielles.
    """
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()

    imputer_num = SimpleImputer(strategy='median')
    if num_cols:
        X[num_cols] = imputer_num.fit_transform(X[num_cols])

    imputer_cat = SimpleImputer(strategy='most_frequent')
    if cat_cols:
        X[cat_cols] = imputer_cat.fit_transform(X[cat_cols])
    return X

def scale_and_encode(X: pd.DataFrame) -> pd.DataFrame:
    """
    Normalisation des données numériques et encodage des variables textuelles.
    Nettoie les noms de colonnes pour la compatibilité avec les modèles de Gradient Boosting.
    """
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()

    # Mise à l'échelle (Standardisation)
    scaler = StandardScaler()
    if num_cols:
        X[num_cols] = scaler.fit_transform(X[num_cols])

    # Encodage (Binaire ou One-Hot)
    le = LabelEncoder()
    for col in cat_cols:
        if X[col].nunique() <= 2:
            X[col] = le.fit_transform(X[col])
        else:
            dummies = pd.get_dummies(X[col], prefix=col)
            # Nettoyage des caractères spéciaux pour Git et LightGBM
            dummies.columns = [re.sub(r'[\[\]{}<>,:"]', '_', str(c)) for c in dummies.columns]
            X = pd.concat([X, dummies], axis=1)
            X.drop(columns=[col], inplace=True)
    return X

# 4. PIPELINE DE PRÉTRAITEMENT FINAL
##################################################################
def preprocess_pipeline(df_input: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Point d'entrée principal du module de traitement.
    Retourne les features (X) et la cible (y) prêtes pour l'entraînement.
    """
    # Optimisation initiale et création de la cible
    df_working = optimize_memory(df_input.copy())
    y = df_working.apply(classifier_gravite, axis=1).values
    
    # Nettoyage et suppression du Data Leakage
    X = drop_leakage_columns(df_working)
    
    # Traitement des valeurs manquantes
    X = handle_missing_values(X)
    
    # Normalisation et Encodage
    X = scale_and_encode(X)
    
    return X, y
