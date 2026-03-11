import pandas as pd
import numpy as np


def handle_outliers(df):
    """
    Handle outliers in the DataFrame by excluding:
    - rows where the value in a column is outside the interquartile range (IQR).
    Returns the DataFrame with outliers handled. 
    """
    df_local = df.copy()
    for column in df_local.columns:
        if df_local[column].dtype in ['float64', 'int64']:
            Q1 = df_local[column].quantile(0.25)
            Q3 = df_local[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR 
            upper_bound = Q3 + 1.5 * IQR
            df_local = df_local[(df_local[column] >= lower_bound) & (df_local[column] <= upper_bound)]
    return df_local
            
            

# Conversion des donnees categorical en numerique
def encode_features(df):
    """
    Encode categorical features in the DataFrame using one-hot encoding."""
    df_local = df.copy()
    for column in df_local.columns:
        if df_local[column].dtype == 'object':
            unique_values = df_local[column].unique()
            unique_values.sort()  # Tri des valeurs uniques pour une encodage cohérent
            mapping = {value: idx for idx, value in enumerate(unique_values)}
            df_local[column] = df_local[column].map(mapping)
    return df_local
            

def handle_missing_values(df):
    """
    On suppose que les outliers ont déjà été traités.
    Les colonnes avec plus de 5% de valeurs manquantes sont supprimées du DataFrame. 
    On considere que ces colonnes ne sont pas pertinentes pour l'analyse.
    Gère les missing values dans le DataFrame en les remplaçant par:
    - la moyenne pour les colonnes numériques 
    - le mode pour les colonnes catégorielles.
    Returns the DataFrame with missing values handled. 
    
    """
    df_local = df.copy()
    for column in df_local.columns:
        if df_local[column].isnull().sum() > 0:
            if df_local[column].dtype in ['float64', 'int64']:
                df_local[column] = df_local[column].fillna(df_local[column].mean())
            else:
                df_local[column] = df_local[column].fillna(df_local[column].mode()[0])
    return df_local






def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    df_opt = df.copy()
    for col in df_opt.columns:
        col_type = df_opt[col].dtype

        if pd.api.types.is_integer_dtype(col_type):
            cmin, cmax = df_opt[col].min(), df_opt[col].max()
            for dtype in [np.int8, np.int16, np.int32, np.int64]:
                if np.iinfo(dtype).min <= cmin and cmax <= np.iinfo(dtype).max:
                    df_opt[col] = df_opt[col].astype(dtype)
                    break

        elif pd.api.types.is_float_dtype(col_type):
            df_opt[col] = df_opt[col].astype(np.float32)

        elif pd.api.types.is_object_dtype(col_type):
            # Catégoriser si peu de modalités
            nunique = df_opt[col].nunique(dropna=False)
            if nunique / max(len(df_opt), 1) < 0.5:
                df_opt[col] = df_opt[col].astype('category')
    return df_opt

def  preprocess_pipeline(df) :
    """
    Pipeline de prétraitement des données qui inclut la gestion des outliers, l'encodage des features et la gestion des valeurs manquantes.
    Returns the preprocessed DataFrame. 
    """
    df_no_outliers = handle_outliers(df) 
    df_encode = encode_features(df_no_outliers)
    df_clean = handle_missing_values(df_encode)
    df_optim = optimize_memory(df_clean)
    return df_optim
    