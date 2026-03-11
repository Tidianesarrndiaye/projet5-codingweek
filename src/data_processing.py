import pandas as pd
import numpy as np


def handle_outliers(df):
    """
    Handle outliers in the DataFrame by replacing values that are outside of 1.5 times the interquartile range (IQR) with the median of the column.
    Returns the DataFrame with outliers handled. 
    """
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            median = df[column].median()
            df[column] = df[column].apply(lambda x: median if x < lower_bound or x > upper_bound else x)
    return df


def encode_features(df):
    """
    Encode categorical features in the DataFrame using one-hot encoding."""
    return pd.get_dummies(df, columns=[col for col in df.columns if df[col].dtype == 'object'])


def handle_missing_values(df):
    """
    On suppose que les outliers ont déjà été traités.
    Gère les missing values dans le DataFrame en les remplaçant par:
    - la moyenne pour les colonnes numériques 
    - le mode pour les colonnes catégorielles.
    Returns the DataFrame with missing values handled. 
    
    """
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            df[column].fillna(df[column].mean(), inplace=True)
        else:
            df[column].fillna(df[column].mode()[0], inplace=True)
    return df






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
    df = handle_outliers(df)
    df = encode_features(df)
    df = handle_missing_values(df)
    df = optimize_memory(df)
    return df
    