# src/data_processing.py

import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. OPTIMISATION MÉMOIRE
def optimize_memory(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    return df

# 2. CLASSIFICATION MÉDICALE
def classifier_gravite(row):
    diag = str(row['Diagnosis']).lower()
    sev = str(row['Severity']).lower()
    if diag == 'no appendicitis': return 0
    elif sev == 'uncomplicated': return 1
    else: return 2

# 3. PIPELINE DE TRAITEMENT
def preprocess_pipeline(df_input):
    df = df_input.copy()
    
    # Création de la cible y
    y = df.apply(classifier_gravite, axis=1).values
    
    # Nettoyage du Leakage
    cols_drop = ['Diagnosis', 'Diagnosis_Presumptive', 'Severity', 'Management', 
                 'Length_of_Stay', 'Admitted', 'us_number', 'Patient_ID', 'Date']
    df = df.drop(columns=[c for c in cols_drop if c in df.columns])
    
    # Séparation Numérique / Catégoriel
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(exclude=['number']).columns
    
    # Imputation simple
    df[num_cols] = SimpleImputer(strategy='median').fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_cols])
    
    # Normalisation pour le SVM
    df[num_cols] = StandardScaler().fit_transform(df[num_cols])
    
    # Encodage final
    le = LabelEncoder()
    for col in cat_cols:
        if df[col].nunique() <= 2:
            df[col] = le.fit_transform(df[col])
        else:
            dummies = pd.get_dummies(df[col], prefix=col)
            # Nettoyage des noms pour éviter les erreurs Git/LightGBM
            dummies.columns = [re.sub(r'[\[\]{}<>,:"]', '_', str(c)) for c in dummies.columns]
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)
            
    return optimize_memory(df), y
