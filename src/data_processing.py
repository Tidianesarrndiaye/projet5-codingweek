# src/data_processing.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder


# OPTIMISATION (MÉMOIRE)
def optimize_memory(df):
    df_local = df.copy()
  
    for col in df_local.select_dtypes(include=["int64", "int32"]).columns:
        df_local[col] = pd.to_numeric(df_local[col], downcast="integer")
    for col in df_local.select_dtypes(include=["float64", "float32"]).columns:
        df_local[col] = pd.to_numeric(df_local[col], downcast="float")
 
    for col in df_local.select_dtypes(include=["object"]).columns:
        nunique = df_local[col].nunique(dropna=False)
        if 1 < nunique < (0.5 * len(df_local)):
            df_local[col] = df_local[col].astype("category")
    return df_local


# NETTOYAGE

path = r"/content/app_data.xlsx"
fichier_original = pd.read_excel(path, decimal=',')

# 3 classes (0:Sain, 1:Simple, 2:Grave)
def classifier_gravite(row):
    if str(row['Diagnosis']).lower() == 'no appendicitis':
        return 0
    elif str(row['Severity']).lower() == 'uncomplicated':
        return 1
    else:
        return 2

y = fichier_original.apply(classifier_gravite, axis=1).values

# Suppression des colonnes pas très pertinente pour l'entrainement
colonnes_a_exclure = [
    'Diagnosis', 'Diagnosis_Presumptive', 'Severity', 'Management', 
    'Length_of_Stay', 'Admitted', 'us_number', 'Patient_ID', 'Date'
]
X = fichier_original.drop(columns=colonnes_a_exclure, errors='ignore')


X = optimize_memory(X)

# PRÉTRAITEMENT 
data_nbn = X.select_dtypes(include=['number']).columns.tolist()
# On inclut 'category' car l'optimisation a pu changer les types
data_bn = X.select_dtypes(include=['object', 'category']).columns.tolist()

# on remplace les vides par l'image du point sur la mediane totale
imputer_num = SimpleImputer(strategy='median')
X[data_nbn] = imputer_num.fit_transform(X[data_nbn])

imputer_cat = SimpleImputer(strategy='most_frequent')
X[data_bn] = imputer_cat.fit_transform(X[data_bn])

# Normalisation 
scaler = StandardScaler()
X[data_nbn] = scaler.fit_transform(X[data_nbn])

# Encodage des données
le = LabelEncoder()
for col in data_bn:
    if X[col].nunique() <= 2:
        X[col] = le.fit_transform(X[col].astype(str))
    else:
        dummies = pd.get_dummies(X[col], prefix=col)
        X = pd.concat([X, dummies], axis=1)
        X.drop(columns=[col], inplace=True)
