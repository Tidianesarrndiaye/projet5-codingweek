# src/data_processing.py

import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_pipeline(fichier_original):
    """
    Pipeline de traitement direct basé sur la structure validée.
    """
    # 1. CRÉATION DE LA CIBLE (y)
    #############################
    def classifier_gravite(row):
        diag = str(row['Diagnosis']).lower()
        sev = str(row['Severity']).lower()
        if diag == 'no appendicitis':
            return 0
        elif sev == 'uncomplicated':
            return 1
        else:
            return 2

    y = fichier_original.apply(classifier_gravite, axis=1).values

    # 2. NETTOYAGE ET EXCLUSION DU LEAKAGE
    ######################################
    colonnes_a_exclure = [
        'Diagnosis', 'Diagnosis_Presumptive', 'Severity', 'Management', 
        'Length_of_Stay', 'Admitted', 'us_number', 'Patient_ID', 'Date'
    ]
    X = fichier_original.drop(columns=[c for c in colonnes_a_exclure if c in fichier_original.columns])

    # 3. TRAITEMENT DES DONNÉES (X)
    ###############################
    data_nbn = X.select_dtypes(include=['number']).columns.tolist()
    data_bn = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Imputation des valeurs manquantes
    if data_nbn:
        X[data_nbn] = SimpleImputer(strategy='median').fit_transform(X[data_nbn])
    if data_bn:
        X[data_bn] = SimpleImputer(strategy='most_frequent').fit_transform(X[data_bn])

    # Normalisation (StandardScaler pour le SVM)
    if data_nbn:
        X[data_nbn] = StandardScaler().fit_transform(X[data_nbn])

    # Encodage des données textuelles
    le = LabelEncoder()
    for col in data_bn:
        if X[col].nunique() <= 2:
            X[col] = le.fit_transform(X[col])
        else:
            dummies = pd.get_dummies(X[col], prefix=col)
            # Nettoyage des noms pour la compatibilité Git/LightGBM
            dummies.columns = [re.sub(r'[\[\]{}<>,:"]', '_', str(c)) for c in dummies.columns]
            X = pd.concat([X, dummies], axis=1)
            X = X.drop(columns=[col])

    # 4. OPTIMISATION MÉMOIRE FINALE
    ################################
    for col in X.columns:
        col_type = X[col].dtype
        if col_type != object:
            c_min, c_max = X[col].min(), X[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    X[col] = X[col].astype(np.int8)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    X[col] = X[col].astype(np.float32)

    return X, y
