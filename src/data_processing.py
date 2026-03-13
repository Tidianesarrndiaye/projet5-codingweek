# src/data_processing.py

import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. OPTIMISATION DE LA MÉMOIRE 
##################################################################
def optimize_memory(fichier_original):
    """
    Cette fonction optimise l'utilisation de la RAM en ajustant les types de données.
    Elle convertit les formats 64-bit (volumineux) en formats plus petits (8/16/32-bit) 
    lorsque les valeurs numériques le permettent, sans perte de précision.
    """
    for col in fichier_original.columns:
        col_type = fichier_original[col].dtype
        
        # Vérification si la colonne est numérique
        if col_type != object and not pd.api.types.is_categorical_dtype(fichier_original[col]):
            c_min = fichier_original[col].min()
            c_max = fichier_original[col].max()
            
            # Optimisation des entiers (ex: Age, scores cliniques)
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    fichier_original[col] = fichier_original[col].astype(np.int8)
            # Optimisation des nombres à virgule (ex: CRP, Température)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    fichier_original[col] = fichier_original[col].astype(np.float32)
        
        # Conversion des objets textuels en type 'category' pour économiser la mémoire
        elif col_type == object:
            fichier_original[col] = fichier_original[col].astype('category')
            
    return fichier_original

# 2. LOGIQUE MÉTIER ET PRÉPARATION DES DONNÉES
##################################################################

def classifier_gravite(row):
    """
    DÉFINITION DE LA VARIABLE CIBLE (Target) :
    Cette fonction croise le diagnostic et la sévérité pour créer 3 classes distinctes :
    0 : Sain (Pas d'appendicite)
    1 : Simple (Appendicite non compliquée)
    2 : Grave (Appendicite compliquée)
    """
    if str(row['Diagnosis']).lower() == 'no appendicitis':
        return 0
    elif str(row['Severity']).lower() == 'uncomplicated':
        return 1
    else:
        return 2

def executer_pretraitement(fichier_original):
    """
    PIPELINE PRINCIPAL : Transforme les données brutes médicales en vecteurs mathématiques.
    """
    # Optimisation de la mémoire pour accélérer les calculs
    fichier_original = optimize_memory(fichier_original)
    
    #  Génération de la colonne cible (y) via la logique de gravité
    y = fichier_original.apply(classifier_gravite, axis=1).values

    #  GESTION DU DATA LEAKAGE (Fuite de données)
    # Suppression des colonnes qui contiennent déjà la réponse ou des informations
    # collectées APRÈS le diagnostic (ex: durée de séjour) pour ne pas fausser l'IA.
    colonnes_a_exclure = [
        'Diagnosis', 'Diagnosis_Presumptive', 'Severity', 'Management', 
        'Length_of_Stay', 'Admitted', 'us_number', 'Patient_ID', 'Date'
    ]
    X = fichier_original.drop(columns=colonnes_a_exclure, errors='ignore')

    #  Identification des types de variables pour un traitement différencié
    data_nbn = X.select_dtypes(include=['number']).columns.tolist()      # Variables numériques
    data_bn = X.select_dtypes(include=['category', 'object']).columns.tolist() # Variables textuelles

    #  IMPUTATION (Gestion des données manquantes)
    # Remplacement des valeurs vides par la médiane (numérique) ou la valeur la plus fréquente (texte).
    imputer_num = SimpleImputer(strategy='median')
    X[data_nbn] = imputer_num.fit_transform(X[data_nbn])

    imputer_cat = SimpleImputer(strategy='most_frequent')
    X[data_bn] = imputer_cat.fit_transform(X[data_bn])

    #  NORMALISATION 
    # Indispensable pour les modèles de type SVM : on ramène toutes les mesures à la même 
    # échelle (moyenne=0, écart-type=1) pour que les grandes valeurs n'écrasent pas les petites.
    scaler = StandardScaler()
    X[data_nbn] = scaler.fit_transform(X[data_nbn])

    #  ENCODAGE DES VARIABLES CATÉGORIELLES (Conversion texte -> nombre)
    le = LabelEncoder()
    for col in data_bn:
        # Encodage binaire (0 ou 1) pour les variables à deux catégories (ex: Sexe)
        if X[col].nunique() <= 2:
            X[col] = le.fit_transform(X[col])
        # Encodage "One-Hot" pour les variables à multiples catégories
        else:
            dummies = pd.get_dummies(X[col], prefix=col)
            # Nettoyage des noms de colonnes pour la compatibilité JSON/LightGBM
            dummies.columns = [re.sub(r'[\[\]{}<>,:"]', '_', str(c)) for c in dummies.columns]
            X = pd.concat([X, dummies], axis=1)
            X.drop(columns=[col], inplace=True)
            
    return X, y
