import pandas as pd
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score

# Import de votre fonction d'optimisation (Exigence projet)
from data_processing import preprocess_pipeline

def train():
    print("🚀 Démarrage de l'entraînement...")
    
    # 1. Chargement des données
    data_path = 'data/raw/regensburg_pediatric_appendicitis.csv'
    if not os.path.exists(data_path):
        print(f"❌ Erreur : Le fichier {data_path} est introuvable.")
        return
    
    df = pd.read_csv(data_path)
    
    # 2. Prétraitement complet (Nettoyage + Optimisation mémoire)
    # Cette fonction doit être dans votre src/data_processing.py
    df_cleaned = preprocess_pipeline(df)
    
    # Séparation Cible/Features (La cible est 'appendicitis')
    X = df_cleaned.drop('appendicitis', axis=1)
    y = df_cleaned['appendicitis']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 3. Entraînement des 3 modèles requis
    models = {
        "random_forest": RandomForestClassifier(random_state=42),
        "svm": SVC(probability=True, random_state=42),
        "lightgbm": lgb.LGBMClassifier(random_state=42)
    }
    
    best_model = None
    best_score = 0
    results = {}

    for name, model in models.items():
        print(f"📦 Entraînement de : {name}...")
        model.fit(X_train, y_train)
        
        # Évaluation (Exigence : ROC-AUC)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_pred_proba)
        results[name] = {"roc_auc": score}
        
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    # 4. Sauvegarde des fichiers nécessaires (Exigence projet)
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Sauvegarde du meilleur modèle
    joblib.dump(best_model, 'models/best_model.pkl')
    
    # Sauvegarde du schéma des colonnes pour l'application Streamlit
    with open('models/feature_schema.json', 'w') as f:
        json.dump(list(X.columns), f)
        
    # Sauvegarde des résultats pour le rapport final
    with open('reports/results.json', 'w') as f:
        json.dump(results, f)

    print(f"✅ Terminé ! Meilleur modèle : {best_name} (ROC-AUC: {best_score:.4f})")
    print("📂 Fichiers créés : models/best_model.pkl et reports/results.json")

# IMPORTANT : Ce bloc permet au script de s'exécuter quand on tape la commande
if __name__ == "__main__":
    train()

