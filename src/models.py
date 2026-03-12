import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

# Configuration visuelle
sns.set_theme(style="whitegrid")

#LightGBM refuse les caractères comme [ ] { } < >
print("Nettoyage des noms de colonnes")
X.columns = [re.sub(r'[\[\]{}<>,:"]', '_', str(col)) for col in X.columns]

# DÉCOUPAGE DES DONNÉES ENTRAINEMENT ET TEST

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f" Données: Train={X_train.shape[0]} patients, Test={X_test.shape[0]} patients")

# ENTRAÎNEMENT DES MODELES 

modeles = {
    "SVM (Optimisé)": SVC(kernel='rbf', C=10, gamma=0.01, class_weight='balanced', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    "LightGBM": lgb.LGBMClassifier(class_weight='balanced', random_state=42, importance_type='gain', verbosity=-1)
}

resultats = {}


print("ENTRAÎNEMENT")

for nom, model in modeles.items():
    
    
    # Apprentissage sur les classes 0, 1 et 2
    model.fit(X_train, y_train)
    
    # Prédiction
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    resultats[nom] = acc
    
    
    # Rapport détaillé 
    print(f"Rapport pour {nom} :")
    print(classification_report(y_test, y_pred, target_names=['Sain (0)', 'Simple (1)', 'Grave (2)']))


#  VISUALISATION DES RÉSULTATS 
plt.figure(figsize=(10, 6))
sns.barplot(x=list(resultats.keys()), y=list(resultats.values()), palette='magma')
plt.title('Performance globale des modèles (Accuracy)')
plt.ylabel('Précision')
for i, v in enumerate(resultats.values()):
    plt.text(i, v + 0.01, f"{v:.2%}", ha='center', fontweight='bold')
plt.show()

# Matrice de confusion des modèles
meilleur_nom = max(resultats, key=resultats.get)
meilleur_model = modeles[meilleur_nom]
y_pred_final = meilleur_model.predict(X_test)

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Sain', 'Simple', 'Grave'], 
            yticklabels=['Sain', 'Simple', 'Grave'])
plt.title(f'Matrice de Confusion : {meilleur_nom} (Meilleur modèle)')
plt.ylabel('Réalité')
plt.xlabel('Prédiction')
plt.show()

print(f"\nCONCLUSION : Le modèle retenu est {meilleur_nom}.")
