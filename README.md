# 🏥 I. Présentation du Projet
Ce projet s'inscrit dans le cadre du développement d'outils d'aide au diagnostic médical de pointe, en exploitant le jeu de données pédiatriques de Regensburg. Il mobilise des techniques avancées de Machine Learning et de Deep Learning pour assister les cliniciens dans l'identification de l'appendicite chez l'enfant. En combinant l'analyse de données biologiques et l'imagerie médicale, cette initiative cherche à apporter une réponse technologique fiable à un défi de santé publique complexe, où la rapidité de décision est cruciale.

# 🧬 II. Description du Projet
Le cœur du projet repose sur une approche multimodale innovante qui traite simultanément deux types de flux d'informations. D'une part, une branche du modèle analyse les données tabulaires (résultats d'examens de sang, constantes cliniques, âge), préalablement nettoyées et normalisées. D'autre part, une branche spécialisée dans le traitement d'images (CNN) extrait les caractéristiques pertinentes des échographies abdominales. Ces deux sources sont ensuite fusionnées pour permettre au système de pondérer les signes cliniques et visuels avant de livrer une prédiction finale.

# 🎯 III. Objectifs du Projet
- Améliorer la précision du diagnostic par rapport aux scores cliniques traditionnels.
- Réduire la variabilité liée aux méthodes d’évaluation classiques.
- Minimiser les erreurs de diagnostic.
- Diminuer le nombre d’interventions chirurgicales inutiles.
- Assurer une prise en charge rapide et efficace des cas critiques.
- Développer un modèle robuste capable de généraliser ses prédictions même avec des données manquantes ou hétérogènes.
- Fournir un support fiable aux équipes médicales en milieu hospitalier.
## 🧩 Documentation du Prompt Engineering
- Documenter les prompts utilisés pour au moins une tâche principale et expliquer leur efficacité.
- Document prêt à rendre: [PROMPT_ENGINEERING.md](PROMPT_ENGINEERING.md)

---

# Repository Structure

```
projet5-codingweek/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── models.py
│   ├── train_model.py
│   └── evaluate.py
│
├── app/
│   └── app.py
│
├── tests/
│   ├── test_sample.py
│   ├── test_data_processing.py
│   └── test_inference.py
│
├── data/
│   ├── raw/
│   │   ├── US_Pictures/
│   │   │   └── B-mode ultrasound images named as subject #>.<view #> *.bmp
│   │   ├── app_data.xlsx  tabular data with 53 features and 782 Instances
│   │   ├── regensburg_pediatric_appendicitis.csv derived from app_data.xlsx
│   │   ├── multiple_in_one: a list of US image names containing multiple snapshots
│   │
│   └── processed/
│       └── (sorties prétraitées)
│
├── models/
│   └── (modèle entraîné best_model.pkl)
│
└── reports/
    └── (graphs, SHAP visuals, results.json)
```

---

#DATA_PROCESSING
🏥 Analyse du Processus de Préparation des Données
Nature et Structure du Jeu de Données
Pour ce projet, nous travaillons sur le dataset de Regensburg, qui compile des dossiers médicaux de patients pédiatriques suspectés d'appendicite. Le fichier source se présente sous la forme d'un tableau complexe regroupant des données hétérogènes : des informations démographiques (âge, sexe), des mesures cliniques (température, douleur) et, surtout, des analyses biologiques détaillées (taux de CRP, numération globulaire comme les leucocytes ou neutrophiles). Ces données présentent deux défis majeurs : une forte proportion de valeurs manquantes (certains tests n'ayant pas été effectués pour tous les patients) et une grande disparité d'échelles entre les variables, rendant indispensable un nettoyage approfondi avant toute phase d'apprentissage.

Stratégie de Traitement et de Nettoyage
Nous avons mis en place un pipeline de traitement rigoureux pour transformer ces relevés bruts en données exploitables par nos algorithmes de Machine Learning. Dans un premier temps, nous procédons à l'élimination des colonnes trop incomplètes qui pourraient fausser les prédictions. Ensuite, nous appliquons une stratégie d'imputation systématique : pour les variables numériques, nous privilégions la médiane afin de limiter l'influence des valeurs extrêmes, tandis que pour les variables catégorielles, nous utilisons le mode. Cette étape est cruciale pour conserver l'intégrité du dataset tout en offrant au modèle une vue complète de chaque patient.

Gestion des Valeurs Aberrantes et Normalisation
Le contexte médical pédiatrique implique souvent des variations physiologiques importantes. Pour traiter les données biologiques sans perdre d'informations vitales, nous utilisons une méthode de "clipping" basée sur l'Écart Interquartile (IQR). Cela permet de plafonner les valeurs hors normes sans supprimer d'individus, ce qui est essentiel vu la taille finie du jeu de données. Enfin, nous appliquons une standardisation (StandardScaler) pour que chaque caractéristique (ex: âge vs nombre de globules blancs) pèse de manière équitable dans le calcul du modèle, garantissant ainsi une convergence optimale des algorithmes.

Prévention de la Fuite de Données (Data Leakage)
Une attention particulière a été portée à la séparation des phases d'ajustement (fit) et de transformation (transform). En calculant tous les paramètres de préparation (médianes, bornes IQR, moyennes) exclusivement sur l'ensemble d'entraînement, nous nous assurons que les données de test restent totalement "neuves" pour le modèle. Cette approche garantit que les performances mesurées reflètent la capacité réelle du modèle à généraliser sur de futurs patients hospitaliers, évitant ainsi toute surestimation des résultats.


### Optimisation mémoire — `optimize_memory(df)`

Exigence du projet :

- downcasting `int64` → `int32` / `int16` / `int8`
- downcasting `float64` → `float32`
- conversion en `category` lorsque pertinent

Permet de réduire la taille du DataFrame et d'améliorer les performances.

## 5. Pipeline complet — `preprocess_pipeline(df)`

Applique automatiquement toutes les étapes de prétraitement :

```
missing values → outliers → encodage → optimisation mémoire
```

Retourne un DataFrame final prêt pour l’entraînement du modèle.

### Équilibre des classes
Le jeu de données est approximativement équilibré :
- ~60 % appendicite
- ~40 % non-appendicite

Par conséquent, aucune technique de suréchantillonnage n’a été appliquée.

### Corrélation des variables
Une matrice de corrélation a été utilisée pour identifier les variables fortement corrélées.


## 4. Run the Training Script

```bash
python src/train_model.py
```

Le script entraîne actuellement les modèles sur `data/processed/features_and_target.csv` avec la cible `Diagnosis`.

---

# 5. What Happens Internally When You Train

Your `train_model.py` script performs these steps automatically.

### 1. Load the dataset
Validates that the file exists and loads it into a Pandas DataFrame.

### 2. Split the dataset

```python
train_test_split(..., test_size=0.2, stratify=y)
```

### 3. Fit preprocessing on the training data

Le préprocessing est porté par un `Pipeline` sklearn et n'est ajusté que sur `X_train` au moment du `fit()`.

### 4. Train 3 models

- SVM
- Random Forest
- LightGBM (if installed)
- CatBoost (optional)

Evaluation metrics:

- ROC-AUC
- Accuracy
- Precision
- Recall
- F1-score

### 5. Select the best model

Based on:

- ROC-AUC (priority)
- F1-score (fallback)

### 6. Save outputs

| File | Location | Description |
|-----|-----|-----|
| Final model | `artifacts/best_model.joblib` | Used by Streamlit app |
| Model artifacts | `artifacts/*.joblib` | Saved estimators |
| ROC plots | `artifacts/roc_curves_*.png` | ROC curves by model |

---

## 6. Résultats actuels

Dataset utilisé après sélection de features :
- `780` patients
- `15` features retenues
- cible `Diagnosis`
- répartition des classes : `463` appendicites, `317` non-appendicites

Résultats obtenus avec `python src/train_model.py` :

| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-----|-----:|-----:|-----:|-----:|-----:|
| `svm_rbf` | `0.9359` | `0.9462` | `0.9462` | `0.9462` | `0.9904` |
| `lgbm` | `0.9487` | `0.9474` | `0.9677` | `0.9574` | `0.9898` |
| `rf` | `0.9551` | `0.9574` | `0.9677` | `0.9626` | `0.9853` |

Le meilleur modèle selon le critère principal `roc_auc` est actuellement `svm_rbf` avec un score de `0.9904`.

---

## 🎉 7. How to Verify Training Succeeded

You should see messages like:

```
Metrics: {...}
Training completed. Best model (by roc_auc): svm_rbf
```

And the following files will be created:

```
artifacts/best_model.joblib
artifacts/rf.joblib
artifacts/svm_rbf.joblib
artifacts/lgbm.joblib
```

---

## 8. Run the Streamlit App

```bash
streamlit run app/app.py
```

This runs the web interface required by the project:

- choose a model in the sidebar
- clinician inputs
- predictions
- SHAP visualizations

---

Example:
- high CRP → increases appendicitis probability
- high leukocyte count → strong positive impact

## 8. Application Interface

A user interface was built using:

- Streamlit

The interface allows doctors to input patient symptoms and receive:

- predicted diagnosis
- explanation of prediction

## 9. Technologies Used

- Python
- Scikit-learn
- lightgbm
- SHAP
- Pandas
- Streamlit
- GitHub

## 10. Conclusion

Machine learning models can assist physicians in diagnosing appendicitis while maintaining transparency through explainable AI techniques.
