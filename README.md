# Projet 5 – Coding Week (Structure minimale)

Ce dépôt contient la structure minimale demandée pour démarrer le projet.

## Prochaines étapes
1. Initialiser Git
2. Créer le repo GitHub
3. Ajouter requirements.txt
4. Ajouter les premiers scripts et tests
5. Télécharger le dataset et commencer l’EDA

---

# Objectifs du projet (Coding Week – Projet 5)

Ce document résume tous les objectifs officiels du projet extraits du PDF de description du projet.

## 🎯 Objectifs principaux du projet
- Développer un modèle de machine learning robuste et explicable.
- Garantir la transparence des prédictions grâce à l’explicabilité SHAP.
- Créer une interface utilisateur intuitive (Streamlit ou Flask).
- Suivre des pratiques professionnelles de développement logiciel (GitHub, automatisation CI/CD).
- Démontrer l’utilisation du prompt engineering en documentant les prompts générés par l’IA et leurs résultats.

## 📊 Objectifs d’analyse des données
- Identifier les valeurs manquantes et décrire comment elles sont traitées.
- Détecter les outliers et documenter la méthode de correction choisie.
- Évaluer l’équilibre du dataset (~50/50 classes). Documenter toute technique d’équilibrage utilisée.
- Analyser les corrélations et décrire comment les variables corrélées sont gérées.

## 🤖 Objectifs de machine learning
- Entraîner et évaluer au moins **trois** modèles parmi les suivants : SVM, Random Forest, LightGBM, CatBoost.
- Utiliser ROC-AUC, accuracy, precision, recall et F1-score pour comparer les modèles.
- Justifier le choix du modèle le plus performant.

## 💾 Objectif d’optimisation mémoire
- Implémenter `optimize_memory(df)` pour réduire l’empreinte mémoire du dataframe en convertissant les types de données.
- Montrer l’amélioration de la mémoire avant/après optimisation dans le notebook.

## 🧠 Objectifs d’explicabilité SHAP
- Générer des graphiques de résumé SHAP.
- Fournir des visualisations interprétables montrant l’importance des variables.

## 🖥️ Objectifs de développement de l’interface
- Construire une interface Streamlit ou Flask pour les cliniciens.
- Permettre la saisie des symptômes, des données démographiques et des données cliniques.
- Afficher les prédictions et les résultats SHAP de manière claire.

## 🧪 Objectifs d’ingénierie logicielle
- Créer un dépôt GitHub professionnel (pas de fork).
- Inclure au moins un test automatisé (par exemple : valeurs manquantes, optimisation mémoire, prédiction du modèle).
- Automatiser les tests avec un workflow CI GitHub Actions.

## 🗂️ Objectifs de répartition des tâches
- Utiliser Jira pour gérer les tâches : To Do, In Progress, Review, Done.

## 🧩 Documentation du Prompt Engineering
- Documenter les prompts utilisés pour au moins une tâche principale et expliquer leur efficacité.

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

# 🚀 How to Run the Project

## 1. Clone the repository with GitHub Desktop
Open GitHub Desktop  
Click **File → Clone repository** : the `projet5-coding-week` repo  
Choose a location on your computer

---

## 2.Install & Activate Your Virtual Environment  

Before running any code, create and activate a virtual environment.

### Windows (PowerShell)
``` shell
.\.venv\Scripts\Activate.ps1
```
You should see **(.venv)** before your current directory.
Si il y a un problème
```shell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Windows (Command Prompt)

``` shell
.\.venv\Scripts\Activate
```

You should see **(.venv)** before your current directory.
### Windows (Git Bash)

```bash
python -m venv .venv
source .venv/Scripts/activate
```

You should see **(.venv)** before your current directory.

### MacOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Confirm it works:

```bash
which python
```
## 3. Analyse exploratoire des données (EDA)

### Valeurs manquantes — `handle_missing_values(df)`
Nous avons analysé les valeurs manquantes et les avons traitées en utilisant :
- Colonnes numériques → remplacées par la moyenne
- Colonnes catégorielles → remplacées par le mode


### Valeurs aberrantes
Les valeurs aberrantes ont été détectées en utilisant :
- des boxplots
- la méthode IQR

Transforme toutes les colonnes catégorielles en variables numériques via **one-hot encoding**.



### Optimisation mémoire — `optimize_memory(df)`

Exigence du projet :

- downcasting `int64` → `int32` / `int16` / `int8`
- downcasting `float64` → `float32`
- conversion en `category` lorsque pertinent

Permet de réduire la taille du DataFrame et d'améliorer les performances.

## 5. Pipeline complet — `preprocess_pipeline(df)`

Applique automatiquement toutes les étapes dans l’ordre :

```
outliers → encodage → missing values → optimisation mémoire
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
python src/train_model.py \
  --input data/processed/features_and_target.csv
  --target diagnosis_no appendicitis \
  --out models/best_model.joblib
```

---

# 5. What Happens Internally When You Train

Your `train_model.py` script performs these steps automatically.

### 1. Load the dataset
Validates that the file exists and loads it into a Pandas DataFrame.

### 2. Preprocess the features

Using `preprocess_pipeline()`:

- missing values handling
- outlier clipping
- one-hot encoding
- memory optimization (required by the project)

### 3. Split the dataset

```python
train_test_split(..., test_size=0.2, stratify=y)
```

### 4. Train at least 3 models

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
| Final model | `models/best_model.joblib` | Used by Streamlit app |
| Evaluation metrics | `reports/results.json` | Required for documentation |

---

## 🎉 6. How to Verify Training Succeeded

You should see messages like:

```
Metrics: {...}
Training completed. Best model: random_forest
```

And the following files will be created:

```
models/best_model.joblib
reports/results.json
```

---

## 7. Run the Streamlit App

```bash
streamlit run app/app.py
```

This runs the web interface required by the project:

- choose a model in the sidebar
- clinician inputs
- predictions
- SHAP visualizations

---


## 5. Model Evaluation

Evaluation metrics used:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Results:

| Model | Accuracy |precision | recall | f1| ROC-AUC |
|------|------|------|
| Random Forest | 0.847 | 0.8448 | 0.765 | 0.803 | 0.949 |
| SVM | 0.828 | 0.930 | 0.625 |  0.747 | 0.937 |
| LightGBM | 0.834 | 0.8064 | 0.781 |  0.93 | 0.932 |

## 6. Selected Model

The best performing model was **Random Forest** due to:

- higher ROC-AUC
- stable performance
- better interpretability

## 7. Explainability with SHAP

We used SHAP values to explain predictions.

SHAP helps identify which features influence the model decision.

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
