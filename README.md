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

## 2. Install Dependencies

Before running any code, create and activate a virtual environment.

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

Then install the required packages:

```bash
pip install -r requirements.txt
```

---

## 3. Train the Model

To train the machine learning model, make sure the dataset is placed in:

```
data/raw/<your_dataset>.csv
```

Replace `<your_dataset>.csv` with the dataset file name.

### 1. Place the Dataset in the Correct Folder

Download the dataset from UCI:

https://archive.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis

Then put the dataset inside:

```
projet5-codingweek/data/raw/
```

Example:

```
projet5-codingweek/data/raw/regensburg_pediatric_appendicitis.csv
```

---

### 2. Activate Your Virtual Environment

Windows + Git Bash:

```bash
source .venv/Scripts/activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

Confirm it works:

```bash
which python
```

---

### 3. Install dependencies if needed

See section **Install Dependencies**

---

### 4. Run the Training Script

```bash
python src/train_model.py \
  --input data/raw/regensburg_pediatric_appendicitis.csv \
  --target appendicitis \
  --out models/best_model.pkl
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
| Final model | `models/best_model.pkl` | Used by Streamlit app |
| Feature schema | `models/feature_schema.json` | Ensures app input columns match training columns |
| Evaluation metrics | `reports/results.json` | Required for documentation |

---

# 🎉 6. How to Verify Training Succeeded

You should see messages like:

```
Training complete. Best model: random_forest
Metrics: {...}
```

And the following files will be created:

```
models/best_model.pkl
models/feature_schema.json
reports/results.json
```

---

# 7. Run the Streamlit App

```bash
streamlit run app/app.py
```

This runs the web interface required by the project:

- clinician inputs
- predictions
- SHAP visualizations

---

# Data Processing Module (`src/data_processing.py`)

Ce module contient toutes les étapes de prétraitement exigées par le projet.

## 1. Outliers — `handle_outliers(df)`
Traite les valeurs extrêmes avec la méthode **IQR (±1.5 × IQR)** puis remplace les outliers par la médiane.

## 2. Encodage — `encode_features(df)`
Transforme toutes les colonnes catégorielles en variables numériques via **one-hot encoding**.

## 3. Valeurs manquantes — `handle_missing_values(df)`

- Colonnes numériques → remplacées par la moyenne
- Colonnes catégorielles → remplacées par le mode

## 4. Optimisation mémoire — `optimize_memory(df)`

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
