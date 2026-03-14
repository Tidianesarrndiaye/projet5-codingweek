# Projet 5 – Coding Week (Structure minimale)

Présentation du Projet
Ce projet s'inscrit dans le cadre du développement d'outils d'aide au diagnostic médical de pointe, en exploitant le jeu de données pédiatriques de Regensburg. Il mobilise des techniques avancées de Machine Learning et de Deep Learning pour assister les cliniciens dans l'identification de l'appendicite chez l'enfant. En combinant l'analyse de données biologiques et l'imagerie médicale, cette initiative cherche à apporter une réponse technologique fiable à un défi de santé publique complexe, où la rapidité de décision est cruciale.

Description Technique
Le cœur du projet repose sur une approche multimodale innovante qui traite simultanément deux types de flux d'informations. D'une part, une branche du modèle analyse les données tabulaires (résultats d'examens de sang, constantes cliniques, âge), préalablement nettoyées et normalisées. D'autre part, une branche spécialisée dans le traitement d'images (CNN) extrait les caractéristiques pertinentes des échographies abdominales. Ces deux sources sont ensuite fusionnées pour permettre au système de pondérer les signes cliniques et visuels avant de livrer une prédiction finale.

Objectifs et Impact
L'objectif principal est d'améliorer la précision du diagnostic par rapport aux scores cliniques traditionnels, souvent sujets à une certaine variabilité. Le projet vise à minimiser les erreurs de diagnostic, réduisant ainsi le nombre d'interventions chirurgicales inutiles tout en garantissant une prise en charge rapide des cas critiques. À terme, l'ambition est de fournir un modèle robuste, capable de généraliser ses prédictions malgré des données parfois manquantes ou hétérogènes, afin d'offrir un support fiable aux équipes médicales en milieu hospitalier.
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

### Workflow leakage-safe
L'ordre de préparation a été corrigé pour éviter la fuite de données :
- définition de la cible `Diagnosis` et retrait des colonnes à exclure
- split train/test stratifié
- prétraitement ajusté sur le train uniquement
- sélection de features sur le train prétraité
- application des mêmes transformations au test

Le notebook [notebooks/eda.ipynb](notebooks/eda.ipynb) exporte désormais un dataset tabulaire leakage-safe de `780` lignes et `16` colonnes : `15` features sélectionnées + la cible `Diagnosis`.

### Valeurs manquantes — `handle_missing_values(df)`
Nous avons analysé les valeurs manquantes et les avons traitées en utilisant :
- Colonnes numériques → remplacées par la médiane
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
