# Projet 5 – Coding Week (Structure minimale)

Ce dépôt contient la structure minimale demandée pour démarrer le projet.

## Prochaines étapes
1. Initialiser Git
2. Créer le repo GitHub
3. Ajouter requirements.txt
4. Ajouter les premiers scripts et tests
5. Télécharger le dataset et commencer l’EDA

# Project Objectives (Coding Week – Project 5)

This document summarizes all official project objectives extracted from the project description PDF.

## 🎯 Main Project Goals
- Develop a robust and explainable machine learning model. 
- Ensure transparency of predictions using SHAP explainability. 
- Create an intuitive user interface (Streamlit or Flask). 
- Follow professional software development practices (GitHub, CI/CD automation). 
- Demonstrate prompt engineering by documenting AI-generated prompts and results. 

## 📊 Data Analysis Objectives
- Identify missing values and describe how they are handled. 
- Detect outliers and document the chosen correction method. 
- Evaluate dataset balance (~50/50 classes). Document any balancing technique used. 
- Analyze correlations and describe how correlated features are managed. 

## 🤖 Machine Learning Objectives
- Train and evaluate at least **three** models from the following: SVM, Random Forest, LightGBM, CatBoost. 
- Use ROC-AUC, accuracy, precision, recall, and F1-score to compare models. 
- Justify the selection of the best-performing model. 

## 💾 Memory Optimization Objective
- Implement `optimize_memory(df)` to reduce dataframe memory footprint by downcasting datatypes. 
- Demonstrate memory improvement before/after optimization in the notebook. 

## 🧠 SHAP Explainability Objectives
- Generate SHAP summary plots. 
- Provide interpretable visualizations showing feature importance. 

## 🖥️ Interface Development Objectives
- Build a Streamlit or Flask interface for clinicians. 
- Allow input of symptoms, demographic, and clinical data. 
- Display predictions and SHAP results in a clear way.

## 🧪 Software Engineering Objectives
- Create a professional GitHub repository (no forks).
- Include at least one automated test (e.g., missing values, memory optimization, model prediction).
- Automate tests using GitHub Actions CI workflow.
## 🗂️ Task Distribution Objectives
- Use Jira to manage tasks: To Do, In Progress, Review, Done.

## 🧩 Prompt Engineering Documentation
- Document prompts used for at least one core task and explain their effectiveness.




## Repository Structure
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
│   └── (vide ou .keep pour le moment)
│
├── src/
│   └── (plus tard: data_processing.py, train_model.py)
│
├── app/
│   └── (plus tard: app.py)
│
├── tests/
│   ├── test_sample.py
│   └── (d’autres tests viendront après)
│
├── data/
│   ├── raw/
│   │   └── (dataset brut .csv)
│   └── processed/
│       └── (sorties prétraitées)
│
├── models/
│   └── (modèle entraîné best_model.pkl)
│
└── reports/
    └── (graphs, SHAP visuals, results.json)
``` 
## 🚀 How to Run the Project
### 1.Clone the repository with Github Desktop
Open GitHub Desktop
Click File → Clone repository: the `projet5-coding-week` repo
Choose a location on your computer

###  <p id="install-dependencies"> 2. Install Dependencies</p>
Before running any code, create and activate a virtual environment:

#### Windows (Git Bash)
```bash
python -m venv .venv
source .venv/Scripts/activate
```
You shoukd see (.venv) up to your current director

#### MacOS / Linux
``` shell 
python3 -m venv .venv
source .venv/bin/activate
```

Then install the required packages(Git Bash in windows/ terminal in MacOS / Linux):
``` bash
pip install -r requirements.txt
```
### 3. Train the Model
To train the machine learning model, make sure the dataset is placed in:
``` 
data/raw/<your_dataset>.csv
```
replace `<your_dataset>.csv` with the file name of the dataset 

#### 1. Place the Dataset in the Correct Folder
Download the dataset from UCI (as required in the project instructions):
[dataset](https://archive.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis)
Then put the `dataset` inside: `projet5-codingweek/data/raw/`

Example:
projet5-codingweek/data/raw/regensburg_pediatric_appendicitis.csv


#### 2. Activate Your Virtual Environment
Windows + Git Bash:
```bash
source .venv/Scripts/activate
```

Mac/Linux:
```shell
source .venv/bin/activate
```

confirm it work:
``` shell
which python
```
#### 3. Install dependencies if not

[see this scope](#install-dependencies)
#### 4. Run the Training Script
Use this command (replace the CSV name if yours is different):
``` shell
python src/train_model.py \
  --input data/raw/regensburg_pediatric_appendicitis.csv \
  --target appendicitis \
  --out models/best_model.pkl
```


### 5. What Happens Internally When You Train
Your train_model.py script performs these steps automatically:
1. Load the dataset
It validates that the file exists and loads it into a Pandas DataFrame. (Code in the file you created.)
2. Preprocess the features
Using your preprocess_pipeline() function:

missing values handling
outlier clipping
one‑hot encoding
memory optimization (required by the project) 1

3. Split the dataset
```shell
train_test_split(..., test_size=0.2, stratify=y)
``` 
4. Train at least 3 models
As required by the project guidelines:
- SVM
- Random Forest
- LightGBM (if installed)
- CatBoost (optional)

The training script compares models using:

ROC‑AUC
Accuracy
Precision
Recall
F1‑Score1

5. Select the best model
Based on:

- ROC‑AUC (priority)
- F1-score (fallback)

6. Save outputs
The script saves:

| File | Location | Description |
|------|----------|-------------|
| Final model | `models/best_model.pkl` | Used by Streamlit app |
| Feature schema | `models/feature_schema.json` | Ensures app input columns match training columns |
| Evaluation metrics | `reports/results.json` | Required for documentation |

🎉 6. How to Verify Training Succeeded
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

### 6. Run the Streamlit App
After training, you can run the Streamlit app to input patient data and see predictions:
``` shell 
streamlit run app/app.py
```

This runs the web interface required by the project:

- clinician inputs
- predictions
- SHAP visualizations (next steps)