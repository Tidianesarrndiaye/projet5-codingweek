import sys
import os

# Ajouter le dossier parent (la racine du projet) au PYTHONPATH

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from data_processing import optimize_memory, train_test_prepare
from models import get_models, get_param_grids
from evaluate import compute_metrics, plot_roc_curves

DATA_PATH = "data/processed/features_and_target.csv"
TARGET = "Diagnosis_no appendicitis"
MODEL_DIR = "artifacts"

import numpy as np
import pandas as pd

def report_low_variance(X, tol=1e-9):
    s = X.std(numeric_only=True)
    low_var = s[s <= tol].index.tolist()
    return low_var

# Exemple après ton preprocess:

def main():
    print()
    try:
        df = pd.read_csv(DATA_PATH)    
    except FileNotFoundError:
        print(f"File not found: {DATA_PATH}")
        return
    preproc, X_train, X_test, y_train, y_test = train_test_prepare(df, target=TARGET)
    low_var_cols = report_low_variance(X_train)
    print("Colonnes variance ~0:", low_var_cols[:20])
    models = get_models()
    grids = get_param_grids()
    results = []
    roc_items = []

    for name, clf in models.items():
        pipe = Pipeline(steps=[("preproc", preproc), ("model", clf)])
        param_grid = grids.get(name, {})
        if param_grid:
            search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, scoring="roc_auc")
            search.fit(X_train, y_train)
            best_est = search.best_estimator_
        else:
            pipe.fit(X_train, y_train)
            best_est = pipe

        y_prob = best_est.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = compute_metrics(y_test, y_prob, y_pred)
        metrics["model"] = name
        results.append(metrics)

        roc_items.append((name, y_test, y_prob))

        # Sauvegarde
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(best_est, f"{MODEL_DIR}/{name}.joblib")

    # Affiche résultats
    res_df = pd.DataFrame(results).sort_values(by="roc_auc", ascending=False)
    print(res_df)

    # ROC
    plot_roc_curves(roc_items)

    # Sauvegarde du meilleur modèle
    best_name = res_df.iloc[0]["model"]
    print(f"Training completed. Best model: {best_name}")
    best_model = joblib.load(f"{MODEL_DIR}/{best_name}.joblib")
    joblib.dump(best_model, f"{MODEL_DIR}/best_model.joblib")
if __name__ == "__main__":
    import os


    main()