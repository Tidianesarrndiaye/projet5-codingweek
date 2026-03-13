# Copilot Workspace Instructions

## Project Overview

**Pediatric Appendicitis Clinical Decision Support System**  
A Streamlit web app that predicts appendicitis probability from clinical variables and explains predictions with SHAP. Built with scikit-learn pipelines, LightGBM, and a full ML training pipeline.

---

## Architecture

```
data/raw/          → Raw Excel data (app_data.xlsx)
data/processed/    → features_and_target.csv (output of eda.ipynb)
src/               → ML pipeline modules
notebooks/eda.ipynb→ EDA + feature selection + exports processed data
artifacts/         → Serialized models (*.joblib) and ROC curves
app/app.py         → Streamlit UI (loads artifacts, accepts input, shows SHAP)
tests/             → pytest tests
```

**Data pipeline**: `app_data.xlsx` → `eda.ipynb` → `features_and_target.csv` → `train_model.py` → `artifacts/*.joblib` → `app.py`

---

## Key Conventions

- **Random state**: always `42`
- **Serialization**: `joblib` (never `pickle`) for sklearn pipelines
- **Model pipeline**: always `sklearn.Pipeline` with steps `preproc` (ColumnTransformer) and `model`
- **Memory optimization**: call `optimize_memory()` from `src/data_processing.py` after any DataFrame load
- **SHAP**: must pass `X_trans = pipeline.named_steps["preproc"].transform(X)` to the explainer, not raw X
- **Feature names**: preserve feature names through ColumnTransformer so SHAP bar plots are labeled
- **Evaluation metric**: primary metric is `roc_auc`; always report accuracy, precision, recall, F1 alongside

---

## Commands

```bash
# Run Streamlit app
streamlit run app/app.py

# Train all models (requires data/processed/features_and_target.csv)
python src/train_model.py

# Run tests
pytest -q

# Activate virtual environment (Windows)
.venv\Scripts\Activate.ps1
```

---

## Web App (app/app.py)

- Framework: **Streamlit**
- Model loaded: `artifacts/lgbm.joblib` (sklearn Pipeline with `preproc` + `model` steps)
- Input: sidebar fields matching **exactly** the feature columns from `data/processed/features_and_target.csv`
- Prediction: `pipeline.predict_proba(X_user)[:, 1][0]`
- SHAP: use `shap.Explainer(clf, X_trans)` where `clf = pipeline.named_steps["model"]`
- Display: probability as `st.metric`, SHAP as `st.pyplot(shap.plots.bar(...))`
- Style: custom CSS loaded from `app/style.css`

**When adding new input fields**, they must match the column names and dtypes from `features_and_target.csv` exactly. Use `pd.DataFrame([{...}])` to build `X_user` with all required columns.

---

## ML Modules

| Module | Responsibility |
|---|---|
| `src/data_processing.py` | Missing values, outliers (IQR clip), encoding, `optimize_memory()`, `train_test_prepare()` |
| `src/models.py` | Model + hyperparameter grid definitions |
| `src/train_model.py` | GridSearchCV loop, saves `*.joblib` artifacts |
| `src/evaluate.py` | Metrics, ROC-AUC, ROC curve plots |
| `src/features.py` | Feature selection: mutual information, RFE, correlation filtering (threshold 0.9) |
| `src/explainability.py` | SHAP wrapper (currently empty — expand here) |

---

## Tests

- Test runner: `pytest`
- Test files: `tests/test_data_processing.py`, `tests/test_inference.py`, `tests/test_optimize_memory.py`
- CI: GitHub Actions (`.github/workflows/ci.yml`) runs on push/PR with Python 3.11
- Always add a test when implementing a new `src/` function

---

## Pitfalls

- `X_user` passed to the model must have **all columns** that were present during training — include columns with `0` as default if not collected from UI
- `shap.Explainer` needs the **classifier step** (`clf`), not the full pipeline
- `app.py` loads the model at **module import time** — if `artifacts/lgbm.joblib` is missing the app crashes on startup; guard with `try/except` or `st.error`
- `data/raw/app_data.xlsx` is not committed (large file) — the notebook exports `features_and_target.csv` which is committed
- Feature selection in `eda.ipynb` may change the feature list; retrain model if features change

---

## When Developing the Web App

1. Check `data/processed/features_and_target.csv` column names to know every required feature
2. Match input widget names exactly to feature column names
3. Build `X_user` DataFrame with all feature columns before calling `model.predict_proba()`
4. Keep SHAP explanation in sync: after changing features, retrain and export a new `lgbm.joblib`
5. Use `st.spinner()` around prediction + SHAP to prevent UI freeze
6. Never `st.rerun()` inside prediction block — use `st.session_state` for stateful UX
