# Prompt Engineering Documentation

## 1. Objective
This document summarizes how AI prompts were designed and iteratively improved during the development of the pediatric appendicitis clinical decision-support system.

Main goal:
- Produce reliable, explainable, and clinically interpretable outputs for data processing, model training, SHAP explainability, and Streamlit interface integration.

## 2. AI-Assisted Scope
Prompt engineering was used to support these core tasks:
- Data preprocessing strategy (missing values, outliers, encoding, memory optimization)
- Modeling pipeline design (multiple models, evaluation, selection)
- Explainability integration (SHAP in model and app)
- UI logic for clinician-facing prediction workflow
- Test and CI support

## 3. Prompt Log (Ready-to-Submit Examples)

| ID | Task | Initial Prompt | Improved Prompt | Why It Improved |
|----|------|----------------|-----------------|-----------------|
| P1 | Data preprocessing | "Clean this medical dataset." | "Build a robust preprocessing pipeline for pediatric appendicitis tabular data: handle missing values (numeric median, categorical mode), clip outliers with IQR, one-hot encode categoricals, and optimize memory. Return modular Python functions with sklearn compatibility and random_state=42 where relevant." | Added explicit methods, constraints, and expected format. Output became reproducible and directly integrable in src/data_processing.py. |
| P2 | Model training | "Train some classifiers and pick best." | "Implement sklearn Pipeline with ColumnTransformer (preproc + model), train SVM-RBF, RandomForest, and LightGBM using GridSearchCV(cv=5, scoring='roc_auc'), report accuracy/precision/recall/F1/ROC-AUC, and save each model + best_model via joblib." | Specified architecture, metrics, and serialization requirement. Reduced ambiguity and ensured compliance with project rubric. |
| P3 | SHAP explainability | "Add SHAP explanation to the app." | "Integrate SHAP for binary classification. Use transformed features from pipeline.named_steps['preproc'].transform(X) before explanation. Show per-patient explanation in Streamlit and return a readable bar plot with stable feature names." | Prevented a common SHAP misuse (raw X with preprocessed model). Improved transparency and interpretability quality. |
| P4 | Streamlit clinical UX | "Create a Streamlit app for prediction." | "Create a clinician-oriented Streamlit UI with patient inputs mapped to model features, probability output as risk level (low/moderate/high), safety disclaimer, and a separate action to generate SHAP explanation. Keep flow simple for rapid triage usage." | Added user context and decision logic, making the interface more practical for clinical support. |
| P5 | Testing and CI | "Set up tests and CI." | "Create pytest checks for preprocessing behavior, inference validity, and data integrity. In GitHub Actions, run pytest with JUnit output and upload test-results.xml as artifact on each push/PR." | Produced auditable QA evidence and aligned workflow with automated validation expectations. |

## 4. One Concrete Prompt Refinement Example

### Version A (too vague)
"Add explainability to my model."

### Version B (effective)
"Add SHAP explainability for this binary classifier pipeline. Use model = pipeline.named_steps['model'] and background/features transformed via pipeline.named_steps['preproc'].transform(X). Preserve feature names in output plots and generate a Streamlit-ready bar chart for one patient prediction."

### Outcome
- Fewer implementation errors
- Correct feature-space alignment
- Better explanation readability for non-technical users

## 5. Evaluation of Prompt Effectiveness

Criteria used:
- Technical correctness (works with existing pipeline)
- Reproducibility (clear constraints and deterministic settings)
- Explainability quality (clinically interpretable outputs)
- Integration readiness (code structure reusable in src/ and app/)

Observed effect:
- Specific prompts consistently produced higher-quality, production-ready code than generic prompts.

## 6. Limits and Mitigations
- Limitation: Generic prompts can create inconsistent code style and missing constraints.
- Mitigation: Include explicit requirements (metrics, architecture, file outputs, random_state=42, SHAP transformation path).

## 7. Reproducibility Note
To reproduce this workflow, reuse the improved prompts above in the same order (P1 -> P5), then validate with:
- pytest tests/
- Model artifact checks in artifacts/
- Streamlit manual run for prediction + SHAP display

## 8. Compliance Statement
Prompt engineering has been explicitly documented for core development tasks, with concrete prompt examples, refinement rationale, and measurable implementation impact.