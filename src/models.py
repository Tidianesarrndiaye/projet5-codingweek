from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def get_models():
    models = {
        "svm_rbf": SVC(kernel="rbf", probability=True, class_weight=None, random_state=42),
        "rf": RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=42),
        "lgbm": LGBMClassifier(n_estimators=600, learning_rate=0.03, random_state=42)
        # "catboost": CatBoostClassifier(verbose=False, random_state=42)  # optionnel
    }
    return models

def get_param_grids():
    grids = {
        "svm_rbf": {
            "model__C": [0.5, 1, 3],
            "model__gamma": ["scale", 0.05, 0.01]
        },
        "rf": {
            "model__n_estimators": [300, 500],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_split": [2, 5]
        },
        "lgbm": {
            "model__n_estimators": [400, 700],
            "model__num_leaves": [31, 63],
            "model__max_depth": [-1, 8],
            "model__min_child_samples": [20, 40]
        },
        "catboost": { "model__depth":[6,8], "model__learning_rate":[0.03,0.1], "model__iterations":[400,800] }
    }
    return grids