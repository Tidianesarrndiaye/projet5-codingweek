from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

def get_model_pipeline(model_type="lgbm", class_weights=None):
    """
    Initialise les modèles avec les paramètres optimisés.
    """
    
    if model_type == "svm":
        # Ton SVM optimisé (C=10, gamma=0.01)
        # class_weight='balanced' aide à mieux détecter les cas graves (classe 2)
        model = SVC(
            kernel='rbf', 
            C=10, 
            gamma=0.01, 
            class_weight='balanced', 
            probability=True,
            random_state=42
        )
        
    elif model_type == "rf":
        # Random Forest classique pour comparaison
        model = RandomForestClassifier(
            n_estimators=100, 
            class_weight='balanced', 
            random_state=42
        )
        
    elif model_type == "lgbm":
        # Ton modèle le plus performant (88.54%)
        model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            class_weight=class_weights if class_weights else 'balanced',
            random_state=42
        )
    
    else:
        raise ValueError(f"Modèle {model_type} non supporté.")
        
    return model

def train_model(model, X_train, y_train):
    """
    Entraîne le modèle sélectionné sur les données fournies.
    """
    model.fit(X_train, y_train)
    return model

def predict_model(model, X_test):
    """
    Génère les prédictions et les probabilités pour l'évaluation.
    """
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    return preds, probs
